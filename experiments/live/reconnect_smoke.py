"""Smoke test of the live reconnect protocol on the real services (#263).

A Python client that speaks the browser's audio protocol
(``transcriber/adapters/inbound/web/js/live-audio.js`` and
``lib/live-link.js``) streams a synthetic wav in real time into a live
session started through the same HTTP routes the page uses
(``POST /api/assist/sessions`` on the transcriber, which forwards to the
orchestrator), follows the session's events, and breaks its own audio
socket on a script:

* every 100 ms piece goes into an outbox with its absolute sample
  offset and stays there until the server's ``{"type": "ack"}``;
* the socket authenticates with ``{"type": "auth"}`` like a browser and
  waits for ``ready``; it sends from ``ready.received_samples`` on (the
  first piece cut at the exact sample) and says ``{"type":
  "caught_up"}`` once everything held is sent;
* a dropped socket is reopened with the page's backoff (0.5, 1, 2, 4,
  8, 8… s, ±20 % jitter) until the session ends; a socket that never
  says ready in 30 s, or says nothing for 20 s while audio waits for
  its ack, is closed and retried;
* «Terminar»: the capture stops, the client waits until everything
  held is acknowledged (at most 60 s of connected time), says
  ``{"type": "stop"}``, waits for ``stopped`` (at most 5 s), then stops
  the session over HTTP as the page does.

A fault is "the network is down from stream second ``at`` for
``seconds``": the audio socket's transport is aborted (no close frame,
as when a tunnel resets) and every reconnection attempt fails at once
until the network is back. The events socket and the suggestion
stream are left alone, so their arrival times stay exact.

Scenarios: ``a`` none (baseline); ``b`` one 3 s drop mid-speech; ``c``
one 60 s drop; ``d`` two drops (3 s and 10 s); ``e`` the network down
for the last 10 s of the wav and 3 s after it, so «Terminar» starts
with about 10 s pending.

Usage::

    python experiments/live/reconnect_smoke.py --base URL --code-file F \\
        --wav W --reference R --scenarios a,b,c,d,e --out RESULT.json

Prints and writes numbers only: counts, seconds, close codes, word
counts and normalised similarities, never transcript text. The access
code is read from ``--code-file`` and never printed. One session at a
time: when another live session is running the script says so and
stops without touching it.
"""

import argparse
import asyncio
import difflib
import json
import random
import re
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

RATE = 16000
PIECE_S = 0.1
BACKOFF_FIRST_S = 0.5
BACKOFF_MAX_S = 8.0
READY_TIMEOUT_S = 30.0
SILENT_S = 20.0
FINISH_CAP_S = 60.0
STOPPED_WAIT_S = 5.0
EVENTS_IDLE_S = 15.0
"""An events socket silent this long is reopened (partials come often)."""
LIVE_SLACK_S = 1.5
"""The link is live again when at most this much waits for its ack."""

SCENARIOS: Dict[str, str] = {
    "a": "baseline, no fault",
    "b": "one 3 s drop mid-speech",
    "c": "one 60 s drop",
    "d": "two drops: 3 s, then 10 s",
    "e": "network down for the last 10 s and 3 s after: stop with 10 s held",
}


def faults_for(name: str, seconds: float) -> List[Tuple[float, float]]:
    """Return the (stream second, outage seconds) faults of a scenario."""
    if name == "a":
        return []
    if name == "b":
        return [(30.0, 3.0)]
    if name == "c":
        return [(20.0, 60.0)]
    if name == "d":
        return [(20.0, 3.0), (60.0, 10.0)]
    if name == "e":
        return [(seconds - 10.0, 13.0)]
    raise ValueError(f"unknown scenario {name}")


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Return the ``q`` quantile (linear interpolation) or ``None``."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(values: Sequence[float]) -> Dict[str, Any]:
    """Return n, p50, p90 and max of ``values`` (rounded)."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x)

    return {
        "n": len(values), "p50": r(percentile(values, 0.5)),
        "p90": r(percentile(values, 0.9)),
        "max": r(max(values) if values else None),
    }


def words(text: str) -> List[str]:
    """Return the lower-case words of ``text``, accents and marks removed."""
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", plain)


def similarity(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """Return the matching ratio of two word lists (0..1, 3 decimals)."""
    if not a and not b:
        return None
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False)
                 .ratio(), 3)


@dataclass
class Piece:
    """Unacknowledged PCM: its first sample (per channel) and bytes."""

    offset: int
    samples: int
    data: bytes


@dataclass
class Drop:
    """One scripted outage and what the client saw around it."""

    at_stream: float
    seconds: float
    dropped_at: float = 0.0
    held_at_drop: int = 0
    ready_at: Optional[float] = None
    received_at_ready: Optional[int] = None
    end_at_ready: Optional[int] = None
    attempts: int = 0
    caught_up_sent_at: Optional[float] = None
    live_at: Optional[float] = None


@dataclass
class Link:
    """The browser's audio link: outbox, reconnection, resend, finish."""

    url: str
    code: str
    channels: int
    log: List[Dict[str, Any]] = field(default_factory=list)
    pieces: List[Piece] = field(default_factory=list)
    end: int = 0
    acked: int = 0
    send_at: int = 0
    sent_samples: int = 0
    caught_up: bool = False
    ws: Optional[ClientConnection] = None
    ready: bool = False
    ended: Optional[str] = None
    stopping: bool = False
    down_until: float = 0.0
    attempt: int = 0
    last_heard: float = 0.0
    sockets: List[Dict[str, Any]] = field(default_factory=list)
    drops: List[Drop] = field(default_factory=list)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    first_ready: asyncio.Event = field(default_factory=asyncio.Event)
    first_ready_at: Optional[float] = None
    clamps: int = 0

    @property
    def frame(self) -> int:
        """Bytes per sample frame (16-bit, channels interleaved)."""
        return 2 * self.channels

    @property
    def pending_s(self) -> float:
        """Seconds captured and not yet acknowledged."""
        return (self.end - self.acked) / RATE

    def push(self, data: bytes) -> None:
        """Keep one captured piece and wake the sender."""
        samples = len(data) // self.frame
        self.pieces.append(Piece(self.end, samples, data))
        self.end += samples
        self.wake.set()

    def ack(self, samples: int) -> None:
        """Forget everything before ``samples``."""
        if samples <= self.acked:
            return
        self.acked = samples
        while self.pieces and \
                self.pieces[0].offset + self.pieces[0].samples <= samples:
            self.pieces.pop(0)

    def drop(self, seconds: float, at_stream: float) -> None:
        """The network goes down for ``seconds``: abort the socket."""
        now = time.monotonic()
        self.down_until = now + seconds
        self.drops.append(Drop(at_stream, seconds, now,
                               held_at_drop=self.end - self.acked))
        ws = self.ws
        if ws is not None:
            ws.transport.abort()

    async def run(self) -> None:
        """Connect, reconnecting with backoff, until stopped or ended."""
        while not self.stopping and self.ended is None:
            reason = await self._connection()
            if self.stopping or self.ended is not None:
                return
            if reason == "ready_seen":
                self.attempt = 0
            base = min(BACKOFF_FIRST_S * 2 ** self.attempt, BACKOFF_MAX_S)
            self.attempt += 1
            await asyncio.sleep(base * (0.8 + 0.4 * random.random()))

    async def _connection(self) -> str:
        """One socket's life; returns how it ended."""
        opened = time.monotonic()
        entry: Dict[str, Any] = {"opened": opened}
        self.sockets.append(entry)
        for drop in self.drops:
            if drop.ready_at is None:
                drop.attempts += 1
        if opened < self.down_until:
            entry.update(result="network_down")
            return "failed"
        try:
            ws = await asyncio.wait_for(
                connect(self.url, max_size=None, open_timeout=10,
                        ping_interval=None), 10)
        except InvalidStatus as exc:
            entry.update(
                result=f"connect_failed:http_{exc.response.status_code}")
            return "failed"
        except (OSError, asyncio.TimeoutError) as exc:
            entry.update(result=f"connect_failed:{type(exc).__name__}")
            return "failed"
        self.ws = ws
        self.ready = False
        self.caught_up = False
        saw_ready = False
        reader = asyncio.create_task(self._read(ws, entry))
        try:
            await ws.send(json.dumps({"type": "auth",
                                      "access_code": self.code}))
            try:
                await asyncio.wait_for(self._wait_ready(ws),
                                       READY_TIMEOUT_S)
            except asyncio.TimeoutError:
                entry["result"] = "ready_timeout"
                await ws.close()
            if self.ready:
                saw_ready = True
                await self._send_loop(ws)
        except ConnectionClosed:
            pass
        finally:
            await asyncio.wait([reader], timeout=STOPPED_WAIT_S + 1)
            reader.cancel()
            self.ws = None
            self.ready = False
            entry["closed"] = time.monotonic()
            entry["close_code"] = ws.close_code
            entry["close_by_server"] = ws.protocol.close_rcvd is not None
        return "ready_seen" if saw_ready else "failed"

    async def _wait_ready(self, ws: ClientConnection) -> None:
        """Return once the reader saw ``ready`` (or the socket ended)."""
        while not self.ready and ws.close_code is None:
            await asyncio.sleep(0.02)

    async def _read(self, ws: ClientConnection, entry: Dict[str, Any]
                    ) -> None:
        """Handle the server's messages on one socket."""
        try:
            async for raw in ws:
                self.last_heard = time.monotonic()
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "ready":
                    self._ready(message, entry)
                elif kind == "ack":
                    self.ack(int(message["samples"]))
                    self._check_live()
                    self.wake.set()
                elif kind == "stopped":
                    self.stopped.set()
                elif kind == "gap_clamped":
                    self.clamps += 1
                elif kind == "error":
                    entry["refusal"] = message.get("code")
                    if message.get("code") in (
                            "not_found", "wrong_state", "replaced"):
                        self.ended = str(message.get("code"))
        except ConnectionClosed:
            pass
        finally:
            self.wake.set()

    def _ready(self, message: Dict[str, Any], entry: Dict[str, Any]) -> None:
        """A socket said ready: resend from what the server has."""
        now = time.monotonic()
        received = int(message.get("received_samples") or 0)
        entry["ready"] = now
        entry["received_samples"] = received
        self.ack(received)
        self.send_at = received
        self.ready = True
        self.last_heard = now
        if self.first_ready_at is None:
            self.first_ready_at = now
            self.first_ready.set()
        for drop in self.drops:
            if drop.ready_at is None:
                drop.ready_at = now
                drop.received_at_ready = received
                drop.end_at_ready = self.end
        self.wake.set()

    def _check_live(self) -> None:
        """Note when the link is live again after a drop."""
        if self.end - self.acked > LIVE_SLACK_S * RATE:
            return
        for drop in self.drops:
            if drop.ready_at is not None and drop.live_at is None:
                drop.live_at = time.monotonic()

    async def _send_loop(self, ws: ClientConnection) -> None:
        """Send what the server lacks; watch for a silent socket."""
        while ws.close_code is None and not self.stopping:
            self.wake.clear()
            for piece in list(self.pieces):
                last = piece.offset + piece.samples
                if last <= self.send_at:
                    continue
                cut = max(self.send_at - piece.offset, 0) * self.frame
                await ws.send(piece.data[cut:])
                self.sent_samples += last - max(self.send_at, piece.offset)
                self.send_at = last
            if not self.caught_up and self.send_at >= self.end:
                self.caught_up = True
                await ws.send(json.dumps({"type": "caught_up"}))
                for drop in self.drops:
                    if drop.ready_at is not None and \
                            drop.caught_up_sent_at is None:
                        drop.caught_up_sent_at = time.monotonic()
            waiting = self.end > self.acked
            if waiting and time.monotonic() - self.last_heard >= SILENT_S:
                await ws.close()
                return
            try:
                await asyncio.wait_for(self.wake.wait(), SILENT_S / 4)
            except asyncio.TimeoutError:
                pass

    async def finish(self) -> Dict[str, Any]:
        """«Terminar»: wait for the acks, say stop, wait for stopped."""
        started = time.monotonic()
        pending_at_start = self.pending_s
        connected = 0.0
        while self.end > self.acked and self.ended is None:
            await asyncio.sleep(0.25)
            if self.ready:
                connected += 0.25
            if connected >= FINISH_CAP_S:
                break
        all_acked = time.monotonic()
        lost_s = self.pending_s
        self.stopping = True
        self.wake.set()
        ws = self.ws
        said_stop = False
        if ws is not None and self.ready and ws.close_code is None:
            await ws.send(json.dumps({"type": "stop"}))
            said_stop = True
            try:
                await asyncio.wait_for(self.stopped.wait(), STOPPED_WAIT_S)
            except asyncio.TimeoutError:
                pass
            await ws.close()
        return {
            "pending_at_finish_s": round(pending_at_start, 2),
            "wait_for_acks_s": round(all_acked - started, 2),
            "lost_at_finish_s": round(lost_s, 2),
            "said_stop": said_stop, "stopped_seen": self.stopped.is_set(),
        }


async def follow_events(
    base_ws: str, session: str, code: str,
    items: List[Tuple[float, Dict[str, Any]]], closes: List[Any],
) -> None:
    """Collect the session's ``live.event.v1`` events until it ends.

    A socket that brings nothing for :data:`EVENTS_IDLE_S` is taken for
    dead (a tunnel reset can leave it half open, with no close) and
    reopened from the next ``seq``, as the page does with its stream.
    """
    seq = 0
    while True:
        url = f"{base_ws}/api/live/{session}/events?from_seq={seq}"
        try:
            async with connect(url, max_size=None,
                               ping_interval=None) as ws:
                await ws.send(json.dumps({"type": "auth",
                                          "access_code": code}))
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(),
                                                     EVENTS_IDLE_S)
                    except asyncio.TimeoutError:
                        closes.append("idle")
                        break
                    items.append((time.monotonic(), json.loads(raw)))
                    seq += 1
        except ConnectionClosed as exc:
            code_seen = exc.rcvd.code if exc.rcvd else 1006
            closes.append(code_seen)
            if code_seen == 1000:
                return
        except InvalidStatus as exc:
            closes.append(f"http_{exc.response.status_code}")
        except OSError:
            closes.append("connect_failed")
        await asyncio.sleep(1.0)


async def follow_feed(
    client: httpx.AsyncClient, base: str, session: str,
    items: List[Dict[str, Any]], reconnects: List[float],
) -> None:
    """Follow the page's suggestion stream (SSE) until its ``end``."""
    after = 0
    while True:
        url = (f"{base}/api/assist/sessions/{session}/suggestions/stream"
               f"?after={after}")
        try:
            async with client.stream("GET", url, timeout=httpx.Timeout(
                    10, read=60)) as response:
                if response.status_code == 404:
                    return
                response.raise_for_status()
                name, data, number = "message", [], None
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        name = line[6:].strip()
                    elif line.startswith("id:"):
                        number = int(line[3:].strip())
                    elif line.startswith("data:"):
                        data.append(line[5:].strip())
                    elif not line:
                        if name == "end":
                            return
                        if data:
                            items.append(json.loads("\n".join(data)))
                            if number is not None:
                                after = number
                        name, data, number = "message", [], None
        except (httpx.HTTPError, ValueError):
            pass
        reconnects.append(time.monotonic())
        await asyncio.sleep(2.0)


def read_wav(path: Path) -> Tuple[int, bytes, float]:
    """Return (channels, 16 kHz s16le frames, seconds) of a wav."""
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == RATE, "16 kHz wav needed"
        assert handle.getsampwidth() == 2, "16-bit wav needed"
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
        return channels, frames, handle.getnframes() / RATE


async def capture(link: Link, frames: bytes, faults: Sequence[
        Tuple[float, float]]) -> float:
    """Feed the wav to the link in real time, firing the faults.

    Returns the monotonic time of stream second 0.
    """
    per_piece = int(RATE * PIECE_S) * link.frame
    todo = sorted(faults)
    t0 = time.monotonic()
    for number, first in enumerate(range(0, len(frames), per_piece)):
        await asyncio.sleep(max(0.0, t0 + (number + 1) * PIECE_S
                                - time.monotonic()))
        link.push(frames[first:first + per_piece])
        stream_s = link.end / RATE
        while todo and todo[0][0] <= stream_s:
            at, seconds = todo.pop(0)
            link.drop(seconds, at)
    return t0


async def busy(client: httpx.AsyncClient, base: str) -> List[str]:
    """Return the ids of the live sessions not ended (another's or ours)."""
    answer = await client.get(f"{base}/api/live/sessions")
    answer.raise_for_status()
    return [s["session_id"] for s in answer.json()["sessions"]
            if s.get("state") != "ended"]


def phase_lags(
    finals: Sequence[Tuple[float, Dict[str, Any]]], t0: float,
    first_drop: Optional[float], live_again: Optional[float],
) -> Dict[str, Any]:
    """Return final lags (server ``lag_ms`` and client arrival) by phase.

    ``before`` arrived before the first drop, ``catch_up`` between it
    and the moment the link was live again (plus 2 s), ``after`` later.
    """
    groups: Dict[str, Dict[str, List[float]]] = {
        k: {"server": [], "client": []}
        for k in ("before", "catch_up", "after")}
    for at, event in finals:
        if first_drop is None or at < first_drop:
            key = "before"
        elif live_again is None or at < live_again + 2.0:
            key = "catch_up"
        else:
            key = "after"
        timings = event.get("timings") or {}
        if timings.get("lag_ms") is not None:
            groups[key]["server"].append(float(timings["lag_ms"]))
        end = event.get("utterance", {}).get("end")
        if end is not None:
            groups[key]["client"].append((at - (t0 + float(end))) * 1000)
    return {k: {"lag_ms": stats(v["server"]),
                "client_lag_ms": stats(v["client"])}
            for k, v in groups.items()}


def transcript_summary(value: Any) -> Any:
    """Return counts and flags of the hand-off, never its text."""
    if not isinstance(value, dict):
        return value if value is None else type(value).__name__
    out: Dict[str, Any] = {"keys": sorted(value)}
    for key, item in value.items():
        if isinstance(item, list):
            out[f"{key}_count"] = len(item)
        elif isinstance(item, (bool, int, float)) or item is None:
            out[key] = item
        elif isinstance(item, dict):
            out[f"{key}_keys"] = sorted(item)
        else:
            out[f"{key}_present"] = True
    return out


async def run_scenario(
    name: str, args: argparse.Namespace, code: str,
    reference_words: List[str], baseline_words: Optional[List[str]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Run one scenario; return its numbers and its words (in memory)."""
    channels, frames, seconds = read_wav(args.wav)
    if args.max_seconds:
        seconds = min(seconds, args.max_seconds)
        frames = frames[:int(seconds * RATE) * 2 * channels]
    base = args.base.rstrip("/")
    base_ws = base.replace("http", "ws", 1)
    session = f"smoke-{name}-{int(time.time())}"
    headers = {"X-Access-Code": code}
    result: Dict[str, Any] = {
        "scenario": name, "what": SCENARIOS[name], "session": session,
        "base": base, "wav_seconds": round(seconds, 1),
        "channels": channels,
        "faults": [{"at_s": a, "seconds": s}
                   for a, s in faults_for(name, seconds)],
    }
    async with httpx.AsyncClient(timeout=60, headers=headers) as client:
        others = await busy(client, base)
        if others:
            result["aborted"] = "busy: another live session is running"
            return result, []
        made = await client.post(f"{base}/api/assist/sessions", json={
            "product": args.product, "audio": args.audio,
            "session_id": session})
        if made.status_code >= 400:
            result["aborted"] = f"start refused: {made.status_code}"
            return result, []
        started = time.monotonic()
        link = Link(f"{base_ws}/api/live/{session}/audio", code, channels)
        events: List[Tuple[float, Dict[str, Any]]] = []
        event_closes: List[Any] = []
        feed: List[Dict[str, Any]] = []
        feed_reconnects: List[float] = []
        follower = asyncio.create_task(follow_events(
            base_ws, session, code, events, event_closes))
        feeder = asyncio.create_task(follow_feed(
            client, base, session, feed, feed_reconnects))
        runner = asyncio.create_task(link.run())
        t0 = started
        try:
            await asyncio.wait_for(link.first_ready.wait(), 180)
            t0 = await capture(link, frames, faults_for(name, seconds))
            finish = await link.finish()
            await asyncio.wait([runner], timeout=10)
            stop = await client.post(
                f"{base}/api/assist/sessions/{session}/stop", timeout=120)
            result["http_stop_status"] = stop.status_code
            await asyncio.wait([follower, feeder], timeout=120)
        finally:
            for task in (runner, follower, feeder):
                task.cancel()
            if session in await busy(client, base):
                await client.post(
                    f"{base}/api/assist/sessions/{session}/stop",
                    timeout=120)
                result["cleanup_stop"] = True
        info = (await client.get(f"{base}/api/live/sessions/{session}")
                ).json()
        orch = (await client.get(f"{base}/api/assist/sessions/{session}")
                ).json()
    assert link.first_ready_at is not None
    finals: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    partial_times: List[float] = []
    warnings: List[Dict[str, Any]] = []
    kinds: Dict[str, int] = {}
    for at, event in events:
        kind = str(event.get("type"))
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "final":
            finals[event["utterance"]["id"]] = (at, event)
        elif kind == "partial":
            partial_times.append(at)
        elif kind == "warning":
            warning = event.get("warning") or {}
            warnings.append({
                "code": warning.get("code"),
                "seconds": warning.get("seconds"),
                "at_s": round(at - t0, 1)})
    ordered = sorted(finals.values(),
                     key=lambda f: (f[1]["utterance"].get("start", 0),
                                    f[1]["utterance"].get("channel") or 0))
    said: List[str] = []
    for _, event in ordered:
        said.extend(words(str(event["utterance"].get("text", ""))))
    first_drop = link.drops[0].dropped_at if link.drops else None
    live_again = link.drops[-1].live_at if link.drops else None
    drops: List[Dict[str, Any]] = []
    for drop in link.drops:
        backlog = None
        if drop.end_at_ready is not None and \
                drop.received_at_ready is not None:
            backlog = (drop.end_at_ready - drop.received_at_ready) / RATE
        row: Dict[str, Any] = {
            "at_stream_s": drop.at_stream, "outage_s": drop.seconds,
            "held_at_drop_s": round(drop.held_at_drop / RATE, 2),
            "connect_attempts": drop.attempts,
        }
        if drop.ready_at is not None:
            row["drop_to_ready_s"] = round(drop.ready_at - drop.dropped_at,
                                           2)
            row["ready_after_network_back_s"] = round(
                drop.ready_at - drop.dropped_at - drop.seconds, 2)
            row["backlog_at_ready_s"] = (None if backlog is None
                                         else round(backlog, 2))
        if drop.caught_up_sent_at is not None and drop.ready_at is not None:
            row["ready_to_caught_up_sent_s"] = round(
                drop.caught_up_sent_at - drop.ready_at, 2)
        if drop.live_at is not None and drop.ready_at is not None:
            took = drop.live_at - drop.ready_at
            row["ready_to_live_s"] = round(took, 2)
            if backlog and took > 0:
                row["catch_up_x_realtime"] = round(
                    (backlog + took) / took, 1)
        back = drop.ready_at
        if back is not None:
            after = [t for t in partial_times if t > back]
            row["first_partial_after_ready_s"] = (
                round(after[0] - back, 2) if after else None)
        drops.append(row)
    captured = link.end
    received = int(info.get("received_samples") or 0)
    result.update({
        "device": info.get("device"), "model": info.get("model"),
        "end_reason": info.get("end_reason"), "state": info.get("state"),
        "source_at_end": info.get("source"),
        "error_present": bool(info.get("error")),
        "server_finals": info.get("finals"),
        "rejected_events": info.get("rejected_events"),
        "captured_samples": captured, "received_samples": received,
        "samples_equal": captured == received,
        "sent_samples": link.sent_samples,
        "resent_samples": link.sent_samples - captured,
        "gap_clamped": link.clamps,
        "ready_after_start_s": round(link.first_ready_at - started, 2),
        "finish": finish,
        "drops": drops,
        "audio_sockets": [
            {"result": s.get("result", "ok"),
             "close_code": s.get("close_code"),
             "close_by_server": s.get("close_by_server"),
             "refusal": s.get("refusal"),
             "received_samples_at_ready": s.get("received_samples")}
            for s in link.sockets],
        "event_kinds": kinds,
        "events_socket_closes": event_closes,
        "warnings": warnings,
        "finals": len(finals), "words": len(said),
        "similarity_to_baseline": (None if baseline_words is None
                                   else similarity(said, baseline_words)),
        "similarity_to_script": similarity(said, reference_words),
        "partials": len(partial_times),
        "partials_after_last_ready": (
            sum(1 for t in partial_times
                if link.drops and link.drops[-1].ready_at is not None
                and t > link.drops[-1].ready_at)),
        "lags": phase_lags([f for f in ordered], t0, first_drop,
                           live_again),
        "feed_items": len(feed),
        "feed_suggestions": sum(1 for i in feed
                                if i.get("type") == "suggestion"),
        "feed_reconnects": len(feed_reconnects),
        "orchestrator_state": orch.get("state") if isinstance(
            orch, dict) else None,
        "transcript": transcript_summary(info.get("transcript")),
    })
    return result, said


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the scenarios one after another (one session at a time)."""
    code = args.code_file.read_text(encoding="utf-8").strip()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    script = [w for u in sorted(reference["utterances"],
                                key=lambda u: (u["start"], u["channel"]))
              for w in words(u["text"])]
    out: Dict[str, Any] = {
        "base": args.base, "product": args.product, "audio": args.audio,
        "wav": args.wav.name, "script_words": len(script),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": []}
    baseline: Optional[List[str]] = None
    for name in args.scenarios.split(","):
        result, said = await run_scenario(name, args, code, script,
                                          baseline)
        if name == "a" and said:
            baseline = said
        out["runs"].append(result)
        print(json.dumps({k: result.get(k) for k in (
            "scenario", "aborted", "samples_equal", "end_reason",
            "finals", "words", "similarity_to_baseline",
            "similarity_to_script")}))
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        if result.get("aborted"):
            break
        await asyncio.sleep(args.pause)
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out


def main() -> None:
    """Parse the arguments, run, write the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True,
                        help="the transcriber, e.g. http://127.0.0.1:8770")
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--product", default="live_assist.support")
    parser.add_argument("--audio", choices=["stereo", "mic"],
                        default="stereo")
    parser.add_argument("--scenarios", default="a,b,c,d,e")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--pause", type=float, default=5.0,
                        help="seconds between sessions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
