"""End of the user's turn → first audio of the voice bot (VB-5).

Synthetic user turns: Spanish questions about the ``support.demo``
domain, spoken by a Piper voice that is not the bot's
(``es_ES-sharvard-medium``), resampled to 16 kHz. They are streamed in
real time into a ``voicebot`` live session of a transcriber through its
audio socket (100 ms pieces on a monotonic schedule, as the page's
microphone; silence between questions, so the transcriber's VAD hang
closes each turn), while this client follows the session's feed and
times, on its own clock:

* *voice end*: the moment a microphone would have delivered the last
  voiced sample of the question (trailing silence of the rendering cut
  at 1 % of its peak);
* *final*: arrival of the user's ``final`` event in the feed;
* *first audio*: arrival of the turn's first ``speech`` message with
  audio, i.e. what the page can start to play.

Each turn's record adds the orchestrator's own stages
(``knowledge_ms``, ``first_text_ms``, ``first_sentence_ms``,
``first_tts_ms``, ``first_audio_ms`` from the final's arrival) and the
transcriber's ``event_lag_ms``. After each answer the client waits for
the bot's audio to play out (its duration) plus a pause, so no turn is
interrupted. Usage::

    python experiments/voicebot/first_audio.py --orchestrator URL \\
        --transcriber URL --code-file F --token-file T --out R.json \\
        [--turns N] [--stream-url URL]

``--stream-url`` follows the feed somewhere else than the orchestrator
(the transcriber's ``/api/assist`` proxy, as the page does). Real Haiku
calls are the orchestrator's, one per answered turn (``turns_by_status``).
Each row keeps the sample rate of the turn's first audio (22050: Piper,
24000: a natural voice, #214) and the result the feed's ``voice``
messages (who spoke, and why not the chosen voice).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import numpy.typing as npt
import websockets

from orchestrator.adapters.outbound.piper_tts import PiperTts
from orchestrator.application.ports import CancelToken
from orchestrator.domain.speech import tts_request

Pcm = npt.NDArray[np.int16]
RATE = 16000
CHUNK = RATE // 10
USER_VOICE = "es_ES-sharvard-medium"
QUESTIONS = (
    "¿Cuántos días tengo para devolver un producto?",
    "¿Cuánto tarda en llegar el reembolso?",
    "Me ha llegado una lámpara rota, ¿qué hago?",
    "¿Puedo devolver un colchón que ya he abierto?",
    "¿Cuánto cuesta devolver algo si he cambiado de opinión?",
    "¿Puedo recoger el pedido en una tienda?",
    "Quiero cambiar una alfombra por otra de otro color.",
    "¿Qué necesito para empezar una devolución?",
    "Han pasado cuarenta días, ¿todavía puedo devolverlo?",
    "¿Cómo puedo pedir la factura de mi compra?",
    "Mi pedido no ha llegado todavía, ¿qué puedo hacer?",
    "¿Hacéis envíos a las islas?",
    "Me cobraron dos veces el mismo pedido.",
    "¿Puedo pagar a plazos?",
    "Quiero hablar con un responsable, por favor.",
)
LEAD_S = 1.0
PAUSE_S = 1.0
TURN_TIMEOUT_S = 20.0


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
    """Return n, p50, p90 and max of ``values`` (ms, rounded), as LV-12."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x)

    return {
        "n": len(values), "p50": r(percentile(values, 0.5)),
        "p90": r(percentile(values, 0.9)),
        "max": r(max(values) if values else None),
    }


def render(tts: PiperTts, text: str) -> Tuple[Pcm, int]:
    """Return a question as 16 kHz int16 and its last voiced sample."""
    [chunk] = list(tts.synthesize(
        tts_request("q", text, USER_VOICE, "es", audio_format="pcm_s16le",
                    stream=False), CancelToken()))
    source = np.frombuffer(chunk.audio, dtype=np.int16).astype(np.float64)
    times = np.arange(int(len(source) * RATE / chunk.sample_rate)) / RATE
    audio = np.interp(times, np.arange(len(source)) / chunk.sample_rate,
                      source).astype(np.int16)
    loud = np.nonzero(np.abs(audio) > 0.01 * np.abs(audio).max())[0]
    return audio, int(loud[-1]) + 1


class Follower:
    """The session's feed with the arrival time of every message."""

    def __init__(self) -> None:
        """Start empty."""
        self.items: List[Tuple[float, Dict[str, Any]]] = []
        self.ended = asyncio.Event()

    async def run(self, client: httpx.AsyncClient, url: str,
                  headers: Dict[str, str]) -> None:
        """Collect messages until the stream ends."""
        async with client.stream("GET", url, headers=headers,
                                 timeout=None) as response:
            response.raise_for_status()
            name, data = "message", []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and data:
                    if name == "end":
                        break
                    self.items.append(
                        (time.monotonic(), json.loads("\n".join(data))))
                    name, data = "message", []
                elif not line:
                    name = "message"
        self.ended.set()

    def since(self, at: float) -> List[Tuple[float, Dict[str, Any]]]:
        """Messages that arrived after ``at``."""
        return [(t, m) for t, m in self.items if t >= at]


class Mic:
    """Sends audio in real time: 100 ms pieces on a fixed schedule."""

    def __init__(self, socket: Any):
        """Start the schedule now."""
        self.socket = socket
        self.t0 = time.monotonic()
        self.sent = 0

    async def send(self, samples: Pcm) -> None:
        """Send ``samples`` piece by piece, each once it is 'recorded'."""
        for start in range(0, len(samples), CHUNK):
            piece = samples[start:start + CHUNK]
            self.sent += len(piece)
            await asyncio.sleep(max(0.0, self.t0 + self.sent / RATE
                                    - time.monotonic()))
            await self.socket.send(piece.tobytes())

    async def silence(self, seconds: float) -> None:
        """Send ``seconds`` of silence."""
        await self.send(np.zeros(int(seconds * RATE), dtype=np.int16))

    def at(self, sample: int) -> float:
        """Wall time at which ``sample`` of the stream is recorded."""
        return self.t0 + sample / RATE


async def one_turn(
    mic: Mic, follower: Follower, audio: Pcm, voiced: int,
) -> Dict[str, Any]:
    """Say one question, wait for the answer to play out; time it."""
    start = mic.sent
    await mic.send(audio)
    voice_end = mic.at(start + voiced)
    row: Dict[str, Any] = {}
    deadline = time.monotonic() + TURN_TIMEOUT_S
    record: Optional[Dict[str, Any]] = None
    while time.monotonic() < deadline:
        await mic.silence(0.1)
        for at, item in follower.since(voice_end):
            kind = item.get("type")
            if kind == "event" and item["event"].get("type") == "final" \
                    and "final_ms" not in row:
                row["final_ms"] = (at - voice_end) * 1000.0
            elif kind == "speech" and item.get("audio") \
                    and "first_audio_ms" not in row:
                row["first_audio_ms"] = (at - voice_end) * 1000.0
                row["turn"] = item["turn"]
                row["first_sample_rate"] = item.get("sample_rate")
            elif kind == "turn" and item["record"]["turn"] == row.get(
                    "turn"):
                record = item["record"]
        if record is not None:
            break
    if record is None:
        row["status"] = "no_answer"
        return row
    stages = record.get("stages") or {}
    row.update({
        "status": record["status"], "spoken": len(record["spoken"]),
        "audio_ms": record["audio_ms"],
        "knowledge_status": record["knowledge_status"],
        **{key: stages.get(key) for key in (
            "event_lag_ms", "knowledge_ms", "first_text_ms",
            "first_sentence_ms", "first_tts_ms", "reasoner_ms",
            "total_ms")},
        "orchestrator_first_audio_ms": stages.get("first_audio_ms"),
        "reasoner_first_token_ms": (stages.get("reasoner") or {}).get(
            "first_token_ms"),
        "reply_chars": len(record.get("reply") or ""),
    })
    if "final_ms" in row and "first_audio_ms" in row:
        row["final_to_first_audio_ms"] = (
            row["first_audio_ms"] - row["final_ms"])
    # let the answer play out, then pause, as a person would
    await mic.silence(record["audio_ms"] / 1000.0 + PAUSE_S)
    return row


KEYS = ("first_audio_ms", "final_ms", "final_to_first_audio_ms",
        "event_lag_ms", "knowledge_ms", "first_text_ms",
        "first_sentence_ms", "first_tts_ms", "orchestrator_first_audio_ms",
        "reasoner_first_token_ms", "reasoner_ms", "audio_ms")


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the turns and return the result."""
    code = args.code_file.read_text(encoding="utf-8").strip()
    token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
    session = args.session or f"vb5-{int(time.time())}"
    tts = PiperTts()
    try:
        questions = [(q, *render(tts, q))
                     for q in QUESTIONS[:args.turns]]
    finally:
        tts.close()
    auth = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60) as client:
        made = await client.post(
            f"{args.orchestrator}/sessions", headers=auth,
            json={"product": "voicebot", "audio": "mic",
                  "session_id": session})
        made.raise_for_status()
        follower = Follower()
        stream = args.stream_url or args.orchestrator
        headers = ({"X-Access-Code": code} if args.stream_url else auth)
        task = asyncio.create_task(follower.run(
            client, f"{stream}/sessions/{session}/suggestions/stream",
            headers))
        ws = args.transcriber.replace("http", "ws", 1)
        rows: List[Dict[str, Any]] = []
        async with websockets.connect(
            f"{ws}/api/live/{session}/audio", max_size=None,
            additional_headers={"X-Access-Code": code},
        ) as socket:
            ready = json.loads(await socket.recv())
            assert ready["type"] == "ready", ready
            mic = Mic(socket)
            await mic.silence(LEAD_S)
            for text, audio, voiced in questions:
                row = await one_turn(mic, follower, audio, voiced)
                row.update({"question": text,
                            "voice_ms": round(voiced * 1000 / RATE)})
                rows.append(row)
            await socket.send(json.dumps({"type": "stop"}))
            # acks may come first (#263): read until the server says stopped
            while json.loads(await socket.recv()).get("type") != "stopped":
                pass
        await asyncio.wait_for(follower.ended.wait(), 60)
        task.cancel()
        info = (await client.get(f"{args.orchestrator}/sessions/{session}",
                                 headers=auth)).json()
    answered = [r for r in rows if r.get("status") == "ok"]
    return {
        "session": session, "turns": len(rows),
        "turns_by_status": {
            s: sum(1 for r in rows if r.get("status") == s)
            for s in sorted({str(r.get("status")) for r in rows})},
        "ok_turns": {key: stats([r[key] for r in answered
                                 if r.get(key) is not None])
                     for key in KEYS},
        "within_2500_ms": sum(1 for r in answered
                              if r.get("first_audio_ms", 1e9) <= 2500),
        "stream_url": "proxy" if args.stream_url else "orchestrator",
        "orchestrator_session": {k: info.get(k) for k in (
            "state", "calls", "skipped", "events")},
        "voice_messages": [m for _, m in follower.items
                           if m.get("type") in ("voice", "voice_summary")],
        "rows": rows,
    }


def main() -> None:
    """Parse the arguments, measure, write the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orchestrator", required=True)
    parser.add_argument("--transcriber", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--stream-url", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=len(QUESTIONS))
    parser.add_argument("--session", default=None)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
