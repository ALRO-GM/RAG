"""End-to-end latency of the live loop on a rendered call (LV-12).

Streams a wav (stereo or mono) in real time into a live session of a
transcriber through its audio socket, while an orchestrator follows the
session, and times every stage on this client's clock::

    speech end → final → suggestion on screen
                 (knowledge, reasoner)

* *speech end*: the moment the microphone would have delivered the
  last sample of the reference sentence (audio is sent in 100 ms
  chunks on a monotonic schedule; ``--ready`` waits for the socket's
  ``ready``, as the live page does);
* *final*: arrival of the ``final`` event of that sentence in the
  orchestrator's SSE stream (matched to the reference by time overlap);
* *suggestion*: arrival of the ``suggestion`` item whose trigger is a
  final; its ``knowledge_ms`` and ``reasoner_ms`` come from the record,
  what is left of the wait is queueing, debounce and transport.

Usage::

    python experiments/live/live_latency.py --orchestrator URL \\
        --transcriber URL --code-file F --product live_assist.support \\
        --audio stereo|mic --wav W --reference R --out RESULT.json \\
        [--max-seconds N] [--session ID]

Prints nothing but the JSON path; real Haiku calls are those of the
orchestrator's reasoner, counted in the result (``calls``).
"""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import websockets

CHUNK_S = 0.1


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
    """Return n, p50, p90 and max of ``values`` (ms, rounded)."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x)

    return {
        "n": len(values), "p50": r(percentile(values, 0.5)),
        "p90": r(percentile(values, 0.9)),
        "max": r(max(values) if values else None),
    }


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return the seconds two spans share."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_reference(
    span: Tuple[float, float], reference: Sequence[Dict[str, Any]],
    channel: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Return the reference sentence sharing most time with ``span``."""
    best, best_share = None, 0.0
    for sentence in reference:
        if channel is not None and sentence.get("channel") != channel:
            continue
        share = overlap(span, (sentence["start"], sentence["end"]))
        if share > best_share:
            best, best_share = sentence, share
    return best


async def sse(
    client: httpx.AsyncClient, url: str, items: List[Tuple[float, Any]],
) -> None:
    """Collect the (arrival time, JSON) of every SSE message until end."""
    async with client.stream("GET", url, timeout=None) as response:
        response.raise_for_status()
        name, data = "message", []
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line and data:
                if name == "end":
                    return
                items.append((time.monotonic(), json.loads("\n".join(data))))
                name, data = "message", []
            elif not line:
                name = "message"


async def stream_audio(
    url: str, code: str, wav: Path, max_seconds: Optional[float],
) -> Tuple[float, float]:
    """Send ``wav`` in real time; return (t0, time the socket was ready)."""
    with wave.open(str(wav), "rb") as handle:
        rate, width = handle.getframerate(), (
            handle.getnchannels() * handle.getsampwidth())
        seconds = handle.getnframes() / rate
        if max_seconds is not None:
            seconds = min(seconds, max_seconds)
        frames = handle.readframes(int(rate * seconds))
    per_chunk = int(rate * CHUNK_S) * width
    async with websockets.connect(
        url, additional_headers={"X-Access-Code": code}, max_size=None,
    ) as audio:
        ready = json.loads(await audio.recv())
        assert ready["type"] == "ready", ready
        ready_at = time.monotonic()
        t0 = time.monotonic()
        for number, first in enumerate(range(0, len(frames), per_chunk)):
            # a microphone hands over a chunk once it is complete
            await asyncio.sleep(max(0.0, t0 + (number + 1) * CHUNK_S
                                    - time.monotonic()))
            await audio.send(frames[first:first + per_chunk])
        await audio.send(json.dumps({"type": "stop"}))
        await audio.recv()
    return t0, ready_at


def analyse(
    items: Sequence[Tuple[float, Any]], t0: float,
    reference: Sequence[Dict[str, Any]], mono: bool,
) -> Dict[str, Any]:
    """Join finals, suggestions and the reference into stage latencies."""
    finals: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for at, item in items:
        if item.get("type") == "event" and \
                item["event"].get("type") == "final":
            utterance = item["event"]["utterance"]
            finals.setdefault(utterance["id"], (at, utterance))
    speech_to_final: List[float] = []
    matched: Dict[str, Dict[str, Any]] = {}
    for uid, (at, utterance) in finals.items():
        sentence = match_reference(
            (utterance["start"], utterance["end"]), reference,
            None if mono else utterance.get("channel"),
        )
        if sentence is None:
            continue
        matched[uid] = sentence
        speech_to_final.append((at - (t0 + sentence["end"])) * 1000)
    rows: List[Dict[str, Any]] = []
    for at, item in items:
        if item.get("type") != "suggestion":
            continue
        record = item["record"]
        stages = record.get("stages", {})
        trigger = record["trigger"]["utterance"]
        row: Dict[str, Any] = {
            "call": record["call"], "status": record["status"],
            "trigger": trigger, "knowledge_ms": stages.get("knowledge_ms"),
            "reasoner_ms": stages.get("reasoner_ms"),
            "total_ms": stages.get("total_ms"),
        }
        if trigger in finals and trigger in matched:
            final_at = finals[trigger][0]
            row["final_to_screen_ms"] = (at - final_at) * 1000
            row["speech_to_screen_ms"] = (
                at - (t0 + matched[trigger]["end"])) * 1000
            row["speech_to_final_ms"] = (
                final_at - (t0 + matched[trigger]["end"])) * 1000
            known = (row["knowledge_ms"] or 0) + (row["reasoner_ms"] or 0)
            row["other_ms"] = row["final_to_screen_ms"] - known
        rows.append(row)
    ok = [r for r in rows if r["status"] == "ok" and "speech_to_screen_ms" in r]
    return {
        "finals": len(finals), "matched_finals": len(matched),
        "calls": len([r for r in rows if r["status"] != "stale"]),
        "calls_by_status": {
            s: sum(1 for r in rows if r["status"] == s)
            for s in sorted({r["status"] for r in rows})},
        "speech_end_to_final_ms": stats(speech_to_final),
        "stages_ok_calls": {
            key: stats([r[key] for r in ok if r.get(key) is not None])
            for key in ("knowledge_ms", "reasoner_ms", "other_ms",
                        "final_to_screen_ms", "speech_to_screen_ms")},
        "rows": rows,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Run one measurement and return its result."""
    code = args.code_file.read_text(encoding="utf-8").strip()
    session = args.session or f"lat-{int(time.time())}"
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    async with httpx.AsyncClient(timeout=60) as client:
        made = await client.post(f"{args.orchestrator}/sessions", json={
            "product": args.product, "audio": args.audio,
            "session_id": session})
        made.raise_for_status()
        items: List[Tuple[float, Any]] = []
        follower = asyncio.create_task(sse(
            client, f"{args.orchestrator}/sessions/{session}"
            "/suggestions/stream", items))
        ws = args.transcriber.replace("http", "ws", 1)
        t0, ready_at = await stream_audio(
            f"{ws}/api/live/{session}/audio", code, args.wav,
            args.max_seconds)
        await asyncio.wait_for(follower, 120)
        info = (await client.get(
            f"{args.orchestrator}/sessions/{session}")).json()
    result = analyse(items, t0, reference["utterances"],
                     args.audio == "mic")
    result.update({
        "session": session, "product": args.product, "audio": args.audio,
        "wav": args.wav.name, "max_seconds": args.max_seconds,
        "orchestrator_session": info,
        "warnings": [
            i["event"]["warning"]["code"] for _, i in items
            if i.get("type") == "event"
            and i["event"].get("type") == "warning"],
    })
    return result


def main() -> None:
    """Parse the arguments, measure, write the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orchestrator", required=True)
    parser.add_argument("--transcriber", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--audio", choices=["stereo", "mic"], required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--session", default=None)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
