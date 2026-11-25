"""Time from creating a live session to its audio socket saying ``ready``.

Creates a session on a running transcriber, opens the audio socket,
waits for ``ready`` and then sends ``--seconds`` of the audio of a wav
in real time, right after ``ready`` as the live page does. Prints JSON:
``ready_s`` (create → ready), the warnings the session emitted and its
state. The session's events and hand-off transcript stay in the
transcriber's ``live/<id>/`` directory (LV-13 reads them). With the
code that holds ``ready`` back until the model is loaded the warnings
are empty; before, ``asr_behind`` showed up.

Usage::

    python experiments/live/start_delay.py URL ACCESS_CODE_FILE WAV \\
        [--seconds 8] [--id ID] [--channels 1|2]
"""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path
from typing import Any, Dict, List

import httpx
import websockets

CHUNK_S = 0.1


async def run(url: str, code: str, wav: Path, seconds: float,
              session: str, channels: int = 1) -> Dict[str, Any]:
    """Return the measurements of one session start."""
    headers = {"X-Access-Code": code}
    with wave.open(str(wav), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(int(rate * seconds))
        width = handle.getnchannels() * handle.getsampwidth()
    per_chunk = int(rate * CHUNK_S) * width
    started = time.monotonic()
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        made = await client.post(f"{url}/api/live/sessions", json={
            "session_id": session, "channels": channels, "language": "es",
        })
        made.raise_for_status()
        created_s = time.monotonic() - started
        ws_url = url.replace("http", "ws", 1)
        async with websockets.connect(
            f"{ws_url}/api/live/{session}/audio", additional_headers=headers,
            max_size=None,
        ) as audio:
            ready = json.loads(await audio.recv())
            ready_s = time.monotonic() - started
            for first in range(0, len(frames), per_chunk):
                await audio.send(frames[first:first + per_chunk])
                await asyncio.sleep(CHUNK_S)
            await audio.send(json.dumps({"type": "stop"}))
            # acks may come first (#263): read until the server says stopped
            while json.loads(await audio.recv()).get("type") != "stopped":
                pass
        info = (await client.get(f"{url}/api/live/sessions/{session}")).json()
        events: List[Dict[str, Any]] = []
        async with websockets.connect(
            f"{ws_url}/api/live/{session}/events", additional_headers=headers,
            max_size=None,
        ) as stream:
            async for message in stream:
                events.append(json.loads(message))
    warnings = [e["warning"]["code"] for e in events
                if e.get("type") == "warning"]
    return {
        "created_s": round(created_s, 2), "ready_s": round(ready_s, 2),
        "ready": ready.get("type"), "warnings": warnings,
        "state": info["state"], "device": info["device"],
    }


def main() -> None:
    """Parse the arguments, measure, print."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("code_file", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--id", default=f"start-{int(time.time())}")
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()
    code = args.code_file.read_text(encoding="utf-8").strip()
    print(json.dumps(asyncio.run(
        run(args.url, code, args.wav, args.seconds, args.id,
            args.channels)), indent=1))


if __name__ == "__main__":
    main()
