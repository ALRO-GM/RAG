"""A stand-in for ``lab/runners/sortformer_stream.py`` (same protocol).

Run as a script by the adapter's tests: no NeMo, no model. A frame
(80 ms) is speaker 0 when its audio is loud, silence otherwise; each
answer holds the frames the audio received so far completes. Modes
(first argument after the script, as the adapter passes ``--device``
too): ``ok``; ``error`` answers every audio with an error;
``crash`` exits after saying it is ready; ``order`` numbers its first
frames wrongly; ``noise`` writes a line that is not JSON first;
``slow`` never answers ``stop``.
"""

import base64
import json
import sys
import time
from typing import Any, Dict

FRAME = 1280


def main() -> int:
    """Serve the protocol in the mode given."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
    out = sys.stdout

    def say(message: Dict[str, Any]) -> None:
        out.write(json.dumps(message) + "\n")
        out.flush()

    if mode == "noise":
        out.write("loading...\n")
    say({"type": "ready", "model": "fake", "frame_seconds": 0.08,
         "speakers": 4, "latency_seconds": 0.08, "device": "cpu"})
    if mode == "crash":
        return 3
    samples = bytearray()
    frames = 0
    for line in sys.stdin:
        request = json.loads(line)
        kind = request["type"]
        if kind == "stop":
            if mode == "slow":
                time.sleep(60)
            return 0
        if kind == "start":
            samples, frames = bytearray(), 0
            say({"type": "started"})
            continue
        if mode == "error":
            say({"type": "error", "message": "model broke"})
            continue
        if kind == "audio":
            samples += base64.b64decode(request["pcm"])
        complete = len(samples) // (2 * FRAME)
        if kind == "flush" and len(samples) % (2 * FRAME):
            complete += 1
        probs = []
        for n in range(frames, complete):
            piece = samples[n * 2 * FRAME:(n + 1) * 2 * FRAME]
            loud = any(abs(int.from_bytes(piece[i:i + 2], "little",
                                          signed=True)) > 3000
                       for i in range(0, len(piece), 2))
            probs.append([0.9 if loud else 0.0, 0.0, 0.0, 0.0])
        first = frames + (5 if mode == "order" else 0)
        frames = complete
        say({"type": "frames", "first": first, "probs": probs})
    return 0


if __name__ == "__main__":
    sys.exit(main())
