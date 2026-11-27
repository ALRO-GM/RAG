"""Write a synthetic large MP4 for the large-upload run (#282).

Lossless H.264 noise at 1 frame per second (incompressible, so the file
is big) and an AAC track of 7 s tones and 1.5 s pauses alternating
220/880 Hz (what the fake speech models hear as two voices). Streamed
to disk a second at a time; it stops once the file passes ``TARGET``
bytes (checked every 30 s, so it ends a little above).

    python experiments/knowledge/large_upload_make_video.py OUT TARGET

``TARGET=1000000000`` gave 1 028 647 728 bytes, 600 s
(docs/KNOWLEDGE.md §4.2).
"""

import os
import sys

import av
import numpy as np

RATE = 16000


def build(out: str, target: int) -> int:
    """Write the MP4 to ``out``; return its length in seconds."""
    rng = np.random.default_rng(282)
    with av.open(out, mode="w", format="mp4") as box:
        video = box.add_stream("libx264", rate=1)
        video.width, video.height = 1280, 720
        video.pix_fmt = "yuv420p"
        video.options = {"qp": "0", "preset": "ultrafast"}
        audio = box.add_stream("aac", rate=RATE, layout="mono")
        size = audio.codec_context.frame_size or 1024
        second = pts = 0
        while True:
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),
                format="rgb24",
            )
            frame.pts = second
            for packet in video.encode(frame):
                box.mux(packet)
            t = np.arange(RATE) / RATE + second
            hz = 220.0 if (second // 8) % 2 == 0 else 880.0
            loud = ((t % 8.5) < 7.0).astype(np.float32)
            samples = (0.5 * np.sin(2 * np.pi * hz * t) * loud).astype(
                np.float32
            )
            for start in range(0, RATE, size):
                part = samples[start:start + size]
                sound = av.AudioFrame.from_ndarray(
                    part.reshape(1, -1), format="fltp", layout="mono"
                )
                sound.sample_rate = RATE
                sound.pts = pts
                pts += part.shape[0]
                for packet in audio.encode(sound):
                    box.mux(packet)
            second += 1
            if second % 30 == 0 and os.path.getsize(out) >= target:
                break
        for packet in video.encode(None):
            box.mux(packet)
        for packet in audio.encode(None):
            box.mux(packet)
    return second


if __name__ == "__main__":
    print(build(sys.argv[1], int(sys.argv[2])), "seconds")
