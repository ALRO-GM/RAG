"""Generated audio for the adapter tests: tones, noise and containers.

Everything is synthesised here (numpy, libsndfile, PyAV); no file is
downloaded and no ``ffmpeg`` binary is needed.
"""

import io
from typing import Sequence, Tuple

import av
import numpy as np
import soundfile

from transcriber.domain.voices import FloatArray

RATE = 16000


def tone(
    seconds: float, rate: int = RATE, hz: float = 220.0, level: float = 0.5
) -> FloatArray:
    """A sine wave of ``hz`` at amplitude ``level``."""
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    wave: FloatArray = (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return wave


def noise(seconds: float, rate: int = RATE, level: float = 0.1) -> FloatArray:
    """Seeded white noise of standard deviation ``level``."""
    rng = np.random.default_rng(7)
    out: FloatArray = rng.normal(0.0, level, int(seconds * rate)).astype(
        np.float32
    )
    return out


def silence(seconds: float, rate: int = RATE) -> FloatArray:
    """Zeros."""
    return np.zeros(int(seconds * rate), dtype=np.float32)


def scene(parts: Sequence[Tuple[str, float]], rate: int = RATE) -> FloatArray:
    """Concatenate ``(what, seconds)`` parts: a pitch in Hz or ``"-"``.

    ``("220", 6.0)`` is six seconds of a 220 Hz voice, ``("-", 1.5)``
    a pause.
    """
    pieces = [
        silence(seconds, rate) if what == "-"
        else tone(seconds, rate, float(what))
        for what, seconds in parts
    ]
    joined: FloatArray = np.concatenate(pieces).astype(np.float32)
    return joined


def pcm_bytes(samples: FloatArray) -> bytes:
    """16-bit little-endian PCM bytes of ``samples``."""
    clipped = np.clip(samples, -1.0, 32767 / 32768)
    return (clipped * 32768).astype("<i2").tobytes()


def sound_file(samples: FloatArray, rate: int, kind: str) -> bytes:
    """``samples`` as a WAV or FLAC file (libsndfile)."""
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, rate, format=kind.upper())
    return buffer.getvalue()


def mp4_aac(samples: FloatArray, rate: int = 44100) -> bytes:
    """``samples`` (mono) as a stereo AAC track in an MP4 container."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("aac", rate=rate, layout="stereo")
        size = stream.codec_context.frame_size or 1024
        for start in range(0, samples.shape[0], size):
            part = samples[start:start + size]
            frame = av.AudioFrame.from_ndarray(
                np.stack([part, part]).astype(np.float32),
                format="fltp", layout="stereo",
            )
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def webm_opus(samples: FloatArray, rate: int = 48000) -> bytes:
    """``samples`` (mono) as Opus in a WebM container (browser format)."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="webm") as container:
        stream = container.add_stream("libopus", rate=rate, layout="mono")
        size = stream.codec_context.frame_size or 960
        ints = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        for start in range(0, ints.shape[0], size):
            frame = av.AudioFrame.from_ndarray(
                ints[start:start + size].reshape(1, -1),
                format="s16", layout="mono",
            )
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def video_only_mp4() -> bytes:
    """An MP4 with one short video stream and no audio."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for n in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), n * 40, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()
