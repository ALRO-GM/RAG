"""Audio reader adapter: decode recordings to mono 16 kHz samples."""

import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.application.skills.ports import Audio
from src.domain.errors import InvalidInputError, NotFoundError

FloatArray = npt.NDArray[np.float32]

DEFAULT_SAMPLE_RATE = 16000
"""What the speech and voice models expect."""


def to_mono(data: FloatArray) -> FloatArray:
    """Average the channels of ``(frames, channels)`` samples."""
    if data.ndim == 1:
        return data.astype(np.float32)
    mono: FloatArray = data.mean(axis=1).astype(np.float32)
    return mono


def resample(samples: FloatArray, rate: int, target: int) -> FloatArray:
    """Resample with a polyphase filter (no aliasing when going down)."""
    if rate == target or samples.size == 0:
        return samples.astype(np.float32)
    from scipy.signal import resample_poly

    divisor = gcd(rate, target)
    out: FloatArray = resample_poly(
        samples, target // divisor, rate // divisor
    ).astype(np.float32)
    return out


class SoundFileAudioReader:
    """Read WAV, FLAC, OGG and MP3 with libsndfile.

    Other containers (M4A, MP4, WEBM: what phones and meeting tools
    record) are decoded through ``ffmpeg`` when it is installed.
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """Deliver audio at ``sample_rate`` Hz."""
        self.sample_rate = sample_rate

    def read(self, path: str) -> Audio:
        """Return the recording at ``path`` as mono samples."""
        file = Path(str(path))
        if not file.is_file():
            raise NotFoundError(f"Audio file not found: {file}")
        import soundfile

        try:
            data, rate = soundfile.read(
                str(file), dtype="float32", always_2d=True
            )
        except (soundfile.LibsndfileError, RuntimeError):
            return self._with_ffmpeg(file)
        samples = resample(to_mono(data), int(rate), self.sample_rate)
        if samples.size == 0:
            raise InvalidInputError(f"{file} contains no audio.")
        return Audio(samples, self.sample_rate)

    def _with_ffmpeg(self, file: Path) -> Audio:
        """Decode with ffmpeg, or explain what to do without it."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise InvalidInputError(
                f"Cannot decode {file}. Convert it to WAV, FLAC, OGG or "
                "MP3, or install ffmpeg to read other formats."
            )
        done = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(file), "-f", "f32le",
             "-ac", "1", "-ar", str(self.sample_rate), "-"],
            capture_output=True, check=False,
        )
        samples = np.frombuffer(done.stdout, dtype=np.float32).copy()
        if done.returncode != 0 or samples.size == 0:
            reason = done.stderr.decode("utf-8", "replace").strip()
            raise InvalidInputError(
                f"Cannot decode {file}: {reason or 'no audio stream'}"
            )
        return Audio(samples, self.sample_rate)
