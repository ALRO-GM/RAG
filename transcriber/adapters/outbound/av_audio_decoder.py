"""Audio decoder: any stored recording to mono 16 kHz float32 samples.

Raw PCM from the phone is read directly; WAV, FLAC and OGG go through
libsndfile (``soundfile``), the fast path; any other container (M4A,
MP4, WEBM, MP3... what phones and meeting tools record) is decoded with
PyAV, which bundles the FFmpeg libraries, so nothing has to be
installed. The resampling helpers are copied from
``src/adapters/outbound/soundfile_audio_reader.py``.
"""

import io
from math import gcd
from typing import TYPE_CHECKING, List

import numpy as np
import numpy.typing as npt

from transcriber.application.ports import PCM_FORMAT, Audio, RecordedAudio
from transcriber.domain.errors import InvalidInputError

if TYPE_CHECKING:  # PyAV is imported only when a file needs it
    from av.audio.frame import AudioFrame

FloatArray = npt.NDArray[np.float32]

DEFAULT_SAMPLE_RATE = 16000
"""What the speech and voice models expect."""

PCM_RATE = 16000
"""Sample rate of :data:`PCM_FORMAT` audio."""

SOUNDFILE_FORMATS = frozenset({"wav", "flac", "ogg"})
"""Formats tried with libsndfile before PyAV."""


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


def pcm_samples(data: bytes) -> FloatArray:
    """Return 16-bit little-endian PCM bytes as floats in [-1, 1)."""
    if len(data) % 2:
        raise InvalidInputError(
            "PCM audio has an odd number of bytes; 16-bit samples need"
            " two each."
        )
    samples: FloatArray = (
        np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    )
    return samples


class AvAudioDecoder:
    """Decode stored recordings into mono samples at ``sample_rate``."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        """Deliver audio at ``sample_rate`` Hz."""
        self.sample_rate = sample_rate

    def decode(self, recording: RecordedAudio) -> Audio:
        """Return ``recording`` as mono float32 samples.

        Raises :class:`InvalidInputError` when the bytes cannot be
        decoded or hold no audio.
        """
        kind = recording.format.lower().lstrip(".")
        if not recording.data:
            raise InvalidInputError("The recording is empty.")
        if kind == PCM_FORMAT:
            samples = resample(
                pcm_samples(recording.data), PCM_RATE, self.sample_rate
            )
        elif kind in SOUNDFILE_FORMATS:
            samples = self._soundfile(recording.data)
        else:
            samples = self._pyav(recording.data, kind)
        if samples.size == 0:
            raise InvalidInputError("The recording contains no audio.")
        return Audio(samples=samples, sample_rate=self.sample_rate)

    def _soundfile(self, data: bytes) -> FloatArray:
        """Decode with libsndfile; fall back to PyAV if it refuses."""
        import soundfile

        try:
            frames, rate = soundfile.read(
                io.BytesIO(data), dtype="float32", always_2d=True
            )
        except (soundfile.LibsndfileError, RuntimeError, TypeError):
            return self._pyav(data, "audio")
        return resample(to_mono(frames), int(rate), self.sample_rate)

    def _pyav(self, data: bytes, kind: str) -> FloatArray:
        """Decode the first audio stream of any container with PyAV."""
        import av

        parts: List[FloatArray] = []
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                if not container.streams.audio:
                    raise InvalidInputError(
                        f"The {kind} recording has no audio stream."
                    )
                # Planar float at the target rate, channels kept: they
                # are averaged below, as on the libsndfile path.
                resampler = av.AudioResampler(
                    format="fltp", rate=self.sample_rate
                )
                stream = container.streams.audio[0]
                for frame in container.decode(stream):
                    for out in resampler.resample(frame):
                        parts.append(_frame_samples(out))
                for out in resampler.resample(None):
                    parts.append(_frame_samples(out))
        except av.FFmpegError as exc:
            raise InvalidInputError(
                f"Cannot decode the {kind} recording: {exc}"
            ) from exc
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)


def _frame_samples(frame: "AudioFrame") -> FloatArray:
    """Return the mean of the channels of a planar PyAV frame."""
    planes = np.asarray(frame.to_ndarray(), dtype=np.float32)
    mono: FloatArray = planes.reshape(planes.shape[0], -1).mean(axis=0)
    return mono.astype(np.float32)
