"""A cheap check that a recording holds speech rather than steady noise.

Some rows of public corpora hold no speech at all although their
reference says people talk: 7 of the first 28 CALLHOME Spanish rows on
the Hugging Face Hub hold steady broadband noise up to 4 kHz
(``docs/transcriber/LAB.md`` §13.6). Every diarizer finds nobody there,
so their DER of about 1 says nothing about the models. The import skips
such recordings, and sessions imported before can be checked again.

:func:`check_speech` cuts the samples into frames of
:data:`FRAME_SECONDS`, keeps the loudest :data:`LOUD_SHARE` of them
(where the speech would be) and measures on each:

* the lag-1 autocorrelation (the sum of ``x[t] * x[t + 1]`` over the sum
  of ``x[t]²``, mean removed): speech carries most of its energy at low
  frequencies, so neighbouring samples move together (near 1); noise
  spread evenly up to 4 kHz at 16 kHz gives much less;
* the spectral flatness between :data:`FLATNESS_LOW_HZ` and
  :data:`FLATNESS_HIGH_HZ` (the geometric over the arithmetic mean of
  the power spectrum, Hann window): near 0 for the harmonics and
  formants of a voice, high for a flat noise spectrum.

Each figure is the median over the frames kept. A recording fails (no
speech) when both say noise: autocorrelation below
:data:`MIN_AUTOCORRELATION` **and** flatness above :data:`MAX_FLATNESS`.
The thresholds lie between the figures measured on the 28 rows of
CALLHOME shard 0 and the 6 AMI test meetings of the first shard
(LAB.md §13.6). Plain numpy arithmetic, no I/O: 0.02–0.05 s per
recording of CALLHOME (5–10 min) and 0.06–0.18 s per AMI meeting
(14–39 min) on the development machine's CPU, after decoding.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

FRAME_SECONDS = 0.032
"""Length of a frame (512 samples at 16 kHz)."""

LOUD_SHARE = 0.3
"""Share of the frames measured: the loudest ones."""

FLATNESS_LOW_HZ = 0.0
"""Lowest frequency of the flatness band (the DC bin is left out)."""

FLATNESS_HIGH_HZ = 4000.0
"""Highest frequency of the flatness band (telephone speech)."""

MIN_AUTOCORRELATION = 0.8
"""Below this lag-1 autocorrelation the loud frames look like noise."""

MAX_FLATNESS = 0.2
"""Above this spectral flatness the loud frames look like noise."""

NO_SPEECH = "no speech in the audio"
"""Why a recording that fails the check is not used."""

_TINY = 1e-20


class SpeechCheck(BaseModel):
    """The figures of :func:`check_speech` and its verdict.

    ``frames`` is how many frames were measured (0 when the audio is
    shorter than a frame: then the check cannot tell and passes).
    """

    model_config = ConfigDict(frozen=True)

    autocorrelation: float
    flatness: float
    frames: int
    speech: bool

    def reason(self) -> Optional[str]:
        """Return why the recording is not used (``None`` when it is)."""
        return None if self.speech else NO_SPEECH

    def describe(self) -> str:
        """Return the figures as a short phrase (for messages)."""
        return (
            f"lag-1 autocorrelation {self.autocorrelation:.3f},"
            f" spectral flatness {self.flatness:.3f}"
        )


def check_speech(
    samples: npt.NDArray[np.float32], sample_rate: int
) -> SpeechCheck:
    """Tell whether mono ``samples`` hold speech (see the module)."""
    size = max(int(round(FRAME_SECONDS * sample_rate)), 2)
    count = int(samples.shape[0]) // size
    if count == 0:
        return SpeechCheck(
            autocorrelation=1.0, flatness=0.0, frames=0, speech=True
        )
    frames = samples[: count * size].reshape(count, size)
    energy = np.einsum(
        "ij,ij->i", frames, frames, dtype=np.float64
    ) / size
    keep = max(int(np.ceil(count * LOUD_SHARE)), 1)
    loudest = np.argsort(energy, kind="stable")[count - keep:]
    loud = frames[np.sort(loudest)].astype(np.float64)
    loud -= loud.mean(axis=1, keepdims=True)
    power = (loud * loud).sum(axis=1)
    lagged = (loud[:, 1:] * loud[:, :-1]).sum(axis=1)
    autocorrelation = float(np.median(lagged / np.maximum(power, _TINY)))
    spectrum = np.abs(
        np.fft.rfft(loud * np.hanning(size), axis=1)
    ) ** 2 + _TINY
    low = max(int(FLATNESS_LOW_HZ * size / sample_rate), 1)
    high = min(
        int(FLATNESS_HIGH_HZ * size / sample_rate), spectrum.shape[1] - 1
    )
    band = spectrum[:, low:high + 1]
    flat = np.exp(np.log(band).mean(axis=1)) / band.mean(axis=1)
    flatness = float(np.median(flat))
    noise = autocorrelation < MIN_AUTOCORRELATION and flatness > MAX_FLATNESS
    return SpeechCheck(
        autocorrelation=round(autocorrelation, 4),
        flatness=round(flatness, 4), frames=keep, speech=not noise,
    )
