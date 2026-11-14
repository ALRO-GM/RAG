"""Speech activity: which parts of a recording carry sound at all.

Speech recognisers invent words over silence (Whisper hears
" Gracias." in five seconds of digital silence). :func:`speech_activity`
marks the frames of the audio whose RMS level reaches a threshold, and
:func:`heard_words` keeps only the recognised words whose time span
holds at least one such frame. The level is a parameter
(:class:`~transcriber.domain.parameters.TranscriptionParameters`) with
an untuned default; ticket TR-05 measures it.
"""

import math
from typing import List, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Word


class SpeechActivity(NamedTuple):
    """Which frames of ``frame`` seconds reach the level of speech.

    ``active[i]`` covers ``[i * frame, (i + 1) * frame)`` seconds.
    """

    frame: float
    active: npt.NDArray[np.bool_]

    def heard(self, start: float, end: float) -> bool:
        """Tell whether any frame touched by ``[start, end]`` is active.

        A span of zero length looks at the frame that holds ``start``;
        a span past the end of the audio holds no active frame.
        """
        count = int(self.active.shape[0])
        first = max(int(math.floor(start / self.frame)), 0)
        last = max(int(math.ceil(end / self.frame)), first + 1)
        if first >= count:
            return False
        return bool(self.active[first:min(last, count)].any())


def speech_activity(
    samples: npt.NDArray[np.float32],
    sample_rate: int,
    frame: float,
    threshold: float,
) -> SpeechActivity:
    """Return the frames of ``samples`` whose RMS level reaches ``threshold``.

    ``samples`` are mono in [-1, 1] at ``sample_rate``; a frame is
    ``frame`` seconds (at least one sample) and the last, partial frame
    counts too. Raises :class:`InvalidInputError` for samples that are
    not 1-D or a rate or frame that is not positive.
    """
    if samples.ndim != 1:
        raise InvalidInputError("Audio samples must be a 1-D array.")
    if sample_rate <= 0 or not frame > 0:
        raise InvalidInputError("Sample rate and frame must be positive.")
    size = max(int(round(frame * sample_rate)), 1)
    count = -(-int(samples.shape[0]) // size)
    padded = np.zeros(count * size, dtype=np.float64)
    padded[:samples.shape[0]] = samples
    squares = np.square(padded).reshape(count, size)
    # Mean over the real samples of each frame (the last may be short).
    lengths = np.full(count, size, dtype=np.float64)
    if count and samples.shape[0] % size:
        lengths[-1] = samples.shape[0] % size
    rms = np.sqrt(squares.sum(axis=1) / np.maximum(lengths, 1.0))
    return SpeechActivity(size / sample_rate, rms >= threshold)


def heard_words(
    words: Sequence[Word], activity: SpeechActivity
) -> List[Word]:
    """Return the words whose span holds at least one active frame."""
    return [w for w in words if activity.heard(w.start, w.end)]
