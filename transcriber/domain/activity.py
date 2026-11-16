"""Speech activity: which parts of a recording carry sound at all.

Speech recognisers invent words over silence (Whisper hears
" Gracias." in five seconds of digital silence). :func:`speech_activity`
marks the frames of the audio whose RMS level reaches a threshold, and
:func:`heard_words` keeps only the recognised words whose time span
holds at least one such frame. The level is a parameter
(:class:`~transcriber.domain.parameters.TranscriptionParameters`) with
an untuned default; ticket TR-05 measures it.

Recognisers also stop early: the Whisper pipeline's last window can end
its text at a closing phrase ("Gracias a todos, ha sido una buena
reunión.") with speech still to come, and no later window reads it
again (#125). :func:`unheard_tail` tells where speech goes on after the
last recognised word, so the use case can have the rest read again.
"""

import math
from typing import List, NamedTuple, Optional, Sequence

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

    def longest_stretch_after(self, time: float, gap: float) -> float:
        """Return the seconds of the longest stretch of speech after ``time``.

        A stretch runs from an active frame to an active frame and
        bridges runs of inactive frames of at most ``gap`` seconds (the
        pauses between words); a longer pause ends it. Scattered active
        frames (a click, a chair) make stretches of a frame or two. The
        frame that holds ``time`` (when it does not start exactly
        there) does not count.
        """
        first = max(int(math.ceil(time / self.frame)), 0)
        found = np.flatnonzero(self.active[first:])
        if not found.size:
            return 0.0
        bridged = max(int(math.floor(gap / self.frame + 1e-9)), 0)
        cuts = np.flatnonzero(np.diff(found) - 1 > bridged) + 1
        longest = max(
            int(part[-1] - part[0]) + 1 for part in np.split(found, cuts)
        )
        return longest * self.frame


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


def unheard_tail(
    words: Sequence[Word], activity: SpeechActivity, minimum: float,
    gap: float,
) -> Optional[float]:
    """Return where the speech after the last of ``words`` begins.

    That is the latest end of ``words``, when the frames after it hold
    a stretch of speech level of at least ``minimum`` seconds (pauses
    of at most ``gap`` seconds bridged:
    :meth:`SpeechActivity.longest_stretch_after`), so the recogniser
    stopped before the speech did; ``None`` otherwise, for no words or
    for a ``minimum`` that is not positive (the check is off).
    """
    if not words or not minimum > 0:
        return None
    end = max(w.end for w in words)
    stretch = activity.longest_stretch_after(end, gap)
    return end if stretch >= minimum else None
