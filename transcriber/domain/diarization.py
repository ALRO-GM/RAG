"""Voices from an external diarizer: which model, labels, vectors, scores.

A transcription can take its voices from a diarizer of the lab (TR-18,
``docs/transcriber/ARCHITECTURE.md`` §6) instead of the built-in
grouping of voice windows. :class:`DiarizerRule` says which model a
session gets from the number of speakers it expects. The rest turns the
diarizer's segments into what the transcription's later stages already
use:

- :func:`segment_spans`: the segments as speaker spans, the labels
  renamed ``SPEAKER_01``, ``SPEAKER_02``... in order of first speech;
- :func:`label_windows`: for each label, short windows of the stretches
  where it speaks alone (:func:`exclusive_regions`), so its voice can
  be embedded and compared with the enrolled voiceprints, as the lab's
  speaker mapping does;
- :func:`segment_voices`: from those windows' embeddings, one raw
  unit-length vector per label (compared with voiceprints) and the
  cosine of every short analysis window with every label (the evidence
  of each sentence's speaker certainty, as the built-in grouping gives).

Plain arithmetic on times and vectors: no model, no I/O.
"""

from typing import (
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.comparison import frame_count, to_frames
from transcriber.domain.lab import BUILTIN_DIARIZER, Segment
from transcriber.domain.speakers import (
    SpeakerSpan,
    analysis_windows,
    speaker_label,
)
from transcriber.domain.voices import FloatArray, centre, unit_rows

FRAME = 0.01
"""Time step (seconds) of the exclusive-speech grid."""

MAX_LABEL_WINDOWS = 120
"""Windows embedded per diarizer label at most (evenly spread)."""

DEFAULT_DIARIZER = "diarizen-wavlm-large-s80-md-v2"
"""Best measured on 4-person far-field meetings (``LAB.md`` §13.8)."""

TWO_SPEAKER_DIARIZER = "nemotron-3-diarization"
"""Best measured on 2-person telephone calls (``LAB.md`` §13.8)."""


class DiarizerRule(BaseModel):
    """Which diarizer a transcription uses.

    ``model`` is the catalogue id used by default (``builtin``: the
    service's own grouping of voice windows); ``by_speakers`` maps a
    number of expected speakers (the session's ``num_speakers``) to the
    model used instead. The defaults are the best of ``LAB.md`` §13.8:
    DiariZen (non-commercial weights) and, for two people, Nemotron-3.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(default=DEFAULT_DIARIZER, min_length=1)
    by_speakers: Dict[int, str] = Field(
        default_factory=lambda: {2: TWO_SPEAKER_DIARIZER}
    )

    def model_for(self, num_speakers: Optional[int]) -> str:
        """Return the model for a session expecting ``num_speakers``."""
        if num_speakers is not None and num_speakers in self.by_speakers:
            return self.by_speakers[num_speakers]
        return self.model

    def models(self) -> List[str]:
        """Return every model the rule may pick, default first."""
        return list(dict.fromkeys([self.model, *self.by_speakers.values()]))


BUILTIN_RULE = DiarizerRule(model=BUILTIN_DIARIZER, by_speakers={})
"""The rule that always uses the built-in grouping."""


class Diarization(BaseModel):
    """What an external diarizer heard in a recording.

    ``segments`` may overlap (two people at once); ``overlaps`` are the
    stretches the diarizer itself marks as two or more voices.
    ``licence`` and ``commercial`` are the catalogue's terms of the
    model's weights; ``version`` the weights' revision when known.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(min_length=1)
    licence: Optional[str] = None
    commercial: Optional[bool] = None
    version: Optional[str] = None
    segments: List[Segment] = Field(default_factory=list)
    overlaps: List[Tuple[float, float]] = Field(default_factory=list)
    peak_gpu_mb: Optional[float] = None


def label_order(segments: Sequence[Segment]) -> Dict[str, int]:
    """Return each diarizer label's number, in order of first speech."""
    order: Dict[str, int] = {}
    for segment in sorted(segments, key=lambda s: (s.start, s.end)):
        order.setdefault(segment.speaker, len(order))
    return order


def segment_spans(
    segments: Sequence[Segment], order: Mapping[str, int]
) -> List[SpeakerSpan]:
    """Return the segments as spans labelled ``speaker_label(number)``.

    Overlapping spans are kept: a word takes the span it overlaps most
    (:func:`~transcriber.domain.speakers.assign_speakers`).
    """
    return [
        SpeakerSpan(
            speaker=speaker_label(order[s.speaker]), start=s.start,
            end=s.end,
        )
        for s in sorted(segments, key=lambda s: (s.start, s.end))
        if s.end > s.start
    ]


def exclusive_regions(
    segments: Sequence[Segment], duration: float, minimum: float = 0.4
) -> Dict[str, List[Tuple[float, float]]]:
    """Return, per label, the stretches where it is the only speaker.

    Stretches shorter than ``minimum`` seconds are dropped.
    """
    frames = frame_count(duration, FRAME)
    grid = to_frames(segments, frames, FRAME)
    alone = grid.active & (grid.active.sum(axis=1) == 1)[:, None]
    regions: Dict[str, List[Tuple[float, float]]] = {}
    for column, name in enumerate(grid.names):
        on = np.concatenate(([False], alone[:, column], [False]))
        edges = np.flatnonzero(on[1:] != on[:-1])
        found = [
            (round(a * FRAME, 3), round(b * FRAME, 3))
            for a, b in zip(edges[::2], edges[1::2])
            if (b - a) * FRAME >= minimum
        ]
        regions[name] = found
    return regions


def spread(
    windows: Sequence[Tuple[float, float]], limit: int
) -> List[Tuple[float, float]]:
    """Return at most ``limit`` of ``windows``, evenly spread."""
    if len(windows) <= limit:
        return list(windows)
    picks = np.linspace(0, len(windows) - 1, limit).round().astype(int)
    return [windows[int(i)] for i in picks]


def label_windows(
    segments: Sequence[Segment],
    order: Mapping[str, int],
    duration: float,
    length: float,
    step: float,
    minimum: float,
    limit: int = MAX_LABEL_WINDOWS,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Return windows of each label's own voice and the label of each.

    The windows (``length``, ``step``, ``minimum`` as in
    :func:`~transcriber.domain.speakers.analysis_windows`) cut the
    stretches where the label speaks alone; a label that is never alone
    long enough is cut over its segments instead (overlapped speech is
    better than no vector). At most ``limit`` per label, evenly spread.
    A label too short for any window gets none.
    """
    regions = exclusive_regions(segments, duration, minimum)
    windows: List[Tuple[float, float]] = []
    owners: List[int] = []
    for label, number in sorted(order.items(), key=lambda item: item[1]):
        own = analysis_windows(regions.get(label, []), length, step, minimum)
        if not own:
            own = analysis_windows(
                [(s.start, s.end) for s in segments if s.speaker == label],
                length, step, minimum,
            )
        own = spread(own, limit)
        windows.extend(own)
        owners.extend([number] * len(own))
    return windows, owners


class SegmentVoices(NamedTuple):
    """The voice of each diarizer label and the windows' cosines.

    ``vectors`` has one raw unit-length mean per label (row ``n`` is
    label ``n``; a label with no window is a zero row, which matches no
    voiceprint). ``fine_scores`` holds, per short analysis window
    (row) and label (column), the cosine of the window with the label's
    voice; ``None`` with fewer than two heard labels or no window.
    """

    vectors: FloatArray
    fine_scores: Optional[FloatArray] = None


def segment_voices(
    label_vectors: FloatArray,
    owners: Sequence[int],
    labels: int,
    fine: FloatArray,
    centre_windows: bool = True,
) -> SegmentVoices:
    """Return the voices of ``labels`` diarizer labels (see the class).

    ``label_vectors`` are the embeddings of the windows of
    :func:`label_windows` (row ``i`` belongs to label ``owners[i]``);
    ``fine`` the embeddings of the short analysis windows over the
    recognised speech. With ``centre_windows`` the cosines are taken on
    vectors centred on the mean of the label windows, as the built-in
    grouping centres on its long windows; the vectors returned for
    matching are the raw unit means either way.
    """
    width = int(label_vectors.shape[1]) if label_vectors.ndim == 2 else 0
    if len(owners) != label_vectors.shape[0]:
        raise ValueError("One owner per window vector is required.")
    raw = unit_rows(label_vectors) if width else label_vectors
    vectors = np.zeros((labels, width), dtype=np.float32)
    heard: List[int] = []
    for number in range(labels):
        rows = [i for i, owner in enumerate(owners) if owner == number]
        if rows:
            vectors[number] = _unit(raw[rows].mean(axis=0))
            heard.append(number)
    if len(heard) < 2 or fine.ndim != 2 or fine.shape[0] == 0:
        return SegmentVoices(vectors)
    if fine.shape[1] != width:
        raise ValueError("Label windows and short windows differ in size.")
    if centre_windows:
        mean: FloatArray = label_vectors.mean(axis=0).astype(np.float32)
        windows, short = centre(label_vectors, mean), centre(fine, mean)
    else:
        windows, short = raw, unit_rows(fine)
    centroids = np.zeros((labels, width), dtype=np.float32)
    for number in heard:
        rows = [i for i, owner in enumerate(owners) if owner == number]
        centroids[number] = _unit(windows[rows].mean(axis=0))
    scores: FloatArray = (short @ centroids.T).astype(np.float32)
    # A label with no window cannot be preferred by any window.
    for number in range(labels):
        if number not in heard:
            scores[:, number] = -1.0
    return SegmentVoices(vectors, scores)


def _unit(vector: FloatArray) -> FloatArray:
    """Return ``vector`` scaled to unit length (zero stays zero)."""
    norm = float(np.linalg.norm(vector))
    out: FloatArray = (vector / norm if norm > 0 else vector).astype(
        np.float32
    )
    return out
