"""Comparison of pipelines without ground truth: agreement, not accuracy.

Everything here is plain arithmetic on times, labels and words
(``docs/transcriber/LAB.md`` §4 defines each number). Time is cut into
frames of ``frame`` seconds (10 ms); a frame belongs to a speaker when
the frame's middle lies in one of the speaker's segments, so a frame
holds a *set* of speakers (empty: nobody; two or more: overlap).

* :func:`compare_tracks`: two diarizations, one taken as reference:
  agreement over time after the best one-to-one mapping of their
  labels and with the same names, the DER-style components (missed,
  false alarm, confusion), change points matched within a tolerance and
  the overlap each marks with its intersection over union.
* :func:`jaccard_error`: the Jaccard error rate of two grids (used
  against a corpus reference, :mod:`transcriber.domain.lab_reference`).
* :func:`word_error` / :func:`compare_transcripts`: word error rate of
  one recogniser's words against another's, after normalisation, and
  the share of the matched words whose times agree.
* :func:`consensus`: per frame, the set of speakers most of the tracks
  give, one vote per vote group (near-duplicates share a vote); a tie
  is no consensus. :func:`agreement_with` measures a track against it.
* :func:`fragments`: the recording cut at every change point of any
  track, with each track's answer and how much the tracks disagree.

The word "accuracy" is never used: without the owner's labels
(:mod:`transcriber.domain.lab_scoring`) none of these numbers says who
is right.
"""

import math
import re
import unicodedata
from typing import (
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment

from transcriber.domain.lab import ANONYMOUS_PREFIX, Segment
from transcriber.domain.models import Word

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

NO_CONSENSUS = -1
"""State id of a frame on which the vote is tied."""


class ComparisonParameters(BaseModel):
    """The tolerances of the comparison (seconds).

    ``frame``: length of a time frame; ``collar``: frames this close to
    a boundary are not scored (diarizations rarely agree on the exact
    instant a turn starts); ``change_tolerance``: two change points
    this close are the same change; ``cut_merge``: fragment cuts closer
    than this are merged; ``word_time_tolerance``: a matched word whose
    start and end are both this close in two transcripts agrees in
    time.
    """

    model_config = ConfigDict(frozen=True)

    frame: float = Field(default=0.01, gt=0, le=1)
    collar: float = Field(default=0.25, ge=0, le=5)
    change_tolerance: float = Field(default=0.5, gt=0, le=5)
    cut_merge: float = Field(default=0.2, ge=0, le=5)
    word_time_tolerance: float = Field(default=0.2, ge=0, le=5)


class Track(NamedTuple):
    """A diarization in comparable form.

    ``segments`` carry names (enrolled ids, or ``~label`` for a voice no
    one was found for); ``vote_group`` is the consensus vote it shares.
    """

    id: str
    segments: Sequence[Segment]
    vote_group: str


class Frames(NamedTuple):
    """A track on the frame grid: one column per name."""

    names: List[str]
    active: BoolArray

    def speakers(self) -> IntArray:
        """Return how many speakers each frame holds."""
        counted: IntArray = self.active.sum(axis=1).astype(np.int64)
        return counted


def frame_count(duration: float, frame: float) -> int:
    """Return how many frames cover ``duration`` seconds."""
    return max(int(math.ceil(duration / frame - 1e-9)), 0)


def frame_index(seconds: float, frame: float) -> int:
    """Return the first frame whose middle is at or after ``seconds``."""
    return max(int(math.ceil(seconds / frame - 0.5 - 1e-9)), 0)


def to_frames(
    segments: Sequence[Segment], frames: int, frame: float,
    names: Optional[Sequence[str]] = None,
) -> Frames:
    """Return ``segments`` on a grid of ``frames`` frames.

    ``names`` fixes the columns (a name with no segment is an empty
    column); by default the segments' speakers, sorted.
    """
    columns = sorted({s.speaker for s in segments}) if names is None else (
        list(names)
    )
    where = {name: n for n, name in enumerate(columns)}
    active = np.zeros((frames, len(columns)), dtype=bool)
    for segment in segments:
        column = where.get(segment.speaker)
        if column is None:
            continue
        first = min(frame_index(segment.start, frame), frames)
        last = min(frame_index(segment.end, frame), frames)
        active[first:last, column] = True
    return Frames(columns, active)


def local_names(track: Track) -> List[Segment]:
    """Return the track's segments with anonymous names made unique.

    ``~S1`` of one track is not ``~S1`` of another: anonymous names are
    prefixed with the track id (``none+x~S1``) before tracks are compared
    by name. The result has a ``+`` or ``~``, which no enrolled id has.
    """
    return [
        s.model_copy(update={"speaker": f"{track.id}{s.speaker}"})
        if s.speaker.startswith(ANONYMOUS_PREFIX) else s
        for s in track.segments
    ]


def collar_mask(
    segments: Sequence[Segment], frames: int, frame: float, collar: float
) -> BoolArray:
    """Return the frames within ``collar`` seconds of a boundary."""
    mask = np.zeros(frames + 1, dtype=np.int64)
    if collar <= 0 or frames == 0:
        return np.zeros(frames, dtype=bool)
    for segment in segments:
        for edge in (segment.start, segment.end):
            first = min(frame_index(edge - collar, frame), frames)
            last = min(frame_index(edge + collar, frame), frames)
            mask[first] += 1
            mask[last] -= 1
    return np.cumsum(mask)[:frames] > 0


class ChangePoints(BaseModel):
    """Change points of two tracks matched within a tolerance.

    ``f1`` is the harmonic mean of the share of each track's changes
    that the other has (``None`` when neither has any change).
    """

    model_config = ConfigDict(frozen=True)

    reference: int
    hypothesis: int
    matched: int
    f1: Optional[float] = None


class OverlapAgreement(BaseModel):
    """Seconds of overlap (two or more speakers) each track marks.

    ``iou``: seconds both mark over seconds either marks (``None`` when
    neither marks any).
    """

    model_config = ConfigDict(frozen=True)

    reference_seconds: float
    hypothesis_seconds: float
    both_seconds: float
    iou: Optional[float] = None


class TrackComparison(BaseModel):
    """Two diarizations compared, ``reference`` taken as the reference.

    ``agreement``: share of the scored frames with speech on either
    side where both give the same set of speakers after the best
    one-to-one mapping of their labels (symmetric); ``name_agreement``:
    the same with the names as given (enrolled ids), no mapping.
    Frames within the collar of either track's boundaries are not
    scored. ``missed``, ``false_alarm`` and ``confusion`` are seconds
    of speaker time (a frame with two speakers counts twice) against
    ``reference_speech``, scored outside the collar of the reference's
    boundaries only; ``error_rate`` is their sum over
    ``reference_speech`` (the DER, were the reference right).
    """

    model_config = ConfigDict(frozen=True)

    reference: str
    hypothesis: str
    agreement: Optional[float] = None
    name_agreement: Optional[float] = None
    scored_seconds: float = 0.0
    reference_speech: float = 0.0
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    error_rate: Optional[float] = None
    change_points: ChangePoints
    overlap: OverlapAgreement


def _share(part: float, whole: float) -> Optional[float]:
    """Return ``part / whole``, ``None`` when ``whole`` is zero."""
    return float(part / whole) if whole > 0 else None


def _states_equal(left: BoolArray, right: BoolArray) -> BoolArray:
    """Tell per frame whether two equally-shaped activity grids agree."""
    if left.shape[1] == 0:
        return np.ones(left.shape[0], dtype=bool)
    return np.asarray(np.all(left == right, axis=1))


def _aligned(
    left: Frames, right: Frames
) -> Tuple[BoolArray, BoolArray]:
    """Return both grids over the union of their names."""
    names = sorted(set(left.names) | set(right.names))
    out = []
    for grid in (left, right):
        wide = np.zeros((grid.active.shape[0], len(names)), dtype=bool)
        for n, name in enumerate(grid.names):
            wide[:, names.index(name)] = grid.active[:, n]
        out.append(wide)
    return out[0], out[1]


def _mapped(reference: Frames, hypothesis: Frames, scored: BoolArray) -> Tuple[
    BoolArray, BoolArray
]:
    """Return both grids with the hypothesis's labels mapped one to one.

    The mapping maximises the frames the paired labels share on the
    ``scored`` frames; unpaired labels of either side get columns of
    their own.
    """
    ref = reference.active
    hyp = hypothesis.active
    kr, kh = ref.shape[1], hyp.shape[1]
    pairs: List[Tuple[int, int]] = []
    if kr and kh:
        shared = (
            ref[scored].astype(np.int64).T @ hyp[scored].astype(np.int64)
        )
        rows, cols = linear_sum_assignment(shared, maximize=True)
        pairs = [
            (int(r), int(c)) for r, c in zip(rows, cols) if shared[r, c] > 0
        ]
    paired_h = {c for _, c in pairs}
    extra = [c for c in range(kh) if c not in paired_h]
    frames = ref.shape[0]
    left = np.zeros((frames, kr + len(extra)), dtype=bool)
    right = np.zeros_like(left)
    left[:, :kr] = ref
    for r, c in pairs:
        right[:, r] = hyp[:, c]
    for n, c in enumerate(extra):
        right[:, kr + n] = hyp[:, c]
    return left, right


def mapped(
    reference: Frames, hypothesis: Frames, scored: BoolArray
) -> Tuple[BoolArray, BoolArray]:
    """Return both grids with the hypothesis mapped one to one.

    The public name of the mapping DER uses: column ``n`` of both
    results is reference speaker ``reference.names[n]`` and the
    hypothesis label paired with it (columns past the reference's are
    hypothesis labels left unpaired).
    """
    return _mapped(reference, hypothesis, scored)


def change_times(grid: Frames, frame: float) -> List[float]:
    """Return the times at which the set of speakers changes."""
    active = grid.active
    if active.shape[0] < 2:
        return []
    if active.shape[1] == 0:
        return []
    changed = np.any(active[1:] != active[:-1], axis=1)
    return [float((i + 1) * frame) for i in np.flatnonzero(changed)]


def match_times(
    left: Sequence[float], right: Sequence[float], tolerance: float
) -> int:
    """Return how many times of ``left`` pair with one of ``right``.

    Pairs are one to one and at most ``tolerance`` apart; both lists
    are walked in time order and each time takes the earliest free
    partner in reach (optimal for points on a line).
    """
    a, b = sorted(left), sorted(right)
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return matched


def _change_points(
    reference: Frames, hypothesis: Frames, p: ComparisonParameters
) -> ChangePoints:
    """Compare the change points of two grids."""
    ref = change_times(reference, p.frame)
    hyp = change_times(hypothesis, p.frame)
    matched = match_times(ref, hyp, p.change_tolerance)
    f1 = None
    if ref or hyp:
        f1 = 2 * matched / (len(ref) + len(hyp))
    return ChangePoints(
        reference=len(ref), hypothesis=len(hyp), matched=matched, f1=f1
    )


def _overlap(
    reference: Frames, hypothesis: Frames, frame: float
) -> OverlapAgreement:
    """Compare the overlap the two grids mark."""
    ref = reference.speakers() >= 2
    hyp = hypothesis.speakers() >= 2
    both = float(np.sum(ref & hyp)) * frame
    either = float(np.sum(ref | hyp)) * frame
    return OverlapAgreement(
        reference_seconds=float(np.sum(ref)) * frame,
        hypothesis_seconds=float(np.sum(hyp)) * frame,
        both_seconds=both, iou=_share(both, either),
    )


def error_components(
    reference: BoolArray, hypothesis: BoolArray, scored: BoolArray
) -> Tuple[float, float, float, float]:
    """Return frames of (reference speech, missed, false alarm, confusion).

    Both grids must share their columns (same speaker per column). Per
    frame, with R and H the sets of speakers: missed ``max(|R|-|H|, 0)``,
    false alarm ``max(|H|-|R|, 0)``, confusion ``min(|R|, |H|) -
    |R ∩ H|``; summed over the ``scored`` frames (NIST's frame-level
    definition).
    """
    ref = reference[scored]
    hyp = hypothesis[scored]
    r = ref.sum(axis=1)
    h = hyp.sum(axis=1)
    both = (ref & hyp).sum(axis=1)
    return (
        float(r.sum()),
        float(np.maximum(r - h, 0).sum()),
        float(np.maximum(h - r, 0).sum()),
        float((np.minimum(r, h) - both).sum()),
    )


def jaccard_error(
    reference: BoolArray, hypothesis: BoolArray, scored: BoolArray
) -> Optional[float]:
    """Return the Jaccard error rate of two grids (``None``: no speaker).

    The columns of each grid are its own speakers. Reference and
    hypothesis speakers are paired one to one so that the sum of their
    intersection over union (on the ``scored`` frames) is the largest;
    a reference speaker's error is 1 minus the intersection over union
    with its partner (1 without one), and the rate is the mean over the
    reference speakers that speak on a scored frame. A hypothesis
    speaker left without a partner adds nothing.
    """
    ref = reference[scored].astype(np.int64)
    hyp = hypothesis[scored].astype(np.int64)
    sizes = ref.sum(axis=0)
    speaking = np.flatnonzero(sizes > 0)
    if speaking.size == 0:
        return None
    ref = ref[:, speaking]
    sizes = sizes[speaking]
    errors = np.ones(speaking.size, dtype=np.float64)
    if hyp.shape[1]:
        shared = ref.T @ hyp
        union = sizes[:, None] + hyp.sum(axis=0)[None, :] - shared
        iou = shared / np.maximum(union, 1)
        rows, cols = linear_sum_assignment(iou, maximize=True)
        errors[rows] = 1.0 - iou[rows, cols]
    return float(errors.mean())


def compare_tracks(
    reference: Track, hypothesis: Track, duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> TrackComparison:
    """Compare two diarizations of the same audio (see the class)."""
    frames = frame_count(duration, p.frame)
    ref_segments = local_names(reference)
    hyp_segments = local_names(hypothesis)
    ref = to_frames(ref_segments, frames, p.frame)
    hyp = to_frames(hyp_segments, frames, p.frame)
    ref_collar = collar_mask(ref_segments, frames, p.frame, p.collar)
    either_collar = ref_collar | collar_mask(
        hyp_segments, frames, p.frame, p.collar
    )
    scored = ~either_collar
    left, right = _mapped(ref, hyp, scored)
    speech = scored & (left.any(axis=1) | right.any(axis=1))
    same = _states_equal(left, right) & speech
    named_left, named_right = _aligned(ref, hyp)
    named_same = _states_equal(named_left, named_right) & speech
    der_scored = ~ref_collar
    der_left, der_right = _mapped(ref, hyp, der_scored)
    total, missed, false_alarm, confusion = error_components(
        der_left, der_right, der_scored
    )
    spoken = float(np.sum(speech))
    return TrackComparison(
        reference=reference.id, hypothesis=hypothesis.id,
        agreement=_share(float(np.sum(same)), spoken),
        name_agreement=_share(float(np.sum(named_same)), spoken),
        scored_seconds=spoken * p.frame,
        reference_speech=total * p.frame,
        missed=missed * p.frame, false_alarm=false_alarm * p.frame,
        confusion=confusion * p.frame,
        error_rate=_share(missed + false_alarm + confusion, total),
        change_points=_change_points(ref, hyp, p),
        overlap=_overlap(ref, hyp, p.frame),
    )


class TrackSummary(BaseModel):
    """What one diarization found.

    ``speakers``: names with at least one second of speech;
    ``speech_seconds``: time with at least one speaker;
    ``overlap_seconds``: time with two or more.
    """

    model_config = ConfigDict(frozen=True)

    track: str
    speakers: List[str]
    speech_seconds: float
    overlap_seconds: float
    speaker_seconds: Dict[str, float]


def summarise_track(
    track: Track, duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> TrackSummary:
    """Return what ``track`` found (see :class:`TrackSummary`)."""
    grid = to_frames(track.segments, frame_count(duration, p.frame), p.frame)
    count = grid.speakers()
    seconds = {
        name: float(grid.active[:, n].sum()) * p.frame
        for n, name in enumerate(grid.names)
    }
    return TrackSummary(
        track=track.id,
        speakers=sorted(n for n, s in seconds.items() if s >= 1.0),
        speech_seconds=float(np.sum(count >= 1)) * p.frame,
        overlap_seconds=float(np.sum(count >= 2)) * p.frame,
        speaker_seconds={n: round(s, 3) for n, s in seconds.items()},
    )


# ---------------------------------------------------------------- consensus


class StateTable(NamedTuple):
    """Every track as one state id per frame, over shared states.

    ``states[i]`` is the set of names of state ``i`` (anonymous names
    made track-local), ``ids[track]`` the state of each frame.
    """

    states: List[FrozenSet[str]]
    ids: Dict[str, IntArray]


def state_table(
    tracks: Sequence[Track], duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> StateTable:
    """Return the frame states of ``tracks`` (see :class:`StateTable`)."""
    frames = frame_count(duration, p.frame)
    states: List[FrozenSet[str]] = []
    index: Dict[FrozenSet[str], int] = {}
    ids: Dict[str, IntArray] = {}
    for track in tracks:
        grid = to_frames(local_names(track), frames, p.frame)
        rows: npt.NDArray[np.bool_]
        inverse: npt.NDArray[np.int64]
        if grid.active.shape[1] == 0:
            rows = np.zeros((1, 0), dtype=bool)
            inverse = np.zeros(frames, dtype=np.int64)
        else:
            unique, found = np.unique(
                grid.active, axis=0, return_inverse=True
            )
            rows = np.asarray(unique, dtype=bool)
            inverse = np.asarray(found, dtype=np.int64)
        local = []
        for row in rows:
            state = frozenset(
                name for name, on in zip(grid.names, row) if on
            )
            if state not in index:
                index[state] = len(states)
                states.append(state)
            local.append(index[state])
        lookup = np.array(local, dtype=np.int64)
        ids[track.id] = lookup[np.asarray(inverse).reshape(-1)] if frames \
            else np.zeros(0, dtype=np.int64)
    return StateTable(states, ids)


def vote_weights(tracks: Sequence[Track]) -> Dict[str, float]:
    """Return each track's vote: one per vote group, shared equally."""
    sizes: Dict[str, int] = {}
    for track in tracks:
        sizes[track.vote_group] = sizes.get(track.vote_group, 0) + 1
    return {t.id: 1.0 / sizes[t.vote_group] for t in tracks}


class Consensus(NamedTuple):
    """The consensus state of each frame (:data:`NO_CONSENSUS` on ties)."""

    table: StateTable
    frame_states: IntArray
    weights: Dict[str, float]


def consensus(
    tracks: Sequence[Track], duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> Consensus:
    """Return, per frame, the set of speakers with the most votes.

    Each vote group (near-duplicates: one method with other weights, or
    one diarizer on several denoised versions) has one vote, shared by
    its tracks. A frame whose highest vote is tied has no consensus.
    """
    table = state_table(tracks, duration, p)
    weights = vote_weights(tracks)
    frames = frame_count(duration, p.frame)
    count = len(table.states)
    votes = np.zeros((frames, max(count, 1)), dtype=np.float64)
    rows = np.arange(frames)
    for track in tracks:
        votes[rows, table.ids[track.id]] += weights[track.id]
    if frames == 0 or not tracks:
        return Consensus(
            table, np.full(frames, NO_CONSENSUS, dtype=np.int64), weights
        )
    best = votes.max(axis=1)
    winners = (votes >= best[:, None] - 1e-9).sum(axis=1)
    chosen = votes.argmax(axis=1).astype(np.int64)
    chosen[winners > 1] = NO_CONSENSUS
    return Consensus(table, chosen, weights)


class ConsensusAgreement(BaseModel):
    """How often a track gives the consensus set of speakers.

    Counted on the frames that have a consensus and speech in the
    consensus or in the track (both silent frames are left out, so
    silence does not inflate it).
    """

    model_config = ConfigDict(frozen=True)

    track: str
    agreement: Optional[float] = None
    scored_seconds: float = 0.0


def agreement_with(
    agreed: Consensus, track: str, frame: float
) -> ConsensusAgreement:
    """Return how far ``track`` agrees with the consensus."""
    ids = agreed.table.ids[track]
    silent = {
        n for n, state in enumerate(agreed.table.states) if not state
    }
    has = agreed.frame_states != NO_CONSENSUS
    quiet = np.isin(agreed.frame_states, list(silent)) & np.isin(
        ids, list(silent)
    )
    scored = has & ~quiet
    same = scored & (ids == agreed.frame_states)
    total = float(np.sum(scored))
    return ConsensusAgreement(
        track=track, agreement=_share(float(np.sum(same)), total),
        scored_seconds=total * frame,
    )


class StateSpan(BaseModel):
    """A stretch with one consensus answer.

    ``speakers`` is the agreed set (empty: nobody); ``agreed`` is false
    where the vote was tied (then ``speakers`` is empty).
    """

    model_config = ConfigDict(frozen=True)

    start: float
    end: float
    speakers: List[str]
    agreed: bool = True


def consensus_spans(agreed: Consensus, frame: float) -> List[StateSpan]:
    """Return the consensus as stretches of constant answer."""
    states = agreed.frame_states
    spans: List[StateSpan] = []
    if states.shape[0] == 0:
        return spans
    edges = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = [0, *edges.tolist()]
    ends = [*edges.tolist(), int(states.shape[0])]
    for first, last in zip(starts, ends):
        state = int(states[first])
        names: List[str] = (
            sorted(agreed.table.states[state]) if state >= 0 else []
        )
        spans.append(StateSpan(
            start=round(first * frame, 3), end=round(last * frame, 3),
            speakers=names, agreed=state >= 0,
        ))
    return spans


def consensus_coverage(agreed: Consensus) -> Optional[float]:
    """Return the share of frames that have a consensus."""
    total = float(agreed.frame_states.shape[0])
    return _share(float(np.sum(agreed.frame_states >= 0)), total)


# ------------------------------------------------------------------- words


def attribute(
    words: Sequence[Word], segments: Sequence[Segment]
) -> List[Optional[str]]:
    """Return the speaker of each word: the segment it overlaps most.

    The rule of :func:`transcriber.domain.speakers.assign_speakers`
    (a word over no segment takes the nearest one in time; among equal
    overlaps the one with the nearer edge), computed on arrays; ``None``
    when there is no segment.
    """
    if not segments:
        return [None] * len(words)
    if not words:
        return []
    ws = np.array([w.start for w in words])[:, None]
    we = np.array([w.end for w in words])[:, None]
    ss = np.array([s.start for s in segments])[None, :]
    se = np.array([s.end for s in segments])[None, :]
    overlap = np.minimum(we, se) - np.maximum(ws, ss)
    middle = (ws + we) / 2
    distance = np.minimum(np.abs(middle - ss), np.abs(middle - se))
    best = overlap.max(axis=1, keepdims=True)
    candidates = overlap >= best - 1e-12
    chosen = np.where(candidates, distance, np.inf).argmin(axis=1)
    names = [s.speaker for s in segments]
    return [names[int(i)] for i in chosen]


def word_agreement(
    left: Sequence[Optional[str]], right: Sequence[Optional[str]]
) -> Optional[float]:
    """Return the share of words given the same speaker in both lists."""
    if not left:
        return None
    same = sum(1 for a, b in zip(left, right) if a is not None and a == b)
    return same / len(left)


def words_on_consensus(
    words: Sequence[Word], speakers: Sequence[Optional[str]],
    agreed: Consensus, frame: float,
) -> Optional[float]:
    """Return the share of words whose speaker is in the consensus set.

    Only words whose middle falls on a frame with a consensus naming at
    least one speaker are counted.
    """
    states = agreed.frame_states
    hits = counted = 0
    for word, speaker in zip(words, speakers):
        index = int(((word.start + word.end) / 2) / frame)
        if index >= states.shape[0] or states[index] < 0:
            continue
        state = agreed.table.states[int(states[index])]
        if not state:
            continue
        counted += 1
        hits += speaker is not None and speaker in state
    return _share(hits, counted)


# ------------------------------------------------------------- recognition

_PUNCTUATION = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalise_text(text: str) -> List[str]:
    """Return the tokens of ``text`` for word error rate.

    Unicode NFKC, lower case, punctuation removed (``¿qué?`` → ``qué``),
    split on white space; numbers are kept as written (``25`` and
    ``veinticinco`` differ).
    """
    plain = unicodedata.normalize("NFKC", text).lower()
    return _PUNCTUATION.sub(" ", plain).split()


def tokens_of(words: Sequence[Word]) -> Tuple[List[str], List[int]]:
    """Return the normalised tokens of ``words`` and each one's word."""
    tokens: List[str] = []
    owners: List[int] = []
    for n, word in enumerate(words):
        for token in normalise_text(word.text):
            tokens.append(token)
            owners.append(n)
    return tokens, owners


class Alignment(NamedTuple):
    """An edit-distance alignment of two token lists.

    ``errors`` is substitutions + deletions + insertions; ``pairs`` the
    (reference, hypothesis) indices of the tokens that matched.
    """

    errors: int
    reference: int
    pairs: List[Tuple[int, int]]


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Return the minimum edit alignment of two token lists.

    The distance rows are computed with arrays (insertions through a
    running minimum), then one optimal path is traced back.
    """
    n, m = len(reference), len(hypothesis)
    if n == 0 or m == 0:
        return Alignment(max(n, m), n, [])
    vocabulary: Dict[str, int] = {}
    ref = np.array([vocabulary.setdefault(t, len(vocabulary))
                    for t in reference], dtype=np.int64)
    hyp = np.array([vocabulary.setdefault(t, len(vocabulary))
                    for t in hypothesis], dtype=np.int64)
    table = np.zeros((n + 1, m + 1), dtype=np.int32)
    table[0] = np.arange(m + 1)
    steps = np.arange(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        previous = table[i - 1]
        cost = (hyp != ref[i - 1]).astype(np.int32)
        row = np.empty(m + 1, dtype=np.int32)
        row[0] = i
        row[1:] = np.minimum(previous[1:] + 1, previous[:-1] + cost)
        # insertions: row[j] = min(row[j], row[j-1] + 1), all at once
        table[i] = np.minimum.accumulate(row - steps) + steps
    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        same = ref[i - 1] == hyp[j - 1]
        if table[i, j] == table[i - 1, j - 1] + (0 if same else 1):
            if same:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif table[i, j] == table[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return Alignment(int(table[n, m]), n, pairs)


def word_error(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> Optional[float]:
    """Return the word error rate of ``hypothesis`` (``None``: no words)."""
    result = align(reference, hypothesis)
    return _share(result.errors, result.reference)


class TranscriptComparison(BaseModel):
    """One recogniser's words against another's, taken as reference.

    ``wer``: word error rate after :func:`normalise_text` (an agreement
    measure here: the reference is another model, not the truth).
    ``time_agreement``: share of the matched tokens whose words start
    and end within the tolerance in both.
    """

    model_config = ConfigDict(frozen=True)

    reference: str
    hypothesis: str
    wer: Optional[float] = None
    reference_tokens: int = 0
    matched_tokens: int = 0
    time_agreement: Optional[float] = None


def compare_transcripts(
    reference_id: str, reference: Sequence[Word],
    hypothesis_id: str, hypothesis: Sequence[Word],
    p: ComparisonParameters = ComparisonParameters(),
) -> TranscriptComparison:
    """Compare two recognisers' words (see :class:`TranscriptComparison`)."""
    ref_tokens, ref_owner = tokens_of(reference)
    hyp_tokens, hyp_owner = tokens_of(hypothesis)
    result = align(ref_tokens, hyp_tokens)
    timed = 0
    for i, j in result.pairs:
        a = reference[ref_owner[i]]
        b = hypothesis[hyp_owner[j]]
        if (abs(a.start - b.start) <= p.word_time_tolerance
                and abs(a.end - b.end) <= p.word_time_tolerance):
            timed += 1
    return TranscriptComparison(
        reference=reference_id, hypothesis=hypothesis_id,
        wer=_share(result.errors, result.reference),
        reference_tokens=result.reference,
        matched_tokens=len(result.pairs),
        time_agreement=_share(timed, len(result.pairs)),
    )


# --------------------------------------------------------------- fragments


def cut_points(
    tracks: Sequence[Track], duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> List[float]:
    """Return where the fragments start: every change of any track.

    Cuts closer than ``cut_merge`` are merged into their mean; the
    recording's start and end are always cuts.
    """
    frames = frame_count(duration, p.frame)
    times: List[float] = []
    for track in tracks:
        grid = to_frames(track.segments, frames, p.frame)
        times.extend(change_times(grid, p.frame))
    times = sorted(t for t in times if 0 < t < duration)
    merged: List[List[float]] = []
    for t in times:
        if merged and t - merged[-1][-1] < p.cut_merge:
            merged[-1].append(t)
        else:
            merged.append([t])
    cuts = [round(sum(group) / len(group), 3) for group in merged]
    cuts = [c for c in cuts if p.cut_merge <= c <= duration - p.cut_merge]
    return [0.0, *cuts, round(duration, 3)]


class Fragment(BaseModel):
    """A piece of the recording between two cuts, with each answer.

    ``answers`` maps a track id to the set of speakers it gives for
    most of the fragment. ``distinct_answers`` counts the different
    sets; ``disagreement`` is one minus the vote share of the most
    voted set (one vote per vote group, as in the consensus): 0 when
    every track gives the same answer.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    start: float
    end: float
    answers: Dict[str, List[str]]
    distinct_answers: int
    disagreement: float
    modal: List[str]


def fragments(
    tracks: Sequence[Track], duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> List[Fragment]:
    """Return the fragments of the recording (see :class:`Fragment`)."""
    cuts = cut_points(tracks, duration, p)
    table = state_table(tracks, duration, p)
    weights = vote_weights(tracks)
    groups = len({t.vote_group for t in tracks}) or 1
    out: List[Fragment] = []
    frames = frame_count(duration, p.frame)
    for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
        first = min(frame_index(start, p.frame), frames)
        last = max(min(frame_index(end, p.frame), frames), first + 1)
        answers: Dict[str, List[str]] = {}
        votes: Dict[FrozenSet[str], float] = {}
        for track in tracks:
            ids = table.ids[track.id][first:last]
            if ids.shape[0] == 0:
                state: FrozenSet[str] = frozenset()
            else:
                state = table.states[int(np.bincount(ids).argmax())]
            answers[track.id] = sorted(_shown(state, track.id))
            votes[state] = votes.get(state, 0.0) + weights[track.id]
        modal_state, modal_vote = max(
            votes.items(), key=lambda kv: (kv[1], -len(kv[0])),
            default=(frozenset(), 0.0),
        )
        out.append(Fragment(
            index=index, start=start, end=end, answers=answers,
            distinct_answers=len(votes),
            disagreement=round(1.0 - modal_vote / groups, 6)
            if tracks else 0.0,
            modal=sorted(_shown(modal_state, None)),
        ))
    return out


def _shown(state: FrozenSet[str], track: Optional[str]) -> List[str]:
    """Return the names of ``state`` as shown, track prefixes removed."""
    shown = []
    for name in state:
        if track is not None and name.startswith(track + ANONYMOUS_PREFIX):
            name = name[len(track):]
        shown.append(name)
    return shown


def words_between(
    words: Sequence[Word], start: float, end: float
) -> List[int]:
    """Return the indices of the words whose middle is in [start, end)."""
    return [
        n for n, w in enumerate(words)
        if start <= (w.start + w.end) / 2 < end
    ]


def text_of(words: Sequence[Word], indices: Sequence[int]) -> str:
    """Return the words at ``indices`` joined with spaces."""
    return " ".join(words[n].text.strip() for n in indices).strip()


def distinct_texts(texts: Mapping[str, str]) -> int:
    """Return how many different normalised texts ``texts`` holds."""
    return len({" ".join(normalise_text(t)) for t in texts.values()})
