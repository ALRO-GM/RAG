"""Certainty: how well the voice windows support each attribution.

Diarization labels every short analysis window with the voice group it
is closest to (a cosine with each group's centroid; see
:class:`~transcriber.domain.voices.VoiceGroups.fine_scores`). Those
cosines are kept here as :class:`VoiceEvidence` and used twice:

- :func:`keep_sentences_whole` stops a weakly supported speaker change
  from cutting a sentence in two (the pieces otherwise go to different
  people), and tells which stretches it moved;
- :func:`assess_utterances` gives every sentence its
  ``speaker_certainty`` (a similarity margin), ``doubtful``,
  ``alternative_speaker`` and ``overlap``; a sentence that holds a
  moved stretch is only as certain as that stretch.

The margin of a speaker over a stretch of audio is the mean, over the
short windows that overlap the stretch (each weighted by the seconds
it overlaps), of the window's cosine with that speaker's voice minus
its cosine with the closest other voice. It is a difference of cosines
in the space the voices were grouped in (centred on the recording), so
it lies in [-2, 2]: positive when the windows prefer the speaker, about
0 when they cannot tell, negative when they prefer someone else. It is
not a probability and was not calibrated against labelled real speech
(``docs/transcriber/IDENTIFICATION.md``). Plain arithmetic, no model.

The windows are indexed once per call (:class:`WindowIndex`: sorted by
start, found by binary search), so the work grows with the windows
each stretch overlaps, not with every window of the recording.
"""

import bisect
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from transcriber.domain.models import Utterance, Word

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

# A margin below this makes a sentence doubtful, and another voice
# ahead of the sentence's speaker by at least this much in one window
# inside the sentence makes it an overlap. The rationale is in
# TranscriptionParameters.doubt_margin.
DEFAULT_DOUBT_MARGIN = 0.1

# The longest stretch (seconds) keep_sentences_whole may give to
# another speaker. Chosen on the TR-05 tuning readers among 1.5, 3, 6
# and unbounded (docs/transcriber/IDENTIFICATION.md, appendix): 6 s was
# best in every condition there; unbounded was worse with the number
# of speakers estimated.
DEFAULT_MAX_MOVE = 6.0

# A word that ends a sentence: final . ! ? … possibly repeated and
# followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”’)\]]*$")


class VoiceEvidence(NamedTuple):
    """The cosines each short window had with each voice of a session.

    ``windows`` are ``(start, end)`` seconds, ``scores`` one row per
    window and one column per voice group, and ``voices`` the speaker
    each column stands for (``None`` for a group that kept no word and
    so is nobody in the transcript; it is left out of every
    comparison).
    """

    windows: Sequence[Tuple[float, float]]
    scores: FloatArray
    voices: Sequence[Optional[str]]


class Support(NamedTuple):
    """How the windows over a stretch of audio support one speaker.

    ``margin`` is defined in the module docstring; ``runner_up`` is the
    other voice with the highest weighted mean cosine over the stretch.
    ``votes`` counts the overlapping windows whose best voice (among
    the named ones) is the speaker.
    """

    margin: float
    runner_up: str
    votes: int


class WindowIndex:
    """The windows of a :class:`VoiceEvidence`, sorted for searching.

    Windows are sorted by start; a window overlapping ``[start, end]``
    starts after ``start`` minus the longest window and before ``end``,
    so two binary searches bound the candidates.
    """

    def __init__(self, windows: Sequence[Tuple[float, float]]) -> None:
        """Sort ``windows`` by start (stable) and note the longest."""
        pairs = np.array(windows, dtype=np.float64).reshape(-1, 2)
        self._order: IntArray = np.argsort(
            pairs[:, 0], kind="stable"
        ).astype(np.int64)
        self._left = pairs[self._order, 0]
        self._right = pairs[self._order, 1]
        self._longest = (
            float(np.max(self._right - self._left)) if len(pairs) else 0.0
        )

    def overlapping(
        self, start: float, end: float
    ) -> Tuple[IntArray, npt.NDArray[np.float64]]:
        """Return the rows overlapping ``[start, end]`` and their weights.

        The weight of a window is the seconds it shares with the
        stretch. A stretch of no length (a one-instant word) takes the
        windows that contain it, with equal weights. Rows come in the
        order of the windows' starts.
        """
        lo = int(np.searchsorted(self._left, start - self._longest, "left"))
        hi = int(np.searchsorted(self._left, max(end, start), "right"))
        left, right = self._left[lo:hi], self._right[lo:hi]
        rows = self._order[lo:hi]
        if end > start:
            shared = np.minimum(end, right) - np.maximum(start, left)
            keep = shared > 0
            return rows[keep], shared[keep]
        keep = (left <= start) & (start <= right)
        return rows[keep], np.ones(int(keep.sum()), dtype=np.float64)

    def inside(self, start: float, end: float) -> IntArray:
        """Return the rows of the windows lying wholly in ``[start, end]``."""
        lo = int(np.searchsorted(self._left, start, "left"))
        hi = int(np.searchsorted(self._left, end, "right"))
        keep = self._right[lo:hi] <= end
        rows: IntArray = self._order[lo:hi][keep]
        return rows


def _columns(evidence: VoiceEvidence) -> Dict[str, int]:
    """Return the column of each named voice of ``evidence``."""
    return {v: n for n, v in enumerate(evidence.voices) if v is not None}


def support(
    evidence: VoiceEvidence,
    start: float,
    end: float,
    speaker: str,
    index: Optional[WindowIndex] = None,
) -> Optional[Support]:
    """Return how the windows over ``[start, end]`` support ``speaker``.

    ``None`` when no window overlaps the stretch, when ``speaker`` is
    not one of the voices or when there is no other voice to compare
    with (one voice in the session). ``index`` is the windows' index
    when the caller already built one (one call per stretch is then
    sub-linear in the number of windows).
    """
    columns = _columns(evidence)
    if speaker not in columns or len(columns) < 2:
        return None
    found = index if index is not None else WindowIndex(evidence.windows)
    rows, weights = found.overlapping(start, end)
    if rows.size == 0:
        return None
    own = columns[speaker]
    others = [n for v, n in columns.items() if v != speaker]
    block = evidence.scores[rows].astype(np.float64)
    weight = weights / weights.sum()
    best_other = block[:, others].max(axis=1)
    margin = float(weight @ (block[:, own] - best_other))
    means = weight @ block[:, others]
    runner_up = evidence.voices[others[int(np.argmax(means))]]
    assert runner_up is not None  # only named columns are compared
    votes = int(np.sum(block[:, own] > best_other))
    return Support(round(margin, 4), runner_up, votes)


def _sentences(words: Sequence[Word], gap: float) -> List[Tuple[int, int]]:
    """Return ``(first, last + 1)`` of each sentence of ``words``.

    A sentence ends at a word with final punctuation, or before a
    silence longer than ``gap`` seconds (recognisers often leave out
    punctuation; a pause is the other cue).
    """
    spans: List[Tuple[int, int]] = []
    first = 0
    for n, word in enumerate(words):
        last = n + 1 == len(words)
        ends = bool(_SENTENCE_END.search(word.text.strip()))
        pause = not last and words[n + 1].start - word.end > gap
        if last or ends or pause:
            spans.append((first, n + 1))
            first = n + 1
    return spans


def _runs(words: Sequence[Word], first: int, end: int) -> List[List[int]]:
    """Group positions ``first..end-1`` into runs of one speaker."""
    runs: List[List[int]] = []
    for n in range(first, end):
        if runs and words[runs[-1][0]].speaker == words[n].speaker:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


class WholeSentences(NamedTuple):
    """What :func:`keep_sentences_whole` did.

    ``words`` are the words with their speakers after the rule;
    ``moved`` the ``(start, end)`` of each stretch given to another
    speaker (in the order they moved; they may overlap).
    """

    words: List[Word]
    moved: List[Tuple[float, float]]


class _Rule(NamedTuple):
    """The settings and the windows of one :func:`keep_sentences_whole`."""

    evidence: VoiceEvidence
    windows: WindowIndex
    cut: float
    max_move: float


def keep_sentences_whole(
    words: Sequence[Word],
    evidence: Optional[VoiceEvidence],
    cut: float = DEFAULT_DOUBT_MARGIN,
    gap: float = 1.0,
    max_move: float = DEFAULT_MAX_MOVE,
) -> WholeSentences:
    """Undo weakly supported speaker changes inside a sentence.

    ``words`` are in time order and carry the diarizer's speakers, the
    ones ``evidence.voices`` names. Inside each sentence (see
    :func:`_sentences`; ``gap`` is the pause that ends one), a run of
    words of one speaker may go to the speaker of a neighbouring run of
    the same sentence. Runs are tried shortest first, one move at a
    time, and a run moves only when all of these hold:

    - it lasts at most ``max_move`` seconds;
    - its own speaker is weak over it: fewer than two windows prefer
      that speaker, or its :func:`support` margin is below ``cut`` (or
      unknown);
    - the evidence favours the move: at most one window prefers the
      run's speaker (a single stray window), or the run's own margin
      is at most 0, or the neighbour's margin over the run is above the
      run's own. A run whose windows prefer its speaker, even by less
      than ``cut``, does not go to a neighbour they prefer less.

    The neighbour is the one with the best margin over the run, or
    without windows the speaker of the longer neighbouring run. A run
    that does not move keeps its speaker, and the sentence is then cut
    at the change (each part becomes its own turn). Without evidence
    nothing changes. The stretches moved are returned so the
    sentences holding them can say so (:func:`assess_utterances`).
    """
    out = list(words)
    moved: List[Tuple[float, float]] = []
    if evidence is None:
        return WholeSentences(out, moved)
    rule = _Rule(evidence, WindowIndex(evidence.windows), cut, max_move)
    for first, end in _sentences(out, gap):
        while True:
            span = _merge_weakest(out, first, end, rule)
            if span is None:
                break
            moved.append(span)
    return WholeSentences(out, moved)


def _merge_weakest(
    words: List[Word], first: int, end: int, rule: _Rule
) -> Optional[Tuple[float, float]]:
    """Give the shortest movable run of a sentence to a neighbour.

    Changes ``words`` in place and returns the stretch that moved, or
    ``None``. Shortest first, so a long run is never handed to a short
    weak neighbour. Every move joins two runs, so repeating it ends.
    """
    runs = _runs(words, first, end)
    if len(runs) < 2:
        return None
    order = sorted(
        range(len(runs)),
        key=lambda k: words[runs[k][-1]].end - words[runs[k][0]].start,
    )
    for k in order:
        run = runs[k]
        start, stop = words[run[0]].start, words[run[-1]].end
        speaker = words[run[0]].speaker
        if speaker is None or stop - start > rule.max_move:
            continue
        own = support(rule.evidence, start, stop, speaker, rule.windows)
        if own is not None and own.votes >= 2 and own.margin >= rule.cut:
            continue
        target = _neighbour(words, runs, k, rule, start, stop)
        if target is None or target == speaker:
            continue
        theirs = support(rule.evidence, start, stop, target, rule.windows)
        favoured = (
            own is None or own.votes <= 1 or own.margin <= 0
            or (theirs is not None and theirs.margin > own.margin)
        )
        if not favoured:
            continue
        for n in run:
            words[n] = words[n].model_copy(update={"speaker": target})
        return start, stop
    return None


def _neighbour(
    words: Sequence[Word],
    runs: Sequence[List[int]],
    k: int,
    rule: _Rule,
    start: float,
    stop: float,
) -> Optional[str]:
    """Return the neighbouring speaker run ``k`` could take.

    Among the speakers of the runs just before and after (as they are
    now), the one with the best margin over the run's windows; when the
    windows cannot tell, the speaker of the longer neighbouring run.
    """
    sides: Dict[str, int] = {}
    for j in (k - 1, k + 1):
        if 0 <= j < len(runs):
            who = words[runs[j][0]].speaker
            if who is not None:
                sides[who] = max(sides.get(who, 0), len(runs[j]))
    if not sides:
        return None
    scored = {
        who: held.margin for who in sides
        if (held := support(rule.evidence, start, stop, who, rule.windows))
        is not None
    }
    if scored:
        return max(scored, key=lambda w: scored[w])
    return max(sides, key=lambda w: sides[w])


def assess_utterances(
    utterances: Sequence[Utterance],
    evidence: Optional[VoiceEvidence],
    cut: float = DEFAULT_DOUBT_MARGIN,
    moved: Sequence[Tuple[float, float]] = (),
) -> List[Utterance]:
    """Give each utterance the certainty of its speaker.

    ``evidence.voices`` must name the transcript's speakers. For each
    utterance, over ``[start, end]``:

    - ``speaker_certainty`` is the :func:`support` margin of its speaker
      (``None`` when it cannot be computed: no window, or one voice).
      When a stretch in ``moved`` (given to this speaker by
      :func:`keep_sentences_whole`) lies in the utterance, it is the
      lower of that and the margin over the moved stretch;
    - ``doubtful`` is ``True`` when that value is below ``cut``, or
      when a moved stretch in the utterance has no window to judge it;
    - ``overlap`` is ``True`` when a window lying wholly inside the
      utterance prefers another voice of the transcript by at least
      ``cut``: two voices are heard in one sentence. The windows cannot
      tell simultaneous speech from a quick exchange, and a sentence
      shorter than a window has no window inside it (``False``);
    - ``alternative_speaker`` is, for a doubtful utterance, the other
      voice the windows over its weakest stretch prefer most; for an
      overlap that is not doubtful, the voice of the window that
      flagged it (with three or more voices it may differ from the
      runner-up over the whole sentence); otherwise ``None``.

    Without evidence (or a margin) all four stay ``None``.
    """
    index = WindowIndex(evidence.windows) if evidence is not None else None
    spans = _Spans(moved)
    out: List[Utterance] = []
    for utterance in utterances:
        held = (
            None if evidence is None or index is None else support(
                evidence, utterance.start, utterance.end, utterance.speaker,
                index,
            )
        )
        if evidence is None or index is None or held is None:
            out.append(utterance.model_copy(update={
                "speaker_certainty": None, "doubtful": None,
                "overlap": None, "alternative_speaker": None,
            }))
            continue
        weakest, unjudged = held, False
        for left, right in spans.inside(utterance.start, utterance.end):
            piece = support(evidence, left, right, utterance.speaker, index)
            if piece is None:
                unjudged = True
            elif piece.margin < weakest.margin:
                weakest = piece
        doubtful = unjudged or weakest.margin < cut
        intruder = _other_voice_inside(evidence, index, utterance, cut)
        alternative = (
            weakest.runner_up if doubtful
            else intruder
        )
        out.append(utterance.model_copy(update={
            "speaker_certainty": weakest.margin,
            "doubtful": doubtful,
            "overlap": intruder is not None,
            "alternative_speaker": alternative,
        }))
    return out


class _Spans:
    """The moved stretches, sorted by start for searching."""

    def __init__(self, spans: Sequence[Tuple[float, float]]) -> None:
        """Sort ``spans`` and note the longest."""
        self._spans = sorted(spans)
        self._lefts = [left for left, _ in self._spans]
        self._longest = max((r - left for left, r in self._spans), default=0.0)

    def inside(self, start: float, end: float) -> List[Tuple[float, float]]:
        """Return the parts of the stretches inside ``[start, end]``."""
        first = bisect.bisect_left(self._lefts, start - self._longest)
        parts: List[Tuple[float, float]] = []
        for left, right in self._spans[first:]:
            if left > end:
                break
            low, high = max(left, start), min(right, end)
            if high > low or (high == low and left <= high <= right):
                parts.append((low, high))
        return parts


def _other_voice_inside(
    evidence: VoiceEvidence,
    index: WindowIndex,
    utterance: Utterance,
    cut: float,
) -> Optional[str]:
    """Return the other voice a window inside ``utterance`` prefers.

    Only windows wholly inside the utterance count, so a window that
    straddles a change of turn at the sentence's edge does not. Of the
    windows where another voice leads the utterance's speaker by at
    least ``cut``, the voice with the largest lead is returned; ``None``
    when there is none.
    """
    columns = _columns(evidence)
    own = columns.get(utterance.speaker)
    others = [n for v, n in columns.items() if v != utterance.speaker]
    if own is None or not others:
        return None
    rows = index.inside(utterance.start, utterance.end)
    if rows.size == 0:
        return None
    block = evidence.scores[rows][:, others].astype(np.float64)
    lead = block - evidence.scores[rows][:, [own]].astype(np.float64)
    best = np.unravel_index(int(np.argmax(lead)), lead.shape)
    if float(lead[best]) < cut:
        return None
    return evidence.voices[others[int(best[1])]]
