"""Certainty: how well the voice windows support each attribution.

Diarization labels every short analysis window with the voice group it
is closest to (a cosine with each group's centroid; see
:class:`~transcriber.domain.voices.VoiceGroups.fine_scores`). Those
cosines are kept here as :class:`VoiceEvidence` and used twice:

- :func:`keep_sentences_whole` stops a weakly supported speaker change
  from cutting a sentence in two (the pieces otherwise go to different
  people);
- :func:`assess_utterances` gives every sentence its
  ``speaker_certainty`` (a similarity margin), ``doubtful``,
  ``alternative_speaker`` and ``overlap``.

The margin of a speaker over a stretch of audio is the mean, over the
short windows that overlap the stretch (each weighted by the seconds
it overlaps), of the window's cosine with that speaker's voice minus
its cosine with the closest other voice. It is a difference of cosines
in the space the voices were grouped in (centred on the recording), so
it lies in [-2, 2]: positive when the windows prefer the speaker, about
0 when they cannot tell, negative when they prefer someone else. It is
not a probability and was not calibrated against labelled real speech
(``docs/transcriber/IDENTIFICATION.md``). Plain arithmetic, no model.
"""

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from transcriber.domain.models import Utterance, Word

FloatArray = npt.NDArray[np.float32]

# A margin below this makes a sentence doubtful, and another voice
# ahead of the sentence's speaker by at least this much in one window
# inside the sentence makes it an overlap. The rationale is in
# TranscriptionParameters.doubt_margin.
DEFAULT_DOUBT_MARGIN = 0.1

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
    """

    margin: float
    runner_up: str


def support(
    evidence: VoiceEvidence, start: float, end: float, speaker: str
) -> Optional[Support]:
    """Return how the windows over ``[start, end]`` support ``speaker``.

    ``None`` when no window overlaps the stretch, when ``speaker`` is
    not one of the voices or when there is no other voice to compare
    with (one voice in the session).
    """
    columns = {v: n for n, v in enumerate(evidence.voices) if v is not None}
    if speaker not in columns or len(columns) < 2:
        return None
    rows, weights = _overlapping(evidence.windows, start, end)
    if not rows:
        return None
    own = columns[speaker]
    others = [n for v, n in columns.items() if v != speaker]
    block = evidence.scores[rows].astype(np.float64)
    weight = np.array(weights, dtype=np.float64)
    weight /= weight.sum()
    best_other = block[:, others].max(axis=1)
    margin = float(weight @ (block[:, own] - best_other))
    means = weight @ block[:, others]
    runner_up = evidence.voices[others[int(np.argmax(means))]]
    assert runner_up is not None  # only named columns are compared
    return Support(round(margin, 4), runner_up)


def _overlapping(
    windows: Sequence[Tuple[float, float]], start: float, end: float
) -> Tuple[List[int], List[float]]:
    """Return the windows overlapping ``[start, end]`` and their weights.

    The weight of a window is the seconds it shares with the stretch. A
    stretch of no length (a one-instant word) takes the windows that
    contain it, with equal weights.
    """
    rows: List[int] = []
    weights: List[float] = []
    for n, (left, right) in enumerate(windows):
        shared = min(end, right) - max(start, left)
        if shared > 0:
            rows.append(n)
            weights.append(shared)
        elif end <= start and left <= start <= right:
            rows.append(n)
            weights.append(1.0)
    return rows, weights


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


def keep_sentences_whole(
    words: Sequence[Word],
    evidence: Optional[VoiceEvidence],
    cut: float = DEFAULT_DOUBT_MARGIN,
    gap: float = 1.0,
) -> List[Word]:
    """Undo weakly supported speaker changes inside a sentence.

    ``words`` are in time order and carry the diarizer's speakers, the
    ones ``evidence.voices`` names. Inside each sentence (see
    :func:`_sentences`; ``gap`` is the pause that ends one), a run of
    words of one speaker whose evidence is weak is given to the speaker
    of a neighbouring run of the same sentence. The evidence is weak
    when at most one window over the run prefers the run's speaker (a
    single stray window) or when the run's :func:`support` margin is
    below ``cut`` (or unknown). Weak runs are handled shortest first,
    one at a time. A run goes to the neighbour the windows over it
    prefer, or with no evidence to the speaker of the longer
    neighbouring run. A run with at least two windows of its own
    and a margin of at least ``cut`` keeps its speaker: two people can
    speak inside one sentence. Without evidence nothing changes.
    """
    out = list(words)
    if evidence is None:
        return out
    for first, end in _sentences(out, gap):
        while _merge_weakest(out, first, end, evidence, cut):
            pass
    return out


def _merge_weakest(
    words: List[Word], first: int, end: int, evidence: VoiceEvidence,
    cut: float,
) -> bool:
    """Give the shortest weak run of a sentence to a neighbour.

    Changes ``words`` in place and tells whether a run moved. Shortest
    first, so a long run is never handed to a short weak neighbour.
    Every move joins two runs, so repeating it ends.
    """
    runs = _runs(words, first, end)
    if len(runs) < 2:
        return False
    order = sorted(
        range(len(runs)),
        key=lambda k: words[runs[k][-1]].end - words[runs[k][0]].start,
    )
    for k in order:
        run = runs[k]
        start, stop = words[run[0]].start, words[run[-1]].end
        speaker = words[run[0]].speaker
        if speaker is None or _strong(evidence, start, stop, speaker, cut):
            continue
        target = _neighbour(words, runs, k, evidence, start, stop)
        if target is None or target == speaker:
            continue
        for n in run:
            words[n] = words[n].model_copy(update={"speaker": target})
        return True
    return False


def _strong(
    evidence: VoiceEvidence, start: float, stop: float, speaker: str,
    cut: float,
) -> bool:
    """Tell whether the windows over ``[start, stop]`` back ``speaker``.

    At least two overlapping windows must prefer ``speaker`` and its
    margin must reach ``cut``.
    """
    held = support(evidence, start, stop, speaker)
    if held is None or held.margin < cut:
        return False
    rows, _ = _overlapping(evidence.windows, start, stop)
    own = list(evidence.voices).index(speaker)
    votes = sum(
        1 for row in rows if int(np.argmax(evidence.scores[row])) == own
    )
    return votes >= 2


def _neighbour(
    words: Sequence[Word],
    runs: Sequence[List[int]],
    k: int,
    evidence: VoiceEvidence,
    start: float,
    stop: float,
) -> Optional[str]:
    """Return the neighbouring speaker run ``k`` should take.

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
        if (held := support(evidence, start, stop, who)) is not None
    }
    if scored:
        return max(scored, key=lambda w: scored[w])
    return max(sides, key=lambda w: sides[w])


def assess_utterances(
    utterances: Sequence[Utterance],
    evidence: Optional[VoiceEvidence],
    cut: float = DEFAULT_DOUBT_MARGIN,
) -> List[Utterance]:
    """Give each utterance the certainty of its speaker.

    ``evidence.voices`` must name the transcript's speakers. For each
    utterance, over ``[start, end]``:

    - ``speaker_certainty`` is the :func:`support` margin of its speaker
      (``None`` when it cannot be computed: no window, or one voice);
    - ``doubtful`` is ``True`` when that margin is below ``cut``;
    - ``overlap`` is ``True`` when a window lying wholly inside the
      utterance prefers another voice of the transcript by at least
      ``cut``: two voices are heard in one sentence. The windows cannot
      tell simultaneous speech from a quick exchange, and a sentence
      shorter than a window has no window inside it (``False``);
    - ``alternative_speaker`` is the other voice the windows prefer
      most, given only for a doubtful or overlapping utterance.

    Without evidence (or a margin) all four stay ``None``.
    """
    out: List[Utterance] = []
    for utterance in utterances:
        held = (
            None if evidence is None else support(
                evidence, utterance.start, utterance.end, utterance.speaker
            )
        )
        if evidence is None or held is None:
            out.append(utterance.model_copy(update={
                "speaker_certainty": None, "doubtful": None,
                "overlap": None, "alternative_speaker": None,
            }))
            continue
        doubtful = held.margin < cut
        overlap = _other_voice_inside(evidence, utterance, cut)
        out.append(utterance.model_copy(update={
            "speaker_certainty": held.margin,
            "doubtful": doubtful,
            "overlap": overlap,
            "alternative_speaker": (
                held.runner_up if doubtful or overlap else None
            ),
        }))
    return out


def _other_voice_inside(
    evidence: VoiceEvidence, utterance: Utterance, cut: float
) -> bool:
    """Tell whether a window inside ``utterance`` prefers another voice.

    Only windows wholly inside the utterance count, so a window that
    straddles a change of turn at the sentence's edge does not.
    """
    columns = {v: n for n, v in enumerate(evidence.voices) if v is not None}
    own = columns.get(utterance.speaker)
    if own is None:
        return False
    others = [n for v, n in columns.items() if v != utterance.speaker]
    for row, (left, right) in enumerate(evidence.windows):
        if left >= utterance.start and right <= utterance.end and others:
            scores = evidence.scores[row]
            if float(scores[others].max() - scores[own]) >= cut:
                return True
    return False
