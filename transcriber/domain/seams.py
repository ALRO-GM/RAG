"""Seams: words a recogniser returned twice where its windows overlap.

A recogniser that reads long audio in overlapping windows has to join
the windows' words. When the join fails, the words of the overlap come
out twice: the stream runs forward to the end of one window, then jumps
back to the start of the next and repeats (often with slightly
different words or times) what was already said. The transformers
Whisper pipeline does this with ``return_timestamps="word"`` and
``chunk_length_s=30``: its windows are 30 s long and start every 20 s,
so a real seam goes back by up to the 10 s the windows share; on the
first real recordings the stream jumped back to exactly 20, 80, 180,
240, 260 and 360 s (``docs/transcriber/ARCHITECTURE.md`` §3). Word
times also wobble by a few hundredths of a second (a word starting
just before the end of the previous one); that is not a seam.

A recogniser also returns, now and then, one word with an impossible
span (a 15 s ``um``, or a word from 0 s to the end of the recording);
left as it is, it looks like a jump back over everything it spans.
Such a word is shortened to a plausible length first
(:data:`MAX_WORD_SECONDS`, :data:`MAX_SECONDS_PER_LETTER`).

:func:`resolve_seams` keeps one copy of each doubled passage. A word
without a counterpart in the other copy is dropped only when the other
copy has more different words at the same moment, or as many and comes
from the window where that moment is farther from its edge; so one
word never displaces two. Plain arithmetic on word times and letters,
no model.
"""

import re
from typing import Dict, List, Sequence, Tuple

from transcriber.domain.models import Word

# A step back in time shorter than this (seconds) is jitter of the word
# times, not a seam: the word is kept in place, its start clamped. A
# real seam of the Whisper pipeline goes back by up to 10 s; the
# jitter seen is a few hundredths. The value between is a judgement.
DEFAULT_MIN_STEP = 1.0

# Two copies of a word may be timed this far apart (seconds, between
# their midpoints) and still be the same word heard twice. A judgement:
# the two windows time the same word differently by fractions of a
# second.
SAME_TIME = 1.0

# The longest a word can plausibly last (seconds): MAX_WORD_SECONDS (a
# drawn-out "mmm" or "eh"), or MAX_SECONDS_PER_LETTER times its letters
# when that is more. Speech runs at roughly ten to
# fifteen letters a second, so 0.4 s a letter is several times slower
# than slow speech. A judgement: a longer span is a timing error, not a
# word, and the word is shortened to this (_plausible).
MAX_WORD_SECONDS = 3.0
MAX_SECONDS_PER_LETTER = 0.4

_LETTERS = re.compile(r"\W+")


def _letters(word: Word) -> str:
    """Return the letters of ``word``, lower case, without punctuation."""
    return _LETTERS.sub("", word.text).lower()


def _middle(word: Word) -> float:
    """Return the midpoint of ``word`` in seconds."""
    return (word.start + word.end) / 2


def _same(left: Word, right: Word) -> bool:
    """Tell whether two words are the same word heard twice.

    Same letters (case and punctuation ignored, not empty) and times
    that overlap or whose midpoints are at most :data:`SAME_TIME`
    apart.
    """
    letters = _letters(left)
    return bool(letters) and letters == _letters(right) and (
        min(left.end, right.end) > max(left.start, right.start)
        or abs(_middle(left) - _middle(right)) <= SAME_TIME
    )


def longest_plausible(word: Word) -> float:
    """Return the most seconds ``word`` can last (see the constants)."""
    return max(MAX_WORD_SECONDS, MAX_SECONDS_PER_LETTER * len(_letters(word)))


def _plausible(words: Sequence[Word], min_step: float) -> List[Word]:
    """Return ``words`` with every word longer than it can last shortened.

    A word that jumps back by more than ``min_step`` from the start of
    the word before it keeps its end (its start is what jumped; a
    recogniser also glues a window's first word to the window's start)
    and any other keeps its start. The text order is unchanged.
    """
    out: List[Word] = []
    for word in words:
        bound = longest_plausible(word)
        if word.end - word.start > bound:
            jumped = bool(out) and word.start < out[-1].start - min_step
            word = word.model_copy(update=(
                {"start": word.end - bound} if jumped
                else {"end": word.start + bound}
            ))
        out.append(word)
    return out


def _overlap(left: Word, right: Word) -> bool:
    """Tell whether two words share some time."""
    return min(left.end, right.end) > max(left.start, right.start)


def _forward(words: Sequence[Word]) -> List[Word]:
    """Clamp each start to the one before it, keeping the text order.

    A word that starts before the previous one starts at the same
    time (and ends no earlier), so the result is sorted by start.
    """
    out: List[Word] = []
    for word in words:
        if out and word.start < out[-1].start:
            start = out[-1].start
            word = word.model_copy(update={
                "start": start, "end": max(word.end, start),
            })
        out.append(word)
    return out


def _repeats(run: Sequence[Word], word: Word, min_step: float) -> bool:
    """Tell whether ``word`` is :func:`_same` as one of ``run`` it re-covers.

    Only the words of ``run`` that end after ``word`` starts count (the
    stretch a step back covers again). The run is scanned from its end
    and the scan stops where no earlier word can be the same: the
    starts of a run never fall more than ``min_step`` below the latest
    one, and a same word starts at most :data:`SAME_TIME` plus its
    longest plausible length before ``word``.
    """
    floor = word.start - SAME_TIME - longest_plausible(word) - min_step
    for other in reversed(run):
        if other.start < floor:
            return False
        if other.end > word.start and _same(other, word):
            return True
    return False


def _runs(words: Sequence[Word], min_step: float) -> List[List[Word]]:
    """Cut ``words`` where they jump back by more than ``min_step``.

    A smaller step back also starts a new run when the word is the
    same word as one it steps back over (:func:`_repeats`): the
    recogniser said it again, as at a seam, and :func:`_join` keeps it
    once. Words that do not step back are never compared, so a
    repetition said in order ("no no no") stays whole. Each run is in
    the recogniser's order with its small steps back clamped
    (:func:`_forward`).
    """
    runs: List[List[Word]] = []
    latest = 0.0
    for word in words:
        if not runs or word.start < latest - min_step or (
            word.start < latest and _repeats(runs[-1], word, min_step)
        ):
            runs.append([word])
            latest = word.start
        else:
            runs[-1].append(word)
            latest = max(latest, word.start)
    return [_forward(run) for run in runs]


def _common(
    first: Sequence[Word], second: Sequence[Word]
) -> List[Tuple[int, int]]:
    """Return the longest common subsequence of two lists of words.

    Pairs ``(i, j)`` of positions whose words are :func:`_same`, in
    order (both indices increase).
    """
    n, m = len(first), len(second)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if _same(first[i], second[j]):
                best[i][j] = best[i + 1][j + 1] + 1
            else:
                best[i][j] = max(best[i + 1][j], best[i][j + 1])
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if _same(first[i], second[j]) and \
                best[i][j] == best[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _settle(
    earlier: Sequence[Word], later: Sequence[Word], middle: float
) -> List[Word]:
    """Keep the unmatched words of both copies between two anchors.

    Words of the two copies that share time are different words heard
    at the same moment: of each such group (words linked by shared
    time) only one copy stays. The copy with more words in the group
    stays (a copy never displaces more words than it has: one word
    with a wrong span must not erase a passage); with as many on each
    side, the copy from the window where the moment is farther from
    the window's edge, the earlier window before ``middle`` and the
    later one from it on. A word that shares no time with the other
    copy stays. Survivors come sorted by start.
    """
    group: Dict[int, int] = {}

    def root(k: int) -> int:
        """Return the group of item ``k`` (union-find)."""
        while group.get(k, k) != k:
            k = group[k]
        return k

    count = len(earlier)
    for i, left in enumerate(earlier):
        for j, right in enumerate(later):
            if _overlap(left, right):
                group[root(i)] = root(count + j)
    members: Dict[int, List[int]] = {}
    for k in range(count + len(later)):
        members.setdefault(root(k), []).append(k)
    kept: List[Word] = []
    for items in members.values():
        words = [earlier[k] if k < count else later[k - count]
                 for k in items]
        if all(k < count for k in items) or all(k >= count for k in items):
            kept.extend(words)
            continue
        firsts = sum(1 for k in items if k < count)
        centre = (min(w.start for w in words) + max(w.end for w in words)) / 2
        from_earlier = (
            centre < middle if 2 * firsts == len(items)
            else 2 * firsts > len(items)
        )
        kept.extend(
            word for k, word in zip(items, words)
            if (k < count) == from_earlier
        )
    return sorted(kept, key=lambda w: w.start)


def _join(kept: List[Word], run: List[Word]) -> List[Word]:
    """Join a run that went back in time to the words kept so far.

    The doubled stretch runs from the run's first start to the end of
    what is kept; the run's words starting up to :data:`SAME_TIME`
    after that end are compared too (the later window may time the
    last doubled word just past it). The two copies are aligned by
    their words
    (:func:`_common`): a word found in both is kept once, from the
    window where it is farther from the edge (the earlier window before
    the stretch's middle, the later one from it on); the words between
    two aligned words are settled by :func:`_settle`. Words outside the
    stretch are kept. The result is in text order with starts clamped
    forward (:func:`_forward`).
    """
    start = run[0].start
    reach = max(w.end for w in kept)
    if reach <= start:
        return _forward(kept + run)
    split = next(
        (n for n, w in enumerate(kept) if w.end > start), len(kept)
    )
    head, doubled = kept[:split], kept[split:]
    stop = next(
        (n for n, w in enumerate(run) if w.start >= reach + SAME_TIME),
        len(run),
    )
    again, tail = run[:stop], run[stop:]
    middle = (start + reach) / 2
    merged: List[Word] = []
    i = j = 0
    for a, b in _common(doubled, again) + [(len(doubled), len(again))]:
        merged.extend(_settle(doubled[i:a], again[j:b], middle))
        if a < len(doubled):
            pair = (doubled[a], again[b])
            centre = (_middle(pair[0]) + _middle(pair[1])) / 2
            merged.append(pair[0] if centre < middle else pair[1])
        i, j = a + 1, b + 1
    return _forward(head + merged + tail)


def resolve_seams(
    words: Sequence[Word], min_step: float = DEFAULT_MIN_STEP
) -> List[Word]:
    """Return ``words`` in time order with doubled passages kept once.

    ``words`` are in the order the recogniser returned them. A word
    that starts more than ``min_step`` seconds before the latest start
    of the words before it begins a new *run* (the recogniser went back
    to a window's start: a seam); a smaller step back is jitter and the
    word's start is clamped to the previous start, so it stays in text
    order, unless it is the same word as one it steps back over (then
    it begins a run too, and the word is kept once). Before that, a word
    longer than :func:`longest_plausible` is shortened to it
    (:func:`_plausible`). Runs are joined one by one (:func:`_join`):
    the stretch the two runs both cover is aligned word by word; a word
    without a counterpart is dropped only when the other copy has more
    different words at the same moment, or as many and comes from the
    window where the moment is farther from its edge.

    Words already in time order and of plausible lengths come back
    unchanged, and the function is idempotent: its result has no step
    back to resolve.
    """
    runs = _runs(_plausible(words, min_step), min_step)
    if not runs:
        return []
    kept = runs[0]
    for run in runs[1:]:
        kept = _join(kept, run)
    return kept
