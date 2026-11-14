"""Seams: words a recogniser returned twice where its windows overlap.

A recogniser that reads long audio in overlapping windows has to join
the windows' words. When the join fails, the words of the overlap come
out twice: the stream runs forward to the end of one window, then jumps
back to the start of the next and repeats (often with slightly
different words) what was already said. The transformers Whisper
pipeline does this with ``return_timestamps="word"`` and
``chunk_length_s=30``: its windows are 30 s long and start every 20 s,
and on the first real recordings the stream jumped back to exactly 20,
80, 180, 240, 260 and 360 s (``docs/transcriber/ARCHITECTURE.md`` §3).

:func:`resolve_seams` keeps one copy of each overlap. Plain arithmetic
on word times, no model.
"""

import re
from typing import List, Sequence

from transcriber.domain.models import Word

_LETTERS = re.compile(r"\W+")


def _middle(word: Word) -> float:
    """Return the midpoint of ``word`` in seconds."""
    return (word.start + word.end) / 2


def _same(left: Word, right: Word) -> bool:
    """Tell whether two words are the same word heard twice.

    Same letters (case and punctuation ignored) and overlapping times.
    """
    return (
        _LETTERS.sub("", left.text).lower()
        == _LETTERS.sub("", right.text).lower()
        and min(left.end, right.end) >= max(left.start, right.start)
    )


def resolve_seams(words: Sequence[Word]) -> List[Word]:
    """Return ``words`` in time order with repeated passages removed.

    ``words`` are in the order the recogniser returned them. Every word
    that starts before the word returned just before it begins a new
    *run* (the recogniser went back in time: a window seam). Runs are
    joined one by one: the passage covered twice runs from the start of
    the new run to the end of everything kept so far, and is cut at its
    middle (where a recogniser with overlapping windows means to cut).
    Words kept so far stay if their midpoint is before the cut, words of
    the new run if theirs is at or after it; but when the last word kept
    before the cut is heard again in the new run (same letters,
    overlapping times), the new run is taken from the word after that
    copy instead, so the two windows are joined on a word they share
    and nothing near the cut is lost or repeated.

    Words already in time order come back unchanged. The result is
    sorted by start time (stable).
    """
    runs: List[List[Word]] = []
    for word in words:
        if runs and word.start < runs[-1][-1].start:
            runs.append([word])
        elif runs:
            runs[-1].append(word)
        else:
            runs.append([word])
    if not runs:
        return []
    kept = runs[0]
    for run in runs[1:]:
        reach = max(w.end for w in kept)
        cut = (run[0].start + reach) / 2
        before = [w for w in kept if _middle(w) < cut]
        anchor = next(
            (n for n, w in enumerate(run) if before and _same(before[-1], w)),
            None,
        )
        if anchor is not None:
            after = run[anchor + 1:]
        else:
            after = [w for w in run if _middle(w) >= cut]
        kept = before + after
    return sorted(kept, key=lambda w: w.start)
