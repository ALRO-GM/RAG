"""Session-level attribution of names: moderators, votes, matching.

In a conversation of many voices a name said to someone ("señor Ruiz,
usted dijo…") rarely belongs to the nearest other speaker: a debater
rebuts someone who spoke two turns earlier, a moderator hands the floor
to the next speaker. This module (TR-21, issue #116) turns every piece
of evidence into weighted **votes** for voices and then gives each
name to at most one voice and each voice at most one name, over the
whole session:

- :func:`find_moderators` spots the voices that run the conversation
  (many sentences with floor-handing cues: "tiene la palabra", "le
  pregunto", "bienvenido", "over to"…);
- :func:`votes_for` spreads one piece of evidence over voices by its
  direction: the speaker (self-introduction), the next voice to speak
  (a cue or a question), the previous one (thanks: a participant who
  thanks by name usually thanks the moderator who gave the floor), or,
  for a plain mention, mostly the last participants who spoke and a
  little the next (a rebuttal skips the moderators);
- :func:`best_matching` finds the one-to-one assignment of names to
  voices with the highest total score (Hungarian algorithm), and
  :func:`margin` how clearly a pair beats its rivals.

Pure functions over speaker labels and texts; no I/O, no other layer.
"""

import re
from statistics import median
from typing import Dict, FrozenSet, List, Literal, Sequence, Tuple

Towards = Literal["self", "previous", "next", "around"]
"""Whose name it is: the speaker's own (``self``), the person who spoke
just before (``previous``) or speaks just after (``next``), or someone
taking part, most likely among the last who spoke (``around``: a plain
mention, a rebuttal)."""

SELF_WEIGHT = 3.0
"""Votes of a clear self-introduction."""

CUE_WEIGHT = 2.0
"""Votes of a name with a direction (thanks, a question, a cue)."""

MENTION_WEIGHT = 1.0
"""Votes of a plain mention, spread over the likely voices."""

AROUND_SHARES: Tuple[float, ...] = (0.5, 0.25)
"""Share of a plain mention for the last and the last-but-one other
participants who spoke; the rest goes to the next one."""

MODERATOR_CUES = re.compile(
    r"\b(?:tiene (?:usted )?la palabra|turno|le pregunto|les pregunto|"
    r"pregunta para|bienvenid[oa]s?|nos acompañan?|con nosotros|"
    r"vamos con|pasamos a|the floor|over to|let'?s hear|next question|"
    r"welcome)\b",
    re.IGNORECASE,
)
"""Phrases of someone who runs a debate, a panel or a meeting."""

MIN_MODERATOR_CUES = 3
"""Cue sentences a voice needs at least to be taken for a moderator."""

MIN_VOICES_FOR_MODERATOR = 3
"""A conversation of two has no moderator."""


def find_moderators(lines: Sequence[Tuple[str, str]]) -> FrozenSet[str]:
    """Return the voices that run the conversation.

    ``lines`` are ``(speaker, text)`` in order. A voice is a moderator
    when the conversation has at least three voices and it says at
    least :data:`MIN_MODERATOR_CUES` sentences with a floor-handing
    cue, and more than twice the median of all voices plus one (so in
    a debate both presenters are found, and in an ordinary meeting
    nobody, or only the one who chairs it).
    """
    voices = list(dict.fromkeys(speaker for speaker, _ in lines))
    if len(voices) < MIN_VOICES_FOR_MODERATOR:
        return frozenset()
    cues = {v: 0 for v in voices}
    for speaker, text in lines:
        if MODERATOR_CUES.search(text):
            cues[speaker] += 1
    floor = max(MIN_MODERATOR_CUES, 2 * median(cues.values()) + 1)
    return frozenset(v for v, n in cues.items() if n >= floor)


def _others(
    speakers: Sequence[str],
    indices: range,
    said_by: str,
    skip: FrozenSet[str],
    limit: int,
) -> List[str]:
    """Return up to ``limit`` distinct voices along ``indices``.

    Not ``said_by`` and not in ``skip``, in the order met.
    """
    found: List[str] = []
    for i in indices:
        voice = speakers[i]
        if voice == said_by or voice in skip or voice in found:
            continue
        found.append(voice)
        if len(found) == limit:
            break
    return found


def votes_for(
    speakers: Sequence[str],
    position: int,
    towards: Towards,
    moderators: FrozenSet[str] = frozenset(),
    weight: float = CUE_WEIGHT,
) -> List[Tuple[str, float]]:
    """Return the voices a name said at ``position`` points to.

    ``speakers`` is the speaker of every sentence in order and
    ``position`` the sentence that holds the name. ``self`` gives all
    of ``weight`` to its speaker; ``next`` to the first other voice
    after it; ``previous`` to the last other voice before it;
    ``around`` (a plain mention: a rebuttal, which skips the
    moderators when a participant says it) gives half to the last other
    participant who spoke, a quarter to the one before and a quarter to
    the next, said by a moderator half to the previous and half to the
    next voice. Shares with nobody to go to are given to the others; an
    empty list when nobody else speaks.
    """
    said_by = speakers[position]
    if towards == "self":
        return [(said_by, weight)]
    later = range(position + 1, len(speakers))
    earlier = range(position - 1, -1, -1)
    skip = frozenset() if said_by in moderators else moderators
    if towards == "next":
        voices = _others(speakers, later, said_by, frozenset(), 1)
        voices = voices or _others(speakers, earlier, said_by, skip, 1)
        return [(v, weight) for v in voices]
    if towards == "previous":
        voices = _others(speakers, earlier, said_by, frozenset(), 1)
        voices = voices or _others(speakers, later, said_by, frozenset(), 1)
        return [(v, weight) for v in voices]
    if said_by in moderators:
        shares = list(zip(
            _others(speakers, earlier, said_by, frozenset(), 1)
            + _others(speakers, later, said_by, frozenset(), 1),
            (0.5, 0.5),
        ))
    else:
        before = _others(speakers, earlier, said_by, skip, len(AROUND_SHARES))
        after = [
            v for v in _others(speakers, later, said_by, skip, 2)
            if v not in before
        ][:1]
        shares = list(zip(before, AROUND_SHARES))
        shares += [(v, 1.0 - sum(AROUND_SHARES)) for v in after]
    total = sum(share for _, share in shares)
    if not total:
        return []
    merged: Dict[str, float] = {}
    for voice, share in shares:
        merged[voice] = merged.get(voice, 0.0) + weight * share / total
    return list(merged.items())


def best_matching(
    scores: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str]]:
    """Return the one-to-one ``(voice, name)`` pairs of highest total.

    ``scores`` maps a voice and a name to its score; only pairs with a
    positive score can be chosen. Hungarian algorithm on the square
    matrix of voices and names (padded with zeros); ties are broken by
    the order the pairs were first given, so the result is
    reproducible.
    """
    voices = list(dict.fromkeys(v for v, _ in scores))
    names = list(dict.fromkeys(n for _, n in scores))
    size = max(len(voices), len(names))
    if not size:
        return []
    cost = [
        [
            -max(scores.get((voices[i], names[j]), 0.0), 0.0)
            if i < len(voices) and j < len(names) else 0.0
            for j in range(size)
        ]
        for i in range(size)
    ]
    assigned = _hungarian(cost)
    return [
        (voices[i], names[j]) for i, j in enumerate(assigned)
        if i < len(voices) and j < len(names)
        and scores.get((voices[i], names[j]), 0.0) > 0
    ]


def _hungarian(cost: List[List[float]]) -> List[int]:
    """Return, for each row, the column of a minimum-cost assignment.

    The classic O(n³) potentials method on a square matrix.
    """
    n = len(cost)
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    owner = [0] * (n + 1)
    way = [0] * (n + 1)
    for row in range(1, n + 1):
        owner[0] = row
        column = 0
        least = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[column] = True
            current, delta, chosen = owner[column], inf, 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                reduced = cost[current - 1][j - 1] - u[current] - v[j]
                if reduced < least[j]:
                    least[j], way[j] = reduced, column
                if least[j] < delta:
                    delta, chosen = least[j], j
            for j in range(n + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    least[j] -= delta
            column = chosen
            if owner[column] == 0:
                break
        while column:
            previous = way[column]
            owner[column] = owner[previous]
            column = previous
    result = [0] * n
    for j in range(1, n + 1):
        result[owner[j] - 1] = j - 1
    return result


def margin(
    scores: Dict[Tuple[str, str], float], voice: str, name: str
) -> float:
    """Return how far ``(voice, name)`` is ahead of its best rival.

    The rival is the best other name for that voice or the best other
    voice for that name (at least 0).
    """
    own = scores.get((voice, name), 0.0)
    rivals = [
        s for (v, n), s in scores.items()
        if (v == voice) != (n == name)
    ]
    return own - max([0.0] + rivals)
