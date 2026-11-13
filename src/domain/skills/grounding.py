"""Grounding: keep only evidence that the transcript really contains.

A language model can misquote, cite the wrong speaker or invent an
indicator. Every evidence item it proposes is checked here against the
transcript and the rubric, and the rating is withheld when too little
verified evidence remains.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.domain.languages import fold
from src.domain.skills.assessment import Evidence
from src.domain.skills.rubric import Competency, Scale
from src.domain.skills.transcript import Turn

_WORD = re.compile(r"\w+", re.UNICODE)

MIN_QUOTE_TOKENS = 3
MAX_SPAN_RATIO = 1.25
NEGATIONS = frozenset(
    "no not never n't dont cannot nor without neither nunca ni sin "
    "jamas tampoco nadie nada".split()
)


class ProposedEvidence(BaseModel):
    """An evidence item as returned by the judge, not yet verified."""

    turn: Optional[int] = None
    quote: str = ""
    indicator: str = ""


class Judgement(BaseModel):
    """The parsed reply of the judge for one competency."""

    evidence: List[ProposedEvidence] = Field(default_factory=list)
    rating: Optional[int] = None
    rationale: str = ""
    parse_error: bool = False


def quote_tokens(text: str) -> List[str]:
    """Return the accent-folded word tokens of ``text``."""
    return [fold(m.group()) for m in _WORD.finditer(text)]


def find_quote(quote: str, text: str) -> Optional[str]:
    """Return the passage of ``text`` that ``quote`` copies, if any.

    Case, accents and punctuation are ignored. Every word of the
    quote must appear in the text, in the same order, inside a
    passage at most ``MAX_SPAN_RATIO`` times as long as the quote:
    the model may skip a filler word, but a changed, added or
    reordered word makes the quote a paraphrase and it is rejected,
    and so does skipping a negation, which would invert the meaning.
    Quotes under ``MIN_QUOTE_TOKENS`` words are too short to verify.
    The returned passage is the transcript's own wording.
    """
    needle = quote_tokens(quote)
    if len(needle) < MIN_QUOTE_TOKENS:
        return None
    matches = list(_WORD.finditer(text))
    words = [fold(m.group()) for m in matches]
    limit = int(len(needle) * MAX_SPAN_RATIO)
    best: Optional[Tuple[int, int]] = None
    for start, word in enumerate(words):
        if word != needle[0]:
            continue
        position = start
        inverted = False
        for wanted in needle[1:]:
            position += 1
            while position < len(words) and words[position] != wanted:
                inverted = inverted or words[position] in NEGATIONS
                position += 1
            if position >= len(words):
                break
        else:
            if inverted:
                continue
            length = position - start + 1
            if length <= limit and (
                best is None or length < best[1] - best[0] + 1
            ):
                best = (start, position)
    if best is None:
        return None
    return text[matches[best[0]].start():matches[best[1]].end()]


def contains_quote(quote: str, text: str) -> bool:
    """Tell whether ``quote`` is verbatim inside ``text``."""
    return find_quote(quote, text) is not None


def ground(
    proposed: Sequence[ProposedEvidence],
    candidates: Sequence[Turn],
    participant: str,
    competency: Competency,
) -> Tuple[List[Evidence], int]:
    """Return the verified evidence and the number of rejected items.

    An item is kept when its indicator belongs to the competency and
    its quote is found in a candidate turn spoken by the participant.
    If the quote is not in the cited turn but is in another candidate
    turn of the participant, the citation is corrected. The stored
    quote is the transcript's wording and the polarity comes from the
    rubric: neither is taken from the model. Two quotes of the same
    turn for the same indicator count once.
    """
    own: Dict[int, Turn] = {
        t.index: t for t in candidates if t.speaker == participant
    }
    kept: List[Evidence] = []
    seen: set[Tuple[int, str]] = set()
    rejected = 0
    for item in proposed:
        indicator = competency.indicator(item.indicator.strip())
        turn = own.get(item.turn) if item.turn is not None else None
        passage = find_quote(item.quote, turn.text) if turn else None
        if passage is None:
            turn = None
            for other in own.values():
                passage = find_quote(item.quote, other.text)
                if passage is not None:
                    turn = other
                    break
        if indicator is None or turn is None or passage is None:
            rejected += 1
            continue
        if (turn.index, indicator.id) in seen:
            continue
        seen.add((turn.index, indicator.id))
        kept.append(Evidence(
            turn=turn.index,
            quote=passage,
            indicator_id=indicator.id,
            polarity=indicator.polarity,
        ))
    kept.sort(key=lambda e: (e.turn, e.indicator_id))
    return kept, rejected


def incidents(evidence: Sequence[Evidence]) -> int:
    """Return the number of independent incidents: distinct turns."""
    return len({e.turn for e in evidence})


def episodes(evidence: Sequence[Evidence], gap: int = 3) -> int:
    """Return the number of separate episodes the evidence comes from.

    Evidence turns at most ``gap`` turns apart belong to the same
    exchange and count as one episode.
    """
    turns = sorted({e.turn for e in evidence})
    return sum(
        1 for i, turn in enumerate(turns)
        if i == 0 or turn - turns[i - 1] > gap
    )


def decide_rating(
    rating: Optional[int],
    evidence: Sequence[Evidence],
    scale: Scale,
    min_incidents: int,
    min_episodes: int = 1,
    episode_gap: int = 3,
) -> Tuple[Optional[int], List[str]]:
    """Return the rating that the verified evidence supports, and flags.

    The rating is withheld (``None``) when the judge gave none, when it
    is off the scale, or when the evidence has fewer than
    ``min_incidents`` distinct turns or comes from fewer than
    ``min_episodes`` separate episodes. The extremes need breadth: the
    top rating needs two different positive indicators and the bottom
    one two negative incidents, otherwise the rating moves one point
    towards the centre. A rating that contradicts the polarity of all
    its evidence is kept but flagged for human review.
    """
    flags: List[str] = []
    if rating is None:
        return None, flags
    if rating < scale.minimum or rating > scale.maximum:
        return None, ["rating_off_scale"]
    if incidents(evidence) < min_incidents:
        return None, ["below_minimum_evidence"]
    if episodes(evidence, episode_gap) < min_episodes:
        return None, ["single_episode"]
    positive = [e for e in evidence if e.polarity == "positive"]
    negative = [e for e in evidence if e.polarity == "negative"]
    if rating == scale.maximum and len(
        {e.indicator_id for e in positive}
    ) < 2:
        rating -= 1
        flags.append("capped_narrow_evidence")
    elif rating == scale.minimum and incidents(negative) < 2:
        rating += 1
        flags.append("raised_narrow_evidence")
    if (rating > scale.midpoint() and not positive) or (
        rating < scale.midpoint() and not negative
    ):
        flags.append("rating_contradicts_evidence")
    return rating, flags
