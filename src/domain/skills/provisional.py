"""Provisional estimates: a lenient second tier below the firm level.

A firm level (``decide_rating`` in :mod:`src.domain.skills.grounding`)
needs ``min_incidents`` distinct turns from ``min_episodes`` exchanges.
A short interview rarely reaches it, and a report with nothing in it
helps nobody (#103). When there is at least one real piece of evidence
for a participant and competency, :func:`estimate` returns a
provisional level instead, always labelled as such, with its
confidence and the reasons it is not firm. It uses only what steps f
and g already produced: no new judge call.

The rule:

* **Basis.** The verified quotes of the judge (kept or below the
  similarity threshold) and the sentences the judge of step f confirmed
  for that competency. Without any of them there is no estimate.
* **Level.** The judge's own rating when it gave one on the scale;
  otherwise the side the basis leans to: one point above the centre
  when it has more positive than negative items, one below in the
  opposite case, the centre (rounded down) when even.
* **Bounds.** Never an extreme of the scale (on 1–5, between 2 and 4),
  and never on the other side of the centre from evidence that is all
  positive or all negative: then it moves to the centre.
* **Confidence.** ``medium`` when the basis has at least two distinct
  turns from at least two exchanges, the judge proposed a rating, no
  basis quote failed the similarity check and the participant said at
  least :data:`SHORT_PARTICIPATION_WORDS` words; ``low`` otherwise.

The thresholds here are choices for the display, not measured values;
like every tunable value of the analyzer they are provisional until
reference data exists (``docs/skills/DESIGN.md`` §9).
"""

import math
from typing import List, Optional, Sequence, Tuple

from src.domain.skills.assessment import (
    Estimate,
    Evidence,
    Observation,
    ParticipantScorecard,
)
from src.domain.skills.grounding import episodes, incidents
from src.domain.skills.rubric import Scale

SHORT_PARTICIPATION_WORDS = 300
"""Below this many words a participation is called short."""

SHORT_RECORDING_MINUTES = 10.0
"""Below this length a recording is called short."""

REASONS = (
    "few_quotes", "only_confirmations", "single_episode", "weak_match",
    "no_judge_rating", "short_participation",
)
"""Why an estimate is not firm, in the order they are listed."""

WHY_FEW_FIRM = (
    "short_recording", "little_participation", "below_floor",
    "few_quotes_proposed", "quotes_filtered", "not_shown",
)
"""Why a person has few firm levels, in the order they are listed."""


def _lean(basis: Sequence[Evidence], scale: Scale) -> int:
    """Return the level the polarity of ``basis`` points to."""
    centre = scale.midpoint()
    positive = sum(1 for e in basis if e.polarity == "positive")
    negative = len(basis) - positive
    if positive > negative:
        return math.floor(centre) + 1
    if negative > positive:
        return math.ceil(centre) - 1
    return math.floor(centre)


def _bounded(rating: int, basis: Sequence[Evidence], scale: Scale) -> int:
    """Keep ``rating`` off the extremes and on the side of the evidence."""
    low, high = scale.minimum + 1, scale.maximum - 1
    if low > high:
        low = high = math.floor(scale.midpoint())
    rating = min(max(rating, low), high)
    centre = scale.midpoint()
    if all(e.polarity == "positive" for e in basis) and rating < centre:
        rating = math.ceil(centre)
    elif all(e.polarity == "negative" for e in basis) and rating > centre:
        rating = math.floor(centre)
    return rating


def estimate(
    judge_rating: Optional[int],
    quotes: Sequence[Evidence],
    weak: Sequence[Observation],
    confirmed: Sequence[Observation],
    scale: Scale,
    min_incidents: int,
    words: int,
    episode_gap: int = 3,
) -> Tuple[Optional[Estimate], List[str]]:
    """Return the provisional estimate, or ``None`` with nothing to go on.

    ``quotes`` are the verified quotes that passed the similarity
    check, ``weak`` the verified ones that did not and ``confirmed``
    step f's confirmed sentences for the same participant and
    competency. Call it only when there is no firm level. Evidence
    from one exchange (turns at most ``episode_gap`` apart) is reported
    as ``single_episode`` whatever the firm rule asks. The flags
    returned say when the bounds moved the level
    (``estimate_bounded``).
    """
    as_evidence = [
        Evidence(turn=o.turn, quote=o.quote, indicator_id=o.indicator_id,
                 polarity=o.polarity)
        for o in [*weak, *confirmed]
    ]
    basis = [*quotes, *as_evidence]
    if not basis:
        return None, []
    on_scale = (
        judge_rating is not None
        and scale.minimum <= judge_rating <= scale.maximum
    )
    proposed = judge_rating if on_scale else None
    wanted = proposed if proposed is not None else _lean(basis, scale)
    rating = _bounded(wanted, basis, scale)
    flags = ["estimate_bounded"] if rating != wanted else []
    quoted = incidents([*quotes, *as_evidence[:len(weak)]])
    reasons: List[str] = []
    if 0 < quoted < min_incidents:
        reasons.append("few_quotes")
    if quoted == 0:
        reasons.append("only_confirmations")
    if episodes(basis, episode_gap) < 2:
        reasons.append("single_episode")
    if weak:
        reasons.append("weak_match")
    if proposed is None:
        reasons.append("no_judge_rating")
    if words < SHORT_PARTICIPATION_WORDS:
        reasons.append("short_participation")
    medium = (
        incidents(basis) >= 2 and episodes(basis, episode_gap) >= 2
        and proposed is not None and not weak
        and words >= SHORT_PARTICIPATION_WORDS
    )
    return Estimate(
        rating=rating, confidence="medium" if medium else "low",
        reasons=reasons, quotes=quoted, confirmations=len(confirmed),
        needed=min_incidents, words=words, judge_rating=judge_rating,
    ), flags


def why_few_firm(
    card: ParticipantScorecard,
    needed: int,
    minutes: Optional[float],
) -> List[str]:
    """Return why a person has few firm levels, or ``[]`` when not few.

    "Few" is fewer than half of the competencies assessed. The codes
    (:data:`WHY_FEW_FIRM`): a recording under
    :data:`SHORT_RECORDING_MINUTES`; a participation under
    :data:`SHORT_PARTICIPATION_WORDS` words; under the participation
    floor (no judge call at all); the judge proposed on average fewer
    quotes per competency than a level needs; half or more of its
    quotes were discarded by the checks; and, when none of these
    applies, that the conversation did not show these behaviours.
    """
    items = card.assessments
    firm = sum(1 for a in items if a.status == "rated")
    if not items or firm * 2 >= len(items):
        return []
    codes: List[str] = []
    if minutes is not None and minutes < SHORT_RECORDING_MINUTES:
        codes.append("short_recording")
    if card.interaction.words < SHORT_PARTICIPATION_WORDS:
        codes.append("little_participation")
    if any("below_participation_floor" in a.flags for a in items):
        codes.append("below_floor")
    proposed = sum(a.proposed for a in items)
    if "below_floor" not in codes:
        if proposed < needed * len(items):
            codes.append("few_quotes_proposed")
        dropped = sum(a.rejected + a.unsupported for a in items)
        if proposed and dropped * 2 >= proposed:
            codes.append("quotes_filtered")
    return codes or ["not_shown"]
