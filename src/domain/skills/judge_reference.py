"""Reference for judging rubric indicators, and how a judge is scored on it.

``tag --confirm`` asks a judge, for one sentence, which of the offered
indicators it clearly shows, or none (``tag_prompting``). The reference
holds sentences whose answer the author decided beforehand: half show
one indicator (positive or negative), half only look like one. The
scores say how far a judge's confirmations can be trusted (precision,
recall, how often it confirms what is not there), whether the order in
which the indicators are listed changes its answer, and how far two
judges agree. Pure functions: nothing here reads a file or calls a model.
"""

from typing import Dict, List, Literal, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.agreement import cohen_kappa

Order = Literal["forward", "reversed"]
Kind = Literal["true", "wrong"]


class ReferenceItem(BaseModel):
    """One sentence with the author's intended verdict.

    ``expect`` is the indicator the sentence shows, or ``None`` when it
    shows none of the ``offered`` competencies' indicators (``kind``
    ``wrong``: it only resembles ``resembles``). ``context`` lines are
    ``"Speaker: text"``.
    """

    id: str = Field(min_length=1)
    offered: List[str] = Field(min_length=1)
    context: List[str] = Field(default_factory=list)
    speaker: str
    text: str = Field(min_length=1)
    expect: Optional[str] = None
    kind: Kind
    resembles: Optional[str] = None


class ReferenceSet(BaseModel):
    """The reference file: a schema name, a note and the items."""

    schema_name: str = Field(alias="schema")
    note: str = ""
    items: List[ReferenceItem]

    model_config = ConfigDict(populate_by_name=True)


class Outcome(BaseModel):
    """What one judge answered for one item in one listing order.

    ``named`` is the offered indicator it confirmed, ``None`` for none;
    ``outside`` is true when it named an indicator it was not offered
    (counted as none, like ``TagConfirmer``); ``parsed`` is false when
    the reply could not be read. ``first_listed`` is the indicator that
    stood first in the list it was shown.
    """

    item_id: str
    order: Order
    named: Optional[str] = None
    parsed: bool = True
    outside: bool = False
    first_listed: str = ""


def _ratio(part: int, whole: int) -> Optional[float]:
    """Return ``part / whole`` rounded, or ``None`` without a whole."""
    return round(part / whole, 3) if whole else None


def score(
    items: Sequence[ReferenceItem],
    outcomes: Sequence[Outcome],
    negatives: Set[str],
) -> Dict[str, object]:
    """Return the figures of one judge in one listing order.

    Only the items with a readable reply count, except ``unparsed``.
    ``precision``: confirmations naming the intended indicator over all
    confirmations. ``recall``: the same over the items that show one.
    ``false_confirm_rate``: confirmations over the items that show none.
    ``negative_recall`` is recall over the items whose indicator is in
    ``negatives``. ``first_listed_rate``: the share of confirmations that
    named the indicator listed first.
    """
    by_id = {item.id: item for item in items}
    confirmed = correct = wrong_item_confirmed = 0
    true_items = wrong_items = negative_items = negative_hits = 0
    other_indicator = first = unparsed = true_hits = 0
    for outcome in outcomes:
        item = by_id[outcome.item_id]
        if not outcome.parsed:
            unparsed += 1
            continue
        hit = outcome.named is not None
        if item.expect is None:
            wrong_items += 1
            wrong_item_confirmed += int(hit)
        else:
            true_items += 1
            negative = item.expect in negatives
            negative_items += int(negative)
            if outcome.named == item.expect:
                true_hits += 1
                negative_hits += int(negative)
            elif hit:
                other_indicator += 1
        if hit:
            confirmed += 1
            correct += int(outcome.named == item.expect)
            first += int(outcome.named == outcome.first_listed)
    return {
        "items": len(outcomes),
        "unparsed": unparsed,
        "confirmed": confirmed,
        "correct": correct,
        "precision": _ratio(correct, confirmed),
        "recall": _ratio(true_hits, true_items),
        "false_confirm_rate": _ratio(wrong_item_confirmed, wrong_items),
        "wrong_indicator_on_true": other_indicator,
        "negative_recall": _ratio(negative_hits, negative_items),
        "first_listed_rate": _ratio(first, confirmed),
    }


def position_bias(
    forward: Sequence[Outcome], reverse: Sequence[Outcome]
) -> Dict[str, object]:
    """Compare one judge's answers when the listing order is reversed.

    ``flip_rate``: the share of items whose answer changed, over the
    items readable in both orders.
    """
    other = {o.item_id: o for o in reverse}
    both = [
        (a, other[a.item_id]) for a in forward
        if a.item_id in other and a.parsed and other[a.item_id].parsed
    ]
    flips = sum(1 for a, b in both if a.named != b.named)
    return {
        "compared": len(both),
        "flips": flips,
        "flip_rate": _ratio(flips, len(both)),
    }


def agreement(
    a: Sequence[Outcome], b: Sequence[Outcome]
) -> Dict[str, object]:
    """Return how two judges agree over the outcomes both could read.

    ``same_answer``: the same indicator or both none. ``kappa``:
    Cohen's kappa on confirmed against none (nominal); ``None`` when
    either judge gave a single answer throughout, as kappa then says
    nothing (it would read 1.0 or 0.0 whatever the other did).
    """
    other = {(o.item_id, o.order): o for o in b}
    pairs = [
        (x, other[(x.item_id, x.order)]) for x in a
        if (x.item_id, x.order) in other
        and x.parsed and other[(x.item_id, x.order)].parsed
    ]
    same = sum(1 for x, y in pairs if x.named == y.named)
    left = [float(x.named is not None) for x, _ in pairs]
    right = [float(y.named is not None) for _, y in pairs]
    kappa = (
        cohen_kappa(left, right, [0.0, 1.0], weights="none")
        if len(set(left)) > 1 and len(set(right)) > 1 else None
    )
    return {
        "compared": len(pairs),
        "same_answer": _ratio(same, len(pairs)),
        "kappa": None if kappa is None else round(kappa, 3),
    }
