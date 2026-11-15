"""Judges of the analyzer: which one runs, and how two of them compare.

A *judge* is the language model that confirms tags and classifies
sentences. Three choices exist (``docs/skills/DESIGN.md`` §4):

- ``claude_code``: Claude through the Claude Code command line, on the
  owner's subscription; when its quota runs out the local judge takes
  over for the rest of the job, and every item says which judge
  produced it and why the other one did not;
- ``local``: a model running on this machine; nothing leaves it;
- ``both``: the two judges answer the same items, both answers are
  kept, and their agreement is measured (:func:`compare_labels`).

This module is plain data and arithmetic: no model, no clock, no file.
"""

from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

JudgeMode = Literal["claude_code", "local", "both"]
JUDGE_MODES: Tuple[str, ...] = ("claude_code", "local", "both")
"""The judges the web and the settings may choose."""

DEFAULT_JUDGE_MODE: JudgeMode = "claude_code"
"""The judge used when nothing was chosen (owner decision on #74)."""

QuotaState = Literal["available", "exhausted", "unknown"]

QUOTA_RETRY_HOURS = 5.0
"""How long a quota refusal counts when the service gave no reset time.

The subscription's usage limits run in five-hour windows, so after
that the next job tries Claude again.
"""


class JudgeState(BaseModel):
    """What is stored about the judge between jobs.

    ``mode`` is the choice made from the web (``None``: never chosen).
    ``quota_exhausted_at`` is when Claude last refused for quota,
    ``quota_resets`` when the service said it would come back (as it
    wrote it), and ``quota_ok_at`` when Claude last answered.
    """

    mode: Optional[JudgeMode] = None
    quota_exhausted_at: Optional[datetime] = None
    quota_resets: str = ""
    quota_ok_at: Optional[datetime] = None


def quota_status(
    state: JudgeState, now: datetime, retry_hours: float = QUOTA_RETRY_HOURS
) -> QuotaState:
    """Tell whether Claude's quota is available, as far as is known.

    ``exhausted`` while the last refusal is younger than
    ``retry_hours`` and no answer came after it; ``available`` once
    Claude answered after the last refusal (or it is older than that);
    ``unknown`` when Claude was never called.
    """
    refused = state.quota_exhausted_at
    answered = state.quota_ok_at
    if refused is None:
        return "unknown" if answered is None else "available"
    if answered is not None and answered >= refused:
        return "available"
    if now - refused < timedelta(hours=retry_hours):
        return "exhausted"
    return "available"


class LabelComparison(BaseModel):
    """How two judges' labels on the same items compare.

    ``kappa`` is Cohen's kappa over all labels; ``per_category`` holds,
    for each label, the kappa of "this label or not" (one against the
    rest), ``None`` when neither judge used it. ``confusion[a][b]``
    counts items judge A labelled ``a`` and judge B labelled ``b``.
    With reference labels, ``accuracy_a`` and ``accuracy_b`` are the
    share of reference items each judge got right (an item's
    reference may admit several labels).
    """

    items: int
    agreement: Optional[float]
    kappa: Optional[float]
    per_category: Dict[str, Optional[float]]
    confusion: Dict[str, Dict[str, int]]
    with_reference: int = 0
    accuracy_a: Optional[float] = None
    accuracy_b: Optional[float] = None
    per_category_accuracy: Dict[str, Dict[str, Optional[float]]] = Field(
        default_factory=dict
    )


def _kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    """Return Cohen's (nominal) kappa of label pairs."""
    total = len(pairs)
    if total == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / total
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum(
        left[label] * right.get(label, 0) for label in left
    ) / (total * total)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def compare_labels(
    a: Sequence[str],
    b: Sequence[str],
    reference: Optional[Sequence[Optional[Sequence[str]]]] = None,
) -> LabelComparison:
    """Compare two judges' labels, item by item, and with a reference.

    ``a`` and ``b`` are aligned label lists; ``reference``, when given,
    holds for each item the labels counted as right (``None`` or empty:
    the item has no reference and only counts for agreement).
    """
    if len(a) != len(b):
        raise ValueError("Both judges must label the same items.")
    if reference is not None and len(reference) != len(a):
        raise ValueError("The reference must cover the same items.")
    pairs = list(zip(a, b))
    labels = sorted(set(a) | set(b))
    confusion: Dict[str, Dict[str, int]] = {}
    for left, right in pairs:
        row = confusion.setdefault(left, {})
        row[right] = row.get(right, 0) + 1
    per_category: Dict[str, Optional[float]] = {
        label: _kappa([(str(x == label), str(y == label)) for x, y in pairs])
        for label in labels
    }
    result = LabelComparison(
        items=len(pairs),
        agreement=(
            round(sum(1 for x, y in pairs if x == y) / len(pairs), 4)
            if pairs else None
        ),
        kappa=_round(_kappa(pairs)),
        per_category={k: _round(v) for k, v in per_category.items()},
        confusion=confusion,
    )
    if reference is None:
        return result
    scored = [
        (x, y, set(ref)) for (x, y), ref in zip(pairs, reference) if ref
    ]
    if not scored:
        return result
    by_category: Dict[str, List[Tuple[bool, bool]]] = {}
    for x, y, ref in scored:
        for label in ref:
            by_category.setdefault(label, []).append((x in ref, y in ref))
    return result.model_copy(update={
        "with_reference": len(scored),
        "accuracy_a": round(
            sum(1 for x, _, ref in scored if x in ref) / len(scored), 4
        ),
        "accuracy_b": round(
            sum(1 for _, y, ref in scored if y in ref) / len(scored), 4
        ),
        "per_category_accuracy": {
            label: {
                "items": float(len(hits)),
                "a": round(sum(1 for h, _ in hits if h) / len(hits), 4),
                "b": round(sum(1 for _, h in hits if h) / len(hits), 4),
            }
            for label, hits in sorted(by_category.items())
        },
    })


def _round(value: Optional[float]) -> Optional[float]:
    """Round a statistic to four decimals, keeping ``None``."""
    return None if value is None else round(value, 4)
