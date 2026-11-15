"""Calibrated confidences for the dialogue-act tagger (#82).

Two ways to turn what a tagger computes into probabilities that mean
what they say (of the labels given 70 %, about 70 % right;
``docs/skills/CONVERSATION.md`` §5.3.1 c):

- **Temperature scaling** (Guo et al., 2017) for a model with logits
  (the fine-tuned encoder): one number ``T`` divides the logits before
  the softmax; it is fitted on a held-out split by minimising the
  log-loss against the compatible sets, so the ranking never changes.
- **Outcome tables** for a tagger that gives a discrete answer (a rule
  that fired, the single label a language model named): the
  probability of each category given that answer is its share among
  the held-out sentences that got the same answer, smoothed towards the
  base rates (:class:`OutcomeTable`). It is histogram binning with one
  bin per answer, so it gives a calibrated top 3 too.

And the "dudoso" cut-off: :func:`margin_curve` measures, for candidate
margins over the base rate, how many shown labels stay clear and how
often those are right, and :func:`propose_margin` takes the smallest
margin that reaches a target precision.
"""

import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.act_agreement import tagger_labels
from src.domain.skills.act_tagging import Probabilities, Reference
from src.domain.skills.dialogue_acts import CATEGORIES

Array = npt.NDArray[np.float64]

_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


def softmax(logits: Array, temperature: float = 1.0) -> Array:
    """Return the row-wise softmax of ``logits / temperature``."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: Array = e / e.sum(axis=-1, keepdims=True)
    return out


def set_masks(references: Sequence[Reference]) -> Array:
    """Return a 0/1 matrix: sentence × category in its reference set."""
    mask = np.zeros((len(references), len(CATEGORIES)))
    for row, reference in enumerate(references):
        for c in reference.categories:
            mask[row, _INDEX[c]] = 1.0
    return mask


def set_nll(logits: Array, mask: Array, temperature: float) -> float:
    """Return the mean ``−ln P(set)`` at ``temperature``."""
    p = softmax(logits, temperature)
    mass = np.clip((p * mask).sum(axis=1), 1e-15, 1.0)
    return float(-np.log(mass).mean())


def fit_temperature(
    logits: Array, references: Sequence[Reference],
    low: float = 0.05, high: float = 20.0, steps: int = 60,
) -> float:
    """Return the temperature that minimises the held-out log-loss.

    Golden-section search on ``ln T`` in ``[low, high]`` (the negative
    log-likelihood is unimodal in ``T`` for this family).
    """
    if len(references) != len(logits) or not references:
        raise ValueError("need one reference per row of logits")
    mask = set_masks(references)
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc = set_nll(logits, mask, math.exp(c))
    fd = set_nll(logits, mask, math.exp(d))
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = set_nll(logits, mask, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = set_nll(logits, mask, math.exp(d))
    return math.exp((a + b) / 2)


def as_probabilities(rows: Array) -> List[Probabilities]:
    """Return a probability matrix as one category dict per row."""
    return [{c: float(row[i]) for i, c in enumerate(CATEGORIES)}
            for row in rows]


def reference_rates(references: Sequence[Reference]) -> Dict[str, float]:
    """Return each category's share of the references.

    Weights count when given; a compatible set splits its sentence
    equally among its categories.
    """
    if not references:
        raise ValueError("no references")
    rates = {c: 0.0 for c in CATEGORIES}
    for reference in references:
        if reference.labels:
            for lab in reference.labels:
                rates[lab.label] += lab.weight / 100.0
        else:
            for c in reference.categories:
                rates[c] += 1.0 / len(reference.categories)
    return {c: r / len(references) for c, r in rates.items()}


class OutcomeTable(BaseModel):
    """Probabilities per discrete answer, learned on held-out sentences.

    ``rows`` maps an answer (a rule key, a language model's label) to a
    distribution over the 13 categories; ``prior`` answers what was
    never seen. ``counts`` keeps how many sentences each row is
    estimated from.
    """

    model_config = ConfigDict(frozen=True)

    rows: Dict[str, Dict[str, float]]
    counts: Dict[str, int]
    prior: Dict[str, float]
    smoothing: float = Field(ge=0)

    def probabilities(self, outcome: str) -> Probabilities:
        """Return the distribution for ``outcome`` (the prior if unseen)."""
        return dict(self.rows.get(outcome, self.prior))


def fit_outcomes(
    outcomes: Sequence[str], references: Sequence[Reference],
    smoothing: float = 2.0,
    prior: Optional[Mapping[str, float]] = None,
) -> OutcomeTable:
    """Learn an :class:`OutcomeTable` from answers and their references.

    Each sentence adds its reference to its answer's row (weights, or
    an equal split of a compatible set); the row is
    ``(counts + smoothing · prior) / (n + smoothing)``, with ``prior``
    the base rates of all the references unless given.
    """
    if not outcomes or len(outcomes) != len(references):
        raise ValueError("need one reference per outcome")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    base = dict(prior) if prior is not None else reference_rates(references)
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for outcome, reference in zip(outcomes, references):
        row = sums.setdefault(outcome, {c: 0.0 for c in CATEGORIES})
        counts[outcome] = counts.get(outcome, 0) + 1
        for c, share in reference_rates([reference]).items():
            row[c] += share
    rows = {
        o: {c: (row[c] + smoothing * base.get(c, 0.0))
            / (counts[o] + smoothing) for c in CATEGORIES}
        for o, row in sums.items()
    }
    return OutcomeTable(rows=rows, counts=counts,
                        prior={c: base.get(c, 0.0) for c in CATEGORIES},
                        smoothing=smoothing)


class MarginPoint(BaseModel):
    """What one "dudoso" margin does to the shown labels (top 3)."""

    model_config = ConfigDict(frozen=True)

    margin: float
    labels: int = Field(ge=0)
    clear: int = Field(ge=0)
    clear_precision: Optional[float]
    doubtful_precision: Optional[float]
    clear_top1: int = Field(ge=0)
    clear_top1_precision: Optional[float]


def _right(category: str, reference: Reference) -> bool:
    """Tell whether ``category`` is one of the reference's acts."""
    return (category in reference.categories
            or any(lab.label == category for lab in reference.labels))


def margin_curve(
    references: Sequence[Reference],
    predictions: Sequence[Mapping[str, float]],
    rates: Mapping[str, float],
    margins: Sequence[float],
) -> List[MarginPoint]:
    """Return, per margin, how many shown labels are clear and right.

    The shown labels are the tagger's top 3 as
    :func:`~src.domain.skills.act_agreement.tagger_labels` writes them;
    a label is clear when its probability reaches the base rate of its
    category plus the margin, and right when its category is one of the
    reference's acts.
    """
    if len(references) != len(predictions):
        raise ValueError("need one prediction per reference sentence")
    points: List[MarginPoint] = []
    for margin in margins:
        thresholds = {c: rates.get(c, 0.0) + margin for c in CATEGORIES}
        clear = right_clear = doubtful = right_doubtful = 0
        top1 = right_top1 = 0
        for reference, p in zip(references, predictions):
            shown = tagger_labels(p, thresholds=thresholds).labels
            for rank, lab in enumerate(shown):
                hit = _right(lab.label, reference)
                if lab.doubtful:
                    doubtful += 1
                    right_doubtful += hit
                else:
                    clear += 1
                    right_clear += hit
                    if rank == 0:
                        top1 += 1
                        right_top1 += hit
        points.append(MarginPoint(
            margin=round(float(margin), 4), labels=clear + doubtful,
            clear=clear,
            clear_precision=right_clear / clear if clear else None,
            doubtful_precision=(right_doubtful / doubtful
                                if doubtful else None),
            clear_top1=top1,
            clear_top1_precision=right_top1 / top1 if top1 else None,
        ))
    return points


def propose_margin(points: Sequence[MarginPoint],
                   target: float = 0.8) -> Optional[float]:
    """Return the smallest margin whose clear labels reach ``target``.

    ``None`` when no margin reaches it (with any clear label left).
    """
    for point in sorted(points, key=lambda p: p.margin):
        if point.clear and (point.clear_precision or 0.0) >= target:
            return point.margin
    return None


def fold_of(groups: Sequence[str], folds: int) -> List[int]:
    """Return the fold of each item, keeping each group in one fold.

    Groups (dialogues) are numbered in sorted order and dealt round-robin
    to ``folds`` folds, so no dialogue is ever split between training
    and test.
    """
    if folds < 2:
        raise ValueError("need at least 2 folds")
    order = {g: n % folds for n, g in enumerate(sorted(set(groups)))}
    return [order[g] for g in groups]


def cross_fit_outcomes(
    outcomes: Sequence[str], references: Sequence[Reference],
    groups: Sequence[str], folds: int = 5, smoothing: float = 2.0,
) -> List[Probabilities]:
    """Return held-out probabilities of discrete answers by k-fold.

    Each sentence gets the distribution of its answer in a table fitted
    on the other folds only (grouped by dialogue: :func:`fold_of`).
    """
    if not (len(outcomes) == len(references) == len(groups)):
        raise ValueError("outcomes, references and groups must align")
    assignment = fold_of(groups, folds)
    result: List[Probabilities] = [{} for _ in outcomes]
    for fold in range(folds):
        train = [n for n, f in enumerate(assignment) if f != fold]
        if not train:
            continue
        table = fit_outcomes([outcomes[n] for n in train],
                             [references[n] for n in train], smoothing)
        for n, f in enumerate(assignment):
            if f == fold:
                result[n] = table.probabilities(outcomes[n])
    return result
