"""Scores of a dialogue-act tagger against compatible-set references.

The measures are those of ``docs/skills/CONVERSATION.md`` §6.3.1
(top-1 and top-3 accuracy, per-category precision/recall/F1 and
confusion, log-loss, expected calibration error), generalised to the
references of the public corpora, where a source label is compatible
with a *set* of our categories (MRDA's statements are ``informacion``
or ``opinion``):

- a prediction is right when it falls inside the set (top-k: when any
  category of the set is among the first k);
- log-loss is ``−ln`` of the probability given to the whole set, or,
  when the reference carries weights (our set), the cross-entropy
  against them, as :func:`~src.domain.skills.act_agreement.log_loss`;
- for precision, recall and the confusion matrix each sentence needs one
  reference category: the predicted one when it is in the set, else the
  set's category the tagger found most probable (:func:`resolve`).
  Per-category figures are only *measurable* for categories that some
  reference names alone; a category a corpus only ever gives inside a
  set (AMI's ``propuesta|peticion``) gets no figure for that corpus.

With one-category references every measure is the usual one.
"""

import math
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.act_tagging import (
    Probabilities,
    Reference,
    top_category,
)
from src.domain.skills.dialogue_acts import CATEGORIES

EPS = 1e-15
"""Probabilities are clipped to this before a logarithm."""


class CategoryScore(BaseModel):
    """Precision, recall and F1 of one category (``None``: undefined)."""

    model_config = ConfigDict(frozen=True)

    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    measurable: bool


class TaggerScores(BaseModel):
    """Every measure of one tagger on one set of references."""

    model_config = ConfigDict(frozen=True)

    sentences: int = Field(ge=0)
    top1: float
    top3: float
    log_loss: float
    ece: float
    macro_f1: Optional[float]
    per_category: Dict[str, CategoryScore]
    confusion: Dict[str, Dict[str, int]]


def _check(references: Sequence[Reference],
           predictions: Sequence[Mapping[str, float]]) -> None:
    """Refuse empty or unequal inputs."""
    if not references or len(references) != len(predictions):
        raise ValueError("need one prediction per reference sentence")


def ranking(probabilities: Mapping[str, float]) -> List[str]:
    """Return the 13 categories from most to least probable."""
    return sorted(CATEGORIES, key=lambda c: (-probabilities.get(c, 0.0),
                                             CATEGORIES.index(c)))


def top_k_accuracy(references: Sequence[Reference],
                   predictions: Sequence[Mapping[str, float]],
                   k: int = 1) -> float:
    """Return the share of sentences with a right category in the top k."""
    _check(references, predictions)
    if k < 1:
        raise ValueError("k must be >= 1")
    hits = sum(bool(set(ranking(p)[:k]) & set(r.categories))
               for r, p in zip(references, predictions))
    return hits / len(references)


def sentence_log_loss(reference: Reference,
                      probabilities: Mapping[str, float]) -> float:
    """Return one sentence's log-loss (natural log).

    Against the weights when the reference has them, else ``−ln`` of
    the probability of the compatible set.
    """
    if reference.labels:
        return -sum(lab.weight / 100.0 * math.log(
            max(probabilities.get(lab.label, 0.0), EPS))
            for lab in reference.labels)
    mass = sum(probabilities.get(c, 0.0) for c in reference.categories)
    return -math.log(max(min(mass, 1.0), EPS))


def log_loss(references: Sequence[Reference],
             predictions: Sequence[Mapping[str, float]]) -> float:
    """Return the mean :func:`sentence_log_loss`."""
    _check(references, predictions)
    return sum(sentence_log_loss(r, p)
               for r, p in zip(references, predictions)) / len(references)


def expected_calibration_error(
    references: Sequence[Reference],
    predictions: Sequence[Mapping[str, float]],
    bins: int = 10,
) -> float:
    """Return the ECE of the top-1 confidence (equal-width bins).

    A top-1 category is right when it is in the reference set; bins as
    in :func:`~src.domain.skills.act_agreement.expected_calibration_error`.
    """
    _check(references, predictions)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    edges = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    cells: Dict[int, List[float]] = {}
    for reference, p in zip(references, predictions):
        top = top_category(p)
        confidence = float(p.get(top, 0.0))
        index = int(np.searchsorted(edges, confidence, side="left"))
        cell = cells.setdefault(index, [0.0, 0.0])
        cell[0] += float(top in reference.categories)
        cell[1] += confidence
    return sum(abs(h - c) for h, c in cells.values()) / len(references)


def resolve(reference: Reference, probabilities: Mapping[str, float]) -> str:
    """Return the one reference category a prediction is compared with.

    The predicted category when it is in the set; otherwise the set's
    category with the highest predicted probability.
    """
    top = top_category(probabilities)
    if top in reference.categories:
        return top
    return max(reference.categories,
               key=lambda c: (probabilities.get(c, 0.0),
                              -CATEGORIES.index(c)))


def confusion(references: Sequence[Reference],
              predictions: Sequence[Mapping[str, float]]
              ) -> Dict[str, Dict[str, int]]:
    """Return counts per resolved reference (rows) and top-1 (columns).

    Only rows and columns that occur are kept.
    """
    _check(references, predictions)
    table: Dict[str, Dict[str, int]] = {}
    for reference, p in zip(references, predictions):
        row = table.setdefault(resolve(reference, p), {})
        top = top_category(p)
        row[top] = row.get(top, 0) + 1
    return {g: dict(sorted(table[g].items(),
                           key=lambda kv: CATEGORIES.index(kv[0])))
            for g in CATEGORIES if g in table}


def _ratio(num: int, den: int) -> Optional[float]:
    """Return ``num / den`` or ``None`` when ``den`` is 0."""
    return num / den if den else None


def per_category(references: Sequence[Reference],
                 predictions: Sequence[Mapping[str, float]]
                 ) -> Dict[str, CategoryScore]:
    """Return precision, recall and F1 per category (resolved references).

    A category is measurable when some reference names it alone;
    otherwise its figures are ``None``.
    """
    _check(references, predictions)
    alone = {r.categories[0] for r in references if len(r.categories) == 1}
    gold = [resolve(r, p) for r, p in zip(references, predictions)]
    tops = [top_category(p) for p in predictions]
    scores: Dict[str, CategoryScore] = {}
    for c in CATEGORIES:
        tp = sum(g == c and t == c for g, t in zip(gold, tops))
        support = sum(g == c for g in gold)
        predicted = sum(t == c for t in tops)
        measurable = c in alone
        precision = _ratio(tp, predicted) if measurable else None
        recall = _ratio(tp, support) if measurable else None
        f1 = None
        if precision is not None and recall is not None:
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall else 0.0)
        scores[c] = CategoryScore(
            precision=precision, recall=recall, f1=f1, support=support,
            predicted=predicted, measurable=measurable)
    return scores


def macro_f1(scores: Mapping[str, CategoryScore]) -> Optional[float]:
    """Return the mean F1 of measurable categories with references."""
    values = [s.f1 for s in scores.values()
              if s.measurable and s.support and s.f1 is not None]
    return sum(values) / len(values) if values else None


def score(references: Sequence[Reference],
          predictions: Sequence[Probabilities]) -> TaggerScores:
    """Return every measure of a tagger on ``references``."""
    _check(references, predictions)
    categories = per_category(references, predictions)
    return TaggerScores(
        sentences=len(references),
        top1=top_k_accuracy(references, predictions, 1),
        top3=top_k_accuracy(references, predictions, 3),
        log_loss=log_loss(references, predictions),
        ece=expected_calibration_error(references, predictions),
        macro_f1=macro_f1(categories),
        per_category=categories,
        confusion=confusion(references, predictions),
    )
