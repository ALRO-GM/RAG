"""Recall@k evaluation of retrieved sources against a ground truth."""

from typing import Dict, List, Sequence

from src.domain.models import (
    AnsweredQuestion,
    EvaluationReport,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Return the intersection over union of two character ranges."""
    if a.file_path != b.file_path:
        return 0.0
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if inter <= 0:
        return 0.0
    union = max(a.last_character_index, b.last_character_index) - min(
        a.first_character_index, b.first_character_index
    )
    return inter / union if union > 0 else 0.0


def question_recall(
    expected: Sequence[MinimalSource],
    retrieved: Sequence[MinimalSource],
    k: int,
) -> float:
    """Share of expected sources overlapped by one of the top-k results."""
    if not expected:
        return 0.0
    top = retrieved[:k]
    found = sum(
        1 for gt in expected
        if any(iou(gt, r) >= IOU_THRESHOLD for r in top)
    )
    return found / len(expected)


def evaluate(
    results: StudentSearchResults,
    dataset: RagDataset,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> EvaluationReport:
    """Compute mean recall@k for every ``k`` in ``ks``.

    Questions of the ground truth that have no search result count as
    a recall of zero.
    """
    by_id = {
        r.question_id: r.retrieved_sources for r in results.search_results
    }
    truth: List[AnsweredQuestion] = [
        q for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)
    ]
    recall: Dict[int, float] = {}
    for k in ks:
        total = sum(
            question_recall(q.sources, by_id.get(q.question_id, []), k)
            for q in truth
        )
        recall[k] = total / len(truth) if truth else 0.0
    missing = sum(1 for q in truth if q.question_id not in by_id)
    return EvaluationReport(
        recall=recall, n_questions=len(truth), n_missing=missing
    )
