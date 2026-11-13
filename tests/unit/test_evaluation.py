"""Recall@k evaluation: IoU, per-question recall, dataset average."""

from typing import List

import pytest

from src.domain.evaluation import IOU_THRESHOLD, evaluate, iou, question_recall
from src.domain.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)


def src(path: str, first: int, last: int) -> MinimalSource:
    """Return a source."""
    return MinimalSource(
        file_path=path, first_character_index=first,
        last_character_index=last,
    )


def test_iou_identical_and_disjoint() -> None:
    """Identical ranges give 1, disjoint or touching ranges give 0."""
    assert iou(src("a", 0, 10), src("a", 0, 10)) == 1.0
    assert iou(src("a", 0, 10), src("a", 20, 30)) == 0.0
    assert iou(src("a", 0, 10), src("a", 10, 20)) == 0.0


def test_iou_partial_overlap() -> None:
    """Intersection over union of two overlapping ranges."""
    assert iou(src("a", 0, 10), src("a", 5, 15)) == pytest.approx(5 / 15)
    assert iou(src("a", 0, 100), src("a", 10, 20)) == pytest.approx(0.1)


def test_iou_different_files() -> None:
    """Ranges in different files never overlap."""
    assert iou(src("a", 0, 10), src("b", 0, 10)) == 0.0


def test_iou_degenerate_ranges() -> None:
    """Empty or inverted ranges do not divide by zero."""
    assert iou(src("a", 5, 5), src("a", 5, 5)) == 0.0
    assert iou(src("a", 10, 0), src("a", 0, 10)) == 0.0


def test_question_recall() -> None:
    """Share of expected sources found in the top-k."""
    expected = [src("a", 0, 100), src("b", 0, 100)]
    retrieved = [src("c", 0, 10), src("a", 0, 90), src("b", 0, 100)]
    assert question_recall(expected, retrieved, 1) == 0.0
    assert question_recall(expected, retrieved, 2) == 0.5
    assert question_recall(expected, retrieved, 3) == 1.0


def test_question_recall_threshold() -> None:
    """An overlap below the IoU threshold does not count."""
    tiny = int(1000 * IOU_THRESHOLD) - 1
    assert question_recall([src("a", 0, 1000)], [src("a", 0, tiny)], 1) == 0
    assert question_recall(
        [src("a", 0, 1000)], [src("a", 0, tiny + 2)], 1
    ) == 1.0


def test_question_recall_no_expected() -> None:
    """A question without expected sources scores zero."""
    assert question_recall([], [src("a", 0, 1)], 5) == 0.0


def make_dataset() -> RagDataset:
    """Return two answered questions and one unanswered question."""
    questions: List[AnsweredQuestion | UnansweredQuestion] = [
        AnsweredQuestion(question_id="q1", question="one",
                         sources=[src("a", 0, 100)], answer="x"),
        AnsweredQuestion(question_id="q2", question="two",
                         sources=[src("b", 0, 100)], answer="y"),
        UnansweredQuestion(question_id="q3", question="three"),
    ]
    return RagDataset(rag_questions=questions)


def test_evaluate_averages_and_missing() -> None:
    """Missing questions count as zero; unanswered ones are ignored."""
    results = StudentSearchResults(k=3, search_results=[
        MinimalSearchResults(
            question_id="q1", question="one",
            retrieved_sources=[src("z", 0, 1), src("a", 0, 100)],
        ),
        MinimalSearchResults(question_id="q3", question="three",
                             retrieved_sources=[]),
    ])
    report = evaluate(results, make_dataset(), ks=(1, 2))
    assert report.n_questions == 2
    assert report.n_missing == 1
    assert report.recall == {1: 0.0, 2: 0.5}


def test_evaluate_without_ground_truth() -> None:
    """A dataset without answered questions gives zero recall."""
    dataset = RagDataset(rag_questions=[
        UnansweredQuestion(question_id="q", question="?"),
    ])
    report = evaluate(StudentSearchResults(k=1, search_results=[]), dataset)
    assert report.n_questions == 0
    assert report.recall == {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
