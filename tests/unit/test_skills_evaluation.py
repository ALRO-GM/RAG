"""Scoring assessments against reference annotations."""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

from src.application.skills.evaluation import evaluate, rating_scores
from src.application.skills.gold import (
    AnnotatedEvidence,
    Annotation,
    AnnotationSet,
)
from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.interaction import describe
from src.domain.skills.rubric import Scale
from tests.fakes import make_transcript

SCALE = Scale()
TEAM = describe(make_transcript())
Key = Tuple[str, str]


def result(
    competency: str,
    rating: Optional[int] = None,
    turns: Sequence[int] = (),
    candidates: Sequence[int] = (),
    proposed: int = 0,
    rejected: int = 0,
    flags: Sequence[str] = (),
) -> CompetencyAssessment:
    """Return the system's assessment of one competency."""
    return CompetencyAssessment(
        competency_id=competency,
        status="rated" if rating is not None else "insufficient_evidence",
        rating=rating,
        evidence=[
            Evidence(turn=t, quote="q", indicator_id="L1",
                     polarity="positive")
            for t in turns
        ],
        candidate_turns=list(candidates),
        proposed=proposed, rejected=rejected, flags=list(flags),
    )


def session(
    cells: Dict[str, List[CompetencyAssessment]], session_id: str = "s1"
) -> SessionAssessment:
    """Return a session assessment from ``participant → results``."""
    return SessionAssessment(
        session_id=session_id, language="en", catalogue="core",
        catalogue_version="1.0", judge_model="j", embedding_model="",
        min_incidents=2, team=TEAM,
        scorecards=[
            ParticipantScorecard(
                participant=name, interaction=TEAM.of(name),
                assessments=items,
            )
            for name, items in cells.items()
        ],
    )


def gold(
    cells: Mapping[Key, Tuple[Optional[int], Sequence[int]]],
    session_id: str = "s1",
) -> AnnotationSet:
    """Return the reference from ``(participant, id) → (rating, turns)``."""
    return AnnotationSet(
        session_id=session_id, annotator="consensus", catalogue="core",
        catalogue_version="1.0",
        annotations=[
            Annotation(
                participant=p, competency_id=c, rating=rating,
                evidence=[AnnotatedEvidence(turn=t, indicator_id="L1")
                          for t in turns],
            )
            for (p, c), (rating, turns) in cells.items()
        ],
    )


SYSTEM = session({
    "Maya": [
        result("listening", 4, [0, 3], [0, 3], proposed=3, rejected=1),
        result("planning", None, [], [0, 3], flags=["too_few_turns"]),
    ],
    "Tom": [
        result("listening", 2, [1, 4], [1, 4], proposed=2,
               flags=["rating_contradicts_evidence"]),
        result("planning", 5, [7], [4, 7], proposed=4, rejected=3,
               flags=["rating_contradicts_evidence", "x"]),
    ],
})
REFERENCE = gold({
    ("Maya", "listening"): (5, [0, 3]),
    ("Maya", "planning"): (None, []),
    ("Tom", "listening"): (None, [1]),
    ("Tom", "planning"): (3, [4, 7, 9]),
    ("Priya", "planning"): (4, [5]),
})


def test_evaluate_counts_units_and_sessions() -> None:
    """Only the units present on both sides are scored."""
    evaluation = evaluate([SYSTEM], [REFERENCE], SCALE)
    assert (evaluation.sessions, evaluation.units) == (1, 4)


def test_evaluate_micro_precision_recall_f1() -> None:
    """Evidence turns are pooled over the units before dividing."""
    scores = evaluate([SYSTEM], [REFERENCE], SCALE).evidence
    assert scores.reference_turns == 6
    assert scores.precision == pytest.approx(4 / 5)
    assert scores.recall == pytest.approx(4 / 6)
    assert scores.f1 == pytest.approx(2 * 0.8 * (4 / 6) / (0.8 + 4 / 6))


def test_evaluate_retrieval_recall_from_candidate_turns() -> None:
    """The ceiling of the judge: reference turns among the candidates."""
    scores = evaluate([SYSTEM], [REFERENCE], SCALE).evidence
    assert scores.retrieval_recall == pytest.approx(5 / 6)


def test_evaluate_abstention_table() -> None:
    """Each unit falls in one cell of rated × rated."""
    table = evaluate([SYSTEM], [REFERENCE], SCALE).abstention
    assert (table.both_rated, table.both_abstained) == (2, 1)
    assert (table.system_only_rated, table.reference_only_rated) == (1, 0)
    inverse = evaluate(
        [session({"Tom": [result("planning")]})], [REFERENCE], SCALE
    ).abstention
    assert inverse.reference_only_rated == 1


def test_evaluate_quote_verification_rate_and_flags() -> None:
    """Verified / proposed quotes, and how often each flag was raised."""
    evaluation = evaluate([SYSTEM], [REFERENCE], SCALE)
    assert evaluation.quote_verification_rate == pytest.approx(5 / 9)
    assert evaluation.flagged == {
        "too_few_turns": 1, "rating_contradicts_evidence": 2, "x": 1,
    }


def test_evaluate_rating_scores() -> None:
    """Ratings are compared on the units both sides rated."""
    ratings = evaluate([SYSTEM], [REFERENCE], SCALE).ratings
    assert ratings.rated_by_both == 2
    assert ratings.exact == pytest.approx(0.0)
    assert ratings.within_one == pytest.approx(0.5)
    assert ratings.mean_absolute_error == pytest.approx(1.5)
    assert ratings.weighted_kappa is not None
    assert ratings.alpha is not None and ratings.alpha < 0.5


def test_evaluate_per_competency_split() -> None:
    """Each competency gets its own rating scores."""
    split = evaluate([SYSTEM], [REFERENCE], SCALE).per_competency
    assert sorted(split) == ["listening", "planning"]
    assert split["listening"].rated_by_both == 1
    assert split["listening"].within_one == 1.0
    assert split["listening"].mean_absolute_error == 1.0
    assert split["planning"].rated_by_both == 1
    assert split["planning"].mean_absolute_error == 2.0


def test_evaluate_perfect_system() -> None:
    """A system equal to the reference scores 1 everywhere."""
    cells = {("Maya", "listening"): (4, [0, 3]),
             ("Tom", "listening"): (2, [1]),
             ("Priya", "listening"): (5, [5, 2])}
    system = session({
        name: [result("listening", rating, turns, turns, len(turns))]
        for (name, _), (rating, turns) in cells.items()
    })
    evaluation = evaluate([system], [gold(cells)], SCALE)
    scores = evaluation.evidence
    assert (scores.precision, scores.recall, scores.f1) == (1.0, 1.0, 1.0)
    assert scores.retrieval_recall == 1.0
    assert evaluation.ratings.exact == 1.0
    assert evaluation.ratings.mean_absolute_error == 0.0
    assert evaluation.ratings.weighted_kappa == pytest.approx(1.0)
    assert evaluation.ratings.alpha == pytest.approx(1.0)
    assert evaluation.quote_verification_rate == 1.0
    assert evaluation.flagged == {}


def test_evaluate_nothing_predicted_and_nothing_wanted() -> None:
    """Without evidence on either side the ratios are undefined."""
    evaluation = evaluate(
        [session({"Maya": [result("planning")]})], [REFERENCE], SCALE
    )
    scores = evaluation.evidence
    assert scores.reference_turns == 0
    assert (scores.precision, scores.recall, scores.f1) == (None, None, None)
    assert scores.retrieval_recall is None
    assert evaluation.quote_verification_rate is None
    assert evaluation.ratings.rated_by_both == 0
    assert evaluation.ratings.exact is None
    assert evaluation.ratings.weighted_kappa is None
    assert evaluation.ratings.alpha is None
    assert evaluation.abstention.both_abstained == 1


def test_evaluate_no_overlap_between_evidence_sets() -> None:
    """Disjoint evidence gives precision, recall and F1 of zero."""
    system = session({"Maya": [result("listening", 4, [6], [6])]})
    scores = evaluate([system], [REFERENCE], SCALE).evidence
    assert (scores.precision, scores.recall, scores.f1) == (0.0, 0.0, 0.0)
    assert scores.retrieval_recall == 0.0


def test_evaluate_prediction_without_reference_evidence() -> None:
    """Evidence where the reference has none: precision 0, recall n/a."""
    system = session({"Maya": [result("planning", 3, [0], [0])]})
    scores = evaluate([system], [REFERENCE], SCALE).evidence
    assert (scores.precision, scores.recall, scores.f1) == (0.0, None, 0.0)


def test_evaluate_system_without_evidence_scores_zero_f1() -> None:
    """A system that finds nothing of the reference has F1 = 0."""
    system = session({"Maya": [result("listening", None, [], [0, 3])]})
    scores = evaluate([system], [REFERENCE], SCALE).evidence
    assert scores.recall == 0.0
    assert scores.f1 == 0.0


def test_evaluate_counts_unsupported_quotes() -> None:
    """Quotes dropped by the similarity check are summed."""
    first = result("listening", 4, [0, 3], [0, 3])
    first.unsupported = 2
    second = result("planning", None)
    second.unsupported = 1
    evaluation = evaluate(
        [session({"Maya": [first, second]})], [REFERENCE], SCALE
    )
    assert evaluation.unsupported_quotes == 3
    assert evaluate([SYSTEM], [REFERENCE], SCALE).unsupported_quotes == 0


def test_evaluate_rejects_off_scale_ratings_on_either_side() -> None:
    """A rating outside the scale is a user error, not a crash."""
    system = session({"Maya": [result("listening", 9, [0], [0])]})
    with pytest.raises(InvalidInputError, match="off the 1-5 scale"):
        evaluate([system], [REFERENCE], SCALE)
    reference = gold({("Maya", "listening"): (0, [0])})
    with pytest.raises(InvalidInputError, match="Rating.s. 0 are off"):
        evaluate([SYSTEM], [reference], SCALE)


def test_evaluate_several_sessions() -> None:
    """Units are keyed by session: same names in two sessions differ."""
    other = session({"Maya": [result("listening", 1, [0], [0])]}, "s2")
    reference = [REFERENCE, gold({("Maya", "listening"): (1, [0])}, "s2")]
    evaluation = evaluate([SYSTEM, other], reference, SCALE)
    assert (evaluation.sessions, evaluation.units) == (2, 5)
    assert evaluation.ratings.rated_by_both == 3


def test_evaluate_without_a_shared_unit_is_an_error() -> None:
    """Another session, or no data, cannot be evaluated."""
    with pytest.raises(InvalidInputError, match="share no"):
        evaluate([SYSTEM], [gold({("Maya", "listening"): (4, [0])}, "s9")],
                 SCALE)
    with pytest.raises(InvalidInputError, match="share no"):
        evaluate([], [REFERENCE], SCALE)
    with pytest.raises(InvalidInputError, match="share no"):
        evaluate([SYSTEM], [], SCALE)


def test_rating_scores_aligned_lists() -> None:
    """Exact, tolerant and chance-corrected agreement of two lists."""
    scores = rating_scores([1, 2, 3, None, 5], [1, 3, 5, 4, None], SCALE)
    assert scores.rated_by_both == 3
    assert scores.exact == pytest.approx(1 / 3)
    assert scores.within_one == pytest.approx(2 / 3)
    assert scores.mean_absolute_error == pytest.approx(1.0)
    assert scores.weighted_kappa is not None and scores.alpha is not None
