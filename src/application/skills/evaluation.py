"""Use case: score the system's assessments against reference data."""

from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from src.application.skills.gold import (
    Annotation,
    AnnotationSet,
    Unit,
    check_on_scale,
)
from src.domain.errors import InvalidInputError
from src.domain.skills.agreement import (
    Rating,
    cohen_kappa,
    krippendorff_alpha,
    mean_absolute_error,
    percent_agreement,
)
from src.domain.skills.assessment import (
    CompetencyAssessment,
    SessionAssessment,
)
from src.domain.skills.rubric import Scale


class EvidenceScores(BaseModel):
    """Turn-level agreement of the system's evidence with the reference.

    ``retrieval_recall`` is the share of reference evidence turns that
    the retriever put among the candidates (the ceiling of the judge);
    precision, recall and F1 compare the verified evidence turns with
    the reference ones, micro-averaged.
    """

    reference_turns: int
    retrieval_recall: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]


class RatingScores(BaseModel):
    """Agreement of the system's ratings with the reference ratings."""

    rated_by_both: int
    exact: Optional[float]
    within_one: Optional[float]
    mean_absolute_error: Optional[float]
    weighted_kappa: Optional[float]
    alpha: Optional[float]


class AbstentionScores(BaseModel):
    """How the "insufficient evidence" decisions compare."""

    both_rated: int
    both_abstained: int
    system_only_rated: int
    reference_only_rated: int


class SkillsEvaluation(BaseModel):
    """Quality of the system on a reference set.

    ``quote_verification_rate`` is the share of the judge's proposed
    quotes that were found verbatim in the transcript: what remains is
    the hallucination rate that grounding removed.
    ``unsupported_quotes`` counts verbatim quotes dropped because they
    did not resemble the indicator they were filed under.
    """

    sessions: int
    units: int
    evidence: EvidenceScores
    ratings: RatingScores
    abstention: AbstentionScores
    quote_verification_rate: Optional[float]
    unsupported_quotes: int
    flagged: Dict[str, int]
    per_competency: Dict[str, RatingScores]


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """Return a ratio, or ``None`` when the denominator is zero."""
    return numerator / denominator if denominator else None


def rating_scores(
    system: Sequence[Rating], reference: Sequence[Rating], scale: Scale
) -> RatingScores:
    """Compare two aligned lists of ratings."""
    points = [float(p) for p in scale.points()]
    both = sum(
        1 for s, r in zip(system, reference)
        if s is not None and r is not None
    )
    return RatingScores(
        rated_by_both=both,
        exact=percent_agreement(system, reference),
        within_one=percent_agreement(system, reference, tolerance=1.0),
        mean_absolute_error=mean_absolute_error(system, reference),
        weighted_kappa=cohen_kappa(system, reference, points),
        alpha=krippendorff_alpha([system, reference], points, "ordinal")
        if both else None,
    )


def evaluate(
    assessments: Sequence[SessionAssessment],
    reference: Sequence[AnnotationSet],
    scale: Scale,
) -> SkillsEvaluation:
    """Score ``assessments`` on the units the reference covers."""
    gold: Dict[Unit, Annotation] = {}
    for item in reference:
        gold.update(item.by_unit())
    pairs: List[Tuple[Unit, CompetencyAssessment, Annotation]] = []
    for session in assessments:
        for card in session.scorecards:
            for result in card.assessments:
                unit = (
                    session.session_id, card.participant,
                    result.competency_id,
                )
                if unit in gold:
                    pairs.append((unit, result, gold[unit]))
    if not pairs:
        raise InvalidInputError(
            "The assessments and the reference share no "
            "(session, participant, competency)."
        )
    check_on_scale(
        [a.rating for _, _, a in pairs] + [r.rating for _, r, _ in pairs],
        scale,
    )
    hits = predicted = wanted = retrieved = proposed = rejected = 0
    flagged: Dict[str, int] = {}
    counts = {"bb": 0, "nn": 0, "sys": 0, "ref": 0}
    for _, result, annotation in pairs:
        own = {e.turn for e in result.evidence}
        reference_turns = annotation.turns()
        hits += len(own & reference_turns)
        predicted += len(own)
        wanted += len(reference_turns)
        retrieved += len(reference_turns & set(result.candidate_turns))
        proposed += result.proposed
        rejected += result.rejected
        for flag in result.flags:
            flagged[flag] = flagged.get(flag, 0) + 1
        key = {
            (True, True): "bb", (False, False): "nn",
            (True, False): "sys", (False, True): "ref",
        }[(result.rating is not None, annotation.rating is not None)]
        counts[key] += 1
    precision = _ratio(hits, predicted)
    recall = _ratio(hits, wanted)
    f1 = _ratio(2 * hits, predicted + wanted)

    def rows(
        subset: Sequence[Tuple[Unit, CompetencyAssessment, Annotation]],
    ) -> Tuple[List[Rating], List[Rating]]:
        """Return the aligned system and reference ratings."""
        return (
            [None if r.rating is None else float(r.rating)
             for _, r, _ in subset],
            [None if a.rating is None else float(a.rating)
             for _, _, a in subset],
        )

    per_competency = {
        competency_id: rating_scores(
            *rows([p for p in pairs if p[0][2] == competency_id]), scale
        )
        for competency_id in sorted({p[0][2] for p in pairs})
    }
    return SkillsEvaluation(
        sessions=len({p[0][0] for p in pairs}),
        units=len(pairs),
        evidence=EvidenceScores(
            reference_turns=wanted,
            retrieval_recall=_ratio(retrieved, wanted),
            precision=precision,
            recall=recall,
            f1=f1,
        ),
        ratings=rating_scores(*rows(pairs), scale),
        abstention=AbstentionScores(
            both_rated=counts["bb"],
            both_abstained=counts["nn"],
            system_only_rated=counts["sys"],
            reference_only_rated=counts["ref"],
        ),
        quote_verification_rate=_ratio(proposed - rejected, proposed),
        unsupported_quotes=sum(r.unsupported for _, r, _ in pairs),
        flagged=flagged,
        per_competency=per_competency,
    )
