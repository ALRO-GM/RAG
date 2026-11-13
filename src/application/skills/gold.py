"""Reference data: human annotations, their agreement and consensus.

A reference ("gold") set is what the system is tuned and judged
against, so it has to be shown reliable first: several annotators rate
the same sessions independently, their agreement is measured, and only
then are their annotations merged into a consensus.
"""

from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from src.domain.errors import InvalidInputError
from src.domain.skills.agreement import (
    Rating,
    cohen_kappa,
    icc_two_way,
    interpret_alpha,
    krippendorff_alpha,
    percent_agreement,
    set_scores,
)
from src.domain.skills.rubric import RubricCatalogue, Scale
from src.domain.skills.transcript import Transcript

Unit = Tuple[str, str, str]


class AnnotatedEvidence(BaseModel):
    """A turn marked by an annotator as evidence of an indicator."""

    turn: int = Field(ge=0)
    indicator_id: str


class Annotation(BaseModel):
    """One annotator's judgement of a competency of a participant.

    ``rating`` is ``None`` when the annotator found too little
    evidence to rate ("insufficient evidence" is a valid answer).
    """

    participant: str
    competency_id: str
    rating: Optional[int] = None
    evidence: List[AnnotatedEvidence] = Field(default_factory=list)
    notes: str = ""

    def turns(self) -> Set[int]:
        """Return the turns marked as evidence."""
        return {e.turn for e in self.evidence}


class AnnotationSet(BaseModel):
    """Everything one annotator marked on one session."""

    session_id: str
    annotator: str
    catalogue: str
    catalogue_version: str
    annotations: List[Annotation]

    def by_unit(self) -> Dict[Unit, Annotation]:
        """Return the annotations keyed by (session, participant, id)."""
        return {
            (self.session_id, a.participant, a.competency_id): a
            for a in self.annotations
        }


MIN_UNITS_FOR_BAND = 10


class CompetencyAgreement(BaseModel):
    """Agreement of the annotators on one competency.

    ``rated_units`` counts the units at least two annotators rated;
    below ``MIN_UNITS_FOR_BAND`` the alpha is shown but not
    interpreted, because it is too unstable to mean anything.
    """

    competency_id: str
    units: int
    rated_units: int
    alpha: Optional[float]
    band: str
    within_one: Optional[float]
    evidence_f1: Optional[float]


class AgreementReport(BaseModel):
    """Inter-annotator agreement over a set of sessions.

    ``alpha`` is Krippendorff's ordinal alpha of the ratings; ``icc``
    is ICC(2,1) and ``icc_average`` ICC(2,k), computed on the units
    every annotator rated; ``abstention_kappa`` is the mean pairwise
    Cohen's kappa on "rated or not"; ``evidence_f1`` is the mean
    pairwise F1 of the turns marked as evidence.
    """

    annotators: List[str]
    sessions: int
    units: int
    alpha: Optional[float]
    band: str
    icc: Optional[float]
    icc_average: Optional[float]
    within_one: Optional[float]
    abstention_kappa: Optional[float]
    evidence_f1: Optional[float]
    pairwise_kappa: Dict[str, Optional[float]]
    per_competency: List[CompetencyAgreement]


def blank_annotations(
    transcript: Transcript, catalogue: RubricCatalogue, annotator: str
) -> AnnotationSet:
    """Return an empty annotation form: one cell per participant × id."""
    if not annotator.strip():
        raise InvalidInputError("The annotator name is empty.")
    return AnnotationSet(
        session_id=transcript.session_id,
        annotator=annotator.strip(),
        catalogue=catalogue.name,
        catalogue_version=catalogue.version,
        annotations=[
            Annotation(participant=speaker, competency_id=competency.id)
            for speaker in transcript.speakers()
            for competency in catalogue.competencies
        ],
    )


def validate_annotations(
    annotations: AnnotationSet,
    transcript: Transcript,
    catalogue: RubricCatalogue,
) -> List[str]:
    """Return every problem of a filled annotation form (none = valid)."""
    problems: List[str] = []
    if annotations.session_id != transcript.session_id:
        problems.append(
            f"session is {annotations.session_id!r}, transcript is "
            f"{transcript.session_id!r}"
        )
    if (annotations.catalogue, annotations.catalogue_version) != (
        catalogue.name, catalogue.version
    ):
        problems.append(
            f"annotated with catalogue {annotations.catalogue} "
            f"{annotations.catalogue_version}, expected {catalogue.name} "
            f"{catalogue.version}"
        )
    speakers = set(transcript.speakers())
    competencies = {c.id: c for c in catalogue.competencies}
    points = set(catalogue.scale.points())
    seen: Set[Tuple[str, str]] = set()
    for item in annotations.annotations:
        where = f"{item.participant}/{item.competency_id}"
        if (item.participant, item.competency_id) in seen:
            problems.append(f"{where}: annotated twice")
        seen.add((item.participant, item.competency_id))
        if item.participant not in speakers:
            problems.append(f"{where}: unknown participant")
            continue
        competency = competencies.get(item.competency_id)
        if competency is None:
            problems.append(f"{where}: unknown competency")
            continue
        if item.rating is not None and item.rating not in points:
            problems.append(f"{where}: rating {item.rating} off the scale")
        if item.rating is not None and not item.evidence:
            problems.append(f"{where}: rating without evidence")
        for evidence in item.evidence:
            if evidence.turn >= len(transcript.turns):
                problems.append(f"{where}: turn {evidence.turn} not found")
            elif transcript.turns[evidence.turn].speaker != item.participant:
                problems.append(
                    f"{where}: turn {evidence.turn} is not theirs"
                )
            if competency.indicator(evidence.indicator_id) is None:
                problems.append(
                    f"{where}: indicator {evidence.indicator_id!r} is not "
                    f"of {competency.id}"
                )
    missing = [
        f"{s}/{c}" for s in sorted(speakers) for c in competencies
        if (s, c) not in seen
    ]
    if missing:
        problems.append(f"not annotated: {', '.join(missing)}")
    return problems


def _by_annotator(
    sets: Sequence[AnnotationSet],
) -> Dict[str, Dict[Unit, Annotation]]:
    """Group the annotations of every session by annotator."""
    grouped: Dict[str, Dict[Unit, Annotation]] = {}
    for item in sets:
        cells = grouped.setdefault(item.annotator, {})
        for unit, annotation in item.by_unit().items():
            if unit in cells:
                raise InvalidInputError(
                    f"{item.annotator} annotated {'/'.join(unit)} twice."
                )
            cells[unit] = annotation
    return grouped


def check_on_scale(ratings: Sequence[Optional[int]], scale: Scale) -> None:
    """Reject ratings that are not points of the scale."""
    points = set(scale.points())
    off = sorted({r for r in ratings if r is not None and r not in points})
    if off:
        raise InvalidInputError(
            f"Rating(s) {', '.join(str(r) for r in off)} are off the "
            f"{scale.minimum}-{scale.maximum} scale."
        )


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Return the mean of the values that exist."""
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def _evidence_f1(
    grouped: Dict[str, Dict[Unit, Annotation]], units: Sequence[Unit]
) -> Optional[float]:
    """Return the mean pairwise F1 of the evidence turns."""
    scores: List[float] = []
    for a, b in combinations(sorted(grouped), 2):
        for unit in units:
            first, second = grouped[a].get(unit), grouped[b].get(unit)
            if first is None or second is None:
                continue
            if not first.turns() and not second.turns():
                continue
            scores.append(set_scores(first.turns(), second.turns())["f1"])
    return sum(scores) / len(scores) if scores else None


def _ratings(
    grouped: Dict[str, Dict[Unit, Annotation]], units: Sequence[Unit]
) -> List[List[Rating]]:
    """Return one row of ratings per annotator, one column per unit."""
    return [
        [
            None if unit not in cells or cells[unit].rating is None
            else float(cells[unit].rating or 0)
            for unit in units
        ]
        for _, cells in sorted(grouped.items())
    ]


def _within_one(rows: Sequence[Sequence[Rating]]) -> Optional[float]:
    """Return the mean pairwise share of ratings at most a point apart."""
    return _mean([
        percent_agreement(a, b, tolerance=1.0)
        for a, b in combinations(rows, 2)
    ])


def agreement(
    sets: Sequence[AnnotationSet], scale: Scale
) -> AgreementReport:
    """Measure how much the annotators agree with each other."""
    grouped = _by_annotator(sets)
    check_on_scale(
        [a.rating for cells in grouped.values() for a in cells.values()],
        scale,
    )
    if len(grouped) < 2:
        raise InvalidInputError(
            "Agreement needs annotations from at least two annotators."
        )
    names = sorted(grouped)
    units = sorted({u for cells in grouped.values() for u in cells})
    points = [float(p) for p in scale.points()]
    rows = _ratings(grouped, units)
    alpha = krippendorff_alpha(rows, points, "ordinal")
    complete = [
        i for i in range(len(units)) if all(r[i] is not None for r in rows)
    ]
    icc = icc_two_way(
        [[float(r[i] or 0) for i in complete] for r in rows]
    ) if complete else None
    rated = [[None if v is None else 1.0 for v in row] for row in rows]
    covered = [
        [
            (1.0 if value is not None else 0.0) if unit in cells else None
            for unit, value in zip(units, row)
        ]
        for (_, cells), row in zip(sorted(grouped.items()), rated)
    ]
    pairwise = {
        f"{a}|{b}": cohen_kappa(rows[i], rows[j], points)
        for (i, a), (j, b) in combinations(enumerate(names), 2)
    }
    per_competency: List[CompetencyAgreement] = []
    for competency_id in sorted({u[2] for u in units}):
        subset = [u for u in units if u[2] == competency_id]
        sub_rows = _ratings(grouped, subset)
        sub_alpha = krippendorff_alpha(sub_rows, points, "ordinal")
        co_rated = sum(
            1 for i in range(len(subset))
            if sum(row[i] is not None for row in sub_rows) >= 2
        )
        per_competency.append(CompetencyAgreement(
            competency_id=competency_id,
            units=len(subset),
            rated_units=co_rated,
            alpha=sub_alpha,
            band=interpret_alpha(sub_alpha)
            if co_rated >= MIN_UNITS_FOR_BAND else "too few units",
            within_one=_within_one(sub_rows),
            evidence_f1=_evidence_f1(grouped, subset),
        ))
    return AgreementReport(
        annotators=names,
        sessions=len({u[0] for u in units}),
        units=len(units),
        alpha=alpha,
        band=interpret_alpha(alpha)
        if len(complete) >= MIN_UNITS_FOR_BAND or sum(
            1 for i in range(len(units))
            if sum(r[i] is not None for r in rows) >= 2
        ) >= MIN_UNITS_FOR_BAND else "too few units",
        icc=icc[0] if icc else None,
        icc_average=icc[1] if icc else None,
        within_one=_within_one(rows),
        abstention_kappa=_mean([
            cohen_kappa(a, b, [0.0, 1.0], "none")
            for a, b in combinations(covered, 2)
        ]),
        evidence_f1=_evidence_f1(grouped, units),
        pairwise_kappa=pairwise,
        per_competency=per_competency,
    )


def median_low(values: Sequence[int]) -> int:
    """Return the median, the lower of the two middle values on a tie."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def consensus(
    sets: Sequence[AnnotationSet], annotator: str = "consensus"
) -> List[AnnotationSet]:
    """Merge several annotators into one reference set per session.

    A unit is rated when more than half of its annotators rated it;
    the rating is their median (the lower middle value on a tie, the
    conservative choice). A turn is evidence when more than half of
    the annotators marked it, under the indicator most of them chose.
    """
    if not sets:
        raise InvalidInputError("No annotation set given.")
    sessions: Dict[str, List[AnnotationSet]] = {}
    for item in sets:
        sessions.setdefault(item.session_id, []).append(item)
    merged: List[AnnotationSet] = []
    for session_id, group in sessions.items():
        names = [g.annotator for g in group]
        if len(set(names)) < len(names):
            raise InvalidInputError(
                f"Session {session_id} has two annotation sets from the "
                "same annotator."
            )
        versions = {(g.catalogue, g.catalogue_version) for g in group}
        if len(versions) > 1:
            raise InvalidInputError(
                f"Session {session_id} was annotated with different "
                "catalogue versions."
            )
        cells: Dict[Tuple[str, str], List[Annotation]] = {}
        for item in group:
            for a in item.annotations:
                cells.setdefault((a.participant, a.competency_id), []).append(
                    a
                )
        annotations: List[Annotation] = []
        for (participant, competency_id), votes in cells.items():
            ratings = [v.rating for v in votes if v.rating is not None]
            rating = (
                median_low(ratings) if len(ratings) * 2 > len(votes)
                else None
            )
            turn_votes = Counter(t for v in votes for t in v.turns())
            evidence: List[AnnotatedEvidence] = []
            for turn in sorted(turn_votes):
                if turn_votes[turn] * 2 <= len(votes):
                    continue
                indicators = Counter(
                    e.indicator_id for v in votes for e in v.evidence
                    if e.turn == turn
                )
                top = sorted(indicators.items(), key=lambda kv: (-kv[1], kv))
                evidence.append(AnnotatedEvidence(
                    turn=turn, indicator_id=top[0][0]
                ))
            annotations.append(Annotation(
                participant=participant, competency_id=competency_id,
                rating=rating, evidence=evidence,
            ))
        merged.append(AnnotationSet(
            session_id=session_id,
            annotator=annotator,
            catalogue=group[0].catalogue,
            catalogue_version=group[0].catalogue_version,
            annotations=annotations,
        ))
    return merged
