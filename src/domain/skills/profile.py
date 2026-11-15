"""A person's profile across sessions: per-session evidence, aggregated.

One conversation is one situation, and people behave differently from
one situation to the next: the same competency agrees across two
exercises only about .27 (Putka & Hoffman 2013, ``docs/skills/
RESEARCH.md`` §6c). A profile therefore keeps every session's level
and evidence side by side and gives an aggregate level only when
enough situations show the competency. A session without evidence for
a competency is left out of that competency's aggregate; it is never
counted as a low level.

Sessions are joined by the person id the transcriber enrolled the
speaker under (``ParticipantScorecard.person_id``); a scorecard without
one cannot be attributed to a person across sessions.
"""

from statistics import median, pstdev
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import Evidence, SessionAssessment

SITUATION_RELIABILITY = 0.27
"""Agreement of one competency between two situations.

Putka & Hoffman 2013 (verified, JAP 98, Tables 3–5): the same
dimension in two exercises correlates .27. Used as the reliability of a
single situation in :func:`expected_reliability`.
"""

TARGET_RELIABILITY = 0.70
"""Reliability usually asked of a score used to describe a person."""

CONSISTENT_RANGE = 1
"""Largest gap between a person's levels still called consistent."""

TREND_STEP = 0.25
"""Smallest change per session, in levels, read as a trend."""

Trend = Literal["up", "down", "flat", "unknown"]
Exclusion = Literal["no_person_ids", "absent"]


class SessionRef(BaseModel):
    """One session of the profile, in time order.

    ``number`` is its place in the profile (1 = earliest).
    """

    number: int = Field(ge=1)
    session_id: str
    started_at: Optional[str] = None
    conversation_type: Optional[str] = None
    participant: str
    words: int = 0
    turns: int = 0


class SessionLevel(BaseModel):
    """A competency in one session: its level (or none) and evidence."""

    session_id: str
    number: int = Field(ge=1)
    rating: Optional[int] = None
    evidence: List[Evidence] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)


class CompetencyProfile(BaseModel):
    """One competency across the person's sessions.

    ``sessions`` holds one entry per session of the profile. ``rated``
    counts the sessions with a level; ``conversation_types`` are the
    declared kinds of those sessions. ``aggregate`` (the median of the
    levels) exists only when ``rated`` reaches the profile's minimum;
    otherwise the competency is under construction and ``missing`` says
    how many more situations with evidence it needs. ``spread`` is the
    standard deviation and ``low``/``high`` the range of the levels;
    ``expected_reliability`` is the published-figure projection for
    ``rated`` situations, ``measured_reliability`` the same projection
    from the agreement measured among the people of the store, when
    there are enough of them.
    """

    competency_id: str
    sessions: List[SessionLevel]
    rated: int = 0
    conversation_types: List[str] = Field(default_factory=list)
    aggregate: Optional[float] = None
    missing: int = 0
    low: Optional[int] = None
    high: Optional[int] = None
    spread: Optional[float] = None
    consistent: Optional[bool] = None
    trend: Trend = "unknown"
    slope: Optional[float] = None
    expected_reliability: Optional[float] = None
    measured_reliability: Optional[float] = None
    situations_for_target: int = 0

    @property
    def ratings(self) -> List[int]:
        """Return the levels of the sessions that have one, in order."""
        return [s.rating for s in self.sessions if s.rating is not None]


class Excluded(BaseModel):
    """A session asked for that could not join the profile, and why."""

    session_id: str
    reason: Exclusion


class PersonProfile(BaseModel):
    """Everything known of one person over several sessions."""

    person_id: str
    participant: str
    min_situations: int = Field(ge=1)
    catalogue: str
    catalogue_version: str
    judge_models: List[str]
    min_incidents: int
    sessions: List[SessionRef]
    conversation_types: List[str]
    competencies: List[CompetencyProfile]
    excluded: List[Excluded] = Field(default_factory=list)

    def competency(self, competency_id: str) -> CompetencyProfile:
        """Return the profile of one competency."""
        for item in self.competencies:
            if item.competency_id == competency_id:
                return item
        raise InvalidInputError(f"No competency {competency_id!r}.")


def spearman_brown(reliability: float, situations: int) -> float:
    """Return the reliability of a mean over ``situations`` situations."""
    if situations < 1:
        raise InvalidInputError("situations must be at least 1.")
    r = max(0.0, min(1.0, reliability))
    return situations * r / (1 + (situations - 1) * r)


def situations_needed(reliability: float, target: float) -> int:
    """Return about how many situations bring ``reliability`` to ``target``.

    Spearman-Brown solved for the length, rounded to the nearest whole
    situation (RESEARCH.md §6c: "about 6" for .27 → .70).
    """
    if not 0 < reliability < 1 or not 0 < target < 1:
        raise InvalidInputError("reliabilities must be between 0 and 1.")
    length = (target / (1 - target)) * ((1 - reliability) / reliability)
    return max(1, round(length))


def expected_reliability(situations: int) -> float:
    """Return the projected reliability of ``situations`` situations.

    From :data:`SITUATION_RELIABILITY`; a projection from a published
    figure, not a measurement on these data.
    """
    return spearman_brown(SITUATION_RELIABILITY, situations)


def icc1(groups: Sequence[Sequence[float]]) -> Optional[float]:
    """Return the one-way random-effects ICC(1) of unequal groups.

    Each group is one person's levels over sessions. ``None`` when it
    cannot be estimated (fewer than two groups with two values each,
    or no variance at all). Negative estimates are returned as they
    are; :func:`spearman_brown` treats them as zero.
    """
    kept = [list(g) for g in groups if len(g) >= 2]
    if len(kept) < 2:
        return None
    total = sum(len(g) for g in kept)
    grand = sum(sum(g) for g in kept) / total
    between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in kept)
    within = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in kept)
    ms_between = between / (len(kept) - 1)
    ms_within = within / (total - len(kept))
    k0 = (total - sum(len(g) ** 2 for g in kept) / total) / (len(kept) - 1)
    denominator = ms_between + (k0 - 1) * ms_within
    if denominator <= 0:
        return None
    return (ms_between - ms_within) / denominator


def trend(levels: Sequence[Tuple[int, int]]) -> Tuple[Trend, Optional[float]]:
    """Return the direction of ``(session number, level)`` pairs.

    The least-squares slope in levels per session, with at least three
    levels; a slope under :data:`TREND_STEP` either way is flat.
    """
    if len(levels) < 3:
        return "unknown", None
    xs = [float(x) for x, _ in levels]
    ys = [float(y) for _, y in levels]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    spread_x = sum((x - mean_x) ** 2 for x in xs)
    slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / spread_x
    slope = round(slope, 3)
    if slope >= TREND_STEP:
        return "up", slope
    if slope <= -TREND_STEP:
        return "down", slope
    return "flat", slope


def _order(item: SessionAssessment) -> Tuple[int, str, str]:
    """Sort key: dated sessions first, by date, then by id."""
    return (0 if item.started_at else 1, item.started_at or "",
            item.session_id)


def measured_agreement(
    assessments: Sequence[SessionAssessment], competency_id: str,
) -> Optional[float]:
    """Return ICC(1) of one competency among the enrolled people.

    Every enrolled person with a level in at least two sessions is one
    group; at least three such people are needed, else ``None``.
    """
    groups: Dict[str, List[float]] = {}
    for item in assessments:
        for card in item.scorecards:
            if card.person_id is None:
                continue
            cell = card.assessment(competency_id)
            if cell is not None and cell.rating is not None:
                groups.setdefault(card.person_id, []).append(
                    float(cell.rating)
                )
    usable = [g for g in groups.values() if len(g) >= 2]
    if len(usable) < 3:
        return None
    return icc1(usable)


def _competency(
    competency_id: str,
    entries: List[SessionLevel],
    refs: Dict[str, SessionRef],
    min_situations: int,
    agreement: Optional[float],
) -> CompetencyProfile:
    """Aggregate one competency over the sessions that show it."""
    rated = [e for e in entries if e.rating is not None]
    levels = [e.rating for e in rated if e.rating is not None]
    kinds = sorted({
        kind for e in rated
        if (kind := refs[e.session_id].conversation_type) is not None
    })
    profile = CompetencyProfile(
        competency_id=competency_id, sessions=entries, rated=len(rated),
        conversation_types=kinds,
        missing=max(0, min_situations - len(rated)),
        situations_for_target=situations_needed(
            SITUATION_RELIABILITY, TARGET_RELIABILITY
        ),
    )
    if not levels:
        return profile
    direction, slope = trend([(e.number, e.rating or 0) for e in rated])
    update: Dict[str, object] = {
        "low": min(levels), "high": max(levels),
        "consistent": (
            max(levels) - min(levels) <= CONSISTENT_RANGE
            if len(levels) >= 2 else None
        ),
        "spread": round(pstdev(levels), 3) if len(levels) >= 2 else None,
        "trend": direction, "slope": slope,
    }
    if len(levels) >= min_situations:
        update["aggregate"] = float(median(levels))
    if len(levels) >= 3:
        update["expected_reliability"] = round(
            expected_reliability(len(levels)), 3
        )
        if agreement is not None:
            update["measured_reliability"] = round(
                spearman_brown(agreement, len(levels)), 3
            )
    return profile.model_copy(update=update)


def build_profile(
    person_id: str,
    assessments: Sequence[SessionAssessment],
    min_situations: int,
    population: Optional[Sequence[SessionAssessment]] = None,
) -> PersonProfile:
    """Return the profile of ``person_id`` over ``assessments``.

    Sessions where the person does not appear (or that carry no person
    id at all) are listed in ``excluded`` instead. ``population`` is
    every assessment of the store, used to measure how much one
    person's levels agree across sessions; it defaults to
    ``assessments``. The competencies are those of the first session,
    in its order; every session must use the same catalogue.
    """
    if min_situations < 1:
        raise InvalidInputError("min_situations must be at least 1.")
    found: List[SessionAssessment] = []
    excluded: List[Excluded] = []
    for item in assessments:
        if item.scorecard_of_person(person_id) is not None:
            found.append(item)
        elif all(card.person_id is None for card in item.scorecards):
            excluded.append(Excluded(
                session_id=item.session_id, reason="no_person_ids"
            ))
        else:
            excluded.append(Excluded(
                session_id=item.session_id, reason="absent"
            ))
    if not found:
        raise InvalidInputError(
            f"No session has a scorecard for person {person_id!r}. "
            "Only sessions transcribed with enrolled speakers carry a "
            "person id."
        )
    seen: Dict[str, int] = {}
    for item in found:
        seen[item.session_id] = seen.get(item.session_id, 0) + 1
    twice = sorted(k for k, v in seen.items() if v > 1)
    if twice:
        raise InvalidInputError(
            f"Session(s) given twice: {', '.join(twice)}."
        )
    catalogues = {(a.catalogue, a.catalogue_version) for a in found}
    if len(catalogues) > 1:
        raise InvalidInputError(
            "The sessions were assessed with different rubrics: "
            + ", ".join(f"{c} {v}" for c, v in sorted(catalogues)) + "."
        )
    found.sort(key=_order)
    refs: Dict[str, SessionRef] = {}
    entries: Dict[str, List[SessionLevel]] = {}
    order: List[str] = []
    for number, item in enumerate(found, start=1):
        card = item.scorecard_of_person(person_id)
        assert card is not None
        refs[item.session_id] = SessionRef(
            number=number, session_id=item.session_id,
            started_at=item.started_at,
            conversation_type=item.conversation_type,
            participant=card.participant, words=card.interaction.words,
            turns=card.interaction.turns,
        )
        for cell in card.assessments:
            if cell.competency_id not in entries:
                entries[cell.competency_id] = []
                order.append(cell.competency_id)
            entries[cell.competency_id].append(SessionLevel(
                session_id=item.session_id, number=number,
                rating=cell.rating, evidence=list(cell.evidence),
                flags=list(cell.flags),
            ))
    pool = list(population) if population is not None else list(found)
    for competency_id in order:
        _fill_absent(entries[competency_id], refs)
    competencies = [
        _competency(
            competency_id, entries[competency_id], refs, min_situations,
            measured_agreement(pool, competency_id),
        )
        for competency_id in order
    ]
    last = refs[found[-1].session_id]
    return PersonProfile(
        person_id=person_id, participant=last.participant,
        min_situations=min_situations, catalogue=found[0].catalogue,
        catalogue_version=found[0].catalogue_version,
        judge_models=sorted({a.judge_model for a in found}),
        min_incidents=max(a.min_incidents for a in found),
        sessions=list(refs.values()),
        conversation_types=sorted({
            r.conversation_type for r in refs.values()
            if r.conversation_type is not None
        }),
        competencies=competencies, excluded=excluded,
    )


def _fill_absent(
    entries: List[SessionLevel], refs: Dict[str, SessionRef]
) -> None:
    """Add an empty entry for each session that did not assess it.

    A competency left out of a session (``--competencies``) shows as
    "no evidence" there, so every strip has one cell per session.
    """
    present = {e.session_id for e in entries}
    for ref in refs.values():
        if ref.session_id not in present:
            entries.append(SessionLevel(
                session_id=ref.session_id, number=ref.number,
                flags=["not_assessed"],
            ))
    entries.sort(key=lambda e: e.number)
