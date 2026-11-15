"""Build the profile mode of the person report: several sessions at once.

A view of a :class:`~src.domain.skills.profile.PersonProfile`: for each
competency the level of every session side by side, the overall level
when enough sessions show it (otherwise "under construction"), how
consistent the levels are, and the strengths and points to improve
that repeat across sessions with their verified quotes. The wording
comes from ``profile_report_texts``, ``person_report_texts`` and the
rubric catalogue; nothing is written by a model.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Mapping, Optional, Tuple

from src.application.skills.person_report import (
    RECOMMENDED_SITUATIONS,
    Band,
    Quote,
    band,
    listing,
    note,
)
from src.application.skills.person_report_texts import (
    TEXTS,
    conversation_type_label,
)
from src.application.skills.profile_report_texts import PROFILE_TEXTS
from src.domain.errors import InvalidInputError
from src.domain.skills.profile import (
    TARGET_RELIABILITY,
    CompetencyProfile,
    PersonProfile,
    SessionLevel,
)
from src.domain.skills.rubric import Competency, RubricCatalogue
from src.domain.skills.transcript import Transcript

Status = Literal["aggregated", "building", "none"]

REPEAT_MINIMUM = 2
"""Sessions in the same band before a strength or weakness "repeats"."""

QUOTES_PER_SESSION = 1
"""Verified quotes shown per session in a repeated point."""


@dataclass(frozen=True)
class SessionView:
    """One conversation of the profile, as the reader sees it."""

    number: int
    session_id: str
    date: Optional[str]
    kind: Optional[str]
    turns: int


@dataclass(frozen=True)
class Cell:
    """One competency in one session: its level and band, or none."""

    number: int
    rating: Optional[int]
    band: Band
    incidents: int


@dataclass(frozen=True)
class ProfileRow:
    """One competency across the sessions, ready to render."""

    competency_id: str
    name: str
    status: Status
    aggregate: Optional[float]
    band: Band
    rated: int
    missing: int
    cells: List[Cell]
    low: Optional[int]
    high: Optional[int]
    consistent: Optional[bool]
    trend: str
    expected_reliability: Optional[float]
    measured_reliability: Optional[float]
    situations_for_target: int
    kinds: List[str]


@dataclass(frozen=True)
class SessionQuote:
    """A verified quote and the number of the session it comes from."""

    number: int
    quote: Quote


@dataclass(frozen=True)
class RepeatedPoint:
    """A competency in the same band in two sessions or more."""

    competency_id: str
    name: str
    definition: str
    count: int
    rated: int
    aggregate: Optional[float]
    quotes: List[SessionQuote]
    practice: List[str]


@dataclass(frozen=True)
class ProfileReport:
    """Everything the profile page shows, ready to render."""

    language: str
    participant: str
    person_id: str
    synthetic: bool
    catalogue: str
    catalogue_version: str
    judges: List[str]
    minimum: int
    maximum: int
    midpoint: float
    min_incidents: int
    min_situations: int
    scale_labels: Dict[int, str]
    sessions: List[SessionView]
    kinds: List[str]
    rows: List[ProfileRow]
    strengths: List[RepeatedPoint]
    weaknesses: List[RepeatedPoint]
    excluded: List[str]
    summary: List[str]
    first: Optional[str]
    last: Optional[str]

    @property
    def aggregated(self) -> List[ProfileRow]:
        """Return the rows that have an overall level."""
        return [row for row in self.rows if row.status == "aggregated"]

    @property
    def building(self) -> List[ProfileRow]:
        """Return the rows still under construction."""
        return [row for row in self.rows if row.status == "building"]

    @property
    def target(self) -> int:
        """Return the situations one competency needs for .70."""
        return max((r.situations_for_target for r in self.rows), default=0)


def date_label(value: Optional[str], language: str) -> Optional[str]:
    """Return an ISO 8601 date as ``6 oct 2026`` (``6 Oct 2026``)."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    months = PROFILE_TEXTS[language]["months"].split()
    return PROFILE_TEXTS[language]["date"].format(
        day=moment.day, month=months[moment.month - 1], year=moment.year
    )


def level_text(value: float, language: str) -> str:
    """Return a level as ``3`` or ``3,5`` (``3.5``)."""
    if value == int(value):
        return str(int(value))
    text = f"{value:.1f}"
    return text.replace(".", ",") if language == "es" else text


def decimal(value: float, language: str) -> str:
    """Return a reliability as ``0,53`` (``0.53``)."""
    text = f"{value:.2f}"
    return text.replace(".", ",") if language == "es" else text


def level_band(value: Optional[float], midpoint: float) -> Band:
    """Place an overall level (maybe a half) against the scale centre."""
    if value is None:
        return "none"
    if value > midpoint:
        return "strong"
    return "weak" if value < midpoint else "middle"


def _status(item: CompetencyProfile) -> Status:
    """Return whether a competency is aggregated, building or empty."""
    if item.aggregate is not None:
        return "aggregated"
    return "building" if item.rated else "none"


def _quote(
    level: SessionLevel,
    participant: str,
    competency: Competency,
    transcript: Optional[Transcript],
    positive: bool,
) -> List[SessionQuote]:
    """Return the first quotes of one sign from one session."""
    found: List[SessionQuote] = []
    for item in level.evidence:
        if (item.polarity == "positive") != positive:
            continue
        seconds = None
        if transcript is not None and 0 <= item.turn < len(transcript.turns):
            seconds = transcript.turns[item.turn].start
        indicator = competency.indicator(item.indicator_id)
        found.append(SessionQuote(number=level.number, quote=Quote(
            text=item.quote, speaker=participant, turn=item.turn,
            seconds=seconds,
            indicator=indicator.text if indicator else item.indicator_id,
            positive=positive,
        )))
        if len(found) == QUOTES_PER_SESSION:
            break
    return found


def _repeated(
    item: CompetencyProfile,
    competency: Competency,
    participant: str,
    transcripts: Mapping[str, Transcript],
    midpoint: float,
    kind: Band,
) -> Optional[RepeatedPoint]:
    """Return the competency as a repeated point of ``kind``, if it is."""
    levels = [
        s for s in item.sessions if band(s.rating, midpoint) == kind
    ]
    if len(levels) < REPEAT_MINIMUM:
        return None
    positive = kind == "strong"
    quotes = [
        q for s in levels
        for q in _quote(
            s, participant, competency, transcripts.get(s.session_id),
            positive,
        )
    ]
    practice = competency.practice.helps if positive else (
        competency.practice.hurts
    )
    return RepeatedPoint(
        competency_id=competency.id, name=competency.name,
        definition=competency.definition, count=len(levels),
        rated=item.rated, aggregate=item.aggregate, quotes=quotes,
        practice=practice[:2],
    )


def _row(
    item: CompetencyProfile,
    competency: Competency,
    midpoint: float,
    language: str,
) -> ProfileRow:
    """Return one competency of the strip chart."""
    aggregate = item.aggregate
    return ProfileRow(
        competency_id=competency.id, name=competency.name,
        status=_status(item), aggregate=aggregate,
        band=level_band(aggregate, midpoint),
        rated=item.rated, missing=item.missing,
        cells=[
            Cell(number=s.number, rating=s.rating,
                 band=band(s.rating, midpoint), incidents=len(s.evidence))
            for s in item.sessions
        ],
        low=item.low, high=item.high, consistent=item.consistent,
        trend=item.trend,
        expected_reliability=item.expected_reliability,
        measured_reliability=item.measured_reliability,
        situations_for_target=item.situations_for_target,
        kinds=[conversation_type_label(language, k)
               for k in item.conversation_types],
    )


def _summary(
    language: str,
    profile: PersonProfile,
    rows: List[ProfileRow],
    strengths: List[RepeatedPoint],
    weaknesses: List[RepeatedPoint],
    first: Optional[str],
    last: Optional[str],
) -> List[str]:
    """Return the sentences that open the profile page."""
    words = PROFILE_TEXTS[language]
    count = len(profile.sessions)
    kinds = len(profile.conversation_types)
    dates = (
        words["s_dates"].format(first=first, last=last)
        if first and last and first != last else ""
    )
    if count == 1:
        sentences = [words["s_seen_one"].format(
            participant=profile.participant
        )]
    elif kinds:
        sentences = [words["s_seen"].format(
            participant=profile.participant, count=count, dates=dates,
            kinds=kinds, plural="" if kinds == 1 else "s",
        )]
    else:
        sentences = [words["s_seen_untyped"].format(
            participant=profile.participant, count=count, dates=dates,
        )]
    aggregated = sum(1 for r in rows if r.status == "aggregated")
    building = sum(1 for r in rows if r.status == "building")
    needed = profile.min_situations
    sentences.append(
        words["s_aggregated"].format(
            count=aggregated, total=len(rows), needed=needed, n1=note(1)
        ) if aggregated
        else words["s_aggregated_none"].format(needed=needed, n1=note(1))
    )
    if building:
        sentences.append(
            words["s_building_one"].format(needed=needed) if building == 1
            else words["s_building"].format(count=building, needed=needed)
        )
    sentences.append(
        words["s_repeated_strengths"].format(
            names=listing([p.name for p in strengths], language)
        ) if strengths else words["s_no_repeated_strengths"]
    )
    sentences.append(
        words["s_repeated_weaknesses"].format(
            names=listing([p.name for p in weaknesses], language)
        ) if weaknesses else words["s_no_repeated_weaknesses"]
    )
    sentences.append(
        words["s_draft_short"].format(
            n2=note(2), n3=note(3), recommended=RECOMMENDED_SITUATIONS,
            count=count,
        ) if count < RECOMMENDED_SITUATIONS
        else words["s_draft"].format(n2=note(2))
    )
    return sentences


def build_profile_report(
    profile: PersonProfile,
    catalogue: RubricCatalogue,
    transcripts: Optional[Mapping[str, Transcript]] = None,
    synthetic: bool = False,
) -> ProfileReport:
    """Return the profile page of ``profile`` in the catalogue's language.

    ``transcripts`` (session id → transcript, any subset) only add the
    minute of each quote. A competency without an overall level is
    "under construction" or "not enough evidence", never low.
    """
    language = catalogue.language
    if language not in PROFILE_TEXTS or language not in TEXTS:
        raise InvalidInputError(
            f"No report wording for language {language!r}; choose one of "
            f"{', '.join(sorted(PROFILE_TEXTS))}."
        )
    known = transcripts or {}
    scale = catalogue.scale
    midpoint = scale.midpoint()
    rows: List[ProfileRow] = []
    strengths: List[RepeatedPoint] = []
    weaknesses: List[RepeatedPoint] = []
    for item in profile.competencies:
        competency = catalogue.competency(item.competency_id)
        rows.append(_row(item, competency, midpoint, language))
        buckets: List[Tuple[Band, List[RepeatedPoint]]] = [
            ("strong", strengths), ("weak", weaknesses),
        ]
        for kind, bucket in buckets:
            point = _repeated(
                item, competency, profile.participant, known, midpoint,
                kind,
            )
            if point is not None:
                bucket.append(point)
    order = {"aggregated": 0, "building": 1, "none": 2}
    bands = {"strong": 0, "middle": 1, "weak": 2, "none": 3}
    rows.sort(key=lambda r: (
        order[r.status], bands[r.band], -(r.aggregate or 0), -r.rated,
        r.name,
    ))
    strengths.sort(key=lambda p: (-p.count, p.name))
    weaknesses.sort(key=lambda p: (-p.count, p.name))
    sessions = [
        SessionView(
            number=s.number, session_id=s.session_id,
            date=date_label(s.started_at, language),
            kind=(
                None if s.conversation_type is None
                else conversation_type_label(language, s.conversation_type)
            ),
            turns=s.turns,
        )
        for s in profile.sessions
    ]
    dated = [s.date for s in sessions if s.date]
    first = dated[0] if dated else None
    last = dated[-1] if dated else None
    words = PROFILE_TEXTS[language]
    excluded = [
        words[f"excluded.{e.reason}"].format(
            session=e.session_id, participant=profile.participant
        )
        for e in profile.excluded
    ]
    labels = {
        point: scale.labels[point]
        for point in (scale.minimum, int(midpoint), scale.maximum)
        if point in scale.labels
    }
    return ProfileReport(
        language=language, participant=profile.participant,
        person_id=profile.person_id, synthetic=synthetic,
        catalogue=profile.catalogue,
        catalogue_version=profile.catalogue_version,
        judges=profile.judge_models, minimum=scale.minimum,
        maximum=scale.maximum, midpoint=midpoint,
        min_incidents=profile.min_incidents,
        min_situations=profile.min_situations, scale_labels=labels,
        sessions=sessions,
        kinds=[conversation_type_label(language, k)
               for k in profile.conversation_types],
        rows=rows, strengths=strengths, weaknesses=weaknesses,
        excluded=excluded,
        summary=_summary(
            language, profile, rows, strengths, weaknesses, first, last
        ),
        first=first, last=last,
    )


def reliability_word(value: float, language: str) -> str:
    """Return ``baja``, ``media`` or ``aceptable`` for a reliability."""
    words = PROFILE_TEXTS[language]
    if value >= TARGET_RELIABILITY:
        return words["word_good"]
    return words["word_middle"] if value >= 0.5 else words["word_low"]


def quote_when(item: SessionQuote, language: str) -> str:
    """Return ``Conversación 2 · Ana, minuto 03:10`` for a quote."""
    session = PROFILE_TEXTS[language]["in_session"].format(
        number=item.number
    )
    return f"{session} · {item.quote.when(language)}"
