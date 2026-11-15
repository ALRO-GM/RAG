"""Build the plain-language report on one participant of a session.

The report is a view of a :class:`SessionAssessment`: which
competencies were rated, where each rating sits on the scale, how much
evidence backs it, and the verified quotes. The wording comes from
``person_report_texts`` and the rubric catalogue, so nothing on the
page is written by a model when the report is built.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from src.application.skills.person_report_texts import TEXTS
from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    SessionAssessment,
)
from src.domain.skills.rubric import Competency, RubricCatalogue
from src.domain.skills.speech import clock
from src.domain.skills.transcript import Transcript

Band = Literal["strong", "middle", "weak", "none"]

RECOMMENDED_SITUATIONS = 4
"""Comparable situations for a person-level reliability near .70
(``docs/skills/RESEARCH.md`` §§1, 6: a projection from jackson2016)."""

RADAR_MINIMUM = 5
"""Rated competencies needed before the radar chart is drawn."""

NOTE_MARK = re.compile("\x00(\\d)\x00")
"""How a footnote call is embedded in a sentence (see :func:`note`)."""


def note(number: int) -> str:
    """Return the mark of footnote ``number`` inside a sentence."""
    return f"\x00{number}\x00"


def plain(text: str) -> str:
    """Return ``text`` without its footnote marks."""
    return NOTE_MARK.sub("", text)


@dataclass(frozen=True)
class Quote:
    """A verified sentence of the participant and where it was said."""

    text: str
    speaker: str
    turn: int
    seconds: Optional[float]
    indicator: str
    positive: bool

    def when(self, language: str) -> str:
        """Return who said it and when, e.g. ``Ana, minuto 01:13``."""
        words = TEXTS[language]
        if self.seconds is None:
            return words["said_turn"].format(
                speaker=self.speaker, turn=self.turn
            )
        stamp = clock(self.seconds)
        stamp = stamp[3:] if stamp.startswith("00:") else stamp
        return words["said"].format(speaker=self.speaker, clock=stamp)


@dataclass(frozen=True)
class Row:
    """One competency on the chart: its level and how much backs it."""

    competency_id: str
    name: str
    rating: Optional[int]
    incidents: int
    band: Band


@dataclass(frozen=True)
class Point:
    """A strength, a point to improve or an isolated signal."""

    competency_id: str
    name: str
    definition: str
    rating: Optional[int]
    quotes: List[Quote]
    practice: List[str]
    other: List[Quote] = field(default_factory=list)


@dataclass(frozen=True)
class PersonReport:
    """Everything the person page shows, ready to render."""

    language: str
    participant: str
    session_id: str
    synthetic: bool
    catalogue: str
    catalogue_version: str
    judge_model: str
    minimum: int
    maximum: int
    midpoint: float
    min_incidents: int
    scale_labels: Dict[int, str]
    minutes: Optional[float]
    turns: int
    words: int
    situations: int
    rows: List[Row]
    strengths: List[Point]
    weaknesses: List[Point]
    signals: List[Point]
    summary: List[str]
    conversation_type: Optional[str] = None
    conversation_goal: Optional[str] = None
    conversation_lines: List[str] = field(default_factory=list)

    @property
    def rated(self) -> List[Row]:
        """Return the rows that have a level."""
        return [row for row in self.rows if row.rating is not None]

    @property
    def radar(self) -> bool:
        """Return whether enough competencies are rated for a radar."""
        return len(self.rated) >= RADAR_MINIMUM


def band(rating: Optional[int], midpoint: float) -> Band:
    """Place a rating against the centre of the scale."""
    if rating is None:
        return "none"
    if rating > midpoint:
        return "strong"
    return "weak" if rating < midpoint else "middle"


def listing(names: List[str], language: str) -> str:
    """Join names as ``a, b y c`` (``a, b and c``)."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} {TEXTS[language]['and']} {names[-1]}"


def session_minutes(transcript: Optional[Transcript]) -> Optional[float]:
    """Return the length of the session in minutes, when it is timed."""
    if transcript is None:
        return None
    starts = [t.start for t in transcript.turns if t.start is not None]
    ends = [
        t.end if t.end is not None else t.start
        for t in transcript.turns if t.start is not None
    ]
    if not starts:
        return None
    return (max(e for e in ends if e is not None) - min(starts)) / 60


def _quote(
    item: Evidence,
    participant: str,
    competency: Competency,
    transcript: Optional[Transcript],
) -> Quote:
    """Return an evidence item with its time and indicator text."""
    seconds = None
    if transcript is not None and 0 <= item.turn < len(transcript.turns):
        seconds = transcript.turns[item.turn].start
    indicator = competency.indicator(item.indicator_id)
    return Quote(
        text=item.quote, speaker=participant, turn=item.turn,
        seconds=seconds,
        indicator=indicator.text if indicator else item.indicator_id,
        positive=item.polarity == "positive",
    )


def _summary(
    language: str,
    participant: str,
    minutes: Optional[float],
    rows: List[Row],
    strengths: List[Point],
    weaknesses: List[Point],
    signals: int,
) -> List[str]:
    """Return the three to five sentences that open the report."""
    words = TEXTS[language]
    duration = (
        "" if minutes is None
        else words["s_duration_short"] if minutes < 1
        else words["s_duration"].format(minutes=round(minutes))
    )
    rated = sum(1 for row in rows if row.rating is not None)
    total = len(rows)
    sentences = [
        words["s_session"].format(participant=participant, duration=duration)
    ]
    if rated == 0:
        sentences.append(words["s_rated_none"].format(n1=note(1)))
    elif rated == total:
        sentences.append(
            words["s_rated_all"].format(total=total, n1=note(1))
        )
    else:
        sentences.append(words["s_rated"].format(
            rated=rated, total=total, missing=total - rated, n1=note(1)
        ))
    if rated:
        sentences.append(
            words["s_strengths"].format(
                names=listing([p.name for p in strengths], language)
            ) if strengths else words["s_no_strengths"]
        )
        sentences.append(
            words["s_weaknesses"].format(
                names=listing([p.name for p in weaknesses], language)
            ) if weaknesses
            else words["s_no_weaknesses"] if not signals
            else words["s_signal"] if signals == 1
            else words["s_signals"].format(count=signals)
        )
    elif signals:
        sentences.append(
            words["s_signal"] if signals == 1
            else words["s_signals"].format(count=signals)
        )
    sentences.append(words["s_draft"].format(n2=note(2)))
    return sentences


def build_person_report(
    assessment: SessionAssessment,
    catalogue: RubricCatalogue,
    participant: str,
    transcript: Optional[Transcript] = None,
    synthetic: bool = False,
    conversation_lines: Optional[List[str]] = None,
) -> PersonReport:
    """Return the report on ``participant`` in the catalogue's language.

    Strengths are rated competencies above the centre of the scale,
    points to improve those below it; a competency without a rating is
    "not enough evidence", never low. Negative quotes outside the
    points to improve become isolated signals to watch. The transcript
    is optional: it adds the session length, the time of each quote and
    the conversation's declared type and goal (shown as context, never
    used for a level). ``conversation_lines`` are the participant's
    conversation-event counts (``conversation_events.person_context``),
    also shown as context only.
    """
    language = catalogue.language
    if language not in TEXTS:
        raise InvalidInputError(
            f"No report wording for language {language!r}; choose one of "
            f"{', '.join(sorted(TEXTS))}."
        )
    card = assessment.scorecard(participant)
    if card is None:
        raise InvalidInputError(
            f"Unknown participant {participant!r}. Assessed: "
            f"{', '.join(c.participant for c in assessment.scorecards)}."
        )
    scale = catalogue.scale
    midpoint = scale.midpoint()
    rows: List[Row] = []
    strengths: List[Point] = []
    weaknesses: List[Point] = []
    signals: List[Point] = []
    for item in card.assessments:
        competency = catalogue.competency(item.competency_id)
        rows.append(Row(
            competency_id=competency.id, name=competency.name,
            rating=item.rating, incidents=len(item.evidence),
            band=band(item.rating, midpoint),
        ))
        _place(
            item, competency, participant, transcript, midpoint,
            strengths, weaknesses, signals,
        )
    order = {"strong": 0, "middle": 1, "weak": 2, "none": 3}
    rows.sort(key=lambda r: (order[r.band], -(r.rating or 0), r.name))
    strengths.sort(key=lambda p: -(p.rating or 0))
    weaknesses.sort(key=lambda p: p.rating or 0)
    minutes = session_minutes(transcript)
    signal_count = sum(len(p.quotes) for p in signals)
    labels = {
        point: scale.labels[point]
        for point in (scale.minimum, int(midpoint), scale.maximum)
        if point in scale.labels
    }
    return PersonReport(
        language=language, participant=participant,
        session_id=assessment.session_id, synthetic=synthetic,
        catalogue=assessment.catalogue,
        catalogue_version=assessment.catalogue_version,
        judge_model=assessment.judge_model, minimum=scale.minimum,
        maximum=scale.maximum, midpoint=midpoint,
        min_incidents=assessment.min_incidents, scale_labels=labels,
        minutes=minutes, turns=card.interaction.turns,
        words=card.interaction.words, situations=1, rows=rows,
        strengths=strengths, weaknesses=weaknesses, signals=signals,
        summary=_summary(
            language, participant, minutes, rows, strengths, weaknesses,
            signal_count,
        ),
        conversation_type=(
            transcript.conversation_type if transcript else None
        ),
        conversation_goal=(
            transcript.conversation_goal if transcript else None
        ),
        conversation_lines=list(conversation_lines or []),
    )


def _place(
    item: CompetencyAssessment,
    competency: Competency,
    participant: str,
    transcript: Optional[Transcript],
    midpoint: float,
    strengths: List[Point],
    weaknesses: List[Point],
    signals: List[Point],
) -> None:
    """File one competency as a strength, a point to improve or a signal."""
    quotes = [
        _quote(e, participant, competency, transcript) for e in item.evidence
    ]
    good = [q for q in quotes if q.positive]
    bad = [q for q in quotes if not q.positive]
    kind = band(item.rating, midpoint)
    practice = competency.practice

    def point(shown: List[Quote], lines: List[str],
              other: List[Quote]) -> Point:
        """Return a point of this competency."""
        return Point(
            competency_id=competency.id, name=competency.name,
            definition=competency.definition, rating=item.rating,
            quotes=shown, practice=lines[:2], other=other,
        )

    if kind == "strong":
        strengths.append(point(good, practice.helps, []))
    elif kind == "weak":
        weaknesses.append(point(bad, practice.hurts, good))
        return
    if bad:
        signals.append(point(bad, practice.hurts, []))
