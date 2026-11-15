"""A person's profile across sessions: aggregation, use case and page."""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from src.application.skills.profile import ProfileService
from src.application.skills.profile_report import (
    build_profile_report,
    date_label,
    decimal,
    level_band,
    level_text,
    reliability_word,
)
from src.application.skills.profile_report_html import render_profile_html
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.interaction import describe
from src.domain.skills.profile import (
    build_profile,
    expected_reliability,
    icc1,
    measured_agreement,
    situations_needed,
    spearman_brown,
    trend,
)
from src.domain.skills.transcript import from_contract
from src.domain.skills.transcript_contract import read_contract
from tests.fakes import make_transcript
from tests.unit.test_skills_ingest import example
from tests.unit.test_skills_person_report import TIMED, catalogue

Levels = Dict[str, Optional[int]]
"""Competency id → level (``None``: assessed, not enough evidence)."""


def cell(competency: str, rating: Optional[int]) -> CompetencyAssessment:
    """Return one competency with three quotes behind a level."""
    if rating is None:
        return CompetencyAssessment(
            competency_id=competency, status="insufficient_evidence",
        )
    negative = rating < 3
    indicator = (
        ("L3" if competency == "listening" else "P2") if negative
        else ("L1" if competency == "listening" else "P1")
    )
    return CompetencyAssessment(
        competency_id=competency, status="rated", rating=rating,
        evidence=[
            Evidence(turn=turn, quote=f"quote {turn}",
                     indicator_id=indicator,
                     polarity="negative" if negative else "positive")
            for turn in (0, 3, 6)
        ],
    )


def session(
    session_id: str,
    people: Dict[str, Levels],
    started_at: Optional[str] = None,
    kind: Optional[str] = None,
    catalogue_version: str = "1.0",
) -> SessionAssessment:
    """Return a session where each person id has the given levels.

    A person id starting with ``?`` has no person id (anonymous).
    """
    team = describe(TIMED)
    names = ["Maya", "Tom", "Ana", "Luis"]
    cards = []
    for index, (person, levels) in enumerate(people.items()):
        name = names[index]
        cards.append(ParticipantScorecard(
            participant=name,
            person_id=None if person.startswith("?") else person,
            interaction=team.of("Maya"),
            assessments=[cell(c, r) for c, r in levels.items()],
        ))
    return SessionAssessment(
        session_id=session_id, language="en", catalogue="core",
        catalogue_version=catalogue_version, judge_model="fake-judge",
        embedding_model="", min_incidents=3, team=team, scorecards=cards,
        started_at=started_at, conversation_type=kind,
    )


def maya(*levels: Tuple[Optional[int], Optional[int]]) -> List[
        SessionAssessment]:
    """Return one dated session per (listening, planning) pair of Maya."""
    kinds = ["reunion_decision", "feedback", "negociacion", "coaching"]
    return [
        session(
            f"s{number}",
            {"maya": {"listening": listen, "planning": plan}},
            started_at=f"2026-10-0{number}T10:00:00+02:00",
            kind=kinds[(number - 1) % len(kinds)],
        )
        for number, (listen, plan) in enumerate(levels, start=1)
    ]


# ------------------------------------------------------------ numbers


def test_spearman_brown_and_the_published_figure() -> None:
    """.27 per situation: about .53 over 3 and 6 situations for .70."""
    assert spearman_brown(0.5, 1) == 0.5
    assert spearman_brown(0.5, 3) == pytest.approx(0.75)
    assert spearman_brown(-0.2, 4) == 0.0
    assert expected_reliability(3) == pytest.approx(0.526, abs=1e-3)
    assert situations_needed(0.27, 0.70) == 6
    assert situations_needed(0.9, 0.5) == 1
    with pytest.raises(InvalidInputError):
        spearman_brown(0.5, 0)
    with pytest.raises(InvalidInputError):
        situations_needed(0.0, 0.7)


def test_icc1_of_unequal_groups() -> None:
    """One-way ICC(1) with the k0 correction; undefined cases are None."""
    assert icc1([[1, 2], [4, 5], [3, 3]]) == pytest.approx(0.8621, 1e-3)
    assert icc1([[1, 2, 3], [2, 3]]) is not None
    assert icc1([[1, 2]]) is None
    assert icc1([[1], [2], [3]]) is None
    assert icc1([[3, 3], [3, 3]]) is None


@pytest.mark.parametrize("levels, direction", [
    ([(1, 2), (2, 3), (3, 4)], "up"),
    ([(1, 4), (2, 3), (4, 2)], "down"),
    ([(1, 3), (2, 3), (3, 3)], "flat"),
    ([(1, 3), (2, 4)], "unknown"),
])
def test_trend(levels: List[Tuple[int, int]], direction: str) -> None:
    """A slope of a quarter level per session or more is a trend."""
    assert trend(levels)[0] == direction


# ------------------------------------------------------------ profile


def test_sessions_without_evidence_are_not_zeros() -> None:
    """[4, none, 4] gives 4, never 8/3; the empty session is kept."""
    profile = build_profile("maya", maya((4, None), (None, None), (4, 2)), 2)
    listening = profile.competency("listening")
    assert [s.rating for s in listening.sessions] == [4, None, 4]
    assert (listening.rated, listening.aggregate) == (2, 4.0)
    assert (listening.low, listening.high, listening.spread) == (4, 4, 0.0)
    assert listening.consistent is True
    assert listening.expected_reliability is None
    planning = profile.competency("planning")
    assert (planning.rated, planning.aggregate, planning.missing) == (
        1, None, 1,
    )
    with pytest.raises(InvalidInputError, match="No competency"):
        profile.competency("nope")


def test_below_the_minimum_the_competency_is_under_construction() -> None:
    """Two sessions with evidence and three needed: one is missing."""
    profile = build_profile("maya", maya((2, None), (4, None)), 3)
    listening = profile.competency("listening")
    assert listening.aggregate is None and listening.missing == 1
    assert listening.consistent is False and listening.spread == 1.0
    assert listening.trend == "unknown"


def test_three_sessions_give_the_median_spread_trend_and_reliability(
) -> None:
    """Median of the levels, the projected reliability and the trend."""
    profile = build_profile("maya", maya((2, 3), (3, 3), (4, 3), (4, 3)), 3)
    listening = profile.competency("listening")
    assert listening.aggregate == 3.5
    assert listening.trend == "up" and listening.slope == 0.7
    assert listening.expected_reliability == pytest.approx(0.597, 1e-3)
    assert listening.measured_reliability is None
    assert listening.situations_for_target == 6
    assert listening.conversation_types == [
        "coaching", "feedback", "negociacion", "reunion_decision",
    ]
    assert profile.conversation_types == listening.conversation_types
    assert profile.competency("planning").trend == "flat"


def test_sessions_are_ordered_by_date_then_undated() -> None:
    """Dated sessions first, in time order; undated ones after, by id."""
    late = session("a", {"maya": {"listening": 3}},
                   started_at="2026-11-01T10:00:00+01:00")
    early = session("z", {"maya": {"listening": 4}},
                    started_at="2026-10-01T10:00:00+02:00")
    undated = session("m", {"maya": {"listening": 2}})
    profile = build_profile("maya", [undated, late, early], 3)
    assert [(s.number, s.session_id) for s in profile.sessions] == [
        (1, "z"), (2, "a"), (3, "m"),
    ]
    assert profile.competency("listening").ratings == [4, 3, 2]


def test_competency_missing_from_a_session_shows_as_not_assessed() -> None:
    """Every strip has one cell per session."""
    first = session("s1", {"maya": {"listening": 4, "planning": 3}})
    second = session("s2", {"maya": {"listening": 4}})
    planning = build_profile("maya", [first, second], 3).competency(
        "planning"
    )
    assert [(s.number, s.rating, s.flags) for s in planning.sessions] == [
        (1, 3, []), (2, None, ["not_assessed"]),
    ]


def test_sessions_that_cannot_join_are_listed_with_the_reason() -> None:
    """No person ids at all, or the person is not in that session."""
    anonymous = session("anon", {"?maya": {"listening": 4}})
    other = session("other", {"tom": {"listening": 4}})
    profile = build_profile(
        "maya", [*maya((4, None)), anonymous, other], 3
    )
    assert [(e.session_id, e.reason) for e in profile.excluded] == [
        ("anon", "no_person_ids"), ("other", "absent"),
    ]
    assert len(profile.sessions) == 1


@pytest.mark.parametrize("sessions, message", [
    ([session("s1", {"tom": {"listening": 4}})], "No session"),
    (maya((4, 4)) * 2, "given twice"),
    ([*maya((4, 4)), session("s9", {"maya": {"listening": 4}},
                             catalogue_version="2.0")], "different rubrics"),
])
def test_profile_errors(
    sessions: Sequence[SessionAssessment], message: str
) -> None:
    """Nothing to join, a session twice, or two rubrics are errors."""
    with pytest.raises(InvalidInputError, match=message):
        build_profile("maya", sessions, 3)
    with pytest.raises(InvalidInputError):
        build_profile("maya", maya((4, 4)), 0)


def test_measured_agreement_needs_three_people_with_two_levels() -> None:
    """ICC(1) among people; projected for the person's sessions."""
    people = [
        session(f"p{n}", {"maya": {"listening": a}, "tom": {"listening": b},
                          "ana": {"listening": c}, "?x": {"listening": 3}})
        for n, (a, b, c) in enumerate([(4, 2, 3), (4, 2, 3), (5, 1, 3)])
    ]
    agreement = measured_agreement(people, "listening")
    assert agreement is not None and agreement > 0.8
    assert measured_agreement(people[:1], "listening") is None
    profile = build_profile("maya", people, 3)
    assert profile.competency("listening").measured_reliability == (
        pytest.approx(spearman_brown(agreement, 3), abs=1e-3)
    )
    only_two = [
        session(f"q{n}", {"maya": {"listening": 4}, "tom": {"listening": 2}})
        for n in range(3)
    ]
    assert measured_agreement(only_two, "listening") is None


# ---------------------------------------------------------- use case


class FakeStore:
    """An assessment store over a list kept in memory."""

    def __init__(self, items: Sequence[SessionAssessment]):
        """Keep ``items``."""
        self.items = list(items)

    def all(self) -> List[SessionAssessment]:
        """Return every assessment."""
        return list(self.items)

    def get(self, reference: str) -> SessionAssessment:
        """Return one assessment by session id."""
        for item in self.items:
            if item.session_id == reference:
                return item
        raise NotFoundError(f"No assessment {reference!r}.")


def test_service_uses_every_stored_session_of_the_person() -> None:
    """Without a list, the sessions where the id appears; others ignored."""
    other = session("other", {"tom": {"listening": 4}})
    store = FakeStore([*maya((4, 2), (3, 2), (4, 2)), other])
    profile = ProfileService(store, 3).profile("maya")
    assert [s.session_id for s in profile.sessions] == ["s1", "s2", "s3"]
    assert profile.excluded == []
    assert profile.competency("listening").aggregate == 4.0


def test_service_with_a_list_reports_sessions_left_out() -> None:
    """An explicit list keeps its sessions and says why one is out."""
    other = session("other", {"tom": {"listening": 4}})
    store = FakeStore([*maya((4, 2), (3, 2)), other])
    profile = ProfileService(store, 2).profile("maya", ["s1", "other"])
    assert [s.session_id for s in profile.sessions] == ["s1"]
    assert [e.reason for e in profile.excluded] == ["absent"]
    with pytest.raises(NotFoundError):
        ProfileService(store).profile("maya", ["nope"])
    with pytest.raises(InvalidInputError):
        ProfileService(store, 0)


# ------------------------------------------------- transcript identity


def test_contract_carries_person_ids_and_start_to_the_assessment() -> None:
    """Enrolled speakers keep their id; anonymous voices have none."""
    transcript = from_contract(read_contract(example()))
    assert transcript.people == {"Marta": "marta", "Luis": "luis"}
    assert transcript.started_at == "2026-11-14T10:02:00+01:00"
    plain = make_transcript()
    assert plain.people == {} and plain.started_at is None


# --------------------------------------------------------------- page


def render(
    profile_levels: Sequence[Tuple[Optional[int], Optional[int]]],
    language: str = "es",
    minimum: int = 3,
    count: int = 3,
    synthetic: bool = True,
) -> str:
    """Return the profile page of Maya for the given levels."""
    profile = build_profile("maya", maya(*profile_levels), minimum)
    report = build_profile_report(
        profile, catalogue(language, count), {"s1": TIMED}, synthetic
    )
    return render_profile_html(report)


def text_of(page: str) -> str:
    """Return the visible text of a page, roughly."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


def test_page_shows_overall_levels_and_construction() -> None:
    """Overall level only from three sessions; never "low" for missing."""
    page = render([(4, 2), (4, None), (5, 2)])
    words = text_of(page)
    assert page.startswith('<!doctype html>\n<html lang="es">')
    assert "Perfil de Maya" in words
    assert "Datos de ejemplo inventados" in words
    assert "3 conversaciones de grupo, del 1 oct 2026 al 3 oct 2026" in words
    assert "Hay un nivel de conjunto 1 en 1 de 2 competencias" in words
    assert "Otra está en construcción" in words
    assert "falta una más con pruebas" in words
    assert "Puntos fuertes que se repiten en varias conversaciones: " \
           "Escucha activa" in words
    assert "Puntos a mejorar que se repiten: Planificación" in words
    assert "Sin datos suficientes en ninguna conversación" not in words
    assert "Niveles parecidos en las conversaciones con nivel: de 4 a 5" \
        in words
    assert "Con el tiempo, se mantiene" not in words
    assert "Conversación 1 · Maya, minuto 01:01" in words
    assert "Confianza en este nivel" in words
    assert "http" not in page


def test_page_lists_competencies_never_seen_apart() -> None:
    """No level in any session: one plain list, no strip, not low."""
    page = render([(4, None), (4, None), (4, None)])
    words = text_of(page)
    assert "Sin datos suficientes en ninguna conversación" in words
    assert "No es un nivel bajo: no se vieron. Planificación" in words
    assert "El mismo nivel en las 3 conversaciones con nivel: 4" in words
    assert "Con el tiempo, se mantiene" in words
    assert page.count('<ol class="strip">') == 1


def test_page_without_any_overall_level() -> None:
    """Two sessions: everything is under construction, said plainly."""
    words = text_of(render([(4, 2), (None, 2)], count=2))
    assert "Todavía ninguna competencia se vio con pruebas en 3" in words
    assert "Otras 2 están en construcción" in words
    assert "faltan 2 con pruebas" in words
    assert "Ningún punto fuerte se repite" in words
    assert "hacen falta unas 4 situaciones distintas" in words


def test_page_in_english_with_a_single_session() -> None:
    """One conversation: no repeated points, one date in the subtitle."""
    words = text_of(render([(3, None)], language="en", synthetic=False))
    assert "Profile of Maya" in words
    assert "One group conversation, 1 Oct 2026" in words
    assert "appears in one conversation only" in words
    assert "No strength repeats" in words
    assert "No point to improve repeats" in words
    assert "Made-up sample data" not in words


def test_page_left_out_sessions_and_undated_ones() -> None:
    """Sessions without person ids are listed with a plain reason."""
    anonymous = session("anon", {"?maya": {"listening": 4}})
    undated = session("u1", {"maya": {"listening": 4}})
    profile = build_profile("maya", [undated, anonymous], 3)
    words = text_of(render_profile_html(build_profile_report(
        profile, catalogue("es"),
    )))
    assert "Conversaciones que no se incluyeron" in words
    assert "«anon»: la grabación no identifica a las personas" in words
    assert "sin fecha" in words
    assert "Maya aparece en una sola conversación" in words


def test_page_needs_a_known_language() -> None:
    """A catalogue in a language without wording is an error."""
    profile = build_profile("maya", maya((4, 4)), 3)
    other = catalogue("es").model_copy(update={"language": "fr"})
    with pytest.raises(InvalidInputError, match="No report wording"):
        build_profile_report(profile, other)


@pytest.mark.parametrize("value, language, expected", [
    ("2026-10-06T10:00:00+02:00", "es", "6 oct 2026"),
    ("2026-12-31T23:00:00+01:00", "en", "31 Dec 2026"),
    ("not a date", "es", None),
    (None, "es", None),
])
def test_date_label(
    value: Optional[str], language: str, expected: Optional[str]
) -> None:
    """ISO dates in the reader's words; anything else is no date."""
    assert date_label(value, language) == expected


def test_numbers_and_words_for_the_reader() -> None:
    """Decimal commas in Spanish, bands of halves, reliability words."""
    assert level_text(3.0, "es") == "3"
    assert level_text(3.5, "es") == "3,5"
    assert level_text(3.5, "en") == "3.5"
    assert decimal(0.526, "es") == "0,53"
    assert decimal(0.526, "en") == "0.53"
    assert level_band(None, 3) == "none"
    assert level_band(3.5, 3) == "strong"
    assert level_band(2.5, 3) == "weak"
    assert level_band(3.0, 3) == "middle"
    assert reliability_word(0.4, "es") == "baja"
    assert reliability_word(0.6, "es") == "media"
    assert reliability_word(0.75, "en") == "acceptable"
