"""The plain-language person report: what it says and how it renders."""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from src.application.skills.person_report import (
    PersonReport,
    band,
    build_person_report,
    listing,
    plain,
    session_minutes,
)
from src.application.skills.person_report_html import render_person_html
from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Estimate,
    Evidence,
    Observation,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.interaction import describe
from src.domain.skills.rubric import (
    Competency,
    Indicator,
    Practice,
    RubricCatalogue,
)
from src.domain.skills.transcript import Transcript
from tests.fakes import MEETING_EN, make_catalogue, make_transcript

TIMED = make_transcript([
    (speaker, text, 61.0 + 10 * i, 70.0 + 10 * i)
    for i, (speaker, text) in enumerate(MEETING_EN)
])

Cell = Tuple[str, Optional[int], Sequence[Tuple[int, str, str]]]
"""(competency id, rating, [(turn, quote, indicator id)])."""


def catalogue(language: str = "es", count: int = 2) -> RubricCatalogue:
    """Return the test catalogue with practice texts, or ``count`` ones.

    Above two competencies the extra ones are ``c3``, ``c4``... with a
    positive ``cN.P`` and a negative ``cN.N`` indicator.
    """
    base = make_catalogue(language)
    practice = Practice(helps=["helps one", "helps two", "helps three"],
                        hurts=["hurts one", "hurts two"])
    competencies = [
        c.model_copy(update={"practice": practice})
        for c in base.competencies
    ]
    for number in range(3, count + 1):
        competencies.append(Competency(
            id=f"c{number}", name=f"Competency {number}",
            definition=f"Definition {number}.",
            indicators=[
                Indicator(id=f"c{number}.P", text=f"Good {number}"),
                Indicator(id=f"c{number}.N", text=f"Bad {number}",
                          polarity="negative"),
            ],
        ))
    return base.model_copy(update={"competencies": competencies})


def assessment(
    cells: Sequence[Cell], participant: str = "Maya", min_incidents: int = 2
) -> SessionAssessment:
    """Return a session where ``participant`` has the given cells."""
    team = describe(TIMED)
    negative = ("L3", "P2")
    items = [
        CompetencyAssessment(
            competency_id=competency,
            status="rated" if rating is not None else "insufficient_evidence",
            rating=rating,
            evidence=[
                Evidence(
                    turn=turn, quote=quote, indicator_id=indicator,
                    polarity="negative"
                    if indicator in negative or indicator.endswith(".N")
                    else "positive",
                )
                for turn, quote, indicator in marks
            ],
        )
        for competency, rating, marks in cells
    ]
    return SessionAssessment(
        session_id="s1", language="en", catalogue="core",
        catalogue_version="1.0", judge_model="fake-judge",
        embedding_model="", min_incidents=min_incidents, team=team,
        scorecards=[ParticipantScorecard(
            participant=participant, interaction=team.of(participant),
            assessments=items,
        )],
    )


def build(
    cells: Sequence[Cell],
    language: str = "es",
    count: int = 2,
    transcript: Optional[Transcript] = TIMED,
    synthetic: bool = False,
) -> PersonReport:
    """Return the report on Maya for the given cells."""
    return build_person_report(
        assessment(cells), catalogue(language, count), "Maya", transcript,
        synthetic,
    )


EMPTY: List[Cell] = [("listening", None, []), ("planning", None, [])]
ONE: List[Cell] = [
    ("listening", 4, [(0, "What do you mean exactly", "L1"),
                      (3, "So if I understood correctly", "L2")]),
    ("planning", None, []),
]


def many(count: int = 6) -> List[Cell]:
    """Return ``count`` rated competencies with a weak one among them."""
    cells: List[Cell] = [
        ("listening", 5, [(0, "What do you mean", "L1"),
                          (3, "So if I understood", "L2")]),
        ("planning", 2, [(6, "Okay planning", "P2"),
                         (3, "So if I understood", "P1")]),
    ]
    for number in range(3, count + 1):
        cells.append((f"c{number}", 3 + number % 3,
                      [(0, f"quote {number}", f"c{number}.P")]))
    return cells


def test_band_places_a_rating_against_the_centre() -> None:
    """Above the centre is strong, below is weak, none is no data."""
    assert [band(r, 3.0) for r in (5, 4, 3, 2, 1, None)] == [
        "strong", "strong", "middle", "weak", "weak", "none",
    ]
    assert band(3, 2.5) == "strong"


def test_listing_joins_names_in_the_language() -> None:
    """Names read ``a, b y c`` in Spanish and ``a and b`` in English."""
    assert listing([], "es") == ""
    assert listing(["a"], "es") == "a"
    assert listing(["a", "b", "c"], "es") == "a, b y c"
    assert listing(["a", "b"], "en") == "a and b"


def test_session_minutes_from_the_transcript() -> None:
    """Length is last end minus first start; untimed or absent is None."""
    assert session_minutes(TIMED) == pytest.approx((140.0 - 61.0) / 60)
    assert session_minutes(make_transcript()) is None
    assert session_minutes(None) is None
    starts_only = make_transcript().model_copy(update={"turns": [
        t.model_copy(update={"start": 10.0 * i})
        for i, t in enumerate(make_transcript().turns)
    ]})
    assert session_minutes(starts_only) == pytest.approx(70.0 / 60)


def test_no_evidence_is_never_shown_as_low() -> None:
    """Without evidence every row is 'no data' and nothing is weak."""
    report = build(EMPTY)
    assert [row.band for row in report.rows] == ["none", "none"]
    assert not report.strengths and not report.weaknesses
    assert not report.signals and not report.radar
    text = " ".join(plain(s) for s in report.summary)
    assert "No hubo pruebas suficientes para valorar ninguna" in text
    assert "no significa un nivel bajo" in text
    page = render_person_html(report)
    assert page.count('<span class="tag none">Sin datos suficientes') == 2
    assert 'class="tag weak"' not in page
    assert "No hay gráfico de araña" in page
    assert "Perfil de las competencias valoradas" not in page


def test_one_rated_competency() -> None:
    """A strength carries its quotes, clock and practice examples."""
    report = build(ONE)
    assert [p.competency_id for p in report.strengths] == ["listening"]
    strength = report.strengths[0]
    assert [q.text for q in strength.quotes] == [
        "What do you mean exactly", "So if I understood correctly",
    ]
    assert strength.practice == ["helps one", "helps two"]
    assert strength.quotes[0].when("es") == "Maya, minuto 01:01"
    assert strength.quotes[1].indicator == "Resume lo que dijeron los demás"
    text = [plain(s) for s in report.summary]
    assert text[1].startswith("Hay nivel en 1 de 2 competencias; en las "
                              "otras 1 no hubo")
    assert text[2] == "Sus puntos fuertes en esta conversación: " \
        "Escucha activa."
    page = render_person_html(report)
    assert "«What do you mean exactly»" in page
    assert "Maya, minuto 01:01" in page
    assert "<li>helps one</li><li>helps two</li>" in page
    assert "helps three" not in page
    assert "Nivel 4 de 5" in page


def test_many_competencies_draw_the_radar() -> None:
    """Five or more rated competencies add the radar; weak ones listed."""
    report = build(many(6), count=6)
    assert report.radar and len(report.rated) == 6
    assert [p.competency_id for p in report.weaknesses] == ["planning"]
    weak = report.weaknesses[0]
    assert [q.text for q in weak.quotes] == ["Okay planning"]
    assert [q.text for q in weak.other] == ["So if I understood"]
    assert weak.practice == ["hurts one", "hurts two"]
    assert report.rows[0].rating == 5
    assert report.rows[-1].band == "weak"
    page = render_person_html(report)
    assert "Perfil de las competencias valoradas" in page
    assert "También se vio a favor" in page
    assert "Puntos a mejorar: Planificación." in page


def test_four_rated_competencies_draw_no_radar() -> None:
    """Below five rated competencies the radar is replaced by a note."""
    report = build(many(4), count=4)
    assert not report.radar
    assert "aquí hay 4" in render_person_html(report)


def test_negative_quotes_outside_weak_points_are_signals() -> None:
    """A negative quote in a strength is a signal; an unrated one is not.

    Without a firm level the quote goes to the reviews (#103).
    """
    report = build([
        ("listening", 4, [(0, "What do you mean", "L1"),
                          (1, "That is nonsense", "L3")]),
        ("planning", None, [(4, "Planning is a waste", "P2")]),
    ])
    assert [q.text for q in report.strengths[0].quotes] == [
        "What do you mean"
    ]
    assert [p.competency_id for p in report.signals] == ["listening"]
    assert report.signals[0].practice == ["hurts one", "hurts two"]
    text = plain(report.summary[3])
    assert text.endswith("pero sí una conducta aislada que conviene "
                         "vigilar.")
    assert [r.competency_id for r in report.reviews] == ["planning"]
    page = render_person_html(report)
    assert "Conductas aisladas a vigilar" in page


def test_an_unrated_quote_is_an_observation_not_a_signal() -> None:
    """With no rating, a quote is listed for review, not as a signal."""
    report = build([("listening", None, [(1, "That is nonsense", "L3")]),
                    ("planning", None, [])])
    text = [plain(s) for s in report.summary]
    assert not report.signals
    assert not any("conducta aislada" in line for line in text)
    assert text[-1].startswith("Es un borrador para un evaluador humano")
    review, = report.reviews
    assert (review.estimate, review.reasons) == (None, [])
    assert [(q.text, q.source, q.positive) for q in review.observations] == [
        ("That is nonsense", "below_minimum", False)
    ]
    assert report.tiers == [
        "Valoraciones firmes: 0 · estimaciones provisionales: 0 · "
        "observaciones: 1."
    ]
    page = render_person_html(report)
    assert "Sin estimación: solo observaciones" in page
    assert "cita comprobada; no llega al mínimo" in page


@pytest.mark.parametrize("cells, count", [
    (EMPTY, 2), (ONE, 2), (many(6), 6), (many(4), 4),
    ([("listening", 3, [(0, "q", "L1")]), ("planning", 4, [(5, "p", "P1")])],
     2),
])
def test_summary_has_three_to_five_sentences(
    cells: List[Cell], count: int
) -> None:
    """The opening summary stays short whatever the evidence."""
    for language in ("es", "en"):
        summary = build(cells, language, count).summary
        assert 3 <= len(summary) <= 5, summary


def test_all_rated_and_middle_levels() -> None:
    """Every competency rated; a middle level is neither list."""
    report = build([("listening", 3, [(0, "q", "L1")]),
                    ("planning", 4, [(5, "p", "P1")])])
    assert plain(report.summary[1]) == "Hay nivel en las 2 competencias."
    assert [p.competency_id for p in report.strengths] == ["planning"]
    assert not report.weaknesses
    assert "No se vio ningún punto a mejorar" in report.summary[3]
    page = render_person_html(report)
    assert '<span class="tag middle">Intermedio</span>' in page
    assert "<rect" in page


def test_no_external_resource_in_the_page() -> None:
    """The page loads nothing: no URL, no external src or href."""
    page = render_person_html(build(many(6), count=6, synthetic=True))
    assert "http" not in page
    assert "//" not in page.replace("<!doctype", "")
    hrefs = re.findall(r'(?:href|src)="([^"]*)"', page)
    assert hrefs and all(h.startswith("#nota-") for h in hrefs)
    assert page.startswith("<!doctype html>\n<html lang=\"es\">")


def test_quotes_and_names_are_escaped() -> None:
    """Markup in quotes cannot reach the page unescaped."""
    hostile = '<script>alert("x")</script> & \'y\''
    report = build([("listening", 4, [(0, hostile, "L1"),
                                      (3, "<b>bold</b>", "L2")]),
                    ("planning", None, [])])
    page = render_person_html(report)
    assert "<script>" not in page and "<b>bold" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; " \
        "&#x27;y&#x27;" in page


def test_synthetic_input_is_marked() -> None:
    """Synthetic data opens with a banner; real data has none."""
    assert "Datos de ejemplo inventados" in render_person_html(
        build(ONE, synthetic=True)
    )
    assert 'class="banner"' not in render_person_html(build(ONE))


def test_footnotes_explain_the_terms() -> None:
    """Every footnote call points at a note that exists."""
    page = render_person_html(build(ONE))
    calls = set(re.findall(r'href="#nota-(\d)"', page))
    notes = set(re.findall(r'id="nota-(\d)"', page))
    assert calls == notes == {"1", "2", "3", "4", "5"}
    assert "Nivel: 2 = below; 4 = above." not in page
    assert "Nivel: 1 es el más bajo y 5 el más alto." in page


def test_scale_labels_explain_the_levels() -> None:
    """With labels at both ends and the centre, the note lists them."""
    rubric = catalogue()
    rubric = rubric.model_copy(update={"scale": rubric.scale.model_copy(
        update={"labels": {1: "poco", 3: "algo", 5: "mucho"}}
    )})
    report = build_person_report(assessment(ONE), rubric, "Maya", TIMED)
    assert report.scale_labels == {1: "poco", 3: "algo", 5: "mucho"}
    assert "Nivel: 1 = poco; 3 = algo; 5 = mucho." in render_person_html(
        report
    )


def test_english_page() -> None:
    """The English wording is complete and the page says lang=en."""
    report = build(ONE, "en", synthetic=True)
    page = render_person_html(report)
    assert page.startswith('<!doctype html>\n<html lang="en">')
    assert "Report on Maya" in page and "Made-up sample data" in page
    assert "Maya, minute 01:01" in page
    assert "Strengths in this conversation: Active listening." in page


def test_untimed_transcript_or_none_gives_turn_numbers() -> None:
    """Without times, quotes cite the turn and length is unknown."""
    for transcript in (None, make_transcript()):
        report = build(ONE, transcript=transcript)
        assert report.minutes is None
        page = render_person_html(report)
        assert "Maya, intervención 0" in page
        assert "sin hora" in page
        assert "unos" not in plain(report.summary[0])


def test_short_session_says_less_than_a_minute() -> None:
    """Under a minute is written as such, never as zero minutes."""
    short = make_transcript([
        (speaker, text, float(i), float(i) + 1)
        for i, (speaker, text) in enumerate(MEETING_EN)
    ])
    report = build(ONE, transcript=short)
    assert "de menos de un minuto" in report.summary[0]
    assert "&lt;1" in render_person_html(report)


def test_quote_past_an_hour_keeps_hours() -> None:
    """A quote after the first hour shows hh:mm:ss."""
    late = make_transcript([
        (speaker, text, 3725.0 + i, 3726.0 + i)
        for i, (speaker, text) in enumerate(MEETING_EN)
    ])
    report = build(ONE, transcript=late)
    assert report.strengths[0].quotes[0].when("es") == "Maya, minuto 01:02:05"


def test_unknown_participant_and_language_are_errors() -> None:
    """A participant not assessed, or a language without wording."""
    with pytest.raises(InvalidInputError, match="Unknown participant 'Tom'"):
        build_person_report(assessment(ONE), catalogue(), "Tom")
    french = catalogue().model_copy(update={"language": "fr"})
    with pytest.raises(InvalidInputError, match="No report wording"):
        build_person_report(assessment(ONE), french, "Maya")


def test_unknown_indicator_falls_back_to_its_id() -> None:
    """An indicator missing from the rubric is shown by its id."""
    report = build([("listening", 4, [(0, "q", "L9"), (3, "r", "L1")]),
                    ("planning", None, [])])
    assert report.strengths[0].quotes[0].indicator == "L9"


def test_no_voice_emotion_or_speaking_time_as_leadership() -> None:
    """The limits box says what is never inferred."""
    words: Dict[str, str] = {"es": "No se deduce ninguna emoción",
                             "en": "No emotion or personality trait"}
    for language, fragment in words.items():
        page = render_person_html(build(ONE, language))
        assert fragment in page
        assert "liderazgo" in page or "leadership" in page


DESCRIBED = TIMED.model_copy(update={
    "conversation_type": "negociacion",
    "conversation_goal": "Precio <del> servidor",
})


def test_the_declared_type_and_goal_are_shown_as_context() -> None:
    report = build(ONE, transcript=DESCRIBED)
    assert (report.conversation_type, report.conversation_goal) == (
        "negociacion", "Precio <del> servidor")
    page = render_person_html(report)
    assert "Tipo de conversación: negociación." in page
    assert "Objetivo: Precio &lt;del&gt; servidor." in page
    assert "no cambia ningún nivel" in page
    english = render_person_html(build(ONE, "en", transcript=DESCRIBED))
    assert "Kind of conversation: negotiation." in english


def test_without_them_nothing_is_shown_and_the_levels_do_not_change() -> None:
    plain = build(ONE)
    assert (plain.conversation_type, plain.conversation_goal) == (None, None)
    assert "Tipo de conversación" not in render_person_html(plain)
    assert build(ONE, transcript=None).conversation_type is None
    described = build(ONE, transcript=DESCRIBED)
    assert described.rows == plain.rows
    assert described.summary == plain.summary


def test_a_type_without_a_label_is_shown_as_it_came() -> None:
    odd = TIMED.model_copy(update={"conversation_type": "nuevo"})
    assert "Tipo de conversación: nuevo." in render_person_html(
        build(ONE, transcript=odd)
    )


# ------------------------------------------ provisional estimates (#103)


def provisional_report(language: str = "es") -> PersonReport:
    """Return Maya's report with one firm and one provisional cell."""
    base = assessment(ONE)
    card = base.scorecards[0]
    planning = CompetencyAssessment(
        competency_id="planning", status="provisional",
        evidence=[Evidence(turn=6, quote="I will do it", indicator_id="P1",
                           polarity="positive", start=121.0)],
        estimate=Estimate(
            rating=4, confidence="low",
            reasons=["few_quotes", "single_episode", "short_participation"],
            quotes=1, confirmations=1, needed=2, words=40, judge_rating=5,
        ),
        observations=[Observation(
            turn=6, quote="Planning matters to me", indicator_id="P1",
            polarity="positive", source="confirmation", start=125.5,
            sentence_id="t6.s1",
        )],
    )
    cells = [card.assessments[0], planning]
    changed = base.model_copy(update={"scorecards": [
        card.model_copy(update={"assessments": cells})
    ]})
    return build_person_report(
        changed, catalogue(language), "Maya", TIMED
    )


def test_a_provisional_estimate_is_labelled_and_never_a_strength() -> None:
    """The estimate has its own row, card, reasons and observations."""
    report = provisional_report()
    assert [p.competency_id for p in report.strengths] == ["listening"]
    assert not report.weaknesses and not report.signals
    row = next(r for r in report.rows if r.competency_id == "planning")
    assert (row.rating, row.estimate, row.band, row.confidence) == (
        None, 4, "provisional", "low",
    )
    review, = report.provisional
    assert [q.source for q in review.observations] == [
        "below_minimum", "confirmation",
    ]
    assert review.observations[1].when("es") == "Maya, minuto 02:05"
    assert report.observations == 2
    assert report.tiers == [
        "Valoraciones firmes: 1 · estimaciones provisionales: 1 · "
        "observaciones: 2.",
        "En una competencia hay una estimación provisional: una hipótesis "
        "para que el entrevistador la revise, no una valoración.",
    ]
    assert 3 <= len(report.summary) <= 5
    page = render_person_html(report)
    assert ("Estimación provisional — revisar: nivel 4 de 5, confianza "
            "baja") in page
    assert "~4 <small>/ 5</small>" in page
    assert "1 de 2 citas necesarias" in page
    assert "un solo momento de la conversación" in page
    assert "participación corta: 40 palabras" in page
    assert "confirmada por el juez frase a frase" in page
    assert "«Planning matters to me»" in page
    assert "hipótesis para el entrevistador, no valoraciones" in page
    assert 'fill="none" stroke="var(--fg)"' in page


def test_provisional_wording_in_english() -> None:
    """Every new line has its English wording."""
    page = render_person_html(provisional_report("en"))
    assert ("Provisional estimate — review: level 4 of 5, low "
            "confidence") in page
    assert ("Firm levels: 1 · provisional estimates: 1 · observations: "
            "2.") in page
    assert "1 of 2 quotes needed" in page
    assert "hypotheses for the interviewer, not assessments" in page


def test_why_few_firm_levels_is_explained() -> None:
    """Few firm levels: the page says why; enough of them: nothing."""
    report = build(EMPTY)
    assert report.why == [
        "short_recording", "little_participation", "few_quotes_proposed",
    ]
    page = render_person_html(report)
    assert "Por qué hay pocas valoraciones firmes" in page
    assert "La grabación es corta." in page
    assert "Maya habló poco" in page
    assert "No hay estimaciones provisionales ni observaciones" in page
    assert build(many(6), count=6).why == []
