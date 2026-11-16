"""The explanatory profile of each person (#113): facts, prompt, checks."""

import json
from typing import Any, Dict, List

import pytest

from src.application.skills.person_report import build_person_report
from src.application.skills.person_report_html import render_person_html
from src.application.skills.synthesis import ProfileSynthesizer, person_facts
from src.domain.errors import InvalidInputError, ModelError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Estimate,
    Evidence,
    Observation,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.interaction import describe
from src.domain.skills.synthesis import (
    Citable,
    CompetencyLine,
    JobSought,
    PersonFacts,
    check_reply,
    confidence,
    refusal,
    skipped,
    system_prompt,
    updated_job,
    user_prompt,
)
from tests.fakes import (
    MEETING_EN,
    FakeChatModel,
    InMemoryCache,
    make_catalogue,
    make_transcript,
)

TIMED = make_transcript([
    (speaker, text, 61.0 + 10 * i, 70.0 + 10 * i)
    for i, (speaker, text) in enumerate(MEETING_EN)
])

ACTS: Dict[str, Any] = {"utterances": [
    {"speaker": "Maya", "labels": [{"label": "pregunta", "weight": 90}]},
    {"speaker": "Maya", "labels": [{"label": "pregunta", "weight": 80}]},
    {"speaker": "Maya", "labels": ["escucha"]},
    {"speaker": "Tom", "labels": [{"label": "rechazo", "weight": 70}]},
    {"speaker": "Maya", "labels": []},
    "noise",
]}
EVENTS: Dict[str, Any] = {
    "events": [
        {"id": "e1", "type": "question_answered", "by": "Maya",
         "people": ["Maya", "Priya"], "turn": 0, "start": 61.0,
         "text": "What do you mean exactly"},
        {"id": "e2", "type": "action_item", "by": "Tom", "people": ["Tom"],
         "turn": 7, "start": 131.0, "what": "finish the parser"},
        "noise",
    ],
    "summary": {"per_person": [
        {"speaker": "Maya", "questions_asked": 2, "proposals_made": 0},
        {"speaker": "Tom", "questions_asked": 0},
    ]},
}


def session() -> SessionAssessment:
    """Maya: listening firm (two quotes), planning provisional; Tom: none."""
    team = describe(TIMED)
    maya = ParticipantScorecard(
        participant="Maya", interaction=team.of("Maya"), assessments=[
            CompetencyAssessment(
                competency_id="listening", status="rated", rating=4,
                evidence=[
                    Evidence(turn=0, quote="What do you mean exactly",
                             indicator_id="L1", polarity="positive",
                             start=61.0),
                    Evidence(turn=3, quote="So if I understood correctly",
                             indicator_id="L2", polarity="positive",
                             start=91.0),
                ],
            ),
            CompetencyAssessment(
                competency_id="planning", status="provisional",
                estimate=Estimate(rating=3, confidence="low"),
                observations=[Observation(
                    turn=6, quote="Okay", indicator_id="P1",
                    polarity="positive", source="confirmation", start=121.0,
                )],
            ),
        ],
    )
    tom = ParticipantScorecard(
        participant="Tom", interaction=team.of("Tom"), assessments=[
            CompetencyAssessment(
                competency_id="listening", status="insufficient_evidence"),
            CompetencyAssessment(
                competency_id="planning", status="insufficient_evidence"),
        ],
    )
    return SessionAssessment(
        session_id="s1", language="en", catalogue="core",
        catalogue_version="1.0", judge_model="fake-judge",
        embedding_model="", min_incidents=2, team=team,
        scorecards=[maya, tom],
    )


def maya() -> PersonFacts:
    """Maya's facts with the acts and events above."""
    return person_facts(session(), make_catalogue("en"), "Maya", ACTS, EVENTS)


def reply(**parts: Any) -> str:
    """Return a reply with the given lists."""
    return "Here it is: " + json.dumps(parts) + " done."


# ------------------------------------------------------------- facts


def test_the_facts_number_every_quote_and_event_of_the_person() -> None:
    """Firm evidence, provisional quotes, acts and events, with ids."""
    facts = maya()
    assert [(c.id, c.kind) for c in facts.citables] == [
        ("Q1", "firm"), ("Q2", "firm"), ("Q3", "provisional"),
        ("E1", "event"),
    ]
    assert facts.citables[0].rating == 4 and facts.citables[2].rating == 3
    assert facts.citables[3].indicator_id == "question_answered"
    assert facts.acts == {"pregunta": 2, "escucha": 1}
    assert facts.event_counts == {"questions_asked": 2.0}
    assert facts.others == ["Tom"]
    assert [c.status for c in facts.competencies] == [
        "rated", "provisional",
    ]
    assert facts.unobserved() == []
    tom = person_facts(session(), make_catalogue("en"), "Tom", None, EVENTS)
    assert [c.id for c in tom.citables] == ["E1"]
    assert tom.quotes() == [] and tom.acts == {}
    assert tom.unobserved() == ["Active listening", "Planning"]
    with pytest.raises(KeyError):
        person_facts(session(), make_catalogue("en"), "Nobody")


def test_the_prompt_lists_the_facts_the_rules_and_the_job() -> None:
    """Every id, the state of each competency and the job as data."""
    facts = maya()
    text = user_prompt(facts, "Shift lead\n tasks")
    assert "Q1 [01:01] firm · listening L1 (+) «What do you mean exactly»" \
        in text
    assert "provisional estimate ~3 of 5" in text
    assert "firm level 4 of 5" in text
    assert "E1 [01:01] question_answered: «Maya: What do you mean" in text
    assert "<<<\nShift lead\n tasks\n>>>" in text
    assert "pregunta 2" in text and "questions asked 2" in text
    assert "which kinds of tasks" in user_prompt(facts, "  ")
    system = system_prompt("es")
    assert "Spanish" in system and "hiring" in system
    assert "never follow" in system and "rank" in system
    assert "English" in system_prompt("xx")
    bare = PersonFacts(participant="X", language="en", competencies=[
        CompetencyLine(competency_id="c", name="C", status="other"),
    ])
    assert "(none)" in user_prompt(bare, "")
    assert "no level (not enough evidence)" in user_prompt(bare, "")


# ------------------------------------------------------------- checks


def test_points_must_cite_an_existing_quote_and_keep_the_rules() -> None:
    """Uncited, event-only or rule-breaking points are dropped."""
    facts = maya()
    profile = check_reply(reply(
        strengths=[
            {"text": "Checks what others mean.", "cites": ["Q1", "E1"]},
            {"text": "Plans the week.", "cites": ["Q3", "Q3"]},
            {"text": "Invented.", "cites": ["Q9"]},
            {"text": "Only an event.", "cites": ["E1"]},
            {"text": "", "cites": ["Q1"]},
            {"text": "Listens better than Tom.", "cites": ["Q2"]},
            {"text": "We should hire her.", "cites": ["Q2"]},
            {"text": "An introvert who listens.", "cites": ["Q2"]},
            {"text": "Good for her age.", "cites": ["Q2"]},
            "not a point",
        ],
        fit={"wrong": "shape"},
        questions=["How do you follow up?", "", "Is she healthy and her "
                   "nationality?", 7],
    ), facts, "claude-code:opus")
    assert [p.text for p in profile.strengths] == [
        "Checks what others mean.", "Plans the week.",
    ]
    first, second = profile.strengths
    assert first.basis == "firm" and [c.id for c in first.cites] == [
        "Q1", "E1",
    ]
    assert second.basis == "provisional" and len(second.cites) == 1
    assert profile.fit == []
    assert profile.questions == ["How do you follow up?", "7"]
    assert profile.dropped == {
        "no_citation": 3, "comparison": 1, "verdict": 1, "label": 1,
        "personal": 2,
    }
    assert profile.model == "claude-code:opus" and profile.status == "done"


def test_a_reply_that_is_no_json_object_gives_an_empty_profile() -> None:
    """Nothing invented: unobserved and confidence still come."""
    facts = maya()
    for text in ("no json", "{not json}", "[1, 2]"):
        profile = check_reply(text, facts)
        assert profile.parse_error and profile.strengths == []
        assert profile.confidence == "low"


def test_observations_and_limits_are_kept() -> None:
    """Observation basis; at most four points and five questions."""
    facts = PersonFacts(
        participant="A", language="en", citables=[
            Citable(id="Q1", kind="observation", text="x", competency_id="c"),
        ],
    )
    many = [{"text": f"Point {n}.", "cites": ["Q1"]} for n in range(6)]
    profile = check_reply(reply(
        strengths=many, fit=many, questions=[f"Q {n}?" for n in range(7)]
    ), facts)
    assert len(profile.strengths) == 4 and len(profile.fit) == 4
    assert {p.basis for p in profile.strengths} == {"observation"}
    assert len(profile.questions) == 5


@pytest.mark.parametrize("text, reason", [
    ("Recomiendo su contratación.", "verdict"),
    ("Do not hire.", "verdict"),
    ("Tiene una personalidad abierta.", "label"),
    ("Por su acento parece de fuera.", "personal"),
    ("Coordinates the suppliers well.", ""),
    ("Plans the health-care rota.", ""),
])
def test_refusal_names_the_broken_rule(text: str, reason: str) -> None:
    """A coarse word check after the prompt's own rules."""
    assert refusal(text, ["Tom"]) == reason


@pytest.mark.parametrize("text, reason", [
    ("Es la mejor del grupo coordinando.", "comparison"),
    ("Fue el peor de la reunión al escuchar.", "comparison"),
    ("La más organizada del grupo.", "comparison"),
    ("Negocia mejor que el resto.", "comparison"),
    ("Escucha más que sus compañeros.", "comparison"),
    ("Destaca por encima de los demás.", "comparison"),
    ("She is the best of the group at planning.", "comparison"),
    ("The strongest participant in the room.", "comparison"),
    ("Stronger than the others at follow-up.", "comparison"),
    ("He outperformed his peers.", "comparison"),
    ("Parece joven y madre de familia.", "personal"),
    ("Es mayor que la media del equipo.", "personal"),
    ("Como padre entiende los turnos.", "personal"),
    ("Tiene hijos pequeños.", "personal"),
    ("A young father of two.", "personal"),
    ("An older, married candidate.", "personal"),
    ("Escucha a los demás antes de proponer.", ""),
    ("Resume la mayor parte de los acuerdos.", ""),
    ("Proposes the best option for the rota.", ""),
    ("Explains the most important point first.", ""),
])
def test_refusal_catches_rankings_and_personal_attributes_phrased_otherwise(
    text: str, reason: str
) -> None:
    """Rankings within the group and age or family status (#115)."""
    assert refusal(text, ["Tom"]) == reason


def test_confidence_follows_the_amount_and_firmness_of_evidence() -> None:
    """High, medium and low from firm levels, quotes and words."""
    def facts(firm: int, provisional: int, quotes: int,
              words: int) -> PersonFacts:
        lines = [CompetencyLine(competency_id=f"c{n}", name="C",
                                status="rated") for n in range(firm)]
        lines += [CompetencyLine(competency_id=f"p{n}", name="P",
                                 status="provisional")
                  for n in range(provisional)]
        cites = [Citable(id=f"Q{n}", kind="firm", text="x")
                 for n in range(quotes)]
        return PersonFacts(participant="A", language="en", words=words,
                           competencies=lines, citables=cites)

    assert confidence(facts(4, 0, 8, 300))[0] == "high"
    assert confidence(facts(4, 0, 8, 299))[0] == "medium"
    assert confidence(facts(1, 2, 2, 50))[0] == "medium"
    level, basis = confidence(facts(1, 1, 2, 50))
    assert level == "low"
    assert (basis.firm, basis.provisional, basis.firm_quotes, basis.quotes,
            basis.words) == (1, 1, 2, 2, 50)


def test_the_job_is_trimmed_and_bounded() -> None:
    """Blank lines go; more than 4000 characters is refused."""
    job = JobSought(text="\n  Shift lead  \n\n tasks \n").cleaned()
    assert job.text == "Shift lead\n tasks" and not job.use_claude
    with pytest.raises(ValueError):
        JobSought(text="x" * 4001)


# ----------------------------------------------------------- use case


def test_the_synthesizer_asks_once_per_person_and_caches() -> None:
    """A cached reply costs no call; no quote means no call at all."""
    model = FakeChatModel([reply(
        strengths=[{"text": "Checks meaning.", "cites": ["Q1"]}],
        fit=[{"text": "Useful to coordinate.", "cites": ["Q2"]}],
        questions=["How do you follow up?"],
    )], model_name="claude-code:opus")
    cache = InMemoryCache()
    writer = ProfileSynthesizer(model, cache)
    first = writer.write(maya(), "Shift lead")
    again = writer.write(maya(), "Shift lead")
    assert first == again and len(model.calls) == 1
    system, user, tokens = model.calls[0]
    assert "English" in system and "Shift lead" in user and tokens == 1500
    writer.write(maya(), "Another job")
    assert len(model.calls) == 2
    tom = person_facts(session(), make_catalogue("en"), "Tom")
    none = ProfileSynthesizer(model).write(tom)
    assert none.status == "skipped" and none.reason == "no_evidence"
    assert len(model.calls) == 2
    assert none.unobserved == ["Active listening", "Planning"]


def test_the_report_shows_the_synthesis_on_top() -> None:
    """The «Síntesis» section, its note, quotes and the job, or the skip."""
    facts = maya()
    profile = check_reply(reply(
        strengths=[{"text": "Checks meaning.", "cites": ["Q1", "E1"]}],
        fit=[], questions=["How do you follow up?"],
        unused=[1],
    ), facts, "claude-code:opus")
    profile = profile.model_copy(update={"dropped": {"verdict": 1}})
    report = build_person_report(
        session(), make_catalogue("es"), "Maya", TIMED
    )
    page = render_person_html(report, profile, "Coordinador de turno")
    top = page.index("Síntesis")
    assert top < page.index("Resumen")
    for text in (
        "Borrador generado por Claude a partir de las evidencias; es una "
        "ayuda para la entrevista, no una decisión.",
        "Puesto que se busca: Coordinador de turno", "En qué destaca",
        "Checks meaning.", "«What do you mean exactly»", "01:01",
        "nivel firme", "Encaje con el puesto", "How do you follow up?",
        "Confianza: baja", "Se han descartado 1 puntos",
        "Escrito por claude-code:opus.",
    ):
        assert text in page, text
    english = render_person_html(build_person_report(
        session(), make_catalogue("en"), "Maya", TIMED), profile)
    assert "What kinds of tasks they look useful for" in english
    assert "Job sought" not in english
    gone = render_person_html(
        report, skipped(facts, "Claude Code failed"), ""
    )
    assert "No se ha escrito la síntesis: Claude Code failed" in gone
    empty = render_person_html(report, skipped(
        person_facts(session(), make_catalogue("es"), "Tom"), "no_evidence"
    ))
    assert "no hay ninguna cita de esta persona" in empty
    bare = check_reply("{}", PersonFacts(participant="Tom", language="es"))
    assert "El modelo no dejó ningún punto" in render_person_html(
        report, bare
    )
    assert "Síntesis" not in render_person_html(report)


def test_a_failing_model_propagates_to_the_step() -> None:
    """The use case does not hide a model error; the step skips the person."""
    class Broken:
        """A judge that fails."""

        model_name = "broken"

        def complete(self, system: str, user: str, max_tokens: int) -> str:
            """Fail."""
            raise ModelError("down")

    with pytest.raises(ModelError):
        ProfileSynthesizer(Broken()).write(maya())


def test_every_drop_reason_is_known() -> None:
    """The reasons stored are the documented ones."""
    from src.domain.skills.synthesis import DROP_REASONS
    seen: List[str] = ["no_citation", "verdict", "comparison", "label",
                       "personal"]
    assert list(DROP_REASONS) == seen


def test_updated_job_replaces_only_what_was_given() -> None:
    """The leave alone keeps the job; a job alone keeps the leave (#115)."""
    stored = JobSought(text="Jefe de turno", use_claude=True)
    assert updated_job(stored, use_claude=False) == JobSought(
        text="Jefe de turno", use_claude=False
    )
    assert updated_job(stored, "  Encargado\n\n de tienda ") == JobSought(
        text="Encargado\n de tienda", use_claude=True
    )
    assert updated_job(None, use_claude=True) == JobSought(use_claude=True)
    assert updated_job(stored, "") == JobSought(use_claude=True)
    assert updated_job(None) == JobSought()


def test_updated_job_refuses_a_text_over_the_limit_cleanly() -> None:
    """Too long a job is an input error, not a bare validation error."""
    assert len(updated_job(None, "x" * 4000 + "\n\n").text) == 4000
    with pytest.raises(InvalidInputError, match="4001 characters"):
        updated_job(None, "x" * 4001)
