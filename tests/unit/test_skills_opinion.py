"""Always an opinion (#231): verdicts per skill, overall, by situation."""

import json
from typing import Any, List

from src.application.skills.synthesis import ProfileSynthesizer, person_facts
import pytest

from src.domain.skills.opinion import (
    DEFAULT_FRAME,
    FRAMES,
    OPINION_FORMAT_VERSION,
    SkillOpinion,
    frame,
    overall_caveats,
    overall_percent,
    weighted_mean,
    overall_score,
    scaled,
    top_picks,
    verdict_of,
)
from src.domain.skills.synthesis import (
    PersonFacts,
    PersonSynthesis,
    check_reply,
    refusal,
    skipped,
    user_prompt,
)
from tests.fakes import FakeChatModel, make_catalogue
from tests.unit.test_skills_synthesis import ACTS, EVENTS, session


def facts(name: str = "Maya", kind: str = "") -> PersonFacts:
    """Return the facts of a person of the shared session, by type."""
    assessment = session().model_copy(update={"conversation_type": kind})
    return person_facts(
        assessment, make_catalogue("en"), name, ACTS, EVENTS
    )


def opinion(
    id_: str, score: int, basis: str = "firm"
) -> SkillOpinion:
    """Return a skill opinion with that score and basis."""
    return SkillOpinion(
        competency_id=id_, name=id_, score=score,
        verdict=verdict_of(score), conclusion="x",
        basis=basis,  # type: ignore[arg-type]
    )


def reply(**parts: Any) -> str:
    """Return a JSON reply."""
    return json.dumps(parts)


def test_every_known_type_has_a_frame_in_both_languages() -> None:
    """Unknown or missing types fall back to the default question."""
    for kind, texts in FRAMES.items():
        assert texts["en"] and texts["es"], kind
    assert frame("coaching", "en") != frame("entrevista", "en")
    assert "acquired" in frame("coaching", "en")
    assert "resolved" not in frame("otro", "en")
    assert frame(None, "es") == DEFAULT_FRAME["es"]
    assert frame("venta", "fr") == FRAMES["venta"]["en"]


def test_the_conversation_type_frames_the_prompt() -> None:
    """The prompt names the type and the question it answers."""
    coaching = user_prompt(facts(kind="coaching"), "")
    interview = user_prompt(facts(kind="entrevista"), "")
    assert "Conversation type: coaching" in coaching
    assert "acquired" in coaching and "acquired" not in interview
    assert "match what the role" in interview
    assert "Conversation type: unknown" in user_prompt(facts(), "")


def test_verdicts_follow_the_cut_offs() -> None:
    """Scores read as strong, adequate or weak."""
    assert [verdict_of(s) for s in (5, 3.75, 3, 2.5, 2.4, 1)] == [
        "strong", "strong", "adequate", "adequate", "weak", "weak"
    ]


def test_the_judge_opinion_is_kept_but_a_firm_level_is_not_overridden(
) -> None:
    """The analysis's own level wins; provisional moves one point."""
    text = reply(
        strengths=[], fit=[], questions=[],
        skills=[
            {"id": "listening", "score": 1, "conclusion": "Listens well.",
             "cites": ["Q1", "Q9"]},
            {"id": "planning", "score": 5, "conclusion": "Plans ahead."},
        ],
        overall={"verdict": "Coordinates the conversation well."},
    )
    result = check_reply(text, facts(kind="coaching"))
    by_id = {s.competency_id: s for s in result.skills}
    assert by_id["listening"].score == 4
    assert by_id["listening"].source == "judge"
    assert by_id["listening"].basis == "firm"
    assert [c.text for c in by_id["listening"].cites] == [
        "What do you mean exactly"
    ]
    assert by_id["planning"].score == 4 and by_id["planning"].basis == (
        "provisional"
    )
    assert "provisional" in by_id["planning"].caveats
    assert result.overall is not None
    assert result.overall.line == "Coordinates the conversation well."
    assert result.overall.source == "judge"
    assert result.conversation_type == "coaching"
    assert result.format_version == OPINION_FORMAT_VERSION == 2
    assert result.overall is not None and result.overall.percent == round(
        (weighted_mean(result.skills) - 1) / 4 * 100, 1
    )
    assert result.overall.score == round(weighted_mean(result.skills), 1)
    assert result.top_strengths == ["listening", "planning"]
    assert result.top_improvements == []


def test_a_missing_or_refused_entry_falls_back_to_the_rules() -> None:
    """Never a gap: the skill the reply left out, or broke a rule on."""
    text = reply(
        skills=[{"id": "planning", "score": 2,
                 "conclusion": "Recommend hiring her."}],
        overall={"verdict": "Better than the others."},
    )
    result = check_reply(text, facts())
    assert [s.competency_id for s in result.skills] == [
        "listening", "planning"
    ]
    assert {s.source for s in result.skills} == {"rules"}
    planning = result.skills[1]
    assert planning.score == 3 and planning.verdict == "adequate"
    assert result.dropped == {"verdict": 1, "comparison": 1}
    assert result.overall is not None and result.overall.source == "rules"
    assert result.overall.line.startswith("Overall reading:")


def test_nothing_quoted_still_gives_a_verdict_with_caveats() -> None:
    """Tom has no evidence: a neutral reading, flagged, not a refusal."""
    tom = facts("Tom", "venta")
    result = skipped(tom, "model failed")
    assert result.status == "skipped" and len(result.skills) == 2
    first = result.skills[0]
    assert first.score == 3 and first.verdict == "adequate"
    assert first.basis == "participation" and first.confidence == "low"
    assert "no_quotes" in first.caveats
    assert "short_participation" in first.caveats
    assert result.overall is not None
    assert result.overall.frame == frame("venta", "en")
    assert "no_quotes" in result.overall.caveats
    assert "judge_unavailable" in result.overall.caveats
    assert result.overall.confidence == "low"
    spanish = tom.model_copy(update={"language": "es"})
    assert skipped(spanish, "x").skills[0].conclusion.startswith(
        "Active listening: en esta conversación"
    )


def test_quoted_observations_move_a_skill_one_point_either_way() -> None:
    """Without a level, the sign of the quotes leans the middle score."""
    maya = facts()
    plain = skipped(maya, "x")
    assert plain.skills[0].basis == "firm" and plain.skills[1].score == 3
    negative = maya.model_copy(update={
        "competencies": [
            c.model_copy(update={"status": "insufficient_evidence",
                                 "rating": None})
            for c in maya.competencies
        ],
        "citables": [
            q.model_copy(update={"polarity": "negative"})
            for q in maya.citables
        ],
    })
    leaned = skipped(negative, "x")
    assert leaned.skills[0].score == 2 and leaned.skills[0].verdict == "weak"
    assert leaned.skills[0].basis == "observation"
    assert "observations_only" in leaned.skills[0].caveats
    assert leaned.top_improvements[0] == "listening"


def test_overall_score_and_top_picks() -> None:
    """Firm skills weigh most; strengths at most 3, improvements 2."""
    skills = [
        opinion("a", 5), opinion("b", 4), opinion("c", 4, "provisional"),
        opinion("d", 3), opinion("e", 1), opinion("f", 2, "observation"),
    ]
    assert overall_score(skills) == round(
        (5 + 4 + 4 * 0.6 + 3 + 1 + 2 * 0.4) / (4 + 0.6 + 0.4), 1
    )
    assert overall_score([]) == 0.0
    strengths, improvements = top_picks(skills)
    assert strengths == ["a", "b", "c"]
    assert improvements == ["e", "f"]
    assert not set(strengths) & set(improvements)
    assert top_picks([opinion("z", 5)]) == (["z"], [])
    assert overall_caveats(skills) == ["no_quotes"]
    assert overall_caveats([opinion("q", 3, "participation")]) == [
        "no_firm_levels", "no_quotes"
    ]
    assert overall_caveats([opinion("q", 3, "observation")]) == [
        "no_firm_levels", "observations_only", "no_quotes"
    ]


def test_the_synthesizer_gives_opinions_without_a_call_when_nothing_quoted(
) -> None:
    """Tom costs no call but still has a verdict; old files still load."""
    model = FakeChatModel([reply(skills=[])])
    tom = facts("Tom", "coaching")
    result = ProfileSynthesizer(model).write(tom)
    assert model.calls == []
    assert result.status == "done" and result.overall is not None
    assert result.top_strengths == [] and result.reason == "no_evidence"
    old: List[Any] = [{"participant": "Tom", "status": "done"}]
    loaded = PersonSynthesis.model_validate(old[0])
    assert loaded.skills == [] and loaded.overall is None
    assert loaded.format_version == 1


@pytest.mark.parametrize("text", [
    "No encaja con el puesto.",
    "Encaja perfectamente con el puesto.",
    "I would not recommend her for this position.",
    "Good fit for the job; we should move forward with her.",
    "She is a strong candidate for the role.",
    "Es la candidata ideal.",
    "Recomiendo pasar a la siguiente fase.",
    "Sería buena idea avanzar con ella.",
    "A good fit for the role.",
    "Yo la contrataría.",
    "No la recomendaría para el puesto.",
    "Strong hire.",
    "Worth hiring.",
    "He is not a fit for this role.",
    "She is the right person for the job.",
    "Es un perfil idóneo para el puesto.",
    "Recomendaría su incorporación.",
    "Sería una buena incorporación al equipo.",
    "I'd advance her to the next round.",
    "She should progress to the final interview.",
    "Debería avanzar en el proceso.",
    "Clearly suited to the role.",
    "Overall, a great match for this position.",
    "She is a good fit for the role and the team.",
    "Encaja muy bien en el puesto que se busca.",
])
def test_decisions_and_blanket_fit_are_refused(text: str) -> None:
    """The fit reading may describe skills, never decide."""
    assert refusal(text, []) == "verdict"


@pytest.mark.parametrize("text", [
    "Sus respuestas sobre bases de datos encajan con lo que pide el "
    "puesto; las pruebas automáticas quedan por debajo.",
    "Her answers on databases match what the role asks for; automated "
    "testing falls short.",
    "Explains trade-offs clearly, a strength for the coordination tasks.",
    "Her planning matches what the role asks for; follow-up is thinner.",
    "Su escucha activa coincide con lo que pide el puesto; el "
    "seguimiento queda por debajo.",
    "The proposal is a good fit for the budget.",
    "Her questions were a good fit with the customer's needs.",
    "Recomienda a la clienta el plan anual.",
    "Her listening is strong; a good fit for the role's customer-facing "
    "tasks.",
    "The team moved on with the next stage of the project.",
])
def test_describing_the_match_skill_by_skill_is_allowed(text: str) -> None:
    """Strengths against requirements pass."""
    assert refusal(text, []) == ""


def test_the_overall_line_and_empty_conclusions_are_checked_and_counted(
) -> None:
    """A verdict line is replaced and counted; an empty text is `empty`."""
    text = reply(
        skills=[{"id": "planning", "score": 5, "conclusion": ""},
                {"id": "listening", "score": 4, "conclusion": "Fine."}],
        overall={"verdict": "She is a strong candidate for the role."},
    )
    result = check_reply(text, facts(kind="entrevista"))
    assert result.dropped == {"empty": 1, "verdict": 1}
    assert result.skills[1].source == "rules" and result.skills[1].score == 3
    assert result.overall is not None
    assert result.overall.source == "rules"
    assert "candidate" not in result.overall.line
    assert "demonstrated strengths" in result.overall.frame
    assert "hire" in result.overall.frame


def test_participation_only_skills_are_never_strengths_or_improvements(
) -> None:
    """Lists ignore skills read from nothing; neutral observations too."""
    skills = [
        opinion("a", 5, "participation"), opinion("b", 3, "observation"),
        opinion("c", 1, "participation"), opinion("d", 2, "observation"),
    ]
    assert top_picks(skills) == ([], ["d"])
    assert overall_percent(5, 1, 5) == 100.0
    assert overall_percent(3.5, 0, 4) == 87.5


def test_top_picks_use_the_scaled_score_and_the_scale_minimum() -> None:
    """On a 0 to 10 scale 6 is mid-range (adequate), 9 is strong."""
    skills = [opinion("a", 9), opinion("b", 6), opinion("c", 1)]
    strengths, improvements = top_picks(skills, 0, 10)
    assert strengths == ["a", "b"] and improvements == ["c"]
    assert scaled(5, 0, 10) == 3.0 and scaled(4, 1, 5) == 4
