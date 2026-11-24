"""Always an opinion (#231): verdicts per skill, overall, by situation."""

import json
from typing import Any, List

from src.application.skills.synthesis import ProfileSynthesizer, person_facts
from src.domain.skills.opinion import (
    DEFAULT_FRAME,
    FRAMES,
    OPINION_FORMAT_VERSION,
    SkillOpinion,
    frame,
    overall_caveats,
    overall_score,
    top_picks,
    verdict_of,
)
from src.domain.skills.synthesis import (
    PersonFacts,
    PersonSynthesis,
    check_reply,
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
    assert "fit the role" in interview
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
    assert result.format_version == OPINION_FORMAT_VERSION
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
    assert planning.score == 2 and "verdict" in result.dropped
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
    assert result.top_strengths and result.reason == "no_evidence"
    old: List[Any] = [{"participant": "Tom", "status": "done"}]
    loaded = PersonSynthesis.model_validate(old[0])
    assert loaded.skills == [] and loaded.overall is None
