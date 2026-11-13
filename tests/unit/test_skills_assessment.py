"""AssessmentService end to end with a scripted judge."""

import re
from typing import Any, Dict, Optional, Tuple

import pytest

from src.application.skills.assessment import AssessmentService
from src.application.skills.evidence import EvidenceRetriever
from src.domain.errors import InvalidInputError
from src.domain.skills.assessment import (
    CompetencyAssessment,
    SessionAssessment,
)
from src.domain.skills.transcript import Transcript
from tests.fakes import (
    MEETING_EN,
    MEETING_ES,
    FakeChatModel,
    FakeEmbedder,
    FakeLexicalBuilder,
    InMemoryCache,
    RecordingProgress,
    judge_reply,
    make_catalogue,
    make_transcript,
)

CATALOGUE = make_catalogue()
Script = Dict[Tuple[str, str], str]

MAYA_LISTENING = judge_reply(
    [(0, "What do you mean exactly", "L1"),
     (3, "So if I understood correctly, the big file must work", "L2")],
    rating=4, rationale="Asks and summarises.",
)
TOM_LISTENING = judge_reply(
    [(1, "That is nonsense, nobody will test", "L3"),
     (4, "Tom really values everyone's opinion", "L1"),
     (7, "he listened carefully to Priya", "L2")],
    rating=2,
)
SCRIPT: Script = {
    ("Maya", "listening"): MAYA_LISTENING,
    ("Tom", "listening"): TOM_LISTENING,
    ("Tom", "planning"): judge_reply(
        [(4, "Planning is a waste of time", "P2"),
         (7, "I will finish it by Friday", "P1")], rating=3,
    ),
    ("Priya", "planning"): "I am sorry, I cannot produce JSON today.",
}


def scripted(script: Script, default: str = "{}") -> FakeChatModel:
    """Return a judge replying by (participant, competency) of the prompt."""

    def responder(system: str, user: str) -> str:
        """Find who and what the prompt is about."""
        who = re.search(r"Participant to assess: (.+)", user)
        what = re.search(r"Competency: .+ \((\w+)\)", user)
        assert who is not None and what is not None
        return script.get((who.group(1), what.group(1)), default)

    return FakeChatModel(responder=responder)


def service(
    judge: Optional[FakeChatModel] = None,
    embedder: Optional[FakeEmbedder] = None,
    **options: Any,
) -> AssessmentService:
    """Return a service over the fake lexical index and a scripted judge."""
    return AssessmentService(
        EvidenceRetriever(FakeLexicalBuilder(), embedder),
        judge or scripted(SCRIPT), **options,
    )


def assess(
    judge: Optional[FakeChatModel] = None,
    transcript: Optional[Transcript] = None,
    **options: Any,
) -> SessionAssessment:
    """Assess the sample meeting with the scripted judge."""
    return service(judge, **options).assess(
        transcript or make_transcript(), CATALOGUE
    )


def cell(
    result: SessionAssessment, participant: str, competency: str
) -> CompetencyAssessment:
    """Return one cell of the scorecards."""
    card = result.scorecard(participant)
    assert card is not None
    found = card.assessment(competency)
    assert found is not None
    return found


# ---------------------------------------------------------- end to end


def test_verified_evidence_and_rating_are_kept() -> None:
    """Verbatim quotes under real indicators support the judge's rating."""
    item = cell(assess(), "Maya", "listening")
    assert (item.status, item.rating) == ("rated", 4)
    assert [(e.turn, e.indicator_id, e.polarity) for e in item.evidence] == [
        (0, "L1", "positive"), (3, "L2", "positive"),
    ]
    assert [e.quote for e in item.evidence] == [
        "What do you mean exactly",
        "So if I understood correctly, the big file must work",
    ]
    assert item.rationale == "Asks and summarises."
    assert item.candidate_turns == [0, 3]
    assert (item.proposed, item.rejected, item.unsupported) == (2, 0, 0)
    assert item.flags == []


def test_hallucinated_quotes_are_rejected_and_rating_withheld() -> None:
    """Invented quotes are counted; too little real evidence, no rating."""
    item = cell(assess(), "Tom", "listening")
    assert (item.status, item.rating) == ("insufficient_evidence", None)
    assert [(e.turn, e.indicator_id) for e in item.evidence] == [(1, "L3")]
    assert (item.proposed, item.rejected) == (3, 2)
    assert item.flags == ["below_minimum_evidence"]
    assert item.candidate_turns == [1, 4, 7]


def test_min_incidents_one_lets_a_single_incident_rate() -> None:
    """Lowering the rule to one incident releases the rating."""
    item = cell(assess(min_incidents=1), "Tom", "listening")
    assert (item.status, item.rating) == ("rated", 2)


def test_mixed_evidence_is_rated_with_rubric_polarities() -> None:
    """Positive and negative evidence both reach the scorecard."""
    item = cell(assess(), "Tom", "planning")
    assert (item.status, item.rating) == ("rated", 3)
    assert [(e.turn, e.polarity) for e in item.evidence] == [
        (4, "negative"), (7, "positive"),
    ]


def test_unparseable_reply_is_insufficient_evidence_with_a_flag() -> None:
    """A reply without JSON never becomes a rating."""
    item = cell(assess(), "Priya", "planning")
    assert (item.status, item.rating) == ("insufficient_evidence", None)
    assert item.evidence == [] and item.proposed == 0
    assert item.flags == ["unparseable_reply"]
    assert item.candidate_turns == [2, 5]


def test_empty_judgement_is_insufficient_evidence_without_flags() -> None:
    """A judge that cites nothing and rates null is a normal result."""
    item = cell(assess(), "Maya", "planning")
    assert (item.status, item.rating, item.flags) == (
        "insufficient_evidence", None, [],
    )


def test_rating_without_any_evidence_is_withheld() -> None:
    """A bare rating with no quote is not reported."""
    judge = scripted({}, default=judge_reply([], rating=5))
    result = assess(judge)
    assert all(r is None for row in result.ratings().values()
               for r in row.values())
    assert cell(result, "Maya", "listening").flags == [
        "below_minimum_evidence"
    ]


def test_off_scale_rating_is_withheld() -> None:
    """A rating outside the catalogue scale is dropped and flagged."""
    judge = scripted({("Maya", "listening"): MAYA_LISTENING.replace(
        '"rating": 4', '"rating": 9'
    )})
    item = cell(assess(judge), "Maya", "listening")
    assert (item.rating, item.flags) == (None, ["rating_off_scale"])
    assert len(item.evidence) == 2


def test_stored_quote_is_the_transcript_wording() -> None:
    """A lower-cased quote skipping a word is stored as it was said."""
    judge = scripted({("Maya", "listening"): judge_reply(
        [(0, "what do you mean by streaming the file priya", "L1"),
         (3, "so if i understood correctly", "L2"),
         (3, "so if I understood wrongly", "L2")], rating=4,
    )})
    item = cell(assess(judge), "Maya", "listening")
    assert [e.quote for e in item.evidence] == [
        "What do you mean exactly by streaming the file, Priya",
        "So if I understood correctly",
    ]
    assert (item.proposed, item.rejected) == (3, 1)


def test_quote_of_a_context_turn_is_rejected() -> None:
    """Quoting the other speaker shown as context is not evidence."""
    judge = scripted({("Tom", "listening"): judge_reply(
        [(0, "What do you mean exactly by streaming", "L1"),
         (1, "That is nonsense, nobody will test", "L3")], rating=4,
    )})
    item = cell(assess(judge), "Tom", "listening")
    assert [(e.turn, e.indicator_id) for e in item.evidence] == [(1, "L3")]
    assert item.rejected == 1 and item.rating is None


def test_judge_is_not_called_with_too_few_candidate_turns() -> None:
    """Fewer eligible turns than incidents needed: no model call."""
    transcript = make_transcript([
        ("A", "What do you mean exactly by that sentence?"),
        ("B", "Ok."),
        ("A", "So if I understood correctly we all agree."),
        ("B", "I will finish it by Wednesday, the whole thing."),
        ("A", "Great, thanks a lot for doing that."),
        ("B", "Yes."),
    ])
    judge = scripted({}, default=MAYA_LISTENING)
    result = service(judge).assess(transcript, CATALOGUE, ["B"])
    for item in result.scorecards[0].assessments:
        assert item.status == "insufficient_evidence"
        assert item.flags == ["too_few_turns"]
        assert item.candidate_turns == [3]
        assert (item.proposed, item.rating) == (0, None)
    assert judge.calls == []


def test_judge_is_called_once_per_participant_and_competency() -> None:
    """Three speakers × two competencies → six prompts."""
    judge = scripted(SCRIPT)
    result = assess(judge)
    assert len(judge.calls) == 6
    assert [c.participant for c in result.scorecards] == [
        "Maya", "Tom", "Priya",
    ]
    assert all(
        [a.competency_id for a in card.assessments] == [
            "listening", "planning"]
        for card in result.scorecards
    )


def test_prompt_shows_only_candidate_turns_and_the_rules() -> None:
    """The judge sees the rubric, the numbered turns and the limits."""
    judge = scripted(SCRIPT)
    service(judge, max_tokens=321, min_incidents=2).assess(
        make_transcript(), CATALOGUE, ["Maya"], ["listening"]
    )
    (system, user, max_tokens), = judge.calls
    assert max_tokens == 321
    assert "at least 2 different turns" in system
    assert "sentences in English" in system
    assert "[0] Maya:" in user and "[3] Maya:" in user
    assert "[6] Maya" not in user
    assert "(context) Priya:" in user
    assert "L3 (-) Dismisses others" in user


# ---------------------------------------------------------------- cache


def test_cache_hit_avoids_a_second_judge_call() -> None:
    """The same prompts are answered from the cache the second time."""
    judge, cache = scripted(SCRIPT), InMemoryCache()
    first = assess(judge, cache=cache)
    assert (len(judge.calls), cache.sets, cache.hits) == (6, 6, 0)
    second = assess(judge, cache=cache)
    assert (len(judge.calls), cache.sets, cache.hits) == (6, 6, 6)
    assert second.ratings() == first.ratings()


def test_cache_is_keyed_by_the_judge_model() -> None:
    """Another model does not reuse the cached replies."""
    cache = InMemoryCache()
    assess(scripted(SCRIPT), cache=cache)
    other = FakeChatModel([MAYA_LISTENING], model_name="other-judge")
    assess(other, cache=cache)
    assert len(other.calls) == 6


def test_cache_is_keyed_by_the_prompt() -> None:
    """Changing the incident rule changes the prompt, hence the key."""
    judge, cache = scripted(SCRIPT), InMemoryCache()
    assess(judge, cache=cache, min_incidents=2)
    assess(judge, cache=cache, min_incidents=1)
    assert len(judge.calls) == 12


def test_without_cache_the_judge_is_always_called() -> None:
    """No cache: every assessment asks the model."""
    judge = scripted(SCRIPT)
    assess(judge)
    assess(judge)
    assert len(judge.calls) == 12


# ------------------------------------------------- selection and errors


def test_participants_and_competencies_can_be_selected() -> None:
    """Only the requested cells are assessed, in the requested order."""
    judge = scripted(SCRIPT)
    result = service(judge).assess(
        make_transcript(), CATALOGUE, ["Tom", "Maya"], ["planning"]
    )
    assert result.ratings() == {
        "Tom": {"planning": 3}, "Maya": {"planning": None},
    }
    assert [c.participant for c in result.scorecards] == ["Tom", "Maya"]
    assert len(judge.calls) == 2


def test_empty_selection_means_everything() -> None:
    """Empty lists select all speakers and all competencies."""
    result = service().assess(make_transcript(), CATALOGUE, [], [])
    assert len(result.scorecards) == 3
    assert len(result.scorecards[0].assessments) == 2


def test_unknown_participant_is_an_error_naming_the_speakers() -> None:
    """A participant who did not speak is rejected before any call."""
    judge = scripted(SCRIPT)
    with pytest.raises(InvalidInputError) as error:
        service(judge).assess(make_transcript(), CATALOGUE, ["Maya", "Zoe"])
    assert "Unknown participant(s): Zoe" in str(error.value)
    assert "Maya, Tom, Priya" in str(error.value)
    assert judge.calls == []


def test_unknown_competency_is_an_error() -> None:
    """A competency that is not in the catalogue is rejected."""
    judge = scripted(SCRIPT)
    with pytest.raises(InvalidInputError, match="Unknown competency 'x'"):
        service(judge).assess(make_transcript(), CATALOGUE, None, ["x"])
    assert judge.calls == []


@pytest.mark.parametrize("options", [
    {"candidates": 0}, {"min_incidents": 0}, {"candidates": -3},
])
def test_constructor_validation(options: Dict[str, Any]) -> None:
    """Zero candidates or zero incidents make no sense."""
    with pytest.raises(InvalidInputError, match="at least 1"):
        service(**options)


def test_candidates_option_limits_the_turns_shown() -> None:
    """``candidates`` caps the turns given to the judge."""
    result = assess(candidates=2)
    assert cell(result, "Tom", "planning").candidate_turns == [4, 7]


# -------------------------------------------------------------- session


def test_session_metadata_and_interaction() -> None:
    """The result says how it was produced and who spoke how much."""
    result = assess()
    assert result.session_id == "s1"
    assert (result.language, result.catalogue, result.catalogue_version) == (
        "en", "core", "1.0",
    )
    assert (result.judge_model, result.embedding_model) == ("fake-judge", "")
    assert result.min_incidents == 2
    assert result.team.speakers == 3 and result.team.turns == 8
    tom = result.scorecard("Tom")
    assert tom is not None and tom.interaction.turns == 3
    assert result.scorecard("Zoe") is None
    assert tom.assessment("nope") is None
    assert result.seconds >= 0.0


def test_session_records_the_rules_it_was_produced_with() -> None:
    """Every threshold that shaped the result is stored with it."""
    result = assess(
        candidates=5, min_incidents=2, min_episodes=1, episode_gap=4,
        min_similarity=0.25, min_words=7,
    )
    assert result.rules == {
        "min_incidents": 2, "min_episodes": 1, "episode_gap": 4,
        "min_similarity": 0.25, "min_participant_words": 7,
        "candidates": 5,
    }


def test_session_names_the_embedding_model() -> None:
    """With an embedder the result records its name."""
    result = service(embedder=FakeEmbedder("emb-x", 256)).assess(
        make_transcript(), CATALOGUE, ["Maya"]
    )
    assert result.embedding_model == "emb-x"


def test_progress_tracks_every_cell() -> None:
    """One progress loop covers participants × competencies."""
    progress = RecordingProgress()
    assess(progress=progress)
    assert progress.loops == [("Assessing competencies", 6)]


def test_rubric_in_another_language_than_the_transcript() -> None:
    """An English rubric on a Spanish session: prompts in Spanish."""
    judge = scripted({("Luis", "planning"): judge_reply(
        [(4, "Planificar es perder el tiempo", "P2"),
         (7, "lo termino el viernes entonces", "P1")], rating=3,
    )})
    result = service(judge).assess(
        make_transcript(MEETING_ES, "es1"), CATALOGUE, ["Luis"], ["planning"]
    )
    assert result.language == "es"
    assert "sentences in Spanish" in judge.calls[0][0]
    item = cell(result, "Luis", "planning")
    assert (item.rating, len(item.evidence)) == (3, 2)


# ---------------------------------------------- similarity second opinion


def test_similarity_is_recorded_on_the_evidence() -> None:
    """With an embedder every kept quote carries its cosine."""
    result = service(embedder=FakeEmbedder(dim=256)).assess(
        make_transcript(), CATALOGUE, ["Maya"], ["listening"]
    )
    item = cell(result, "Maya", "listening")
    assert len(item.evidence) == 2
    for evidence in item.evidence:
        assert evidence.similarity is not None
        assert 0.0 < evidence.similarity <= 1.0
        assert evidence.similarity == round(evidence.similarity, 4)
    assert item.unsupported == 0


def test_similarity_is_none_without_an_embedder() -> None:
    """Lexical-only: nothing is dropped, even with a high threshold."""
    item = cell(assess(min_similarity=0.99), "Maya", "listening")
    assert [e.similarity for e in item.evidence] == [None, None]
    assert (item.unsupported, item.rating) == (0, 4)


def test_quotes_far_from_their_indicator_are_unsupported() -> None:
    """A real quote filed under an unrelated indicator is dropped."""
    judge = scripted({("Maya", "listening"): judge_reply(
        [(0, "What do you mean exactly", "L1"),
         (3, "So if I understood correctly", "L3")], rating=4,
    )})
    kept_all = service(judge, FakeEmbedder(dim=256)).assess(
        make_transcript(), CATALOGUE, ["Maya"], ["listening"]
    )
    loose = cell(kept_all, "Maya", "listening")
    assert [e.indicator_id for e in loose.evidence] == ["L1", "L3"]
    wrong = loose.evidence[1].similarity
    right = loose.evidence[0].similarity
    assert wrong is not None and right is not None and wrong < 0.2 < right
    strict = service(
        judge, FakeEmbedder(dim=256), min_similarity=0.2
    ).assess(make_transcript(), CATALOGUE, ["Maya"], ["listening"])
    item = cell(strict, "Maya", "listening")
    assert [(e.turn, e.indicator_id) for e in item.evidence] == [(0, "L1")]
    assert (item.proposed, item.rejected, item.unsupported) == (2, 0, 1)
    assert (item.rating, item.flags) == (None, ["below_minimum_evidence"])


def test_threshold_above_every_cosine_drops_all_evidence() -> None:
    """``min_similarity`` 1.01 keeps nothing and counts everything."""
    result = service(
        embedder=FakeEmbedder(dim=256), min_similarity=1.01
    ).assess(make_transcript(), CATALOGUE, ["Maya"], ["listening"])
    item = cell(result, "Maya", "listening")
    assert (item.evidence, item.unsupported, item.rating) == ([], 2, None)


# ----------------------------------------- episodes and participation floor


def test_single_episode_evidence_is_withheld() -> None:
    """Two turns of one exchange do not make a rating with 2 episodes."""
    item = cell(assess(min_episodes=2), "Maya", "listening")
    assert (item.status, item.rating) == ("insufficient_evidence", None)
    assert item.flags == ["single_episode"]
    assert len(item.evidence) == 2


def test_episode_gap_decides_what_one_exchange_is() -> None:
    """With a narrower gap the same turns are two episodes."""
    item = cell(assess(min_episodes=2, episode_gap=2), "Maya", "listening")
    assert (item.rating, item.flags) == (4, [])


def test_top_rating_on_one_indicator_is_capped() -> None:
    """A 5 backed by a single positive indicator becomes a 4."""
    judge = scripted({("Maya", "listening"): judge_reply(
        [(0, "What do you mean exactly", "L1"),
         (3, "the big file must work", "L1")], rating=5,
    )})
    item = cell(assess(judge), "Maya", "listening")
    assert (item.rating, item.flags) == (4, ["capped_narrow_evidence"])


def test_participants_below_the_word_floor_are_not_judged() -> None:
    """Someone who barely spoke gets no rating and costs no model call."""
    transcript = make_transcript([*MEETING_EN, ("Zoe", "I agree, sure.")])
    judge = scripted(SCRIPT)
    result = service(judge, min_words=10).assess(transcript, CATALOGUE)
    quiet = result.scorecard("Zoe")
    assert quiet is not None
    for item in quiet.assessments:
        assert item.status == "insufficient_evidence"
        assert item.flags == ["below_participation_floor"]
        assert (item.rating, item.evidence, item.candidate_turns) == (
            None, [], [],
        )
    assert len(judge.calls) == 6
    assert cell(result, "Maya", "listening").rating == 4


def test_word_floor_is_exclusive_and_off_by_default() -> None:
    """Exactly ``min_words`` words is enough; 0 disables the floor."""
    words = make_transcript().turns
    tom_words = sum(t.n_words() for t in words if t.speaker == "Tom")
    at_floor = assess(min_words=tom_words)
    above = assess(min_words=tom_words + 1)
    assert cell(at_floor, "Tom", "planning").rating == 3
    assert cell(above, "Tom", "planning").flags == [
        "below_participation_floor"
    ]
    assert cell(assess(), "Tom", "planning").rating == 3
