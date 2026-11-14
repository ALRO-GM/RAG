"""SentenceTagger: candidate rubric indicators per sentence."""

import json
from typing import List, Optional

import pytest

from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.tagging import (
    SentenceTag,
    SentenceTagger,
    TagConfirmer,
    TaggedSentence,
    render_timeline,
)
from src.domain.skills.speech import (
    Sentence,
    clock,
    sentence_transcript,
    turn_sentences,
)
from src.domain.skills.tag_prompting import confirm_user_prompt
from src.domain.skills.transcript import from_contract
from src.domain.skills.transcript_contract import read_contract
from tests.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeLexicalBuilder,
    InMemoryCache,
    RecordingProgress,
    make_catalogue,
    make_transcript,
)
from tests.unit.test_skills_ingest import example

CATALOGUE = make_catalogue()

# Two sentences per turn: the second one of Tom's turn echoes indicator
# L3 ("nonsense"), the second of Priya's echoes P1 (a dated commitment).
TURNS = (
    ("Maya", "Good morning everyone. What do you mean exactly by "
             "streaming the file, Priya?", 0.0, 10.0),
    ("Tom", "I am here too now. That is nonsense, nobody will test with "
            "the big file.", 10.0, 20.0),
    ("Priya", "Let me check my notes. I will finish it by Wednesday "
              "afternoon, the reader I mean.", 20.0, 30.0),
    ("Maya", "Okay."),
)


def tagger(
    embedder: Optional[FakeEmbedder] = None, **kwargs: float
) -> SentenceTagger:
    """A tagger over the fake lexical index (and optional embedder)."""
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), embedder, min_words=4
    )
    return SentenceTagger(retriever, **kwargs)  # type: ignore[arg-type]


def by_id(tagged: List[TaggedSentence]) -> dict[str, List[str]]:
    """Indicator ids per sentence id."""
    return {
        t.sentence.id: [tag.indicator_id for tag in t.tags] for t in tagged
    }


def test_every_sentence_is_returned_in_order_with_speaker_and_time() -> None:
    tagged = tagger().tag(make_transcript(TURNS), CATALOGUE)
    assert [t.sentence.id for t in tagged] == [
        "t0.s0", "t0.s1", "t1.s0", "t1.s1", "t2.s0", "t2.s1", "t3.s0",
    ]
    assert tagged[1].sentence.speaker == "Maya"
    assert tagged[3].sentence.speaker == "Tom"
    assert tagged[0].sentence.start == 0.0
    assert tagged[1].sentence.end == 10.0
    assert tagged[6].sentence.start is None


def test_tags_point_at_the_sentence_not_the_whole_turn() -> None:
    tags = by_id(tagger(top=1).tag(make_transcript(TURNS), CATALOGUE))
    assert tags["t0.s1"] == ["L1"]
    assert tags["t1.s1"] == ["L3"]
    assert tags["t2.s1"] == ["P1"]
    assert tags["t3.s0"] == []       # "Okay." is too short to be tagged
    assert "L3" not in tags["t1.s0"]


def test_tag_carries_competency_polarity_and_scores() -> None:
    tagged = tagger(FakeEmbedder(), top=1, min_similarity=0.0).tag(
        make_transcript(TURNS), CATALOGUE
    )
    tag = next(t for t in tagged if t.sentence.id == "t1.s1").tags[0]
    assert tag.indicator_id == "L3"
    assert tag.competency_id == "listening"
    assert tag.polarity == "negative"
    assert tag.score > 0
    assert tag.similarity is not None and -1.0 <= tag.similarity <= 1.0


def test_top_limits_the_tags_per_sentence_best_first() -> None:
    tagged = tagger(top=3).tag(make_transcript(TURNS), CATALOGUE)
    for item in tagged:
        assert len(item.tags) <= 3
        scores = [t.score for t in item.tags]
        assert scores == sorted(scores, reverse=True)


def test_similarity_floor_removes_weak_tags() -> None:
    loose = tagger(FakeEmbedder(), min_similarity=-1.0).tag(
        make_transcript(TURNS), CATALOGUE
    )
    strict = tagger(FakeEmbedder(), min_similarity=0.99).tag(
        make_transcript(TURNS), CATALOGUE
    )
    assert sum(len(t.tags) for t in loose) > 0
    assert sum(len(t.tags) for t in strict) == 0
    assert [t.sentence.id for t in strict] == [t.sentence.id for t in loose]


def test_without_embedder_similarity_is_unknown() -> None:
    tagged = tagger().tag(make_transcript(TURNS), CATALOGUE)
    tags = [tag for t in tagged for tag in t.tags]
    assert tags and all(tag.similarity is None for tag in tags)


def test_timeline_shows_time_speaker_sentence_and_signed_tags() -> None:
    tagged = tagger(FakeEmbedder(), top=1, min_similarity=-1.0).tag(
        make_transcript(TURNS), CATALOGUE
    )
    lines = render_timeline(tagged).splitlines()
    assert len(lines) == 7
    assert lines[0].startswith("[00:00:00] t0.s0 Maya: Good morning")
    nonsense = next(line for line in lines if "nonsense" in line)
    assert "<-L3 " in nonsense
    assert lines[-1] == "[--:--:--] t3.s0 Maya: Okay."
    plain = render_timeline(tagger(top=1).tag(
        make_transcript(TURNS), CATALOGUE
    ))
    assert "<-L3>" in plain


def test_sentence_transcript_and_clock() -> None:
    units = sentence_transcript(make_transcript(TURNS))
    assert [t.index for t in units.turns] == list(range(7))
    assert units.turns[3].speaker == "Tom"
    assert units.session_id == "s1"
    assert clock(None) == "--:--:--"
    assert clock(3725.9) == "01:02:05"
    assert clock(-3) == "00:00:00"


# -- confirmation by the judge (call 1) ----------------------------------

def candidates() -> List[TaggedSentence]:
    """The tagged test meeting, one candidate per sentence."""
    return tagger(top=1).tag(make_transcript(TURNS), CATALOGUE)


def answer(indicator: object, reason: str = "because") -> str:
    """A well-formed classifier reply."""
    return json.dumps({"indicator": indicator, "reason": reason})


def test_judge_names_the_indicator_within_the_offered_competencies() -> None:
    """Retrieval offered L3's competency; the judge may pick L1 in it."""
    judge = FakeChatModel(responder=lambda system, user: answer(
        "L1" if "TARGET sentence, said by Tom" in user and "nonsense" in user
        else None
    ))
    confirmed = TagConfirmer(judge).confirm(candidates(), CATALOGUE)
    by_sentence = {t.sentence.id: t.decision for t in confirmed}
    decision = by_sentence["t1.s1"]
    assert decision is not None and decision.status == "confirmed"
    assert decision.indicator_id == "L1"
    assert decision.competency_id == "listening"
    assert decision.polarity == "positive"
    assert decision.model == "fake-judge"
    other = by_sentence["t0.s1"]
    assert other is not None and other.status == "rejected"


def test_sentences_without_candidates_are_never_sent() -> None:
    judge = FakeChatModel([answer(None)])
    tagged = candidates()
    confirmed = TagConfirmer(judge).confirm(tagged, CATALOGUE)
    with_candidates = sum(1 for t in tagged if t.tags)
    assert len(judge.calls) == with_candidates < len(tagged)
    short = next(t for t in confirmed if t.sentence.id == "t3.s0")
    assert short.tags == [] and short.decision is None


def test_prompt_shows_context_target_and_every_offered_indicator() -> None:
    judge = FakeChatModel([answer(None)])
    TagConfirmer(judge, context=2).confirm(candidates(), CATALOGUE, "es")
    system, user, max_tokens = next(
        call for call in judge.calls if "nonsense" in call[1]
        and "TARGET sentence, said by Tom" in call[1]
    )
    assert "in Spanish" in system and max_tokens == 120
    listening = CATALOGUE.competency("listening")
    for indicator in listening.indicators:
        assert indicator.id in user
    context, target = user.split("TARGET sentence")
    assert "Maya: What do you mean exactly" in context
    assert "Tom: I am here too now." in context
    assert "Good morning everyone." not in context      # only 2 back
    assert "That is nonsense" in target


def test_first_sentence_has_no_context() -> None:
    judge = FakeChatModel([answer(None)])
    TagConfirmer(judge).confirm(
        [t for t in candidates() if t.sentence.id == "t0.s1"], CATALOGUE
    )
    assert "(start of the session)" in judge.calls[0][1]


@pytest.mark.parametrize("reply, status, fragment", [
    (json.dumps({"indicator": "ZZ9", "reason": "x"}), "rejected",
     "'ZZ9', which was not offered"),
    (json.dumps({"indicator": "none"}), "rejected", ""),
    (json.dumps({"indicator": "", "reason": "nothing here"}), "rejected",
     "nothing here"),
    (json.dumps({"indicator": False}), "rejected", ""),
    ("I think it is L3", "unparsed", ""),
    ("{not json}", "unparsed", ""),
    (json.dumps({"reason": "no key"}), "unparsed", ""),
    (json.dumps(["L3"]), "unparsed", ""),
    ("Sure! " + json.dumps({"indicator": "L3", "reason": " a  b "}),
     "confirmed", "a b"),
])
def test_replies_are_checked_not_trusted(
    reply: str, status: str, fragment: str
) -> None:
    confirmed = TagConfirmer(FakeChatModel([reply])).confirm(
        [t for t in candidates() if t.sentence.id == "t1.s1"], CATALOGUE
    )
    decision = confirmed[0].decision
    assert decision is not None and decision.status == status
    assert fragment in decision.reason
    if status != "confirmed":
        assert decision.indicator_id is None


def test_replies_are_cached_by_model_and_prompt() -> None:
    cache = InMemoryCache()
    first = FakeChatModel([answer("L3")])
    TagConfirmer(first, cache).confirm(candidates(), CATALOGUE)
    again = FakeChatModel([answer(None)])
    confirmed = TagConfirmer(again, cache).confirm(candidates(), CATALOGUE)
    assert again.calls == []
    assert any(
        t.decision is not None and t.decision.status == "confirmed"
        for t in confirmed
    )
    other_model = FakeChatModel([answer(None)], model_name="other")
    TagConfirmer(other_model, cache).confirm(candidates(), CATALOGUE)
    assert len(other_model.calls) == len(first.calls)


def test_timeline_shows_decisions_instead_of_candidates() -> None:
    judge = FakeChatModel(responder=lambda system, user: (
        answer("L3") if "said by Tom" in user and "nonsense" in user
        else "garbage" if "said by Priya" in user else answer(None)
    ))
    lines = render_timeline(
        TagConfirmer(judge).confirm(candidates(), CATALOGUE)
    ).splitlines()
    assert next(x for x in lines if "nonsense" in x).endswith("[-L3]")
    assert next(x for x in lines if "streaming" in x).endswith("[none]")
    assert next(x for x in lines if "Wednesday" in x).endswith("[?]")
    assert lines[-1].endswith("Maya: Okay.")


def test_progress_is_reported() -> None:
    progress = RecordingProgress()
    TagConfirmer(FakeChatModel([answer(None)]), progress=progress).confirm(
        candidates(), CATALOGUE
    )
    assert progress.loops == [("Confirming tags", 3)]


def test_prompt_states_the_previous_speaker_when_known() -> None:
    """A ``transcript.v1`` sentence says who spoke before; others do not."""
    listening = [CATALOGUE.competency("listening")]
    plain = Sentence(id="t1.s0", turn=1, index=0, speaker="Luis",
                     text="Eso es una tontería.")
    assert "TARGET sentence, said by Luis:\nEso es" in confirm_user_prompt(
        [], plain, listening
    )
    after = plain.model_copy(update={"previous_speaker": "Marta"})
    assert (
        "TARGET sentence, said by Luis, after Marta spoke:\nEso es"
        in confirm_user_prompt([], after, listening)
    )


def test_confirmation_of_a_contract_transcript_names_who_spoke_before(
) -> None:
    """Through the use case: the judge sees the previous speaker."""
    sentences = [
        s for turn in from_contract(read_contract(example())).turns
        for s in turn_sentences(turn)
    ]
    tagged = [
        TaggedSentence(sentence=s, tags=[SentenceTag(
            indicator_id="L1", competency_id="listening",
            polarity="positive", score=1.0,
        )])
        for s in sentences
    ]
    judge = FakeChatModel([answer(None)])
    TagConfirmer(judge).confirm(tagged, CATALOGUE, "es")
    targets = [user.split("TARGET sentence, ")[1].split(":")[0]
               for _, user, _ in judge.calls]
    assert targets == [
        "said by Marta", "said by Marta", "said by Luis, after Marta spoke",
        "said by S1, after Luis spoke",
    ]
