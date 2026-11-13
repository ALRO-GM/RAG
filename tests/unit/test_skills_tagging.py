"""SentenceTagger: candidate rubric indicators per sentence."""

from typing import List, Optional

from src.application.skills.evidence import EvidenceRetriever
from src.application.skills.tagging import (
    SentenceTagger,
    TaggedSentence,
    render_timeline,
)
from src.domain.skills.speech import clock, sentence_transcript
from tests.fakes import (
    FakeEmbedder,
    FakeLexicalBuilder,
    make_catalogue,
    make_transcript,
)

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
