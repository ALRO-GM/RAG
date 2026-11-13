"""EvidenceRetriever: the turns of a participant closest to a rubric."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from src.application.ports import FloatArray
from src.application.skills.evidence import Candidate, EvidenceRetriever
from src.domain.errors import InvalidInputError
from src.domain.skills.rubric import Competency
from src.domain.skills.transcript import Transcript
from tests.fakes import (
    MEETING_ES,
    FakeEmbedder,
    FakeLexicalBuilder,
    make_catalogue,
    make_transcript,
)

CATALOGUE = make_catalogue()
LISTENING = CATALOGUE.competency("listening")
PLANNING = CATALOGUE.competency("planning")
QUERIES = [i.query() for c in CATALOGUE.competencies for i in c.indicators]

Found = List[Tuple[int, str]]


class ConceptEmbedder:
    """Embed a text on one axis per concept, whatever its language."""

    CONCEPTS: Dict[str, int] = {
        "nonsense": 0, "tontería": 0, "dismisses": 0,
        "wednesday": 1, "miércoles": 1, "deadline": 1,
        "planning": 2, "planificar": 2, "plan": 2,
    }

    def __init__(self) -> None:
        """Start with no call."""
        self.calls = 0

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return "concepts"

    def encode(
        self, texts: Sequence[str], progress: bool = False
    ) -> FloatArray:
        """Return one unit row per text (zero without a known concept)."""
        self.calls += 1
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for row, text in enumerate(texts):
            for word, axis in self.CONCEPTS.items():
                if word in text.lower():
                    out[row, axis] = 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


def lexical(min_words: int = 4) -> EvidenceRetriever:
    """Return a lexical-only retriever over the fake index."""
    return EvidenceRetriever(FakeLexicalBuilder(), None, min_words=min_words)


def hybrid(
    embedder: Optional[FakeEmbedder] = None, weight: float = 1.0
) -> EvidenceRetriever:
    """Return a retriever with the fake embedder."""
    return EvidenceRetriever(
        FakeLexicalBuilder(), embedder or FakeEmbedder(dim=256), weight
    )


def find(
    retriever: EvidenceRetriever,
    participant: str,
    competency: Competency = LISTENING,
    k: int = 8,
    transcript: Optional[Transcript] = None,
    rubric_language: Optional[str] = None,
) -> List[Candidate]:
    """Index a transcript and return the candidates of a participant."""
    index = retriever.index(transcript or make_transcript())
    return retriever.candidates(
        index, participant, competency, k, rubric_language
    )


def pairs(candidates: Sequence[Candidate]) -> Found:
    """Return the (turn, indicator) pairs of the candidates."""
    return [(c.turn, c.indicator_id) for c in candidates]


# ------------------------------------------------------------- indexing


def test_language_is_declared_or_detected() -> None:
    """A declared language wins; otherwise it is detected from the text."""
    retriever = lexical()
    assert retriever.language_of(make_transcript()) == "en"
    assert retriever.language_of(make_transcript(MEETING_ES)) == "es"
    assert retriever.language_of(make_transcript(language="es-ES")) == "es"
    assert retriever.index(make_transcript(MEETING_ES)).language == "es"


def test_index_of_an_unsupported_language_is_a_user_error() -> None:
    """A transcript declared in a language without analyzer is rejected."""
    with pytest.raises(InvalidInputError, match="Unsupported language"):
        lexical().index(make_transcript(language="fr"))


def test_index_without_embedder_has_no_embeddings() -> None:
    """Lexical-only mode builds one lexical index and no vectors."""
    retriever = lexical()
    index = retriever.index(make_transcript())
    assert index.embeddings is None
    assert isinstance(retriever.lexical_builder, FakeLexicalBuilder)
    assert retriever.lexical_builder.builds == 1


def test_index_embeds_every_turn_once_with_the_passage_prefix() -> None:
    """Turns are embedded in one batch, prefixed for asymmetric models."""
    embedder = FakeEmbedder()
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), embedder, passage_prefix="passage: "
    )
    transcript = make_transcript()
    index = retriever.index(transcript)
    assert embedder.calls == 1
    assert embedder.encoded == [
        "passage: " + t.text for t in transcript.turns
    ]
    assert index.embeddings is not None
    assert index.embeddings.shape[0] == len(transcript.turns)


def test_mask_marks_the_eligible_turns_of_a_speaker() -> None:
    """The mask is the speaker's turns of at least ``min_words`` words."""
    index = lexical().index(make_transcript())
    assert index.mask("Maya").tolist() == [
        True, False, False, True, False, False, False, False,
    ]
    assert not index.mask("Nobody").any()
    assert index.mask("Maya") is index.mask("Maya")


# ----------------------------------------------------------- candidates


def test_candidates_are_only_the_participants_own_turns() -> None:
    """Nobody else's turn is returned, however well it matches."""
    transcript = make_transcript()
    for retriever in (lexical(), hybrid()):
        for speaker in transcript.speakers():
            for competency in CATALOGUE.competencies:
                found = find(retriever, speaker, competency)
                assert found
                assert all(
                    transcript.turns[c.turn].speaker == speaker
                    for c in found
                )


def test_candidates_report_the_best_indicator_of_each_turn() -> None:
    """Each turn carries the indicator it matched best."""
    for retriever in (lexical(), hybrid()):
        assert sorted(pairs(find(retriever, "Maya"))) == [
            (0, "L1"), (3, "L2"),
        ]
        assert pairs(find(retriever, "Tom"))[0] == (1, "L3")
        assert pairs(find(retriever, "Tom", PLANNING))[:2] == [
            (4, "P2"), (7, "P1"),
        ]


def test_candidates_are_sorted_by_decreasing_score() -> None:
    """The best turn comes first."""
    found = find(hybrid(), "Tom", PLANNING)
    scores = [c.score for c in found]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_candidates_skip_turns_under_min_words() -> None:
    """A one-word turn is never a candidate unless the floor is lowered."""
    assert 6 not in [c.turn for c in find(lexical(), "Maya")]
    assert 6 in [c.turn for c in find(lexical(min_words=1), "Maya")]


def test_candidates_k_larger_than_the_eligible_turns() -> None:
    """Asking for more turns than exist returns the eligible ones."""
    found = find(lexical(), "Maya", k=50)
    assert sorted(c.turn for c in found) == [0, 3]
    assert all(np.isfinite(c.score) for c in found)


def test_candidates_k_limits_the_result() -> None:
    """``k`` caps the number of turns, keeping the best."""
    assert pairs(find(lexical(), "Tom", PLANNING, k=1)) == [(4, "P2")]
    assert len(find(lexical(), "Tom", PLANNING, k=2)) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_candidates_k_zero_or_negative_is_empty(k: int) -> None:
    """No turn is returned when none is asked for."""
    assert find(lexical(), "Maya", k=k) == []
    assert find(hybrid(), "Maya", k=k) == []


def test_candidates_of_an_unknown_speaker_are_empty() -> None:
    """Somebody who did not speak has no candidate turn."""
    assert find(lexical(), "Nobody") == []
    assert find(hybrid(), "Nobody") == []


def test_candidates_of_a_speaker_with_only_short_turns_are_empty() -> None:
    """A participant who only said 'ok' has no eligible turn."""
    transcript = make_transcript([
        ("A", "What do you mean exactly by that sentence?"), ("B", "Ok."),
        ("A", "So if I understood correctly we agree."), ("B", "Yes."),
    ])
    assert find(lexical(), "B", transcript=transcript) == []


def test_lexical_only_scores_are_the_lexical_scores() -> None:
    """Without embedder the score is the lexical score itself."""
    found = find(lexical(), "Tom", PLANNING)
    assert [(c.turn, c.score) for c in found][:2] == [(4, 3.0), (7, 1.0)]


def test_hybrid_score_adds_the_weighted_cosine() -> None:
    """Hybrid = lexical / max + weight · cosine; weight 0 is lexical."""
    only_lexical = find(hybrid(weight=0.0), "Tom", PLANNING)[0]
    one = find(hybrid(weight=1.0), "Tom", PLANNING)[0]
    two = find(hybrid(weight=2.0), "Tom", PLANNING)[0]
    assert only_lexical.turn == one.turn == two.turn == 4
    assert only_lexical.score == pytest.approx(1.0)
    cosine = one.score - only_lexical.score
    assert 0.0 < cosine <= 1.0
    assert two.score == pytest.approx(1.0 + 2 * cosine, abs=1e-5)


def test_query_embeddings_are_cached_per_indicator_text() -> None:
    """Each indicator query is embedded once, whatever is asked later."""
    embedder = FakeEmbedder(dim=256)
    retriever = hybrid(embedder)
    for transcript in (make_transcript(), make_transcript(session_id="s2")):
        index = retriever.index(transcript)
        for speaker in transcript.speakers():
            for competency in CATALOGUE.competencies:
                retriever.candidates(index, speaker, competency, 8)
    for query in QUERIES:
        assert embedder.encoded.count(query) == 1
    assert embedder.calls == 2 + len(CATALOGUE.competencies)


def test_query_prefix_is_applied_to_the_embedded_query_only() -> None:
    """The query prefix reaches the embedder but not the lexical side."""
    embedder = FakeEmbedder(dim=256)
    retriever = EvidenceRetriever(
        FakeLexicalBuilder(), embedder, query_prefix="query: "
    )
    plain = find(lexical(), "Tom", PLANNING)
    prefixed = find(retriever, "Tom", PLANNING)
    assert "query: " + PLANNING.indicators[0].query() in embedder.encoded
    assert [c.turn for c in prefixed][:2] == [c.turn for c in plain][:2]


def test_cross_language_rubric_is_matched_by_the_dense_part_only() -> None:
    """An English rubric on a Spanish transcript relies on the embedder."""
    spanish = make_transcript(MEETING_ES)
    blind = find(lexical(), "Luis", PLANNING, transcript=spanish,
                 rubric_language="en")
    assert sorted(c.turn for c in blind) == [1, 4, 7]
    assert all(c.score == 0.0 for c in blind)
    retriever = EvidenceRetriever(FakeLexicalBuilder(), ConceptEmbedder())
    found = find(retriever, "Luis", PLANNING, transcript=spanish,
                 rubric_language="en")
    assert pairs(found)[0] == (4, "P2")
    assert found[0].score == pytest.approx(1.0)
    assert pairs(find(retriever, "Marta", PLANNING, transcript=spanish,
                      rubric_language="en"))[0] == (5, "P1")


def test_same_language_rubric_uses_the_spanish_analyzer() -> None:
    """A Spanish rubric matches a Spanish transcript lexically."""
    spanish = make_transcript(MEETING_ES)
    planning = make_catalogue("es").competency("planning")
    found = find(lexical(), "Luis", planning, transcript=spanish)
    assert pairs(found)[:2] == [(4, "P2"), (7, "P1")]
    assert found[0].score > found[1].score > 0.0


def test_rubric_language_must_be_supported() -> None:
    """A rubric in a language without analyzer is a user error."""
    with pytest.raises(InvalidInputError, match="Unsupported language"):
        find(lexical(), "Tom", rubric_language="de")


def test_turns_returns_the_turns_of_the_candidates_in_order() -> None:
    """Candidates are resolved to their turns, best first."""
    retriever = lexical()
    index = retriever.index(make_transcript())
    found = retriever.candidates(index, "Tom", PLANNING, 2)
    assert [t.index for t in retriever.turns(index, found)] == [4, 7]
    assert retriever.turns(index, []) == []


# ----------------------------------------------------------- similarity


def test_similarity_is_none_without_an_embedder() -> None:
    """Lexical-only retrieval has no second opinion."""
    retriever = lexical()
    index = retriever.index(make_transcript())
    assert retriever.similarity(index, 0, LISTENING.indicators[0]) is None


def test_similarity_is_the_cosine_of_turn_and_indicator() -> None:
    """The matching indicator is closer than the others; range [-1, 1]."""
    embedder = FakeEmbedder(dim=256)
    retriever = hybrid(embedder)
    transcript = make_transcript()
    index = retriever.index(transcript)
    own = retriever.similarity(index, 4, PLANNING.indicators[1])
    other = retriever.similarity(index, 4, LISTENING.indicators[1])
    assert own is not None and other is not None
    assert 0.0 <= other < own <= 1.0 + 1e-6
    vectors = embedder.encode(
        [transcript.turns[4].text, PLANNING.indicators[1].query()]
    )
    assert own == pytest.approx(float(vectors[0] @ vectors[1]), abs=1e-6)


def test_similarity_does_not_depend_on_the_semantic_weight() -> None:
    """The cosine is reported unweighted."""
    scores = []
    for weight in (0.0, 1.0, 5.0):
        retriever = hybrid(weight=weight)
        index = retriever.index(make_transcript())
        scores.append(retriever.similarity(index, 4, PLANNING.indicators[1]))
    assert scores[0] == scores[1] == scores[2]


def test_similarity_reuses_the_cached_query_embedding() -> None:
    """Asking twice, or after retrieval, embeds the indicator once."""
    embedder = FakeEmbedder(dim=256)
    retriever = hybrid(embedder)
    index = retriever.index(make_transcript())
    retriever.candidates(index, "Tom", PLANNING, 8)
    calls = embedder.calls
    retriever.similarity(index, 4, PLANNING.indicators[1])
    retriever.similarity(index, 7, PLANNING.indicators[0])
    assert embedder.calls == calls
