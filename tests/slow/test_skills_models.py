"""Soft-skills retrieval with the real multilingual embedding model."""

import pytest

from src.adapters.outbound.bm25_index import BM25Builder
from src.adapters.outbound.fs_transcript_repository import (
    FileSystemTranscriptRepository,
)
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.application.skills.evidence import EvidenceRetriever
from src.config import DEFAULT_SKILLS_EMBEDDING_MODEL
from src.domain.skills.rubric import Competency, Indicator

pytestmark = pytest.mark.slow

TRANSCRIPTS = "examples/skills/transcripts"


def listening(language: str) -> Competency:
    """Return a one-indicator competency in Spanish or English."""
    text = {
        "es": "Parafrasea o resume lo que ha dicho otra persona para "
              "comprobar que lo ha entendido",
        "en": "Paraphrases or summarises what another person said to "
              "check understanding",
    }[language]
    return Competency(
        id="LISTEN", name="Listening", definition="Listens.",
        indicators=[Indicator(id="LISTEN.P1", text=text)],
    )


@pytest.fixture(scope="module")
def retriever() -> EvidenceRetriever:
    """Build a retriever with the default multilingual model."""
    return EvidenceRetriever(
        BM25Builder(),
        SentenceTransformerEmbedder(
            DEFAULT_SKILLS_EMBEDDING_MODEL, max_seq_length=128
        ),
    )


@pytest.mark.parametrize("language", ["es", "en"])
def test_spanish_turn_found_with_either_rubric_language(
    retriever: EvidenceRetriever, language: str
) -> None:
    """Pablo's summary ("si lo he entendido bien") ranks in the top 2."""
    transcript = FileSystemTranscriptRepository(TRANSCRIPTS).find("demo_es")
    index = retriever.index(transcript)
    assert index.language == "es"
    found = retriever.candidates(
        index, "Pablo", listening(language), 2, language
    )
    assert 9 in [c.turn for c in found]
    assert all(transcript.turns[c.turn].speaker == "Pablo" for c in found)


def test_similarity_separates_matching_and_unrelated_turns(
    retriever: EvidenceRetriever,
) -> None:
    """The summary turn is closer to the indicator than a logistics one."""
    transcript = FileSystemTranscriptRepository(TRANSCRIPTS).find("demo_es")
    index = retriever.index(transcript)
    indicator = listening("es").indicators[0]
    summary = retriever.similarity(index, 9, indicator)
    logistics = retriever.similarity(index, 25, indicator)
    assert summary is not None and logistics is not None
    assert summary > logistics


def test_real_catalogue_retrieves_in_both_languages(
    retriever: EvidenceRetriever,
) -> None:
    """The shipped rubric puts the facilitation turn in the top 5."""
    from src.adapters.outbound.json_catalogue_repository import (
        JsonCatalogueRepository,
    )

    catalogues = JsonCatalogueRepository("knowledge/rubrics")
    for session, language, speaker, turn in (
        ("demo_es", "es", "Pablo", 4), ("demo_en", "en", "Sam", 4),
    ):
        transcript = FileSystemTranscriptRepository(TRANSCRIPTS).find(
            session
        )
        index = retriever.index(transcript)
        rubric = catalogues.load("core", language)
        found = retriever.candidates(
            index, speaker, rubric.competency("INCL"), 5, language
        )
        assert turn in [c.turn for c in found]
