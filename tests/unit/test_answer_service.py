"""AnswerService: retrieve, generate, cache."""

from typing import List, Optional

import pytest

from src.application.answering import AnswerService
from src.application.search import SearchService
from src.domain.errors import InvalidInputError
from src.domain.models import Chunk
from src.domain.prompting import NO_ANSWER
from src.domain.text_analysis import tokenize
from tests.fakes import (
    FakeEmbedder,
    FakeGenerator,
    InMemoryCache,
    InMemoryIndexRepository,
    make_snapshot,
)


def answers(
    chunks: List[Chunk], cache: Optional[InMemoryCache] = None
) -> AnswerService:
    """Return an AnswerService over a lexical in-memory index."""
    repo = InMemoryIndexRepository()
    repo.snapshot = make_snapshot(chunks)
    search = SearchService(repo, tokenize, FakeEmbedder(), mode="bm25")
    return AnswerService(search, FakeGenerator(), cache)


def generator(service: AnswerService) -> FakeGenerator:
    """Return the fake generator of ``service``."""
    assert isinstance(service.generator, FakeGenerator)
    return service.generator


def test_answer_uses_retrieved_chunks(chunks: List[Chunk]) -> None:
    """Sources are the retrieved chunks and the generator sees them."""
    service = answers(chunks)
    result = service.answer("FP8 quantization memory", 2)
    assert result.retrieved_sources[0] == chunks[2].to_source()
    assert result.question == "FP8 quantization memory"
    assert result.answer.startswith("answer to")
    assert generator(service).calls == [
        ("FP8 quantization memory", len(result.retrieved_sources))
    ]


@pytest.mark.parametrize("question", ["", "   "])
def test_answer_rejects_empty_question(
    chunks: List[Chunk], question: str
) -> None:
    """``answer`` validates the question before retrieving."""
    service = answers(chunks)
    with pytest.raises(InvalidInputError):
        service.answer(question, 3)
    assert generator(service).calls == []


def test_answer_rejects_bad_k(chunks: List[Chunk]) -> None:
    """``answer`` validates k."""
    with pytest.raises(InvalidInputError):
        answers(chunks).answer("lora", 0)


def test_answer_from_empty_question(chunks: List[Chunk]) -> None:
    """``answer_from`` does not call the model for a blank question."""
    service = answers(chunks)
    assert service.answer_from("  ", chunks) == "The question is empty."
    assert generator(service).calls == []


def test_no_chunks_gives_no_answer(chunks: List[Chunk]) -> None:
    """Without context the refusal is returned without generating."""
    service = answers(chunks)
    assert service.answer_from("anything", []) == NO_ANSWER
    result = service.answer("zzzunknownterm", 3)
    assert result.retrieved_sources == []
    assert result.answer == NO_ANSWER
    assert generator(service).calls == []


def test_answers_are_cached(chunks: List[Chunk]) -> None:
    """The same question and context are generated once."""
    cache = InMemoryCache()
    service = answers(chunks, cache)
    first = service.answer_from("q?", chunks[:2])
    second = service.answer_from("q?", chunks[:2])
    assert first == second
    assert len(generator(service).calls) == 1
    assert cache.hits == 1


def test_cache_key_depends_on_context_and_model(chunks: List[Chunk]) -> None:
    """Another context or another model is a cache miss."""
    cache = InMemoryCache()
    service = answers(chunks, cache)
    service.answer_from("q?", chunks[:2])
    service.answer_from("q?", chunks[:3])
    other = AnswerService(service.search, FakeGenerator("other"), cache)
    other.answer_from("q?", chunks[:2])
    assert cache.hits == 0
    assert len(cache.data) == 3


def test_without_cache(chunks: List[Chunk]) -> None:
    """Without a cache every call generates."""
    service = answers(chunks)
    service.answer_from("q?", chunks)
    service.answer_from("q?", chunks)
    assert len(generator(service).calls) == 2
