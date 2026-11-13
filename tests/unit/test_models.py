"""Domain models."""

import pytest
from pydantic import ValidationError

from src.domain.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    UnansweredQuestion,
)
from tests.fakes import make_chunk


def test_chunk_to_source() -> None:
    """A chunk exposes its location as a MinimalSource."""
    chunk = make_chunk("a.py", 10, "hello", context="def f", kind="python")
    assert chunk.to_source() == MinimalSource(
        file_path="a.py", first_character_index=10, last_character_index=15
    )


def test_chunk_search_text_includes_path_and_context() -> None:
    """The indexed text is path, context and body on separate lines."""
    chunk = make_chunk("a.py", 0, "body", context="def f")
    assert chunk.search_text() == "a.py\ndef f\nbody"
    assert make_chunk("b.md", 0, "x").search_text() == "b.md\n\nx"


def test_unanswered_question_gets_unique_id() -> None:
    """A missing question id is generated."""
    a = UnansweredQuestion(question="q")
    b = UnansweredQuestion(question="q")
    assert a.question_id and a.question_id != b.question_id


def test_dataset_parses_both_question_kinds() -> None:
    """Answered questions keep their sources; others stay unanswered."""
    dataset = RagDataset.model_validate({"rag_questions": [
        {"question_id": "1", "question": "a", "answer": "x",
         "sources": [{"file_path": "f", "first_character_index": 0,
                      "last_character_index": 3}]},
        {"question_id": "2", "question": "b"},
    ]})
    first, second = dataset.rag_questions
    assert isinstance(first, AnsweredQuestion)
    assert first.sources[0].last_character_index == 3
    assert not isinstance(second, AnsweredQuestion)


def test_source_requires_fields() -> None:
    """Missing fields are rejected."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate({"file_path": "x"})


def test_chunk_embedding_text_full_is_the_search_text() -> None:
    """The default style embeds path, context and body."""
    chunk = make_chunk("pkg/a.py", 0, "body text", context="def f")
    assert chunk.embedding_text() == chunk.search_text()
    assert chunk.embedding_text("full") == "pkg/a.py\ndef f\nbody text"


def test_chunk_embedding_text_body_drops_the_path() -> None:
    """The body style keeps the context and the text only."""
    chunk = make_chunk("pkg/a.py", 0, "body text", context="def f")
    assert chunk.embedding_text("body") == "def f\nbody text"
    bare = make_chunk("pkg/a.py", 0, "body text")
    assert bare.embedding_text("body") == "\nbody text"
