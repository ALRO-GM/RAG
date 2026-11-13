"""Grounded prompt construction."""

from typing import List

from src.domain.models import Chunk
from src.domain.prompting import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    build_context,
    format_excerpt,
    user_message,
)
from tests.fakes import make_chunk


def words(text: str) -> int:
    """Count whitespace-separated tokens (a stand-in tokenizer)."""
    return len(text.split())


def test_format_excerpt_with_and_without_context() -> None:
    """The header shows the index, the path and the context if any."""
    with_ctx = make_chunk("a.py", 0, "body", context="def f")
    assert format_excerpt(2, with_ctx, "body") == "[2] a.py (def f)\nbody"
    without = make_chunk("a.md", 0, "body")
    assert format_excerpt(1, without, "x") == "[1] a.md\nx"


def test_build_context_everything_fits(chunks: List[Chunk]) -> None:
    """With a large budget every chunk is included, in order."""
    excerpts = build_context(chunks, words, 10_000)
    assert len(excerpts) == len(chunks)
    for i, (excerpt, chunk) in enumerate(zip(excerpts, chunks), start=1):
        assert excerpt == format_excerpt(i, chunk, chunk.text)


def test_build_context_truncates_first_overflowing_chunk() -> None:
    """The first chunk that does not fit is cut; the rest are dropped."""
    small = make_chunk("a.md", 0, "alpha " * 10)
    big = make_chunk("b.md", 0, "beta " * 500)
    dropped = make_chunk("c.md", 0, "gamma")
    excerpts = build_context([small, big, dropped], words, 200)
    assert len(excerpts) == 2
    assert excerpts[0].startswith("[1] a.md")
    assert excerpts[1].startswith("[2] b.md")
    assert excerpts[1].endswith("[...]")
    assert words(excerpts[0]) + words(excerpts[1]) <= 200
    assert not any("gamma" in e for e in excerpts)


def test_build_context_small_remainder_is_dropped() -> None:
    """A remaining budget of 64 tokens or less is not worth a cut."""
    first = make_chunk("a.md", 0, "alpha " * 90)
    second = make_chunk("b.md", 0, "beta " * 500)
    excerpts = build_context([first, second], words, 150)
    assert len(excerpts) == 1
    assert excerpts[0].startswith("[1] a.md")


def test_build_context_best_chunk_always_present() -> None:
    """Even a first chunk larger than the budget is truncated, not lost."""
    huge = make_chunk("a.md", 0, "word " * 5000)
    excerpts = build_context([huge], words, 100)
    assert len(excerpts) == 1
    assert excerpts[0].endswith("[...]")
    assert words(excerpts[0]) <= 100


def test_build_context_empty() -> None:
    """No chunk gives no excerpt."""
    assert build_context([], words, 100) == []


def test_user_message() -> None:
    """Context excerpts are separated by blank lines before the question."""
    message = user_message("What?", ["[1] a\nx", "[2] b\ny"])
    assert message == (
        "Context:\n[1] a\nx\n\n[2] b\ny\n\nQuestion: What?\nAnswer:"
    )


def test_prompts_mention_refusal() -> None:
    """The system prompt instructs the exact refusal sentence."""
    assert NO_ANSWER.rstrip(".") in SYSTEM_PROMPT
