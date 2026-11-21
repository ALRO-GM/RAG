"""Front matter, JSONL records and vocabulary checks (PL-7)."""

from typing import Any, Dict, List

import pytest

from src.domain.errors import InvalidInputError
from src.domain.knowledge.domains import (
    DomainManifest,
    Vocabularies,
    metadata_problems,
)
from src.domain.knowledge.sources import (
    chunk_jsonl,
    chunk_markdown,
    chunk_source,
    read_metadata,
    split_front_matter,
)
from src.domain.models import ChunkMetadata

MANIFEST = DomainManifest(
    id="support.acme", title="Acme", vocabularies=Vocabularies(
        intent=["refund", "pricing"], script_phase=["opening", "closing"],
        tags=["faq", "vip"],
    ),
)
FREE = DomainManifest(id="free", title="No vocabularies")

DOC = """---
title: Refund policy
intent: refund
author: Ana
tags: [faq]
---
# Asking

Within 30 days.
"""


def test_front_matter_fields_lines_and_body() -> None:
    """Keys, their lines and where the body starts are found."""
    front = split_front_matter("kb.md", DOC)
    assert front.fields["intent"] == "refund"
    assert front.lines == {"title": 2, "intent": 3, "author": 4, "tags": 5}
    assert DOC[front.body_start:].startswith("# Asking")


@pytest.mark.parametrize("text", ["# No front matter\n", "", "--- x\n"])
def test_text_without_front_matter(text: str) -> None:
    """Without an opening ``---`` line everything is body."""
    assert split_front_matter("a.md", text).body_start == 0


@pytest.mark.parametrize("text, error", [
    ("---\ntitle: x\n", "a.md:1: the front matter is not closed"),
    ("---\ntitle: x\nintent: [unclosed\n---\n", "a.md:3: invalid YAML"),
    ("---\n- a\n- b\n---\n", "a.md:2: the front matter must be a mapping"),
])
def test_bad_front_matter(text: str, error: str) -> None:
    """Broken front matter is reported with its line."""
    with pytest.raises(InvalidInputError, match=error):
        split_front_matter("a.md", text)


def test_empty_front_matter_and_dots_end() -> None:
    """An empty block and a ``...`` terminator are accepted."""
    assert split_front_matter("a.md", "---\n---\nbody").fields == {}
    front = split_front_matter("a.md", "---\nintent: x\n...\nbody")
    assert front.fields == {"intent": "x"}


def test_markdown_chunks_inherit_front_matter() -> None:
    """Every chunk carries the file's metadata and its heading title."""
    chunks = chunk_markdown("kb.md", DOC, MANIFEST, 500)
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert (meta.intent, meta.tags, meta.title) == \
        ("refund", ["faq"], "Asking")
    c = chunks[0]
    assert DOC[c.first_character_index:c.last_character_index] == c.text
    assert "author" not in c.text and "intent" not in c.text


def test_values_outside_the_vocabulary_are_refused_with_file_line() -> None:
    """The error names the file, the line and the allowed values."""
    bad = DOC.replace("intent: refund", "intent: shipping")
    with pytest.raises(InvalidInputError,
                       match=r"kb.md:3: intent 'shipping' is not in .*"
                             r"refund, pricing"):
        chunk_markdown("kb.md", bad, MANIFEST, 500)
    tags = DOC.replace("[faq]", "[faq, gold]")
    with pytest.raises(InvalidInputError, match=r"kb.md:5: tags 'gold'"):
        chunk_markdown("kb.md", tags, MANIFEST, 500)
    category = DOC.replace("author: Ana", "category: chisme")
    with pytest.raises(InvalidInputError,
                       match=r"kb.md:4: category 'chisme' is not a "
                             r"dialogue-act category"):
        chunk_markdown("kb.md", category, MANIFEST, 500)


def test_empty_vocabularies_accept_any_value() -> None:
    """A domain without vocabularies takes any intent or tag."""
    bad = DOC.replace("intent: refund", "intent: shipping")
    assert chunk_markdown("kb.md", bad, FREE, 500)[0].metadata.intent == \
        "shipping"


@pytest.mark.parametrize("fields, expected", [
    ({"tags": "faq, vip, faq"}, ["faq", "vip"]),
    ({"tags": ["vip", ""]}, ["vip"]),
    ({"tags": None}, []),
])
def test_tags_from_lists_and_strings(
    fields: Dict[str, Any], expected: List[str]
) -> None:
    """Tags may be a list or a comma-separated string, deduplicated."""
    assert read_metadata(fields, MANIFEST, lambda k: "f:1").tags == expected


@pytest.mark.parametrize("fields, error", [
    ({"tags": 3}, "f:9: tags must be a list"),
    ({"intent": ["refund"]}, "f:9: intent must be a string"),
    ({"title": True}, "f:9: title must be a string"),
    ({"title": "x" * 201}, "longer than 200"),
])
def test_metadata_types_are_checked(
    fields: Dict[str, Any], error: str
) -> None:
    """Wrong types are reported where they were read."""
    with pytest.raises(InvalidInputError, match=error):
        read_metadata(fields, MANIFEST, lambda k: "f:9")


def test_metadata_problems_lists_every_field() -> None:
    """Each refused field is reported once."""
    problems = metadata_problems(ChunkMetadata(
        intent="x", script_phase="y", category="z", tags=["faq", "w"]
    ), MANIFEST)
    assert [f for f, _ in problems] == \
        ["category", "intent", "script_phase", "tags"]
    assert metadata_problems(ChunkMetadata(category="pregunta"), FREE) == []


JSONL = (
    '{"id": "q-7", "text": "How do I ask for a refund?", '
    '"intent": "refund", "category": "pregunta"}\n'
    "\n"
    '{"text": "Prices start at 10 EUR.", "tags": ["vip"]}\n'
)


def test_jsonl_records_become_labelled_chunks() -> None:
    """One chunk per short record, pointing at its line."""
    chunks = chunk_source("kb.jsonl", JSONL, MANIFEST, 500)
    assert [c.metadata.ref for c in chunks] == ["q-7", "3"]
    first, second = chunks
    assert first.text == "How do I ask for a refund?"
    assert first.metadata.intent == "refund"
    assert first.metadata.category == "pregunta"
    assert first.kind == "record"
    line = JSONL.split("\n")[0]
    assert (first.first_character_index, first.last_character_index) == \
        (0, len(line))
    assert JSONL[second.first_character_index:].startswith('{"text": "Pri')
    assert second.metadata.tags == ["vip"]


def test_long_jsonl_records_are_split() -> None:
    """A record longer than a chunk gives several chunks, same ref."""
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(4))
    line = '{"id": "long", "text": ' + f'{text!r}'.replace("'", '"') + "}"
    chunks = chunk_jsonl("kb.jsonl", line, FREE, 200)
    assert len(chunks) > 1
    refs = [c.metadata.ref for c in chunks]
    assert refs[:2] == ["long", "long#2"]
    assert len(set(refs)) == len(refs)


@pytest.mark.parametrize("line, error", [
    ("{bad json", "kb.jsonl:2: invalid JSON"),
    ("[1, 2]", "kb.jsonl:2: a record is an object"),
    ('{"text": "x", "speaker": "S1"}', "kb.jsonl:2: unknown key"),
    ('{"text": ""}', "kb.jsonl:2: text must be a string"),
    ('{"text": "x", "intent": "nope"}', "kb.jsonl:2: intent 'nope'"),
])
def test_bad_jsonl_lines(line: str, error: str) -> None:
    """Each problem names the file and the line."""
    text = '{"text": "fine"}\n' + line + "\n"
    with pytest.raises(InvalidInputError, match=error):
        chunk_jsonl("kb.jsonl", text, MANIFEST, 500)
