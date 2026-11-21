"""Chunk a knowledge domain's source files and analyse their terms.

Sources are Markdown or plain text, optionally with YAML front matter
giving the metadata of every chunk of the file, and JSONL files with
one labelled record per line (PL-7). Text is chunked with the same
:class:`~src.domain.chunking.text_chunker.TextChunker` as the 42
corpus; each chunk's heading path becomes ``metadata.title``. Values
outside the domain's vocabularies are refused with ``file:line``.
"""

import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

from src.domain.chunking.text_chunker import TextChunker
from src.domain.errors import InvalidInputError
from src.domain.knowledge.domains import DomainManifest, metadata_problems
from src.domain.models import Chunk, ChunkMetadata
from src.domain.text_analysis import tokenize

SOURCE_EXTENSIONS = (".md", ".markdown", ".txt", ".jsonl")
"""Files of a domain folder that are indexed."""

METADATA_FIELDS = ("title", "intent", "category", "script_phase", "tags")
"""Keys of front matter and JSONL records read as chunk metadata."""

RECORD_FIELDS = ("id", "text") + METADATA_FIELDS
"""Keys a JSONL record may have."""

MAX_FIELD = 200
"""Longest metadata value (characters)."""

Where = Callable[[str], str]
"""Field name → ``file:line`` of the place it was read from."""


def fold_accents(text: str) -> str:
    """Return ``text`` without diacritics (``migración`` → ``migracion``).

    The lexical tokenizer only knows ASCII letters; folding first keeps
    Spanish words whole instead of splitting them at every accent.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def knowledge_terms(text: str) -> List[str]:
    """Return the lexical terms of ``text`` for a knowledge domain."""
    return tokenize(fold_accents(text))


@dataclass(frozen=True)
class FrontMatter:
    """The YAML block opening a Markdown file, if any.

    ``lines`` gives the 1-based line of each top-level key and
    ``body_start`` the offset where the text after the block begins.
    """

    fields: Dict[str, Any] = field(default_factory=dict)
    lines: Dict[str, int] = field(default_factory=dict)
    body_start: int = 0


def split_front_matter(path: str, text: str) -> FrontMatter:
    """Read the front matter of ``text`` (``---`` … ``---``)."""
    lines = text.split("\n")
    if not lines or lines[0].rstrip("\r") != "---":
        return FrontMatter()
    for end in range(1, len(lines)):
        if lines[end].rstrip("\r") in ("---", "..."):
            break
    else:
        raise InvalidInputError(f"{path}:1: the front matter is not closed.")
    block = "\n".join(lines[1:end])
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 2 if mark is not None else 1
        raise InvalidInputError(f"{path}:{line}: invalid YAML front matter.")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise InvalidInputError(
            f"{path}:2: the front matter must be a mapping of keys."
        )
    positions: Dict[str, int] = {}
    for number, line in enumerate(lines[1:end], start=2):
        key = line.split(":", 1)[0]
        if ":" in line and not line[:1].isspace() and key in data:
            positions.setdefault(key, number)
    body_start = sum(len(line) + 1 for line in lines[:end + 1])
    return FrontMatter(data, positions, min(body_start, len(text)))


def _text_value(value: Any, name: str, where: Where) -> Optional[str]:
    """Return a metadata string (``None`` for empty), checking its type."""
    if value is None:
        return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise InvalidInputError(f"{where(name)}: {name} must be a string.")
    text = str(value).strip()
    if len(text) > MAX_FIELD:
        raise InvalidInputError(
            f"{where(name)}: {name} is longer than {MAX_FIELD} characters."
        )
    return text or None


def _tags(value: Any, where: Where) -> List[str]:
    """Return tags from a list or a comma-separated string."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, list):
        raise InvalidInputError(
            f"{where('tags')}: tags must be a list or a comma-separated "
            "string."
        )
    tags: List[str] = []
    for item in items:
        tag = _text_value(item, "tags", where)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def read_metadata(
    fields: Mapping[str, Any], manifest: DomainManifest, where: Where
) -> ChunkMetadata:
    """Return the metadata in ``fields``, checked against ``manifest``.

    Keys other than :data:`METADATA_FIELDS` are ignored here (front
    matter may carry an author or a date).
    """
    metadata = ChunkMetadata(
        title=_text_value(fields.get("title"), "title", where),
        intent=_text_value(fields.get("intent"), "intent", where),
        category=_text_value(fields.get("category"), "category", where),
        script_phase=_text_value(
            fields.get("script_phase"), "script_phase", where
        ),
        tags=_tags(fields.get("tags"), where),
    )
    problems = metadata_problems(metadata, manifest)
    if problems:
        name, message = problems[0]
        raise InvalidInputError(f"{where(name)}: {message}.")
    return metadata


def chunk_text(
    path: str,
    text: str,
    max_size: int,
    target: Optional[int] = None,
    start: int = 0,
    base: Optional[ChunkMetadata] = None,
) -> List[Chunk]:
    """Chunk ``text[start:]`` of ``path`` with labelled chunks.

    Offsets refer to the whole ``text``; ``base`` is the metadata every
    chunk inherits. A chunk's title is its heading path, or the base
    title when it has none.
    """
    base = base or ChunkMetadata()
    body = text[start:]
    chunks = TextChunker(max_size, target, table_context=True).chunk(
        path, body
    )
    return [
        c.model_copy(update={
            "first_character_index": c.first_character_index + start,
            "last_character_index": c.last_character_index + start,
            "metadata": base.model_copy(
                update={"title": c.context or base.title}, deep=True
            ),
        })
        for c in chunks
    ]


def chunk_markdown(
    path: str,
    text: str,
    manifest: DomainManifest,
    max_size: int,
    target: Optional[int] = None,
) -> List[Chunk]:
    """Chunk a Markdown or text file, labelled by its front matter."""
    front = split_front_matter(path, text)
    base = read_metadata(
        front.fields, manifest, lambda k: f"{path}:{front.lines.get(k, 1)}"
    )
    return chunk_text(path, text, max_size, target, front.body_start, base)


def at_line(path: str, number: int) -> Where:
    """Return a :data:`Where` placing every field at ``path:number``."""
    def where(_field: str) -> str:
        """Return ``path:number``."""
        return f"{path}:{number}"
    return where


def chunk_jsonl(
    path: str,
    text: str,
    manifest: DomainManifest,
    max_size: int,
    target: Optional[int] = None,
) -> List[Chunk]:
    """Chunk a JSONL file: one labelled record per line.

    Each record is ``{"text": ..., "id"?, "title"?, "intent"?,
    "category"?, "script_phase"?, "tags"?}``. Its chunks point at the
    record's line (``first``/``last`` are the line's offsets) and the
    record id (or its line number) becomes ``metadata.ref``; the
    second and later pieces of a long record get ``<id>#2``, ``#3``…
    Blank lines are skipped.
    """
    chunks: List[Chunk] = []
    offset = 0
    for number, line in enumerate(text.split("\n"), start=1):
        start, offset = offset, offset + len(line) + 1
        if not line.strip():
            continue
        where = at_line(path, number)
        try:
            record = json.loads(line)
        except ValueError:
            raise InvalidInputError(f"{where('')}: invalid JSON.")
        if not isinstance(record, dict):
            raise InvalidInputError(f"{where('')}: a record is an object.")
        unknown = sorted(set(record) - set(RECORD_FIELDS))
        if unknown:
            raise InvalidInputError(
                f"{where('')}: unknown key(s) {', '.join(unknown)}; use "
                f"{', '.join(RECORD_FIELDS)}."
            )
        body = record.get("text")
        if not isinstance(body, str) or not body.strip():
            raise InvalidInputError(f"{where('')}: text must be a string.")
        ref = _text_value(record.get("id"), "id", where) or str(number)
        base = read_metadata(record, manifest, where)
        base = base.model_copy(update={"ref": ref})
        end = start + len(line.rstrip("\r"))
        pieces = chunk_text(path, body, max_size, target, base=base)
        for n, piece in enumerate(pieces, start=1):
            metadata = piece.metadata
            if n > 1:
                metadata = metadata.model_copy(update={"ref": f"{ref}#{n}"})
            chunks.append(piece.model_copy(update={
                "first_character_index": start,
                "last_character_index": end,
                "kind": "record",
                "metadata": metadata,
            }))
    return chunks


def chunk_source(
    path: str,
    text: str,
    manifest: DomainManifest,
    max_size: int,
    target: Optional[int] = None,
) -> List[Chunk]:
    """Chunk one source file of a domain according to its extension."""
    if path.lower().endswith(".jsonl"):
        return chunk_jsonl(path, text, manifest, max_size, target)
    return chunk_markdown(path, text, manifest, max_size, target)
