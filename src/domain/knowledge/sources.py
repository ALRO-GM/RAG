"""Chunk a knowledge domain's source files and analyse their terms.

Domain sources are Markdown or plain text. They are chunked with the
same :class:`~src.domain.chunking.text_chunker.TextChunker` as the 42
corpus; each chunk's heading path becomes ``metadata.title``.
"""

import unicodedata
from typing import List, Optional

from src.domain.chunking.text_chunker import TextChunker
from src.domain.models import Chunk, ChunkMetadata
from src.domain.text_analysis import tokenize

SOURCE_EXTENSIONS = (".md", ".markdown", ".txt")
"""Files of a domain folder that are indexed."""


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
