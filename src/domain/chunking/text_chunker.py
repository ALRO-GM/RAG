"""Markdown and plain-text chunking based on document structure."""

import re
from typing import List, Optional, Tuple

from src.domain.chunking.spans import (
    Span,
    line_starts,
    pack_spans,
    split_span_by_lines,
    strip_span,
)
from src.domain.models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")

# (start, end, heading_path)
Section = Tuple[int, int, str]


class TextChunker:
    """Split Markdown along headings, then paragraphs, then lines.

    Each section (a heading and its body) becomes a chunk when it fits;
    larger sections are split on blank lines outside code fences, and
    the paragraphs are packed back up to ``target_size``. Very small
    consecutive sections are merged. Every chunk records its heading
    breadcrumb (``Title > Section > Subsection``) as context. Plain
    text without headings degrades gracefully to paragraph packing.
    """

    def __init__(
        self,
        max_size: int,
        target_size: Optional[int] = None,
        table_context: bool = False,
    ):
        """Create a chunker producing chunks of at most ``max_size``.

        With ``table_context`` a chunk starting in the middle of a
        Markdown table records the column names of that table in its
        context, so the rows stay searchable by their header.
        """
        self.max_size = max_size
        self.target_size = target_size or max_size
        self.table_context = table_context

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Return the chunks of a Markdown or text file."""
        pieces: List[Section] = []
        for start, end, path in self._sections(text):
            for s, e in self._split_section(text, start, end):
                pieces.append((s, e, path))
        return self._pack(file_path, text, pieces)

    def _sections(self, text: str) -> List[Section]:
        """Return heading-delimited sections with their heading path."""
        starts = line_starts(text)
        lines = text.split("\n")
        headings: List[Tuple[int, int, str]] = []
        in_fence = False
        for idx, line in enumerate(lines):
            if _FENCE.match(line):
                in_fence = not in_fence
                continue
            match = None if in_fence else _HEADING.match(line)
            if match:
                headings.append(
                    (starts[idx], len(match.group(1)), match.group(2))
                )
        sections: List[Section] = []
        stack: List[Tuple[int, str]] = []
        bounds = [h[0] for h in headings] + [len(text)]
        if not headings or headings[0][0] > 0:
            sections.append((0, bounds[0], ""))
        for i, (offset, level, title) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
            sections.append((offset, bounds[i + 1], path))
        return sections

    def _split_section(self, text: str, start: int, end: int) -> List[Span]:
        """Split a section into paragraph spans of at most ``max_size``."""
        if end - start <= self.max_size:
            return [(start, end)]
        paragraphs: List[Span] = []
        para_start = start
        in_fence = False
        pos = start
        while pos < end:
            nl = text.find("\n", pos, end)
            line_end = end if nl == -1 else nl + 1
            line = text[pos:line_end]
            if _FENCE.match(line):
                in_fence = not in_fence
            elif not in_fence and not line.strip() and pos > para_start:
                paragraphs.append((para_start, line_end))
                para_start = line_end
            pos = line_end
        if end > para_start:
            paragraphs.append((para_start, end))
        spans: List[Span] = []
        for s, e in paragraphs:
            spans.extend(split_span_by_lines(text, s, e, self.max_size))
        return [
            (g[0][0], g[-1][1])
            for g in pack_spans(spans, self.max_size, self.target_size)
        ]

    def _pack(
        self, file_path: str, text: str, pieces: List[Section]
    ) -> List[Chunk]:
        """Merge small consecutive pieces and build chunks."""
        paths = {(s, e): p for s, e, p in pieces}
        spans = [(s, e) for s, e, _ in pieces]
        chunks: List[Chunk] = []
        for group in pack_spans(spans, self.max_size, self.target_size):
            start, end = strip_span(text, group[0][0], group[-1][1])
            if start >= end:
                continue
            context: List[str] = []
            for span in group:
                p = paths[span]
                if p and p not in context:
                    context.append(p)
            if self.table_context:
                header = table_header(text, start)
                if header:
                    context.append(header)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    kind="text",
                    context=" | ".join(context),
                    text=text[start:end],
                )
            )
        return chunks


def table_header(text: str, offset: int) -> str:
    """Return the column names of the table cut at ``offset``, if any.

    ``offset`` must be the start of a table row that is not the first
    line of its table; otherwise the result is empty.
    """
    if not text.startswith("|", offset):
        return ""
    first = offset
    while first > 0:
        previous = text.rfind("\n", 0, first - 1) + 1
        if not text[previous:first].lstrip().startswith("|"):
            break
        first = previous
    if first == offset:
        return ""
    line = text[first:text.find("\n", first)]
    if _TABLE_RULE.match(line):
        return ""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return " ".join(c for c in cells if c)
