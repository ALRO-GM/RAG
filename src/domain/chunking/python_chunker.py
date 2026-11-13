"""Python code chunking based on the abstract syntax tree."""

import ast
from typing import List, Optional, Tuple

from src.domain.chunking.spans import (
    Span,
    line_starts,
    pack_spans,
    split_span_by_lines,
    strip_span,
)
from src.domain.chunking.text_chunker import TextChunker
from src.domain.models import Chunk

# (start, end, label) where label names the definitions in the span.
Segment = Tuple[int, int, str]

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class PythonChunker:
    """Split Python source along its top-level definitions.

    Functions and classes are kept whole when they fit in a chunk.
    Oversized classes are split into their methods, oversized functions
    into line windows. Small neighbouring segments are packed together
    up to ``target_size``. Each chunk carries a context line listing
    the definitions it contains (``class Foo.bar``), which lets lexical
    search match identifiers even when they are far from the body.
    """

    def __init__(self, max_size: int, target_size: Optional[int] = None):
        """Create a chunker producing chunks of at most ``max_size``."""
        self.max_size = max_size
        self.target_size = target_size or max_size
        self._fallback = TextChunker(max_size, self.target_size)

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Return the chunks of a Python file."""
        if "\r" in text.replace("\r\n", ""):
            # ast counts a lone "\r" as a line break but line_starts()
            # does not, so AST line numbers would not map to offsets.
            return self._fallback.chunk(file_path, text)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return self._fallback.chunk(file_path, text)
        starts = line_starts(text)
        segments = self._body_segments(
            text, starts, tree.body, 0, len(text), ""
        )
        return self._pack(file_path, text, segments)

    def _node_span(
        self, starts: List[int], node: ast.stmt, text_len: int
    ) -> Span:
        """Return the character span of a statement, decorators included."""
        first = node.lineno
        decorators = getattr(node, "decorator_list", [])
        for deco in decorators:
            first = min(first, deco.lineno)
        last = node.end_lineno or node.lineno
        begin = starts[first - 1]
        end = starts[last] if last < len(starts) else text_len
        return begin, end

    def _body_segments(
        self,
        text: str,
        starts: List[int],
        body: List[ast.stmt],
        begin: int,
        end: int,
        prefix: str,
    ) -> List[Segment]:
        """Turn a list of statements into labelled segments."""
        segments: List[Segment] = []
        cursor = begin
        for node in body:
            n_start, n_end = self._node_span(starts, node, len(text))
            n_start = max(n_start, cursor)
            if n_start > cursor:
                segments.extend(
                    self._sized(text, cursor, n_start, prefix.rstrip("."))
                )
            label = prefix.rstrip(".")
            if isinstance(node, _DEFS):
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                label = f"{kind} {prefix}{node.name}"
            if (
                isinstance(node, ast.ClassDef)
                and n_end - n_start > self.max_size
                and node.body
            ):
                segments.extend(
                    self._class_segments(text, starts, node, n_start, n_end,
                                         prefix)
                )
            else:
                segments.extend(self._sized(text, n_start, n_end, label))
            cursor = max(cursor, n_end)
        if end > cursor:
            segments.extend(
                self._sized(text, cursor, end, prefix.rstrip("."))
            )
        return segments

    def _class_segments(
        self,
        text: str,
        starts: List[int],
        node: ast.ClassDef,
        begin: int,
        end: int,
        prefix: str,
    ) -> List[Segment]:
        """Split an oversized class into its header and members."""
        first_member, _ = self._node_span(starts, node.body[0], len(text))
        header = f"class {prefix}{node.name}"
        segments: List[Segment] = []
        if first_member > begin:
            segments.extend(self._sized(text, begin, first_member, header))
        segments.extend(
            self._body_segments(
                text, starts, node.body, first_member, end,
                f"{prefix}{node.name}.",
            )
        )
        return segments

    def _sized(
        self, text: str, begin: int, end: int, label: str
    ) -> List[Segment]:
        """Cut a span into segments no longer than ``max_size``."""
        return [
            (s, e, label)
            for s, e in split_span_by_lines(text, begin, end, self.max_size)
        ]

    def _pack(
        self, file_path: str, text: str, segments: List[Segment]
    ) -> List[Chunk]:
        """Pack segments into chunks and attach their context."""
        labels = {(s, e): lbl for s, e, lbl in segments}
        spans = [(s, e) for s, e, _ in segments]
        chunks: List[Chunk] = []
        for group in pack_spans(spans, self.max_size, self.target_size):
            start, end = strip_span(text, group[0][0], group[-1][1])
            if start >= end:
                continue
            names: List[str] = []
            for span in group:
                lbl = labels[span]
                if lbl and lbl not in names:
                    names.append(lbl)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    kind="python",
                    context=" | ".join(names),
                    text=text[start:end],
                )
            )
        return chunks
