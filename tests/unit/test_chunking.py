"""Chunking: span helpers, Python and text chunkers, registry."""

from typing import List, Optional

import pytest

from src.domain.chunking import (
    ChunkerRegistry,
    PythonChunker,
    TextChunker,
    default_registry,
)
from src.domain.chunking.spans import (
    line_starts,
    offset_to_line,
    pack_spans,
    split_span_by_lines,
    strip_span,
)
from src.domain.chunking.text_chunker import table_header
from src.domain.models import Chunk
from tests.samples import MD_FILE, PY_FILE, TXT_FILE


def assert_valid(chunks: List[Chunk], text: str, max_size: int) -> None:
    """Check the invariants every chunker must respect."""
    previous_end = 0
    for chunk in chunks:
        first = chunk.first_character_index
        last = chunk.last_character_index
        assert 0 <= first < last <= len(text)
        assert last - first <= max_size
        assert chunk.text == text[first:last]
        assert chunk.text == chunk.text.strip()
        assert first >= previous_end, "chunks must not overlap"
        previous_end = last
    covered = "".join(c.text for c in chunks)
    assert "".join(covered.split()) == "".join(text.split()), (
        "every non-whitespace character must be in exactly one chunk"
    )


# --------------------------------------------------------------- spans


def test_line_starts() -> None:
    """Each line start is recorded, including a trailing empty line."""
    assert line_starts("") == [0]
    assert line_starts("ab\ncd\n") == [0, 3, 6]
    assert line_starts("no newline") == [0]


def test_offset_to_line() -> None:
    """Offsets map to their 0-based line."""
    starts = line_starts("ab\ncd\nef")
    assert [offset_to_line(starts, i) for i in (0, 2, 3, 5, 6, 7)] == [
        0, 0, 1, 1, 2, 2,
    ]


def test_split_span_by_lines_cuts_on_newlines() -> None:
    """Cuts land just after a newline and spans are contiguous."""
    text = "aaaa\nbbbb\ncccc\n"
    spans = split_span_by_lines(text, 0, len(text), 10)
    assert spans == [(0, 10), (10, 15)]


def test_split_span_by_lines_long_line_is_cut_hard() -> None:
    """A line longer than max_size is cut at fixed positions."""
    spans = split_span_by_lines("x" * 25, 0, 25, 10)
    assert spans == [(0, 10), (10, 20), (20, 25)]


def test_split_span_by_lines_small_or_empty() -> None:
    """A span that fits is returned as is; an empty span yields nothing."""
    assert split_span_by_lines("abc", 0, 3, 10) == [(0, 3)]
    assert split_span_by_lines("abc", 2, 2, 10) == []


def test_pack_spans_respects_target() -> None:
    """Consecutive spans are grouped up to the target size."""
    spans = [(0, 4), (4, 8), (8, 12), (12, 30)]
    assert pack_spans(spans, 100, 10) == [
        [(0, 4), (4, 8)], [(8, 12)], [(12, 30)],
    ]


def test_pack_spans_target_capped_by_max() -> None:
    """The effective group size is min(target, max)."""
    spans = [(0, 5), (5, 10), (10, 15)]
    assert pack_spans(spans, 10, 1000) == [[(0, 5), (5, 10)], [(10, 15)]]
    assert pack_spans([], 10, 10) == []


def test_strip_span() -> None:
    """Leading and trailing whitespace is excluded."""
    text = "  \n abc \n"
    assert strip_span(text, 0, len(text)) == (4, 7)
    assert strip_span("   ", 0, 3) == (3, 3)


# ------------------------------------------------------- PythonChunker


def test_python_chunker_keeps_definitions_whole() -> None:
    """Small definitions are packed with their context labels."""
    chunks = PythonChunker(2000).chunk("m.py", PY_FILE)
    assert_valid(chunks, PY_FILE, 2000)
    assert len(chunks) == 1
    assert chunks[0].kind == "python"
    assert "def load_lora_adapter" in chunks[0].context
    assert "class KVCacheManager" in chunks[0].context


def test_python_chunker_small_target_separates_definitions() -> None:
    """With a small target each definition gets its own chunk."""
    chunks = PythonChunker(2000, target_size=50).chunk("m.py", PY_FILE)
    assert_valid(chunks, PY_FILE, 2000)
    contexts = [c.context for c in chunks]
    assert "def load_lora_adapter" in contexts
    assert "class KVCacheManager" in contexts
    func = chunks[contexts.index("def load_lora_adapter")]
    assert func.text.startswith("def load_lora_adapter")
    assert func.text.endswith('return os.path.join("adapters", name)')


def test_python_chunker_splits_oversized_class_into_methods() -> None:
    """A class larger than max_size is split along its methods."""
    methods = "".join(
        f"    def method_{i}(self) -> int:\n"
        f"        return {i} + {i} + {i}\n\n"
        for i in range(12)
    )
    text = f"class Big:\n    \"\"\"Doc.\"\"\"\n\n{methods}"
    chunks = PythonChunker(150).chunk("big.py", text)
    assert_valid(chunks, text, 150)
    assert len(chunks) > 1
    assert any("class Big" in c.context for c in chunks)
    assert any("def Big.method_11" in c.context for c in chunks)


def test_python_chunker_splits_oversized_function_by_lines() -> None:
    """A function larger than max_size is cut into line windows."""
    body = "".join(f"    x{i} = {i}\n" for i in range(60))
    text = f"@decorator\ndef long_function():\n{body}"
    chunks = PythonChunker(100).chunk("f.py", text)
    assert_valid(chunks, text, 100)
    assert chunks[0].text.startswith("@decorator")
    assert all(c.context == "def long_function" for c in chunks)


def test_python_chunker_decorators_belong_to_definition() -> None:
    """Decorators are included in the span of their definition."""
    text = "x = 1\n\n\n@property\n@other(1)\ndef f():\n    return 1\n"
    chunks = PythonChunker(2000, target_size=10).chunk("d.py", text)
    assert_valid(chunks, text, 2000)
    assert chunks[-1].text.startswith("@property")
    assert chunks[-1].context == "def f"


def test_python_chunker_syntax_error_falls_back_to_text() -> None:
    """Unparsable Python is chunked as text."""
    text = "def broken(:\n    pass\n\n# Heading-like comment\n"
    chunks = PythonChunker(2000).chunk("bad.py", text)
    assert_valid(chunks, text, 2000)
    assert chunks and all(c.kind == "text" for c in chunks)


def test_python_chunker_carriage_return_newlines() -> None:
    """Old Mac (CR-only) newlines do not crash the chunker."""
    text = "x = 1\ry = 2\rdef f():\r    return 1\r"
    chunks = PythonChunker(2000).chunk("cr.py", text)
    assert_valid(chunks, text, 2000)
    assert chunks


@pytest.mark.parametrize("text", ["", "\n\n   \n", "# only a comment\n"])
def test_python_chunker_degenerate_inputs(text: str) -> None:
    """Empty and whitespace files produce no empty chunk."""
    chunks = PythonChunker(100).chunk("e.py", text)
    assert_valid(chunks, text, 100)
    assert bool(chunks) == bool(text.strip())


def test_python_chunker_crlf() -> None:
    """Windows newlines keep offsets exact."""
    text = PY_FILE.replace("\n", "\r\n")
    chunks = PythonChunker(120).chunk("w.py", text)
    assert_valid(chunks, text, 120)


# --------------------------------------------------------- TextChunker


def test_text_chunker_heading_breadcrumbs() -> None:
    """Each section records its heading path."""
    chunks = TextChunker(2000, target_size=20).chunk("s.md", MD_FILE)
    assert_valid(chunks, MD_FILE, 2000)
    contexts = [c.context for c in chunks]
    assert contexts == ["Serving", "Serving > LoRA",
                        "Serving > Quantization"]
    assert all(c.kind == "text" for c in chunks)


def test_text_chunker_merges_small_sections() -> None:
    """Small sections are merged up to the target size."""
    chunks = TextChunker(2000).chunk("s.md", MD_FILE)
    assert_valid(chunks, MD_FILE, 2000)
    assert len(chunks) == 1
    assert chunks[0].context == (
        "Serving | Serving > LoRA | Serving > Quantization"
    )


def test_text_chunker_heading_levels_pop() -> None:
    """A heading of the same or higher level replaces deeper ones."""
    text = "# A\n\n## B\n\n### C\n\nx\n\n## D\n\ny\n"
    chunks = TextChunker(2000, target_size=1).chunk("h.md", text)
    assert [c.context for c in chunks] == [
        "A", "A > B", "A > B > C", "A > D",
    ]


def test_text_chunker_ignores_headings_in_code_fences() -> None:
    """``#`` lines inside fenced code are not headings."""
    text = "# Title\n\n```bash\n# not a heading\nls\n```\n"
    chunks = TextChunker(2000, target_size=1).chunk("c.md", text)
    assert [c.context for c in chunks] == ["Title"]


def test_text_chunker_plain_text_paragraphs() -> None:
    """Text without headings is split into paragraphs."""
    chunks = TextChunker(70).chunk("n.txt", TXT_FILE)
    assert_valid(chunks, TXT_FILE, 70)
    assert len(chunks) == 2
    assert all(c.context == "" for c in chunks)


def test_text_chunker_large_section_split() -> None:
    """A section larger than max_size is split on paragraphs/lines."""
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 30
                             for i in range(10))
    text = f"# Big\n\n{paragraphs}\n"
    chunks = TextChunker(300).chunk("big.md", text)
    assert_valid(chunks, text, 300)
    assert len(chunks) > 3
    assert all(c.context == "Big" for c in chunks)


def test_text_chunker_long_single_line() -> None:
    """A single line longer than max_size is cut at fixed positions."""
    text = "y" * 1000
    chunks = TextChunker(300).chunk("l.txt", text)
    assert_valid(chunks, text, 300)
    assert len(chunks) == 4


def test_text_chunker_fence_paragraphs_not_split() -> None:
    """Blank lines inside a code fence do not start a new paragraph."""
    code = "```\n" + "line\n\n" * 10 + "```\n"
    text = "intro " * 20 + "\n\n" + code
    chunks = TextChunker(130, target_size=10).chunk("f.md", text)
    assert_valid(chunks, text, 130)
    assert any(c.text.startswith("```") for c in chunks)


@pytest.mark.parametrize("text", ["", " \n\n\t\n"])
def test_text_chunker_empty(text: str) -> None:
    """Blank documents yield no chunk."""
    assert TextChunker(100).chunk("e.md", text) == []


@pytest.mark.parametrize("max_size", [50, 77, 200, 2000])
def test_chunkers_invariants_on_samples(max_size: int) -> None:
    """Both chunkers respect the invariants for several sizes."""
    for text in (PY_FILE * 5, MD_FILE * 5, TXT_FILE * 5):
        assert_valid(PythonChunker(max_size).chunk("x.py", text),
                     text, max_size)
        assert_valid(TextChunker(max_size).chunk("x.md", text),
                     text, max_size)


# ------------------------------------------------------------ registry


class _Marker:
    """A chunker recording the size it was built with."""

    def __init__(self, max_size: int, target: Optional[int]):
        """Remember the sizes."""
        self.max_size = max_size
        self.target = target

    def chunk(self, file_path: str, text: str) -> List[Chunk]:
        """Return a single chunk covering everything."""
        return [Chunk(file_path=file_path, first_character_index=0,
                      last_character_index=len(text), kind="marker",
                      text=text)]


def test_registry_dispatch_by_extension() -> None:
    """Registered extensions win, case-insensitively; else default."""
    registry = ChunkerRegistry(TextChunker)
    registry.register(".LOG", _Marker)
    chunker = registry.for_path("dir/app.log", 10, 5)
    assert isinstance(chunker, _Marker)
    assert (chunker.max_size, chunker.target) == (10, 5)
    assert isinstance(registry.for_path("APP.Log", 10, None), _Marker)
    assert isinstance(registry.for_path("a.md", 10, None), TextChunker)
    assert registry.chunk("x.log", "abc", 10)[0].kind == "marker"


def test_default_registry() -> None:
    """Python files use the AST chunker, everything else the text one."""
    registry = default_registry()
    assert isinstance(registry.for_path("a/b.py", 100, None), PythonChunker)
    assert isinstance(registry.for_path("a/b.md", 100, None), TextChunker)
    assert isinstance(registry.for_path("a/b.rst", 100, None), TextChunker)
    py = registry.chunk("m.py", PY_FILE, 2000)
    assert py[0].kind == "python"


# ------------------------------------------------------- table context

TABLE = (
    "| Name | Default | Description |\n"
    "|------|:-------:|-------------|\n"
    "| alpha | 1 | first option |\n"
    "| beta | 2 | second option |\n"
)
TABLE_DOC = (
    "# Options\n\n"
    "| Name | Default | Description |\n"
    "|---|---|---|\n"
    + "".join(
        f"| option_{i} | {i} | description of option number {i} |\n"
        for i in range(30)
    )
)
COLUMNS = "Name Default Description"


def line_offset(text: str, line: int) -> int:
    """Return the offset at which the 0-based ``line`` starts."""
    return line_starts(text)[line]


@pytest.mark.parametrize("line", [1, 2, 3])
def test_table_header_of_a_row_inside_a_table(line: int) -> None:
    """Any line after the first one gets the column names."""
    assert table_header(TABLE, line_offset(TABLE, line)) == COLUMNS


def test_table_header_of_the_first_line_is_empty() -> None:
    """The header row itself needs no extra context."""
    assert table_header(TABLE, 0) == ""
    text = "Intro\n\n" + TABLE
    assert table_header(text, line_offset(text, 2)) == ""


def test_table_header_outside_a_table_is_empty() -> None:
    """Text that is not a table row has no header."""
    text = "Intro paragraph\n\n" + TABLE + "\nOutro paragraph\n"
    assert table_header(text, 0) == ""
    assert table_header(text, text.index("Outro")) == ""
    assert table_header("no table here", 3) == ""
    assert table_header("", 0) == ""


def test_table_header_stops_at_the_start_of_the_table() -> None:
    """Text or another table above does not leak into the header."""
    other = "| Other | Columns |\n|---|---|\n| x | y |\n"
    text = other + "\nA sentence.\n" + TABLE
    row = text.index("| beta")
    assert table_header(text, row) == COLUMNS
    assert table_header(text, text.index("| x | y |")) == "Other Columns"


def test_table_header_ignores_a_table_starting_with_a_rule() -> None:
    """A separator row is never returned as column names."""
    text = "|---|---|\n| a | 1 |\n| b | 2 |\n"
    assert table_header(text, line_offset(text, 2)) == ""


def test_table_header_drops_empty_cells_and_outer_pipes() -> None:
    """Cells are stripped and joined by one space."""
    text = "|  Flag  |  | Meaning|\n|---|---|---|\n| -v | | verbose |\n"
    assert table_header(text, line_offset(text, 2)) == "Flag Meaning"


def test_text_chunker_table_context_names_columns_mid_table() -> None:
    """Chunks cut inside a table carry its columns after the breadcrumb."""
    chunks = TextChunker(200, table_context=True).chunk("opt.md", TABLE_DOC)
    assert_valid(chunks, TABLE_DOC, 200)
    rows = [c for c in chunks if c.text.startswith("| option_")]
    assert len(rows) >= 3
    assert {c.context for c in rows} == {f"Options | {COLUMNS}"}
    [first] = [c for c in chunks if "| Name |" in c.text]
    assert first.context == "Options"


def test_text_chunker_table_context_is_off_by_default() -> None:
    """Without the option only the heading breadcrumb is recorded."""
    plain = TextChunker(200).chunk("opt.md", TABLE_DOC)
    assert {c.context for c in plain} == {"Options"}
    tabled = TextChunker(200, table_context=True).chunk("opt.md", TABLE_DOC)
    assert [c.to_source() for c in tabled] == [c.to_source() for c in plain]


@pytest.mark.parametrize("text", [MD_FILE, TXT_FILE, TABLE, ""])
def test_text_chunker_table_context_leaves_other_text_alone(
    text: str
) -> None:
    """Text without a table cut, or a table kept whole, is unchanged."""
    assert TextChunker(300, table_context=True).chunk("f.md", text) == (
        TextChunker(300).chunk("f.md", text)
    )


def test_text_chunker_table_context_without_heading() -> None:
    """With no breadcrumb the context is the column names alone."""
    text = TABLE_DOC.split("\n\n", 1)[1]
    chunks = TextChunker(200, table_context=True).chunk("opt.md", text)
    assert chunks[0].context == ""
    assert {c.context for c in chunks[1:]} == {COLUMNS}


def test_default_registry_table_context() -> None:
    """The option reaches the text chunker, not the Python one."""
    tabled = default_registry(table_context=True)
    chunker = tabled.for_path("opt.md", 200, None)
    assert isinstance(chunker, TextChunker)
    assert chunker.table_context is True
    assert (chunker.max_size, chunker.target_size) == (200, 200)
    assert isinstance(tabled.for_path("a.py", 200, None), PythonChunker)
    contexts = {c.context for c in tabled.chunk("opt.md", TABLE_DOC, 200)}
    assert f"Options | {COLUMNS}" in contexts
    plain = default_registry().for_path("opt.md", 200, None)
    assert isinstance(plain, TextChunker)
    assert plain.table_context is False
    assert tabled.chunk("m.py", PY_FILE, 200) == default_registry().chunk(
        "m.py", PY_FILE, 200
    )
