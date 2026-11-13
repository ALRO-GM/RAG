"""Helpers shared by the chunking strategies."""

import bisect
from typing import List, Tuple

Span = Tuple[int, int]


def line_starts(text: str) -> List[int]:
    """Return the character offset at which each line of ``text`` starts."""
    starts = [0]
    pos = text.find("\n")
    while pos != -1:
        starts.append(pos + 1)
        pos = text.find("\n", pos + 1)
    return starts


def split_span_by_lines(
    text: str, start: int, end: int, max_size: int
) -> List[Span]:
    """Split ``text[start:end]`` into spans of at most ``max_size`` chars.

    Cuts happen on line boundaries whenever possible; a single line
    longer than ``max_size`` is cut at fixed character positions.
    """
    spans: List[Span] = []
    cur = start
    while end - cur > max_size:
        limit = cur + max_size
        cut = text.rfind("\n", cur, limit)
        if cut <= cur:
            cut = limit
        else:
            cut += 1
        spans.append((cur, cut))
        cur = cut
    if end > cur:
        spans.append((cur, end))
    return spans


def pack_spans(
    spans: List[Span], max_size: int, target_size: int
) -> List[List[Span]]:
    """Greedily group consecutive spans while they fit in ``target_size``.

    Each input span must already be at most ``max_size`` long. Groups
    are contiguous runs of spans; a span is never split.
    """
    groups: List[List[Span]] = []
    current: List[Span] = []
    for span in spans:
        if current:
            size = span[1] - current[0][0]
            if size > min(target_size, max_size):
                groups.append(current)
                current = []
        current.append(span)
    if current:
        groups.append(current)
    return groups


def strip_span(text: str, start: int, end: int) -> Span:
    """Shrink a span so it does not start or end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def offset_to_line(starts: List[int], offset: int) -> int:
    """Return the 0-based line containing character ``offset``."""
    return bisect.bisect_right(starts, offset) - 1
