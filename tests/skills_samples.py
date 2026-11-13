"""Files of a tiny soft-skills project, for the integration tests."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tests.fakes import MEETING_EN, MEETING_ES, catalogue_data

RUBRICS = "rubrics"
TRANSCRIPTS = "transcripts"
EN = f"{TRANSCRIPTS}/s1.txt"
ES = f"{TRANSCRIPTS}/es1.txt"

Marks = Sequence[Tuple[int, str]]
Cells = Mapping[Tuple[str, str], Tuple[Optional[int], Marks]]

GOLD: Dict[Tuple[str, str], Tuple[Optional[int], Marks]] = {
    ("Maya", "listening"): (4, [(0, "L1"), (3, "L2")]),
    ("Maya", "planning"): (None, []),
    ("Tom", "listening"): (2, [(1, "L3")]),
    ("Tom", "planning"): (3, [(4, "P2"), (7, "P1")]),
    ("Priya", "listening"): (None, []),
    ("Priya", "planning"): (4, [(5, "P1")]),
}
"""A valid, fully filled annotation of the English sample meeting."""


def write_rubrics(
    root: Path, languages: Sequence[str] = ("en", "es")
) -> Path:
    """Write the test catalogue as ``rubrics/core.<language>.json``."""
    directory = root / RUBRICS
    directory.mkdir(parents=True, exist_ok=True)
    for language in languages:
        (directory / f"core.{language}.json").write_text(
            json.dumps(catalogue_data(language), ensure_ascii=False),
            encoding="utf-8",
        )
    return directory


def meeting_text(
    turns: Sequence[Tuple[str, str]] = MEETING_EN, timed: bool = True
) -> str:
    """Return a meeting as ``[mm:ss] Speaker: text`` lines."""
    lines = []
    for position, (speaker, text) in enumerate(turns):
        stamp = f"[{position // 6:02d}:{position % 6 * 10:02d}] "
        lines.append(f"{stamp if timed else ''}{speaker}: {text}")
    return "\n".join(lines) + "\n"


def write_transcripts(root: Path) -> Path:
    """Write the English (timed) and Spanish (untimed) meetings."""
    directory = root / TRANSCRIPTS
    directory.mkdir(parents=True, exist_ok=True)
    (root / EN).write_text(meeting_text(), encoding="utf-8")
    (root / ES).write_text(
        meeting_text(MEETING_ES, timed=False), encoding="utf-8"
    )
    return directory


def annotation_form(
    annotator: str,
    cells: Cells = GOLD,
    session: str = "s1",
    version: str = "1.0",
) -> Dict[str, Any]:
    """Return an annotation set as JSON-ready data."""
    return {
        "session_id": session, "annotator": annotator, "catalogue": "core",
        "catalogue_version": version,
        "annotations": [
            {"participant": participant, "competency_id": competency,
             "rating": rating,
             "evidence": [{"turn": t, "indicator_id": i} for t, i in marks]}
            for (participant, competency), (rating, marks) in cells.items()
        ],
    }


def write_form(
    path: Path, annotator: str, cells: Cells = GOLD, session: str = "s1",
    version: str = "1.0",
) -> Path:
    """Write an annotation set file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(annotation_form(annotator, cells, session, version)),
        encoding="utf-8",
    )
    return path
