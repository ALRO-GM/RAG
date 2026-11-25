"""Suggestion sink: one JSON line per reasoner call; hand-off records."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.domain.records import SuggestionRecord, TurnRecord


class JsonlSink:
    """Append the records to a JSONL file (created, truncated at start).

    Args:
        path: The file to write; its folder is created.
    """

    def __init__(self, path: Path):
        """Create the file empty."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self.records: List[SuggestionRecord] = []

    def write(self, record: SuggestionRecord) -> None:
        """Append ``record`` as a line."""
        self.records.append(record)
        line = json.dumps(record.model_dump(mode="json"),
                          ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class MemorySink:
    """Keep the records in a list (tests, summaries)."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: List[SuggestionRecord] = []

    def write(self, record: SuggestionRecord) -> None:
        """Keep ``record``."""
        self.records.append(record)


class JsonlTurns:
    """Append the voice bot's turn records to a JSONL file (no audio).

    Args:
        path: The file to write; its folder is created.
    """

    def __init__(self, path: Path):
        """Create the file empty."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def write(self, record: TurnRecord) -> None:
        """Append ``record`` as a line."""
        line = json.dumps(record.model_dump(mode="json"),
                          ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class JsonHandoff:
    """Write a session's hand-off record as one JSON file (LV-16).

    Args:
        path: The file to write when the session ends; its folder is
            created then.
    """

    def __init__(self, path: Path):
        """Keep the path; nothing is written yet."""
        self.path = path

    def write(self, record: Dict[str, Any]) -> None:
        """Write ``record`` (names, mode, mandatory items, the round's
        judgement labels), readable by its owner only (mode 600)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


class MemoryHandoff:
    """Keep the hand-off records in a list (tests)."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: List[Dict[str, Any]] = []

    def write(self, record: Dict[str, Any]) -> None:
        """Keep ``record``."""
        self.records.append(record)
