"""Suggestion sink: one JSON line per reasoner call."""

import json
from pathlib import Path
from typing import List

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
