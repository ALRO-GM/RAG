"""Sink adapter: dialogue-act records as one JSON object per line."""

import os
import tempfile
from pathlib import Path
from typing import Iterable

from src.domain.skills.dialogue_acts import ActRecord


class JsonlActRecordSink:
    """Write records to ``path`` (UTF-8 JSONL), replacing it atomically.

    The records go to a temporary file in the same directory that is
    renamed over ``path`` only when every record was written, so a
    failed import never leaves half a file behind.
    """

    def __init__(self, path: str) -> None:
        """Write to ``path`` (its directory is created when needed)."""
        self.path = Path(path)

    def write(self, records: Iterable[ActRecord]) -> int:
        """Write ``records`` and return how many were written."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.",
            suffix=".tmp")
        written = 0
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                for record in records:
                    out.write(record.model_dump_json() + "\n")
                    written += 1
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return written
