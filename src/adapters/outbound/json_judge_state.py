"""Judge state store: one small JSON file.

Holds the judge chosen from the web and the last quota events of
Claude (:class:`~src.domain.skills.judging.JudgeState`); no transcript
text. Written to a temporary file and renamed, so a reader never sees
half a file. A missing or unreadable file reads as an empty state.
"""

import json
import os
from pathlib import Path

from pydantic import ValidationError

from src.domain.errors import CorruptedDataError
from src.domain.skills.judging import JudgeState


class JsonJudgeStateStore:
    """Keep the judge state in ``path`` (created with its directory)."""

    def __init__(self, path: str):
        """Remember where the state lives; nothing is read yet."""
        self.path = Path(path)

    def load(self) -> JudgeState:
        """Return the stored state, or an empty one."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return JudgeState()
        except OSError as exc:
            raise CorruptedDataError(f"Cannot read {self.path}: {exc}")
        try:
            return JudgeState.model_validate(json.loads(raw))
        except (ValueError, ValidationError):
            return JudgeState()

    def save(self, state: JudgeState) -> None:
        """Replace the stored state atomically."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(self.path.suffix + ".part")
            partial.write_text(
                state.model_dump_json(indent=1), encoding="utf-8"
            )
            os.replace(partial, self.path)
        except OSError as exc:
            raise CorruptedDataError(f"Cannot write {self.path}: {exc}")
