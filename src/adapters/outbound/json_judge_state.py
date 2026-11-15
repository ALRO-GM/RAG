"""Judge state store: one small JSON file, updated under a file lock.

Holds the judge chosen from the web and the last quota events of
Claude (:class:`~src.domain.skills.judging.JudgeState`); no transcript
text. Two processes write it (the RAG's web route and an analysis
job), so every change is a read-modify-write under an exclusive lock
on ``<file>.lock`` (``fcntl.flock``), and the new content is written
to a temporary file and renamed, so a reader never sees half a file.
A missing or unreadable file reads as an empty state.
"""

import fcntl
import json
import os
from pathlib import Path
from typing import Callable

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
        """Replace the stored state (under the lock)."""
        self.update(lambda _: state)

    def update(self, change: Callable[[JudgeState], JudgeState]) -> JudgeState:
        """Apply ``change`` to the stored state under the lock."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock = self.path.with_name(self.path.name + ".lock")
            with open(lock, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    state = change(self.load())
                    self._write(state)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError as exc:
            raise CorruptedDataError(f"Cannot write {self.path}: {exc}")
        return state

    def _write(self, state: JudgeState) -> None:
        """Write ``state`` to a temporary file and rename it in place."""
        partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.part")
        partial.write_text(state.model_dump_json(indent=1), encoding="utf-8")
        os.replace(partial, self.path)
