"""Settings store: the ``settings.v1`` document in one JSON file.

Written to a temporary file and renamed in place under an exclusive
file lock, so the RAG server and an analysis child never read half a
file. A file that cannot be parsed reads as no settings (the services
keep their configuration) and is replaced at the next save.
"""

import fcntl
import json
import os
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from src.domain.errors import CorruptedDataError
from src.domain.pipeline_settings import SettingsDocument


class JsonSettingsStore:
    """Keep the settings in ``path`` (created with its directory)."""

    def __init__(self, path: str):
        """Remember where the document lives; nothing is read yet."""
        self.path = Path(path)

    def load(self) -> Optional[SettingsDocument]:
        """Return the stored document, or ``None`` if absent or unreadable."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CorruptedDataError(f"Cannot read {self.path}: {exc}")
        try:
            return SettingsDocument.model_validate(json.loads(raw))
        except (ValueError, ValidationError):
            return None

    def update(
        self,
        change: Callable[[Optional[SettingsDocument]], SettingsDocument],
    ) -> SettingsDocument:
        """Apply ``change`` to the stored document under the lock.

        An exception raised by ``change`` leaves the file as it was.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock = self.path.with_name(self.path.name + ".lock")
            with open(lock, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    document = change(self.load())
                    self._write(document)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError as exc:
            raise CorruptedDataError(f"Cannot write {self.path}: {exc}")
        return document

    def _write(self, document: SettingsDocument) -> None:
        """Write ``document`` to a temporary file and rename it in place."""
        partial = self.path.with_name(f"{self.path.name}.{os.getpid()}.part")
        partial.write_text(
            json.dumps(document.to_json(), indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(partial, self.path)
