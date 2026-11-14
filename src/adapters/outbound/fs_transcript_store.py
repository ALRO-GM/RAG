"""Transcript store adapter: received ``transcript.v1`` documents on disk.

Each session is one file, ``<directory>/<session id>.transcript.v1.json``,
written atomically: the bytes go to a temporary file in the same
directory, are flushed to disk and renamed over the target, so a reader
(or a crash) never sees half a transcript, and a second document for
the same session replaces the first. Session ids are checked against
the contract's pattern before any path is built, so an id cannot reach
outside the directory. :meth:`FileSystemTranscriptStore.find` reads a
stored document as the analyzer's transcript; the analyzer's own
repository finds the same files by session id when both use the same
directory (``transcripts_dir``, the default).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
    RagError,
)
from src.domain.skills.transcript import Transcript, from_contract
from src.domain.skills.transcript_contract import (
    SCHEMA,
    is_session_id,
    read_contract,
)

SUFFIX = f".{SCHEMA}.json"


class FileSystemTranscriptStore:
    """Keep one ``transcript.v1`` document per session in ``directory``."""

    def __init__(self, directory: str) -> None:
        """Store under ``directory`` (created on the first write)."""
        self.directory = Path(directory)

    def path_for(self, session_id: str) -> Path:
        """Return the file of ``session_id``; refuse an unsafe id."""
        if not is_session_id(session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        return self.directory / f"{session_id}{SUFFIX}"

    def put(self, session_id: str, document: Mapping[str, Any]) -> bool:
        """Write ``document`` atomically; return whether one was replaced."""
        target = self.path_for(session_id)
        data = json.dumps(
            dict(document), ensure_ascii=False, indent=1, allow_nan=False
        ).encode("utf-8")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            replaced = target.is_file()
            handle, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=self.directory
            )
            try:
                with os.fdopen(handle, "wb") as stream:
                    stream.write(data + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise RagError(f"Cannot write {target}: {exc}")
        return replaced

    def get(self, session_id: str) -> Dict[str, Any]:
        """Return the stored document of ``session_id``."""
        target = self.path_for(session_id)
        try:
            text = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NotFoundError(f"No transcript for session {session_id!r}.")
        except OSError as exc:
            raise RagError(f"Cannot read {target}: {exc}")
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise CorruptedDataError(f"{target} is not valid JSON: {exc}")
        if not isinstance(document, dict):
            raise CorruptedDataError(f"{target} is not a JSON object.")
        return document

    def find(self, session_id: str) -> Transcript:
        """Return the stored document of ``session_id`` as a transcript.

        Utterance ids become sentence ids
        (:func:`~src.domain.skills.transcript.from_contract`). Raises
        :class:`~src.domain.errors.CorruptedDataError` when the stored
        file no longer follows the contract.
        """
        document = self.get(session_id)
        try:
            contract = read_contract(document)
        except InvalidInputError as exc:
            raise CorruptedDataError(f"{self.path_for(session_id)}: {exc}")
        return from_contract(contract)
