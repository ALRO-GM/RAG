"""Transcript publisher: write ``transcript.v1`` documents to a directory.

The RAG (or a person copying files) reads finished transcripts from
that directory: ``<directory>/<session id>.transcript.v1.json``. The
document is checked against the contract before it is written, and
written atomically, so a reader never sees half a transcript.
"""

from pathlib import Path
from typing import Any, Mapping

from transcriber.adapters.outbound.fs_storage import child_path, write_json
from transcriber.domain.models import SCHEMA, SESSION_ID_PATTERN, Transcript

SUFFIX = f".{SCHEMA}.json"


class DirectoryTranscriptPublisher:
    """Publish transcripts as JSON files in ``directory``."""

    def __init__(self, directory: Path) -> None:
        """Write into ``directory`` (created on the first publish)."""
        self.directory = Path(directory)

    def path_for(self, session_id: str) -> Path:
        """Return the file the transcript of ``session_id`` goes to."""
        return child_path(
            self.directory, session_id, SESSION_ID_PATTERN, "session",
            SUFFIX,
        )

    def publish(self, document: Mapping[str, Any]) -> None:
        """Write one ``transcript.v1`` document, replacing an older one.

        Raises :class:`~transcriber.domain.errors.ContractError` when
        the document does not follow the contract.
        """
        transcript = Transcript.from_contract(document)
        write_json(self.path_for(transcript.session.id), dict(document))
