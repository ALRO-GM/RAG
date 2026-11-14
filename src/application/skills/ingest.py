"""Use cases: receive transcripts from the transcriber and give them back.

The transcriber pushes a finished ``transcript.v1`` document
(``docs/transcriber/ARCHITECTURE.md`` §2); the RAG validates it against
its own reading of the contract and keeps it, one copy per session: a
second document for the same session replaces the first, so a repeated
or retried publish never duplicates a session.
"""

from typing import Any, Dict, NamedTuple

from src.application.skills.ports import TranscriptStore
from src.domain.errors import InvalidInputError
from src.domain.skills.transcript_contract import is_session_id, read_contract


class Ingested(NamedTuple):
    """What :class:`IngestTranscript` did with a document."""

    session_id: str
    replaced: bool
    utterances: int


class IngestTranscript:
    """Validate a ``transcript.v1`` document and store it."""

    def __init__(self, store: TranscriptStore) -> None:
        """Wire the store."""
        self._store = store

    def __call__(self, document: Any) -> Ingested:
        """Store ``document`` under its session id.

        Raises :class:`~src.domain.errors.InvalidInputError` when the
        document breaks the contract; nothing is stored then.
        """
        transcript = read_contract(document)
        session_id = transcript.session.id
        replaced = self._store.put(session_id, document)
        return Ingested(session_id, replaced, len(transcript.utterances))


class ShowIngestedTranscript:
    """Return a stored ``transcript.v1`` document."""

    def __init__(self, store: TranscriptStore) -> None:
        """Wire the store."""
        self._store = store

    def __call__(self, session_id: str) -> Dict[str, Any]:
        """Return the document of ``session_id``.

        Raises :class:`~src.domain.errors.InvalidInputError` for an id
        the contract does not allow and
        :class:`~src.domain.errors.NotFoundError` when none is stored.
        """
        if not is_session_id(session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        return self._store.get(session_id)
