"""Use cases: receive, give back and forget transcripts of the transcriber.

The transcriber pushes a finished ``transcript.v1`` document
(``docs/transcriber/ARCHITECTURE.md`` §2); the RAG validates it against
its own reading of the contract and keeps it, one copy per session: a
second document for the same session replaces the first, so a repeated
or retried publish never duplicates a session. When the transcriber
deletes a session it asks the RAG to delete its copy too
(:class:`ForgetIngestedTranscript`).
"""

from contextlib import nullcontext
from typing import Any, ContextManager, Dict, NamedTuple, Optional, Sequence

from src.application.skills.ports import (
    SessionEraser,
    SessionGuard,
    TranscriptStore,
)
from src.domain.errors import InvalidInputError, NotFoundError
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


class ForgetIngestedTranscript:
    """Delete a stored ``transcript.v1`` document and what came of it.

    ``erasers`` delete what other modules derived from the session (its
    analysis, scorecards, reports, acts): they must not outlive the
    transcript (retention, ``docs/DEPLOYMENT.md``). They run before the
    document is deleted, so an eraser that refuses (an analysis running
    from the command line, #99) leaves the session as it was and the
    deletion can be asked again. ``guard`` (the analysis) holds the
    session while the erasers and the deletion run and then deletes the
    analysis, so no analysis can read the transcript meanwhile and write
    its files again (#121).
    """

    def __init__(
        self,
        store: TranscriptStore,
        erasers: Sequence[SessionEraser] = (),
        guard: Optional[SessionGuard] = None,
    ) -> None:
        """Wire the store, the erasers and the guard."""
        self._store = store
        self._erasers = list(erasers)
        self._guard = guard

    def __call__(self, session_id: str) -> None:
        """Delete the document of ``session_id``.

        Raises :class:`~src.domain.errors.InvalidInputError` for an id
        the contract does not allow,
        :class:`~src.domain.errors.BusyError` when an eraser refuses
        (nothing is deleted then) and
        :class:`~src.domain.errors.NotFoundError` when none is stored
        (what was derived from it is deleted all the same).
        """
        if not is_session_id(session_id):
            raise InvalidInputError(f"Invalid session id {session_id!r}.")
        held: ContextManager[None] = (
            nullcontext() if self._guard is None
            else self._guard.erasing(session_id)
        )
        with held:
            for eraser in self._erasers:
                eraser.erase(session_id)
            deleted = self._store.delete(session_id)
        if not deleted:
            raise NotFoundError(f"No transcript for session {session_id!r}.")
