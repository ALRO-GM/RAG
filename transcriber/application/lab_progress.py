"""The lab's progress across sessions, for the "Pruebas" overview.

:class:`LabOverview` lists every session the lab worked on (it has a
run) or was given (an imported corpus recording, ``docs/transcriber/
LAB.md`` §13), with its corpus and, for its newest run, the steps
counted by :func:`~transcriber.domain.lab_progress.tally_steps` and the
steps left over with their reasons. It also names the newest run that
has results (``done`` or ``partial``), whose figures and audio the
page shows. :class:`ReferenceTurns` gives an imported session's
reference speaker turns (anonymous corpus ids, no words), from which the
page picks a stretch to listen to. Both only read.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from transcriber.application.common import require_session
from transcriber.application.lab import ACTIVE, LabRunStatus
from transcriber.application.lab_ports import LabRepository
from transcriber.application.ports import SessionRepository
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StorageError,
)
from transcriber.domain.lab import Segment
from transcriber.domain.lab_progress import StepTally, tally_steps
from transcriber.domain.lab_reference import SessionReference

RESULT_STATES = ("done", "partial")
"""States of a run whose comparison was stored."""


class SessionProgress(BaseModel):
    """One session in the overview.

    ``corpus`` is ``None`` for the owner's own recordings. ``run_*``
    and ``steps`` describe the newest run (``None`` without any run);
    ``results_run`` is the newest run with stored results.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    title: Optional[str] = None
    duration: float = 0.0
    corpus: Optional[str] = None
    licence: Optional[str] = None
    licence_commercial: Optional[bool] = None
    run_id: Optional[str] = None
    run_state: Optional[str] = None
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    message: Optional[str] = None
    steps: Optional[StepTally] = None
    results_run: Optional[str] = None


class LabOverview:
    """Progress of the newest run of every session the lab knows."""

    def __init__(
        self,
        sessions: SessionRepository,
        repository: LabRepository,
        status: LabRunStatus,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._repository = repository
        self._status = status

    def __call__(self) -> List[SessionProgress]:
        """Return the sessions with a run or a reference, by id."""
        found = []
        for session_id in sorted(self._sessions.ids()):
            entry = self._progress(session_id)
            if entry is not None:
                found.append(entry)
        return found

    def _progress(self, session_id: str) -> Optional[SessionProgress]:
        """Return one session's progress (``None``: nothing to show)."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        reference = self._reference(session_id)
        runs = self._repository.runs(session_id)
        if not runs and reference is None:
            return None
        entry = SessionProgress(
            session_id=session_id, title=session.title,
            duration=(
                reference.duration if reference is not None
                else session.duration
            ),
            corpus=reference.corpus if reference is not None else None,
            licence=reference.licence if reference is not None else None,
            licence_commercial=(
                reference.licence_commercial if reference is not None
                else None
            ),
            results_run=next(
                (r.id for r in runs if r.state in RESULT_STATES), None
            ),
        )
        if not runs:
            return entry
        newest = runs[0]
        state = self._status.state(newest)
        return entry.model_copy(update={
            "run_id": newest.id, "run_state": state,
            "created_at": newest.created_at,
            "finished_at": newest.finished_at, "message": newest.message,
            "steps": tally_steps(newest.steps, state in ACTIVE),
        })

    def _reference(self, session_id: str) -> Optional[SessionReference]:
        """Return the session's reference; an unreadable one is skipped."""
        try:
            return self._repository.reference(session_id)
        except (StorageError, InvalidInputError):
            return None


class ReferenceTurns:
    """The reference speaker turns of an imported session."""

    def __init__(
        self, sessions: SessionRepository, repository: LabRepository
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._repository = repository

    def __call__(self, session_id: str) -> List[Segment]:
        """Return the turns (overlap kept); NotFoundError without any."""
        require_session(self._sessions, session_id)
        reference = self._repository.reference(session_id)
        if reference is None:
            raise NotFoundError(f"Session {session_id} has no reference.")
        return list(reference.segments)
