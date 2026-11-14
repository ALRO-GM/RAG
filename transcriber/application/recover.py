"""Recovery at start-up: no session stays stuck waiting for a lost job.

Jobs live in the memory of the process that runs them. When that
process stops (killed, restarted, interrupted), sessions it left
``queued`` or ``running`` would wait forever: they can be neither
queued again nor deleted while running. :class:`RecoverSessions`, run
by the composition root when the job-running process starts, moves them
to ``failed`` with the reason ``interrupted``, from where they can be
queued again or deleted.
"""

from typing import List

from transcriber.application.ports import SessionLocks, SessionRepository
from transcriber.domain.sessions import ACTIVE, interrupt


class RecoverSessions:
    """Fail every session left queued or running by a stopped process."""

    def __init__(
        self, sessions: SessionRepository, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._locks = locks

    def __call__(self) -> List[str]:
        """Fail the interrupted sessions; return their ids, sorted.

        Only call it when no job of this data directory is running
        (at the start of the process that runs the jobs): a session it
        fails is assumed to have no live job.
        """
        recovered: List[str] = []
        for session_id in sorted(self._sessions.ids()):
            with self._locks.hold(session_id):
                session = self._sessions.get(session_id)
                if session is None or session.state not in ACTIVE:
                    continue
                self._sessions.save(interrupt(session))
                recovered.append(session_id)
        return recovered
