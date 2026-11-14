"""Recovery at start-up: no session stays stuck waiting for a lost job.

Jobs live in the memory of the process that runs them. When that
process stops (killed, restarted, interrupted), sessions it left
``queued`` or ``running`` would wait forever: they can be neither
queued again nor deleted while running. :class:`RecoverSessions`, run
by the composition root when the job-running process starts, moves them
to ``failed`` with the reason ``interrupted``, from where they can be
queued again or deleted. A session whose stored record cannot be read is
skipped and reported, so one damaged file never stops the start-up.
"""

from typing import List, NamedTuple

from transcriber.application.ports import SessionLocks, SessionRepository
from transcriber.domain.errors import StorageError
from transcriber.domain.sessions import ACTIVE, interrupt


class Recovery(NamedTuple):
    """What a recovery did: sessions failed, and sessions it could not read.

    ``damaged`` lists the sessions skipped because their stored record
    cannot be read; they are left untouched for a person to look at.
    """

    recovered: List[str]
    damaged: List[str]


class RecoverSessions:
    """Fail every session left queued or running by a stopped process."""

    def __init__(
        self, sessions: SessionRepository, locks: SessionLocks
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._locks = locks

    def __call__(self) -> Recovery:
        """Fail the interrupted sessions; return what was done, sorted.

        A session that cannot be read (:class:`StorageError`) is left
        as it is and listed in ``damaged``. Only call it when no job of
        this data directory is running (at the start of the process
        that runs the jobs): a session it fails is assumed to have no
        live job.
        """
        recovered: List[str] = []
        damaged: List[str] = []
        for session_id in sorted(self._sessions.ids()):
            with self._locks.hold(session_id):
                try:
                    session = self._sessions.get(session_id)
                except StorageError:
                    damaged.append(session_id)
                    continue
                if session is None or session.state not in ACTIVE:
                    continue
                self._sessions.save(interrupt(session))
                recovered.append(session_id)
        return Recovery(recovered, damaged)
