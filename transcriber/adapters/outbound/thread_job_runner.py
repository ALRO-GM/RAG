"""Job runner: one worker thread, one job at a time, in arrival order.

The workstation has one GPU, so transcriptions never run side by side:
jobs wait in a queue and a single daemon thread runs them. The status
of each session's last job (``queued``, ``running`` with its stage,
fraction and last note, ``done`` or ``failed``) is kept in memory for
the API to report. A session may be submitted again while an earlier
job of it still waits (it was deleted and re-created, or queued again):
both jobs run in order, and the status is the latest job's; the earlier
one finds another ``run_id`` and does nothing. Refusing duplicates is
left to the use cases, which know the session's state. After every job
an optional ``after_job`` hook runs, which the composition root uses to
unload the models.
"""

import logging
import queue
import threading
from typing import Callable, Dict, Optional, Tuple

from transcriber.application.ports import Job, JobStatus
from transcriber.application.transcribe import DONE_STAGE
from transcriber.domain.errors import StateError
from transcriber.domain.models import SessionState

LOG = logging.getLogger(__name__)


class _Token:
    """Identity of one submission (a later one supersedes it)."""


_Item = Tuple[str, _Token, Job]


class _StatusProgress:
    """Progress that records the stage of one session's running job."""

    def __init__(
        self, runner: "ThreadJobRunner", session_id: str, token: _Token
    ) -> None:
        """Remember where to write."""
        self._runner = runner
        self._session_id = session_id
        self._token = token

    def report(self, stage: str, fraction: float) -> None:
        """Record ``stage`` and ``fraction`` (clamped to [0, 1])."""
        self._runner._set(
            self._session_id, self._token, SessionState.RUNNING, stage,
            min(max(float(fraction), 0.0), 1.0),
        )

    def note(self, message: str) -> None:
        """Record ``message`` on the job's current status."""
        self._runner._note(self._session_id, self._token, message)


class ThreadJobRunner:
    """Run submitted jobs one at a time on a background thread."""

    def __init__(
        self, after_job: Optional[Callable[[], None]] = None
    ) -> None:
        """Prepare the queue; the thread starts with the first job."""
        self._after_job = after_job
        self._queue: "queue.Queue[Optional[_Item]]" = queue.Queue()
        self._statuses: Dict[str, JobStatus] = {}
        self._latest: Dict[str, _Token] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def submit(self, session_id: str, job: Job) -> None:
        """Queue ``job`` for ``session_id``.

        The session's status becomes this job's, even if an earlier job
        of the session still waits or runs (that one keeps running but
        no longer reports). Raises :class:`StateError` if the runner
        was closed.
        """
        token = _Token()
        with self._lock:
            if self._closed:
                raise StateError("The job runner is closed.")
            self._latest[session_id] = token
            self._statuses[session_id] = JobStatus(
                session_id=session_id, state=SessionState.QUEUED,
            )
            self._errors.pop(session_id, None)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._work, name="transcriber-jobs", daemon=True
                )
                self._thread.start()
            self._queue.put((session_id, token, job))

    def status(self, session_id: str) -> Optional[JobStatus]:
        """Return the status of the session's job, if one was submitted."""
        with self._lock:
            return self._statuses.get(session_id)

    def error(self, session_id: str) -> Optional[str]:
        """Return why the session's last job failed, if it did."""
        with self._lock:
            return self._errors.get(session_id)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait until every queued job has finished; tell whether it did."""
        done = threading.Event()

        def joined() -> None:
            """Block on the queue, then signal."""
            self._queue.join()
            done.set()

        threading.Thread(target=joined, daemon=True).start()
        return done.wait(timeout)

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop taking jobs, let the queued ones finish, stop the thread."""
        with self._lock:
            self._closed = True
            thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join(timeout)

    def _set(
        self,
        session_id: str,
        token: _Token,
        state: SessionState,
        stage: Optional[str],
        fraction: float,
        error: Optional[str] = None,
    ) -> None:
        """Record the status of the job ``token`` if it is the latest.

        The note given while the job ran is kept; ``error`` is recorded
        as the job's failure.
        """
        with self._lock:
            if self._latest.get(session_id) is not token:
                return
            previous = self._statuses.get(session_id)
            self._statuses[session_id] = JobStatus(
                session_id=session_id, state=state, stage=stage,
                fraction=fraction,
                message=previous.message if previous is not None else None,
            )
            if error is not None:
                self._errors[session_id] = error

    def _note(self, session_id: str, token: _Token, message: str) -> None:
        """Attach ``message`` to the status of job ``token`` if latest."""
        with self._lock:
            current = self._statuses.get(session_id)
            if self._latest.get(session_id) is not token or current is None:
                return
            self._statuses[session_id] = current.model_copy(
                update={"message": message}
            )

    def _work(self) -> None:
        """Take jobs off the queue and run them until closed."""
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._run(*item)
            finally:
                self._queue.task_done()

    def _run(self, session_id: str, token: _Token, job: Job) -> None:
        """Run one job and record how it ended; never raise."""
        progress = _StatusProgress(self, session_id, token)
        self._set(session_id, token, SessionState.RUNNING, None, 0.0)
        try:
            job(progress)
        except Exception as exc:
            LOG.exception("Job of session %s failed", session_id)
            last = self.status(session_id)
            self._set(
                session_id, token, SessionState.FAILED,
                last.stage if last is not None else None,
                last.fraction if last is not None else 0.0,
                error=str(exc).strip() or type(exc).__name__,
            )
        else:
            self._set(
                session_id, token, SessionState.DONE, DONE_STAGE, 1.0
            )
        finally:
            if self._after_job is not None:
                try:
                    self._after_job()
                except Exception:
                    LOG.exception("Cleaning up after a job failed")
