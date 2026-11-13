"""Job runner: one worker thread, one job at a time, in arrival order.

The workstation has one GPU, so transcriptions never run side by side:
jobs wait in a queue and a single daemon thread runs them. The status
of each session's last job (``queued``, ``running`` with its stage and
fraction, ``done`` or ``failed``) is kept in memory for the API to
report. After every job an optional ``after_job`` hook runs, which the
composition root uses to unload the models.
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

_ACTIVE = (SessionState.QUEUED, SessionState.RUNNING)


class _StatusProgress:
    """Progress that records the stage of one session's running job."""

    def __init__(self, runner: "ThreadJobRunner", session_id: str) -> None:
        """Remember where to write."""
        self._runner = runner
        self._session_id = session_id

    def report(self, stage: str, fraction: float) -> None:
        """Record ``stage`` and ``fraction`` (clamped to [0, 1])."""
        self._runner._set(
            self._session_id, SessionState.RUNNING, stage,
            min(max(float(fraction), 0.0), 1.0),
        )


class ThreadJobRunner:
    """Run submitted jobs one at a time on a background thread."""

    def __init__(
        self, after_job: Optional[Callable[[], None]] = None
    ) -> None:
        """Prepare the queue; the thread starts with the first job."""
        self._after_job = after_job
        self._queue: "queue.Queue[Optional[Tuple[str, Job]]]" = queue.Queue()
        self._statuses: Dict[str, JobStatus] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def submit(self, session_id: str, job: Job) -> None:
        """Queue ``job`` for ``session_id``.

        Raises :class:`StateError` if that session already has a job
        waiting or running, or if the runner was closed.
        """
        with self._lock:
            if self._closed:
                raise StateError("The job runner is closed.")
            current = self._statuses.get(session_id)
            if current is not None and current.state in _ACTIVE:
                raise StateError(
                    f"Session {session_id} already has a job"
                    f" {current.state.value}."
                )
            self._statuses[session_id] = JobStatus(
                session_id=session_id, state=SessionState.QUEUED,
            )
            self._errors.pop(session_id, None)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._work, name="transcriber-jobs", daemon=True
                )
                self._thread.start()
            self._queue.put((session_id, job))

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
        state: SessionState,
        stage: Optional[str],
        fraction: float,
    ) -> None:
        """Record the status of ``session_id``'s job."""
        with self._lock:
            self._statuses[session_id] = JobStatus(
                session_id=session_id, state=state, stage=stage,
                fraction=fraction,
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

    def _run(self, session_id: str, job: Job) -> None:
        """Run one job and record how it ended; never raise."""
        progress = _StatusProgress(self, session_id)
        self._set(session_id, SessionState.RUNNING, None, 0.0)
        try:
            job(progress)
        except Exception as exc:
            LOG.exception("Job of session %s failed", session_id)
            last = self.status(session_id)
            with self._lock:
                self._errors[session_id] = (
                    str(exc).strip() or type(exc).__name__
                )
            self._set(
                session_id, SessionState.FAILED,
                last.stage if last is not None else None,
                last.fraction if last is not None else 0.0,
            )
        else:
            self._set(session_id, SessionState.DONE, DONE_STAGE, 1.0)
        finally:
            if self._after_job is not None:
                try:
                    self._after_job()
                except Exception:
                    LOG.exception("Cleaning up after a job failed")
