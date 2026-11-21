"""Job queue adapter: one background thread runs jobs in order.

The knowledge uploads are extracted, chunked and embedded here, one at
a time, so an HTTP request returns at once and two indexings of the
same domain never overlap.
"""

import logging
import queue
import threading
from typing import Callable, Optional

LOG = logging.getLogger(__name__)


class ThreadJobQueue:
    """Run submitted jobs on one daemon thread, first in, first out.

    The thread starts with the first job. A job's exception is logged
    and the next job runs.
    """

    def __init__(self, name: str = "knowledge-worker") -> None:
        """Create the queue; no thread yet."""
        self.name = name
        self._jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def submit(self, job: Callable[[], None]) -> None:
        """Queue ``job`` and make sure the worker runs."""
        self._jobs.put(job)
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name=self.name, daemon=True
                )
                self._thread.start()

    def _run(self) -> None:
        """Run jobs until the process ends."""
        while True:
            job = self._jobs.get()
            try:
                job()
            except Exception:  # a job must never stop the worker
                LOG.exception("Background job failed")
            finally:
                self._jobs.task_done()

    def join(self) -> None:
        """Wait until every submitted job has run (tests, shutdown)."""
        self._jobs.join()
