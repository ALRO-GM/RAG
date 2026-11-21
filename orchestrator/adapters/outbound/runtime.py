"""Real clock and executors: wall time, inline calls and threads."""

import threading
import time
from typing import Callable, List


class SystemClock:
    """Monotonic time and ``time.sleep``."""

    def now(self) -> float:
        """Return monotonic seconds."""
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""
        time.sleep(seconds)


class InlineExecutor:
    """Run every call to the end before returning (``--speed max``)."""

    def submit(self, job: Callable[[], None]) -> None:
        """Run ``job`` now."""
        job()

    def wait_idle(self) -> None:
        """Nothing is ever pending."""


class ThreadExecutor:
    """Run each call on its own thread (real-time replay and live)."""

    def __init__(self) -> None:
        """Start with no threads."""
        self._threads: List[threading.Thread] = []

    def submit(self, job: Callable[[], None]) -> None:
        """Run ``job`` on a new thread."""
        thread = threading.Thread(target=job, daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(thread)
        thread.start()

    def wait_idle(self) -> None:
        """Join every thread started so far."""
        for thread in list(self._threads):
            thread.join()
