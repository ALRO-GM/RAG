"""Inbound adapter: the thread that moves the recovery jobs on (#265).

:class:`RecoveryWorker` sweeps the expired audio when it starts and
every ``sweep_seconds`` (an hour), and moves every due recovery job one
step on every ``tick_seconds`` (:meth:`~transcriber.application.
live_recovery.LiveRecovery.tick`). Only the process that runs the jobs
starts it (``transcriber serve``). It never stops on an error.
"""

import logging
import threading
import time
from typing import Callable, Optional

from transcriber.application.live_recovery import LiveRecovery

LOG = logging.getLogger(__name__)


class RecoveryWorker:
    """Drive a :class:`LiveRecovery` from a daemon thread."""

    def __init__(
        self, recovery: LiveRecovery, tick_seconds: float,
        sweep_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep the service and the periods; nothing runs yet."""
        self._recovery = recovery
        self._tick = tick_seconds
        self._sweep = sweep_seconds
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_sweep: Optional[float] = None

    def once(self) -> None:
        """Sweep when due, then move the jobs on; never raises."""
        now = self._monotonic()
        try:
            if self._last_sweep is None or now - self._last_sweep \
                    >= self._sweep:
                self._last_sweep = now
                self._recovery.sweep()
            self._recovery.tick()
        except Exception as exc:  # noqa: BLE001 - keep the worker alive
            LOG.warning("Live recovery round failed (%s)", type(exc).__name__)

    def start(self) -> None:
        """Start the thread (once)."""
        if self._thread is not None:
            return

        def run() -> None:
            while True:
                self.once()
                if self._stop.wait(self._tick):
                    return

        self._thread = threading.Thread(target=run, name="live-recovery",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the thread after its current round."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
