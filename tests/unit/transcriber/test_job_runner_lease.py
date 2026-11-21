"""The job runner with a fake lease: deferral while a live lease is held."""

import threading
from typing import Optional

from tests.unit.transcriber.fakes import NOW
from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
from transcriber.domain.gpu_lease import (
    AckFile, LeaseFile, Priority, new_lease,
)

WAIT = 10.0


class FakeLease:
    """A lease somebody else holds, until the test drops it."""

    def __init__(self, priority: Optional[Priority]) -> None:
        self.priority = priority

    def acquire(self, need_mib: int, starts_at: object = None) -> LeaseFile:
        raise NotImplementedError

    def release(self) -> None:
        self.priority = None

    def current(self) -> Optional[LeaseFile]:
        if self.priority is None:
            return None
        return new_lease("other", self.priority, 1, NOW, 0)

    def ack(self) -> Optional[AckFile]:
        return None

    def freed(self) -> bool:
        return False

    def clear(self) -> bool:
        return False


def test_a_job_waits_for_the_live_lease_then_runs() -> None:
    lease = FakeLease("live")
    runner = ThreadJobRunner(lease=lease, lease_poll=0.01)
    ran = threading.Event()
    runner.submit("s", lambda progress: ran.set())
    assert not ran.wait(0.2)
    lease.release()
    assert ran.wait(WAIT)
    runner.close(WAIT)


def test_waited_for_work_is_deferred_too() -> None:
    lease = FakeLease("live")
    runner = ThreadJobRunner(lease=lease, lease_poll=0.01)
    threading.Timer(0.1, lease.release).start()
    assert runner.run(lambda: 7) == 7
    runner.close(WAIT)


def test_no_lease_no_wait() -> None:
    runner = ThreadJobRunner(lease=FakeLease(None))
    assert runner.run(lambda: 1) == 1
    runner.close(WAIT)
