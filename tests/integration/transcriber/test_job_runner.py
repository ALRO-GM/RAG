"""The thread job runner: one job at a time, statuses, failures."""

import threading
from typing import List

import pytest

from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
from transcriber.application.ports import Job, Progress
from transcriber.domain.errors import StateError
from transcriber.domain.models import SessionState

WAIT = 10.0


@pytest.fixture()
def runner() -> ThreadJobRunner:
    """A runner without clean-up hook."""
    return ThreadJobRunner()


def test_jobs_run_one_at_a_time_in_order() -> None:
    cleaned: List[str] = []
    runner = ThreadJobRunner(after_job=lambda: cleaned.append("x"))
    log: List[str] = []
    running = 0
    most = 0
    lock = threading.Lock()

    def job(name: str) -> Job:
        def run(progress: Progress) -> None:
            nonlocal running, most
            with lock:
                running += 1
                most = max(most, running)
            log.append(f"{name} start")
            threading.Event().wait(0.02)
            log.append(f"{name} end")
            with lock:
                running -= 1
        return run

    for name in ("a", "b", "c"):
        runner.submit(name, job(name))
    assert runner.wait(WAIT)
    assert most == 1
    assert log == [
        "a start", "a end", "b start", "b end", "c start", "c end",
    ]
    assert cleaned == ["x", "x", "x"]
    for name in ("a", "b", "c"):
        status = runner.status(name)
        assert status is not None
        assert (status.state, status.stage, status.fraction) == (
            SessionState.DONE, "done", 1.0,
        )
    runner.close(WAIT)


def test_status_while_queued_and_running(runner: ThreadJobRunner) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow(progress: Progress) -> None:
        progress.report("recognise", 0.25)
        started.set()
        release.wait(WAIT)

    runner.submit("first", slow)
    runner.submit("second", lambda progress: None)
    assert started.wait(WAIT)
    first = runner.status("first")
    second = runner.status("second")
    assert first is not None and second is not None
    assert (first.state, first.stage, first.fraction) == (
        SessionState.RUNNING, "recognise", 0.25,
    )
    assert (second.state, second.stage) == (SessionState.QUEUED, None)
    assert runner.status("never") is None
    with pytest.raises(StateError, match="already has a job"):
        runner.submit("second", lambda progress: None)
    release.set()
    assert runner.wait(WAIT)
    done = runner.status("second")
    assert done is not None and done.state is SessionState.DONE
    runner.close(WAIT)


def test_a_failing_job_is_reported_and_the_next_one_runs(
    runner: ThreadJobRunner,
) -> None:
    def broken(progress: Progress) -> None:
        progress.report("voices", 0.5)
        raise RuntimeError("CUDA out of memory")

    runner.submit("bad", broken)
    runner.submit("good", lambda progress: progress.report("decode", 7.0))
    assert runner.wait(WAIT)
    bad = runner.status("bad")
    assert bad is not None
    assert (bad.state, bad.stage, bad.fraction) == (
        SessionState.FAILED, "voices", 0.5,
    )
    assert runner.error("bad") == "CUDA out of memory"
    good = runner.status("good")
    assert good is not None and good.state is SessionState.DONE
    assert runner.error("good") is None
    # A failed session can be submitted again.
    runner.submit("bad", lambda progress: None)
    assert runner.wait(WAIT)
    again = runner.status("bad")
    assert again is not None and again.state is SessionState.DONE
    assert runner.error("bad") is None
    runner.close(WAIT)


def test_a_failing_clean_up_does_not_stop_the_runner() -> None:
    def explode() -> None:
        raise RuntimeError("cannot free")

    runner = ThreadJobRunner(after_job=explode)
    runner.submit("a", lambda progress: None)
    runner.submit("b", lambda progress: None)
    assert runner.wait(WAIT)
    status = runner.status("b")
    assert status is not None and status.state is SessionState.DONE
    runner.close(WAIT)


def test_a_closed_runner_refuses_jobs(runner: ThreadJobRunner) -> None:
    runner.close(WAIT)  # never started: nothing to join
    with pytest.raises(StateError, match="closed"):
        runner.submit("a", lambda progress: None)


def test_close_waits_for_queued_jobs(runner: ThreadJobRunner) -> None:
    ran: List[str] = []
    runner.submit("a", lambda progress: ran.append("a"))
    runner.submit("b", lambda progress: ran.append("b"))
    runner.close(WAIT)
    assert ran == ["a", "b"]
