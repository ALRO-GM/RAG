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
    release.set()
    assert runner.wait(WAIT)
    done = runner.status("second")
    assert done is not None and done.state is SessionState.DONE
    runner.close(WAIT)


def test_a_later_submission_owns_the_status(runner: ThreadJobRunner) -> None:
    """An earlier job of the same session runs but no longer reports."""
    started = threading.Event()
    release = threading.Event()
    ran: List[str] = []

    def hold(progress: Progress) -> None:
        started.set()
        release.wait(WAIT)

    def stale(progress: Progress) -> None:
        ran.append("stale")
        progress.report("recognise", 0.3)
        progress.note("stale note")
        raise RuntimeError("stale failure")

    def fresh(progress: Progress) -> None:
        ran.append("fresh")
        progress.note("fresh note")

    runner.submit("busy", hold)
    assert started.wait(WAIT)
    runner.submit("s", stale)
    runner.submit("s", fresh)
    queued = runner.status("s")
    assert queued is not None and queued.state is SessionState.QUEUED
    release.set()
    assert runner.wait(WAIT)
    assert ran == ["stale", "fresh"]
    status = runner.status("s")
    assert status is not None
    assert (status.state, status.message) == (SessionState.DONE, "fresh note")
    assert runner.error("s") is None
    runner.close(WAIT)


def test_a_note_is_kept_on_the_status(runner: ThreadJobRunner) -> None:
    def noted(progress: Progress) -> None:
        progress.report("identify", 0.8)
        progress.note("2 voiceprints skipped")

    runner.submit("s", noted)
    assert runner.wait(WAIT)
    status = runner.status("s")
    assert status is not None
    assert (status.state, status.message) == (
        SessionState.DONE, "2 voiceprints skipped",
    )
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


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
def test_a_job_raising_a_base_exception_keeps_the_runner_alive(
    runner: ThreadJobRunner, error: type
) -> None:
    def leaving(progress: Progress) -> None:
        progress.report("recognise", 0.3)
        raise error("stop")

    runner.submit("bad", leaving)
    runner.submit("good", lambda progress: None)
    assert runner.wait(WAIT)
    bad = runner.status("bad")
    assert bad is not None
    assert (bad.state, bad.stage) == (SessionState.FAILED, "recognise")
    good = runner.status("good")
    assert good is not None and good.state is SessionState.DONE
    runner.close(WAIT)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_fraction_that_is_not_a_number_keeps_the_previous_one(
    runner: ThreadJobRunner, value: float
) -> None:
    seen: List[float] = []

    def odd(progress: Progress) -> None:
        progress.report("voices", 0.4)
        progress.report("identify", value)
        status = runner.status("s")
        assert status is not None
        seen.append(status.fraction)

    runner.submit("s", odd)
    assert runner.wait(WAIT)
    status = runner.status("s")
    assert status is not None and status.state is SessionState.DONE
    assert seen == [0.4]
    runner.close(WAIT)


@pytest.mark.parametrize("fails", [False, True])
def test_a_progress_kept_after_its_job_changes_nothing(
    runner: ThreadJobRunner, fails: bool
) -> None:
    kept: List[Progress] = []

    def leaky(progress: Progress) -> None:
        kept.append(progress)
        if fails:
            raise RuntimeError("boom")

    runner.submit("s", leaky)
    assert runner.wait(WAIT)
    before = runner.status("s")
    kept[0].report("recognise", 0.5)
    kept[0].note("late")
    assert runner.status("s") == before
    assert before is not None and before.state in (
        SessionState.DONE, SessionState.FAILED,
    )
    runner.close(WAIT)


class TestRun:
    def test_the_work_runs_on_the_worker_after_queued_jobs(self) -> None:
        cleaned: List[str] = []
        runner = ThreadJobRunner(after_job=lambda: cleaned.append("x"))
        log: List[str] = []
        release = threading.Event()

        def blocking(progress: Progress) -> None:
            release.wait(WAIT)
            log.append("job")

        runner.submit("s", blocking)
        threads: List[str] = []

        def work() -> int:
            threads.append(threading.current_thread().name)
            log.append("work")
            return 42

        threading.Timer(0.05, release.set).start()
        assert runner.run(work) == 42
        assert log == ["job", "work"]
        assert threads == ["transcriber-jobs"]
        assert cleaned == ["x", "x"]  # the models are released after it
        runner.close(WAIT)

    def test_errors_reach_the_caller_and_the_runner_goes_on(
        self, runner: ThreadJobRunner
    ) -> None:
        def broken() -> None:
            raise ValueError("bad sample")

        with pytest.raises(ValueError, match="bad sample"):
            runner.run(broken)

        def leaving() -> None:
            raise SystemExit(3)

        with pytest.raises(SystemExit):
            runner.run(leaving)
        assert runner.run(lambda: "still here") == "still here"
        runner.close(WAIT)

    def test_inside_a_job_the_work_runs_at_once(
        self, runner: ThreadJobRunner
    ) -> None:
        results: List[str] = []
        runner.submit(
            "s", lambda progress: results.append(runner.run(lambda: "ok"))
        )
        assert runner.wait(WAIT)
        assert results == ["ok"]
        runner.close(WAIT)

    def test_a_closed_runner_refuses_it(self, runner: ThreadJobRunner) -> None:
        runner.close(WAIT)
        with pytest.raises(StateError, match="closed"):
            runner.run(lambda: None)
