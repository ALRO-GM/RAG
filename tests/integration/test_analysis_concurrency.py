"""Terminal runs and simultaneous requests of the analysis API (#99)."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.outbound.fs_analysis_store import (
    SHARED_WAIT,
    STARTS_FILE,
    FileAnalysisLock,
    FileSystemAnalysisStore,
    JsonStartLog,
)
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.ingest import (
    ForgetIngestedTranscript,
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.application.skills.ports import AnalysisLock
from src.application.skills.session_analysis import (
    AnalysisRequests,
    AnalysisResults,
    StartLimits,
)
from src.domain.errors import BusyError, LimitError, NotFoundError
from src.domain.skills.analysis_run import StepRecord, new_run
from src.domain.skills.synthesis import JobSought
from src.domain.skills.transcript_contract import is_session_id
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator
from tests.integration.test_analysis_api import FakeLauncher, RefusingLauncher
from tests.integration.test_analysis_review import Clock, Sleeper
from tests.integration.test_transcript_ingest import AUTH, TOKEN, example
from tests.unit.test_skills_session_analysis import T0

HOLD = (
    "import fcntl, os, sys, time\n"
    "f = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
    "fcntl.flock(f, fcntl.LOCK_EX)\n"
    "os.write(f, str(os.getpid()).encode())\n"
    "print('held', flush=True)\n"
    "time.sleep(60)\n"
)
"""A stand-in for ``skills analyze`` run from the terminal: holds the lock."""


@pytest.fixture
def terminal(tmp_path: Path) -> Iterator["Terminal"]:
    """A terminal run factory; every run it started is killed at the end."""
    runs = Terminal(tmp_path / "analysis")
    yield runs
    runs.kill_all()


class Terminal:
    """Start processes that hold a session's analysis lock."""

    def __init__(self, directory: Path) -> None:
        """Lock files go under ``directory/<session>/``."""
        self.directory = directory
        self.children: List["subprocess.Popen[str]"] = []

    def run(self, session_id: str) -> "subprocess.Popen[str]":
        """Hold ``session_id`` from another process until killed."""
        folder = self.directory / session_id
        folder.mkdir(parents=True, exist_ok=True)
        child = subprocess.Popen(
            [sys.executable, "-c", HOLD, str(folder / "analysis.lock")],
            stdout=subprocess.PIPE, text=True,
        )
        self.children.append(child)
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        return child

    def kill_all(self) -> None:
        """Kill every run still alive."""
        for child in self.children:
            child.kill()
            child.wait()


def stored(folder: Path) -> FileSystemAnalysisStore:
    """A store with a finished analysis of ``demo-es``."""
    store = FileSystemAnalysisStore(str(folder / "analysis"))
    run = new_run("demo-es", T0)
    for step in ("sentences", "reports"):
        run = run.with_step(StepRecord(name=step, status="done"))
    store.save(run)
    store.write("demo-es", "sentences.json", '{"old": true}')
    store.write("demo-es", "report-1.html", "<p>Marta</p>")
    return store


def requests_for(
    folder: Path,
    launcher: FakeLauncher,
    lock: AnalysisLock,
    limits: StartLimits = StartLimits(),
    clock: Optional[Clock] = None,
) -> AnalysisRequests:
    """The API's use case over ``demo-es`` and ``demo-en``."""
    transcripts = FileSystemTranscriptStore(str(folder / "transcripts"))
    transcripts.put("demo-es", example())
    other = example()
    other["session"] = {**other["session"], "id": "demo-en"}
    transcripts.put("demo-en", other)
    return AnalysisRequests(
        AnalysisResults(stored(folder)), transcripts, launcher, lock,
        limits, clock or Clock(),
    )


def client_for(folder: Path, requests: AnalysisRequests) -> TestClient:
    """The HTTP API, deleting a session deletes its analysis."""
    search = SearchService(
        FileSystemIndexRepository(str(folder / "index")), tokenize,
        FakeEmbedder(),
    )
    transcripts = requests.transcripts
    routes = TranscriptRoutes(
        IngestTranscript(transcripts), ShowIngestedTranscript(transcripts),
        TOKEN,
        forget=ForgetIngestedTranscript(transcripts, guard=requests),
        analysis=requests,
    )
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()), routes
    ))


def test_a_terminal_run_refuses_the_delete_until_it_ends(
    tmp_path: Path, terminal: Terminal,
) -> None:
    """409 and nothing deleted while the CLI holds the lock; then 204."""
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    client = client_for(
        tmp_path, requests_for(tmp_path, FakeLauncher(), lock)
    )
    child = terminal.run("demo-es")
    refused = client.delete("/transcripts/demo-es", headers=AUTH)
    assert refused.status_code == 409
    assert f"process {child.pid}" in refused.json()["detail"]
    assert client.get("/transcripts/demo-es", headers=AUTH).status_code \
        == 200
    folder = tmp_path / "analysis" / "demo-es"
    assert (folder / "run.json").is_file()
    assert lock.holder("demo-es") == child.pid
    child.kill()
    child.wait()
    assert client.delete("/transcripts/demo-es", headers=AUTH).status_code \
        == 204
    assert not folder.exists()
    assert client.get("/transcripts/demo-es", headers=AUTH).status_code \
        == 404


def test_no_run_can_take_the_session_while_its_transcript_is_deleted(
    tmp_path: Path,
) -> None:
    """The lock is held until the transcript is gone (#121).

    A terminal run that starts in between finds the session held and,
    once the deletion is over, finds no transcript: it never writes the
    erased analysis again.
    """
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    requests = requests_for(tmp_path, FakeLauncher(), lock)
    transcripts = requests.transcripts
    assert isinstance(transcripts, FileSystemTranscriptStore)
    folder = tmp_path / "analysis" / "demo-es"
    seen: List[Optional[int]] = []
    delete = transcripts.delete

    def deleting(session_id: str) -> bool:
        """Try to start a terminal run during the deletion."""
        seen.append(FileAnalysisLock(str(tmp_path / "analysis"),
                                     pid=7).acquire(session_id))
        assert (folder / "run.json").is_file()
        return delete(session_id)

    transcripts.delete = deleting  # type: ignore[method-assign]
    ForgetIngestedTranscript(transcripts, guard=requests)("demo-es")
    assert seen == [None]
    assert not (folder / "run.json").exists()
    after = FileAnalysisLock(str(tmp_path / "analysis"), pid=8)
    assert after.acquire("demo-es") == 8
    after.release("demo-es")
    with pytest.raises(NotFoundError):
        transcripts.get("demo-es")


def test_a_start_racing_a_deletion_writes_nothing(tmp_path: Path) -> None:
    """The session is deleted right after the first transcript check.

    The start looks again under its lock: no launch, no ``job.json``
    in the erased session (#121 review).
    """
    launcher = FakeLauncher()
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    requests = requests_for(tmp_path, launcher, lock)
    transcripts = requests.transcripts
    assert isinstance(transcripts, FileSystemTranscriptStore)
    real_get = transcripts.get
    forget = ForgetIngestedTranscript(transcripts, guard=requests)
    calls: List[str] = []

    def get_then_delete(session_id: str) -> Any:
        """Answer, then let the deletion run (first call only)."""
        document = real_get(session_id)
        if not calls:
            calls.append(session_id)
            forget(session_id)
        return document

    transcripts.get = get_then_delete  # type: ignore[method-assign]
    with pytest.raises(NotFoundError):
        requests.start("demo-es", job=JobSought(text="Nuevo"))
    folder = tmp_path / "analysis" / "demo-es"
    assert launcher.started == []
    assert not (folder / "job.json").exists()
    assert not (folder / "run.json").exists()


def test_the_launcher_counts_terminal_runs(
    tmp_path: Path, terminal: Terminal,
) -> None:
    """A terminal run of another session takes the one running place."""
    launcher = Sleeper(tmp_path, str(tmp_path / "t"),
                       str(tmp_path / "analysis"), max_running=1, grace=2)
    other = terminal.run("demo-en")
    assert FileAnalysisLock(str(tmp_path / "analysis")).held() == ["demo-en"]
    with pytest.raises(LimitError, match="at most 1"):
        launcher.start("demo-es", False, False)
    other.kill()
    other.wait()
    launcher.start("demo-es", False, False)
    try:
        assert launcher.running("demo-es")
    finally:
        launcher.stop("demo-es")


def test_held_ignores_free_locks_and_other_files(tmp_path: Path) -> None:
    """Released locks, stray files and a missing directory hold nothing."""
    lock = FileAnalysisLock(str(tmp_path / "missing"))
    assert lock.held() == []
    lock = FileAnalysisLock(str(tmp_path))
    assert lock.acquire("s1") is not None
    lock.release("s1")
    (tmp_path / "s2").mkdir()
    (tmp_path / STARTS_FILE).write_text("[]")
    assert lock.held() == []


def test_the_start_log_never_takes_a_session_name(tmp_path: Path) -> None:
    """No session id can name the start log; a session ``starts.json``
    is analysed, held and erased apart from it (#99 review)."""
    assert not is_session_id(STARTS_FILE)
    log = JsonStartLog(str(tmp_path))
    log.save([(5.0, "starts.json")])
    store = FileSystemAnalysisStore(str(tmp_path))
    store.save(new_run("starts.json", T0))
    lock = FileAnalysisLock(str(tmp_path))
    assert lock.acquire("starts.json") is not None
    assert lock.held() == ["starts.json"]
    lock.release("starts.json")
    store.erase("starts.json")
    assert log.load() == [(5.0, "starts.json")]
    assert sorted(p.name for p in tmp_path.iterdir()) == [STARTS_FILE]


def test_a_status_poll_never_makes_the_session_busy(tmp_path: Path) -> None:
    """A reader's instant shared lock is waited for; a holder is not."""
    lock = FileAnalysisLock(str(tmp_path))
    path = tmp_path / "demo-es" / "analysis.lock"
    path.parent.mkdir()
    reader = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(reader, fcntl.LOCK_SH)
        timer = threading.Timer(
            SHARED_WAIT / 5, fcntl.flock, (reader, fcntl.LOCK_UN)
        )
        timer.start()
        assert lock.acquire("demo-es") == lock.pid
        timer.join()
        lock.release("demo-es")
        fcntl.flock(reader, fcntl.LOCK_EX)  # a real holder
        assert FileAnalysisLock(str(tmp_path), pid=7).acquire(
            "demo-es"
        ) is None
    finally:
        os.close(reader)


class SlowLauncher(FakeLauncher):
    """Takes a while to start, so two requests overlap."""

    def start(
        self, session_id: str, restart: bool, synthetic: bool,
        steps: Sequence[str] = (),
    ) -> None:
        """Wait, then start."""
        time.sleep(0.2)
        super().start(session_id, restart, synthetic, steps)


def at_once(
    requests: AnalysisRequests, calls: Sequence[tuple[str, str]]
) -> List[str]:
    """Run ``(session, job)`` starts in parallel; return each outcome."""
    outcomes = [""] * len(calls)
    barrier = threading.Barrier(len(calls))

    def call(index: int, session_id: str, job: str) -> None:
        barrier.wait()
        try:
            requests.start(session_id, steps=["synthesis"],
                           job=JobSought(text=job))
            outcomes[index] = "started"
        except BusyError:
            outcomes[index] = "busy"
        except LimitError:
            outcomes[index] = "limit"

    threads = [threading.Thread(target=call, args=(i, s, j))
               for i, (s, j) in enumerate(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


def test_two_starts_of_one_session_never_mix_their_jobs(
    tmp_path: Path,
) -> None:
    """One is accepted; the refused one never touches ``job.json``."""
    launcher = SlowLauncher()
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    requests = requests_for(tmp_path, launcher, lock)
    outcomes = at_once(requests, [("demo-es", "Primero"),
                                  ("demo-es", "Segundo")])
    assert sorted(outcomes) == ["busy", "started"]
    accepted = "Primero" if outcomes[0] == "started" else "Segundo"
    job = tmp_path / "analysis" / "demo-es" / "job.json"
    assert json.loads(job.read_text())["text"] == accepted
    assert len(launcher.started) == 1


def test_two_starts_cannot_both_pass_the_daily_limit(tmp_path: Path) -> None:
    """With one start a day left, two sessions at once: one is refused."""
    launcher = SlowLauncher()
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    requests = requests_for(tmp_path, launcher, lock, StartLimits(0, 1))
    outcomes = at_once(requests, [("demo-es", "A"), ("demo-en", "B")])
    assert sorted(outcomes) == ["limit", "started"]
    assert len(launcher.started) == 1


def test_the_start_limits_survive_a_server_restart(tmp_path: Path) -> None:
    """A new server reads the starts the previous one kept."""
    clock = Clock()
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    first = requests_for(tmp_path, FakeLauncher(), lock, clock=clock)
    first.starts = JsonStartLog(str(tmp_path / "analysis"))
    first.start("demo-es")
    clock.now += 10
    second = AnalysisRequests(
        first.results, first.transcripts, FakeLauncher(), lock,
        StartLimits(), clock, JsonStartLog(str(tmp_path / "analysis")),
    )
    with pytest.raises(LimitError, match="wait 291 s"):
        second.start("demo-es")
    assert second.starts.load() == [(1000.0, "demo-es")]


def test_an_unreadable_start_log_counts_as_none(tmp_path: Path) -> None:
    """A damaged file is no start; a directory in the way is an error."""
    log = JsonStartLog(str(tmp_path))
    assert log.load() == []
    (tmp_path / STARTS_FILE).write_text("{not json")
    assert log.load() == []
    log.save([(5.0, "s1")])
    assert log.load() == [(5.0, "s1")]
    blocked = JsonStartLog(str(tmp_path / STARTS_FILE / "x"))
    with pytest.raises(Exception, match="Cannot write"):
        blocked.save([])


def test_a_restart_hides_the_previous_results_until_they_are_cleared(
    tmp_path: Path,
) -> None:
    """No old step result or report between the launch and the clear."""
    launcher = FakeLauncher()
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    requests = requests_for(tmp_path, launcher, lock)
    client = client_for(tmp_path, requests)
    step = "/analysis/sessions/demo-es/steps/sentences"
    report = "/analysis/sessions/demo-es/reports/1"
    assert client.get(step, headers=AUTH).json() == {"old": True}
    assert client.post("/analysis/sessions/demo-es", json={"restart": True},
                       headers=AUTH).status_code == 202
    assert client.get(step, headers=AUTH).status_code == 404
    assert client.get(report, headers=AUTH).status_code == 404
    # The child cleared the old run and finished step a of the new one.
    store = requests.results.store
    store.clear("demo-es")
    store.save(new_run("demo-es", T0.replace(year=T0.year + 1)).with_step(
        StepRecord(name="sentences", status="done")
    ))
    store.write("demo-es", "sentences.json", '{"new": true}')
    assert client.get(step, headers=AUTH).json() == {"new": True}


class PausedLauncher(FakeLauncher):
    """Stops inside ``start`` until told to go on."""

    def __init__(self) -> None:
        """Not entered yet."""
        super().__init__()
        self.entered = threading.Event()
        self.go = threading.Event()

    def start(
        self, session_id: str, restart: bool, synthetic: bool,
        steps: Sequence[str] = (),
    ) -> None:
        """Say it entered, wait, then start."""
        self.entered.set()
        assert self.go.wait(5)
        super().start(session_id, restart, synthetic, steps)


def test_a_status_poll_during_a_restart_keeps_the_old_run_hidden(
    tmp_path: Path,
) -> None:
    """Between ``hide`` and the launch a poll must not show it (#121)."""
    launcher = PausedLauncher()
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    requests = requests_for(tmp_path, launcher, lock)
    starting = threading.Thread(
        target=requests.start, args=("demo-es",), kwargs={"restart": True},
    )
    starting.start()
    try:
        assert launcher.entered.wait(5)
        assert requests.status("demo-es")["active"] is False
        with pytest.raises(NotFoundError, match="no result now"):
            requests.results.result("demo-es", "sentences")
    finally:
        launcher.go.set()
        starting.join()
    assert requests.status("demo-es")["active"] is True
    with pytest.raises(NotFoundError, match="no result now"):
        requests.results.result("demo-es", "sentences")


def test_a_restart_that_never_ran_shows_the_results_again(
    tmp_path: Path,
) -> None:
    """Refused, or its process died before clearing: the old run is back."""
    lock = FileAnalysisLock(str(tmp_path / "analysis"))
    refused = requests_for(tmp_path, RefusingLauncher(), lock)
    with pytest.raises(LimitError):
        refused.start("demo-es", restart=True)
    assert refused.results.result("demo-es", "sentences") == {"old": True}
    launcher = FakeLauncher()
    requests = requests_for(tmp_path, launcher, lock)
    requests.start("demo-es", restart=True)
    with pytest.raises(Exception, match="no result now"):
        requests.results.result("demo-es", "sentences")
    launcher.active.clear()  # died before clearing anything
    assert requests.status("demo-es")["active"] is False
    assert requests.results.result("demo-es", "sentences") == {"old": True}


def test_a_run_of_a_session_without_transcript_leaves_no_lock_file(
    tmp_path: Path,
) -> None:
    """The lock is taken first; a missing session leaves nothing (#121)."""
    from tests.unit.test_skills_session_analysis import build

    directory = tmp_path / "analysis"
    lock = FileAnalysisLock(str(directory))
    analysis = build(lock=lock)
    with pytest.raises(NotFoundError):
        analysis.run("tipo-errado")
    assert not (directory / "tipo-errado").exists()
    kept = directory / "s2"
    kept.mkdir(parents=True)
    (kept / "run.json").write_text("{}")
    with pytest.raises(NotFoundError):
        analysis.run("s2")
    assert sorted(p.name for p in kept.iterdir()) == [
        "analysis.lock", "run.json",
    ]
    assert lock.holder("s2") is None and lock.acquire("s2") is not None
    lock.release("s2")
