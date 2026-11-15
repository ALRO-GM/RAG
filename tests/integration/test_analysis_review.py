"""Review fixes of #88: retention, limits, cancel, time-out, environment."""

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.inbound.skills_cli import exit_on_sigterm
from src.adapters.outbound.fs_analysis_store import (
    FileSystemAnalysisStore,
    FileSystemSessionOutputs,
)
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.adapters.outbound.subprocess_analysis import (
    SubprocessAnalysisLauncher,
    child_environment,
    descendants,
    kill_groups,
)
from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.ingest import (
    ForgetIngestedTranscript,
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.application.skills.session_analysis import (
    AnalysisRequests,
    AnalysisResults,
    StartLimits,
)
from src.domain.errors import BusyError, LimitError, NotFoundError
from src.domain.skills.analysis_run import StepRecord, new_run
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryAnalysisLock
from tests.integration.test_analysis_api import FakeLauncher
from tests.integration.test_transcript_ingest import AUTH, TOKEN, example
from tests.unit.test_skills_session_analysis import T0


class Clock:
    """A settable clock in seconds."""

    def __init__(self) -> None:
        """Start at 1000 s."""
        self.now = 1000.0

    def __call__(self) -> float:
        """Return the time."""
        return self.now


def setup(
    folder: Path,
    launcher: FakeLauncher,
    lock: Optional[InMemoryAnalysisLock] = None,
    limits: StartLimits = StartLimits(),
    clock: Optional[Clock] = None,
) -> TestClient:
    """The API with a stored transcript, a finished analysis and outputs."""
    transcripts = FileSystemTranscriptStore(str(folder / "transcripts"))
    transcripts.put("demo-es", example())
    store = FileSystemAnalysisStore(str(folder / "analysis"))
    run = new_run("demo-es", T0)
    for step in ("sentences", "reports"):
        run = run.with_step(StepRecord(name=step, status="done"))
    store.save(run)
    store.write("demo-es", "sentences.json", "{}")
    store.write("demo-es", "summary.json", "{}")
    store.write("demo-es", "report-1.html", "<p>Marta</p>")
    out = folder / "out"
    out.mkdir()
    (out / "demo-es.assessment.json").write_text(json.dumps(
        {"scorecards": [{"participant": "Marta"}]}))
    (out / "demo-es.scorecard.md").write_text("x")
    (out / "demo-es.Marta.report.html").write_text("x")
    (out / "demo-es2.assessment.json").write_text("{}")  # another session
    (folder / "transcripts" / "demo-es.acts.json").write_text("{}")
    requests = AnalysisRequests(
        AnalysisResults(store), transcripts, launcher,
        lock or InMemoryAnalysisLock(), limits, clock or Clock(),
    )
    search = SearchService(
        FileSystemIndexRepository(str(folder / "index")), tokenize,
        FakeEmbedder(),
    )
    routes = TranscriptRoutes(
        IngestTranscript(transcripts), ShowIngestedTranscript(transcripts),
        TOKEN,
        forget=ForgetIngestedTranscript(transcripts, [
            requests,
            FileSystemSessionOutputs(str(out), str(folder / "transcripts")),
        ]),
        analysis=requests,
    )
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()), routes
    ))


ROUTES = (
    ("GET", "/analysis/sessions/demo-es"),
    ("GET", "/analysis/sessions/demo-es/steps/sentences"),
    ("GET", "/analysis/sessions/demo-es/steps/reports"),
    ("GET", "/analysis/sessions/demo-es/reports/1"),
)


def test_deleting_the_transcript_deletes_the_analysis(tmp_path: Path) -> None:
    """Every analysis route 404s, the directory and outputs are gone."""
    launcher = FakeLauncher()
    client = setup(tmp_path, launcher)
    for method, path in ROUTES:
        assert client.request(method, path, headers=AUTH).status_code == 200
    gone = client.delete("/transcripts/demo-es", headers=AUTH)
    assert gone.status_code == 204
    for method, path in ROUTES:
        assert client.request(method, path, headers=AUTH).status_code == 404
    assert not (tmp_path / "analysis" / "demo-es").exists()
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "demo-es2.assessment.json"
    ]
    assert not (tmp_path / "transcripts" / "demo-es.acts.json").exists()
    # Deleting again: no transcript (404), nothing left to erase.
    again = client.delete("/transcripts/demo-es", headers=AUTH)
    assert again.status_code == 404


def test_deleting_while_the_analysis_runs_stops_it(tmp_path: Path) -> None:
    """The running child is stopped before its files are removed."""
    launcher = FakeLauncher()
    client = setup(tmp_path, launcher)
    assert client.post(
        "/analysis/sessions/demo-es", headers=AUTH
    ).status_code == 202
    assert client.delete("/transcripts/demo-es", headers=AUTH).status_code \
        == 204
    assert launcher.stopped == ["demo-es"]
    assert not (tmp_path / "analysis" / "demo-es").exists()


def test_status_right_after_a_start_is_active_with_no_steps(
    tmp_path: Path,
) -> None:
    """Before the worker writes its record the page keeps polling."""
    launcher = FakeLauncher()
    client = setup(tmp_path, launcher)
    store = FileSystemAnalysisStore(str(tmp_path / "analysis"))
    store.erase("demo-es")
    assert client.get(
        "/analysis/sessions/demo-es", headers=AUTH
    ).status_code == 404
    client.post("/analysis/sessions/demo-es", headers=AUTH)
    status = client.get("/analysis/sessions/demo-es", headers=AUTH).json()
    assert status == {"session_id": "demo-es", "status": "running",
                      "steps": [], "open_issues": [], "active": True}


def test_start_limits_per_session_and_per_day(tmp_path: Path) -> None:
    """Too soon after the last start, or one too many: 429 with why."""
    clock = Clock()
    launcher = FakeLauncher()
    client = setup(tmp_path, launcher, limits=StartLimits(300, 2),
                   clock=clock)
    assert client.post(
        "/analysis/sessions/demo-es", headers=AUTH
    ).status_code == 202
    launcher.active.clear()
    clock.now += 10
    soon = client.post("/analysis/sessions/demo-es", headers=AUTH)
    assert soon.status_code == 429
    assert "wait 291 s" in soon.json()["detail"]
    clock.now += 300
    assert client.post(
        "/analysis/sessions/demo-es", headers=AUTH
    ).status_code == 202
    launcher.active.clear()
    clock.now += 400
    full = client.post("/analysis/sessions/demo-es", headers=AUTH)
    assert full.status_code == 429
    assert "last 24 hours" in full.json()["detail"]
    clock.now += 86400
    assert client.post(
        "/analysis/sessions/demo-es", headers=AUTH
    ).status_code == 202


def test_cancel_route(tmp_path: Path) -> None:
    """204 when this server's child is stopped; 409 from the CLI; 404."""
    launcher = FakeLauncher()
    lock = InMemoryAnalysisLock()
    client = setup(tmp_path, launcher, lock)
    path = "/analysis/sessions/demo-es/run"
    assert client.delete(path).status_code == 401
    assert client.delete(path, headers=AUTH).status_code == 404
    client.post("/analysis/sessions/demo-es", headers=AUTH)
    assert client.delete(path, headers=AUTH).status_code == 204
    assert launcher.stopped == ["demo-es"]
    lock.taken["demo-es"] = 55
    terminal = client.delete(path, headers=AUTH)
    assert terminal.status_code == 409
    assert "command line" in terminal.json()["detail"]


class Sleeper(SubprocessAnalysisLauncher):
    """The real launcher running a sleep that reports its environment."""

    def command(
        self, session_id: str, restart: bool, synthetic: bool,
        steps: Sequence[str] = (),
    ) -> List[str]:
        """Print the variables seen, then sleep."""
        return [sys.executable, "-c",
                "import os, sys, time; "
                "print(sorted(os.environ)); sys.stdout.flush(); "
                "time.sleep(30)"]


def wait_until(condition: object, seconds: float = 10.0) -> bool:
    """Poll ``condition`` until it holds or ``seconds`` pass."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():  # type: ignore[operator]
            return True
        time.sleep(0.05)
    return False


def test_the_launcher_limits_stops_times_out_and_hides_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One child at a time; stop kills its group; the time-out stops it."""
    monkeypatch.setenv("RAG_TRANSCRIPTS_TOKEN", "secret-token")
    monkeypatch.setenv("SOME_API_KEY", "k")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    launcher = Sleeper(tmp_path, str(tmp_path / "t"), str(tmp_path / "a"),
                       timeout=60, max_running=1, grace=2)
    launcher.start("s1", False, False)
    with pytest.raises(LimitError, match="at most 1"):
        launcher.start("s2", False, False)
    with pytest.raises(BusyError):
        launcher.start("s1", False, False)
    log = tmp_path / "a" / "s1" / "analyze.log"
    assert wait_until(lambda: log.read_text().strip() != "")
    seen = log.read_text()
    assert "HF_HUB_OFFLINE" in seen
    assert "RAG_TRANSCRIPTS_TOKEN" not in seen and "SOME_API_KEY" not in seen
    assert launcher.stop("s1") is True
    assert not launcher.running("s1")
    assert launcher.stop("s1") is False
    quick = Sleeper(tmp_path, str(tmp_path / "t"), str(tmp_path / "a"),
                    timeout=0.3, grace=2)
    quick.start("s3", False, False)
    assert wait_until(lambda: not quick.running("s3"))
    last = Sleeper(tmp_path, str(tmp_path / "t"), str(tmp_path / "a"),
                   grace=2)
    last.start("s4", False, False)
    last.stop_all()
    assert not last.running("s4")
    assert child_environment({"A_TOKEN": "x", "PATH": "/bin"}) == {
        "PATH": "/bin"
    }


def test_forget_without_analysis_and_outputs_is_harmless(
    tmp_path: Path,
) -> None:
    """Nothing to erase: no error; a bad id is refused by the store."""
    store = FileSystemAnalysisStore(str(tmp_path / "analysis"))
    store.erase("nothing")
    store.clear("nothing")
    outputs = FileSystemSessionOutputs(str(tmp_path / "o"),
                                       str(tmp_path / "t"))
    outputs.erase("nothing")
    with pytest.raises(Exception):
        outputs.erase("../x")
    transcripts = FileSystemTranscriptStore(str(tmp_path / "t"))
    with pytest.raises(NotFoundError):
        ForgetIngestedTranscript(transcripts, [outputs])("nothing")


class Spawner(SubprocessAnalysisLauncher):
    """The real launcher running a child that starts a judge-like helper.

    The helper (a grandchild) runs in a session of its own, as ``claude
    -p`` does, and writes its pid to ``helper.pid``.
    """

    def command(
        self, session_id: str, restart: bool, synthetic: bool,
        steps: Sequence[str] = (),
    ) -> List[str]:
        """Start a sleeping grandchild in a new session, then sleep."""
        target = self.analysis_dir / "helper.pid"
        return [sys.executable, "-c",
                "import subprocess, sys, time; "
                "p = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'], start_new_session=True); "
                f"open({str(target)!r}, 'w').write(str(p.pid)); "
                "time.sleep(60)"]


def gone(pid: int) -> bool:
    """Tell whether ``pid`` has ended (absent or a zombie)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rpartition(")")[2].split()[0] == "Z"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(),
                    reason="needs /proc")
def test_stopping_kills_a_helper_in_its_own_session(tmp_path: Path) -> None:
    """A cancelled analysis leaves no judge process behind."""
    launcher = Spawner(tmp_path, str(tmp_path / "t"), str(tmp_path / "a"),
                       grace=2)
    launcher.start("s1", False, False)
    marker = tmp_path / "a" / "helper.pid"
    assert wait_until(lambda: marker.exists() and marker.read_text() != "")
    helper = int(marker.read_text())
    assert not gone(helper)
    assert launcher.stop("s1") is True
    assert wait_until(lambda: gone(helper))


def test_descendants_and_kill_groups_read_proc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tree comes from ``stat``; zombies, odd entries and reused pids
    are skipped; no ``/proc`` means no descendants."""
    proc = tmp_path / "proc"
    for pid, text in {
        10: "10 (a b) S 1 10 x", 11: "11 (judge)) S 10 11 x",
        12: "12 (helper) S 11 11 x", 13: "13 (dead) Z 10 13 x",
        14: "14 (x) S", 99: "99 (other) S 1 99 x",
    }.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "stat").write_text(text)
    (proc / "self").mkdir()
    (proc / "15").mkdir()  # no stat file
    assert sorted(descendants(10, proc)) == [(11, 11), (12, 11)]
    assert descendants(10, tmp_path / "missing") == []
    killed: List[int] = []
    monkeypatch.setattr(
        "src.adapters.outbound.subprocess_analysis.os.killpg",
        lambda group, sig: killed.append(group),
    )
    kill_groups(10, [(11, 11), (12, 11), (16, 16), (99, 98), (10, 10)], proc)
    assert killed == [11, 11]

    def refuse(group: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(
        "src.adapters.outbound.subprocess_analysis.os.killpg", refuse
    )
    kill_groups(10, [(11, 11)], proc)


def test_sigterm_becomes_a_clean_exit_while_analysing() -> None:
    """SIGTERM raises SystemExit(143) inside the block, so ``finally``
    blocks (the judge's group kill) run; the old handler comes back."""
    before = signal.getsignal(signal.SIGTERM)
    cleaned: List[str] = []
    with pytest.raises(SystemExit) as stopped:
        with exit_on_sigterm():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
            finally:
                cleaned.append("judge killed")
    assert stopped.value.code == 143
    assert cleaned == ["judge killed"]
    assert signal.getsignal(signal.SIGTERM) == before
    outcome: List[str] = []

    def in_thread() -> None:
        with exit_on_sigterm():
            outcome.append("ran")

    worker = threading.Thread(target=in_thread)
    worker.start()
    worker.join()
    assert outcome == ["ran"]
