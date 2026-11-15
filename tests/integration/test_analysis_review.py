"""Review fixes of #88: retention, limits, cancel, time-out, environment."""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
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
        self, session_id: str, restart: bool, synthetic: bool
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
