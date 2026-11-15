"""The RAG's ``/analysis/sessions`` routes and the child-process launcher."""

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.outbound.fs_analysis_store import FileSystemAnalysisStore
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.adapters.outbound.subprocess_analysis import (
    SubprocessAnalysisLauncher,
)
from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.application.skills.session_analysis import (
    AnalysisRequests,
    AnalysisResults,
)
from src.config import SkillsSettings
from src.domain.errors import BusyError, RagError
from src.domain.skills.analysis_run import StepRecord, new_run
from src.domain.text_analysis import tokenize
from src.skills_bootstrap import SkillsContainer
from tests.fakes import FakeEmbedder, FakeGenerator
from tests.integration.test_transcript_ingest import AUTH, TOKEN, example
from tests.unit.test_skills_session_analysis import T0


class FakeLauncher:
    """Records starts; a session is running until ``finish``."""

    def __init__(self) -> None:
        """Nothing started."""
        self.started: List[Tuple[str, bool, bool]] = []
        self.active: set[str] = set()

    def start(self, session_id: str, restart: bool, synthetic: bool) -> None:
        """Remember the start and mark the session running."""
        self.started.append((session_id, restart, synthetic))
        self.active.add(session_id)

    def running(self, session_id: str) -> bool:
        """Tell whether the session was started and not finished."""
        return session_id in self.active


def client_for(folder: Path, launcher: FakeLauncher) -> TestClient:
    """The API with a stored transcript and one finished analysis."""
    transcripts = FileSystemTranscriptStore(str(folder / "transcripts"))
    transcripts.put("demo-es", example())
    store = FileSystemAnalysisStore(str(folder / "analysis"))
    run = new_run("demo-es", T0).with_step(
        StepRecord(name="sentences", status="done", counts={"sentences": 4})
    )
    store.save(run)
    store.write("demo-es", "sentences.json", json.dumps({"sentences": []}))
    store.write("demo-es", "report-1.html", "<!doctype html><p>Marta</p>")
    search = SearchService(
        FileSystemIndexRepository(str(folder / "index")), tokenize,
        FakeEmbedder(),
    )
    routes = TranscriptRoutes(
        IngestTranscript(transcripts), ShowIngestedTranscript(transcripts),
        TOKEN,
        analysis=AnalysisRequests(
            AnalysisResults(store), transcripts, launcher
        ),
    )
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()), routes
    ))


def test_every_route_needs_the_token(tmp_path: Path) -> None:
    """Without the bearer token: 401, nothing started."""
    launcher = FakeLauncher()
    client = client_for(tmp_path, launcher)
    for method, path in (
        ("POST", "/analysis/sessions/demo-es"),
        ("GET", "/analysis/sessions/demo-es"),
        ("GET", "/analysis/sessions/demo-es/steps/sentences"),
        ("GET", "/analysis/sessions/demo-es/reports/1"),
    ):
        assert client.request(method, path).status_code == 401
    assert launcher.started == []


def test_start_status_results_and_report(tmp_path: Path) -> None:
    """Start once (409 while running), then read what was stored."""
    launcher = FakeLauncher()
    client = client_for(tmp_path, launcher)
    started = client.post(
        "/analysis/sessions/demo-es", json={"synthetic": True}, headers=AUTH
    )
    assert started.status_code == 202
    assert started.json() == {"session_id": "demo-es", "active": True}
    assert launcher.started == [("demo-es", False, True)]
    again = client.post("/analysis/sessions/demo-es", headers=AUTH)
    assert again.status_code == 409
    status = client.get("/analysis/sessions/demo-es", headers=AUTH).json()
    assert status["active"] is True
    assert status["steps"][0]["counts"] == {"sentences": 4}
    launcher.active.clear()
    status = client.get("/analysis/sessions/demo-es", headers=AUTH).json()
    assert status["active"] is False
    step = client.get(
        "/analysis/sessions/demo-es/steps/sentences", headers=AUTH
    )
    assert step.json() == {"sentences": []}
    report = client.get("/analysis/sessions/demo-es/reports/1", headers=AUTH)
    assert report.status_code == 200
    assert report.text.startswith("<!doctype html>")
    assert report.headers["content-type"].startswith("text/html")
    csp = report.headers["content-security-policy"]
    assert csp.startswith("sandbox;") and "script-src" not in csp
    assert "default-src 'none'" in csp
    assert report.headers["content-disposition"].startswith("attachment;")
    assert report.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("method, path, status", [
    ("POST", "/analysis/sessions/unknown", 404),
    ("POST", "/analysis/sessions/bad%20id", 400),
    ("GET", "/analysis/sessions/unknown", 404),
    ("GET", "/analysis/sessions/..%2F..%2Fetc", 404),
    ("GET", "/analysis/sessions/demo-es/steps/run", 400),
    ("GET", "/analysis/sessions/demo-es/steps/acts", 404),
    ("GET", "/analysis/sessions/demo-es/steps/..%2Frun", 404),
    ("GET", "/analysis/sessions/demo-es/reports/0", 400),
    ("GET", "/analysis/sessions/demo-es/reports/7", 404),
    ("GET", "/analysis/sessions/demo-es/reports/x", 422),
    ("GET", "/analysis/sessions/.hidden/reports/1", 400),
])
def test_ids_steps_and_reports_are_checked(
    tmp_path: Path, method: str, path: str, status: int
) -> None:
    """Unknown, invalid or traversing names never reach a file."""
    client = client_for(tmp_path, FakeLauncher())
    assert client.request(method, path, headers=AUTH).status_code == status


def test_a_bad_body_is_refused(tmp_path: Path) -> None:
    """Only the two booleans are accepted."""
    launcher = FakeLauncher()
    client = client_for(tmp_path, launcher)
    answer = client.post(
        "/analysis/sessions/demo-es", content=b"{not json", headers=AUTH
    )
    assert answer.status_code == 422
    assert launcher.started == []


def test_a_launcher_failure_is_a_server_error(tmp_path: Path) -> None:
    """The launcher cannot start: 500, no detail lost."""
    launcher = FakeLauncher()

    def broken(session_id: str, restart: bool, synthetic: bool) -> None:
        """Fail to start."""
        raise RagError("no python")

    launcher.start = broken  # type: ignore[method-assign]
    client = client_for(tmp_path, launcher)
    answer = client.post("/analysis/sessions/demo-es", headers=AUTH)
    assert answer.status_code == 500


class SleepingLauncher(SubprocessAnalysisLauncher):
    """The real launcher running a short sleep instead of the analysis."""

    def command(
        self, session_id: str, restart: bool, synthetic: bool
    ) -> List[str]:
        """Sleep a moment and say which session it was."""
        return [sys.executable, "-c",
                f"import time; print({session_id!r}); time.sleep(0.4)"]


def test_the_launcher_runs_one_child_per_session(tmp_path: Path) -> None:
    """Running while alive, busy on a second start, free once ended."""
    launcher = SleepingLauncher(
        tmp_path, str(tmp_path / "t"), str(tmp_path / "a")
    )
    assert not launcher.running("s1")
    launcher.start("s1", False, False)
    assert launcher.running("s1")
    with pytest.raises(BusyError):
        launcher.start("s1", True, False)
    deadline = time.monotonic() + 10
    while launcher.running("s1") and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not launcher.running("s1")
    assert (tmp_path / "a" / "s1" / "analyze.log").read_text().strip() == (
        "s1"
    )
    with pytest.raises(RagError):
        launcher.start("../x", False, False)


def test_the_launcher_command_and_a_failed_start(tmp_path: Path) -> None:
    """The child gets both directories; a missing python is a RagError."""
    launcher = SubprocessAnalysisLauncher(
        tmp_path, str(tmp_path / "t"), str(tmp_path / "a"),
        python="/no/such/python",
    )
    args = launcher.command("s1", True, True)
    assert args[1:4] == ["-m", "src", "skills"]
    assert args[args.index("--transcripts_dir") + 1] == str(
        tmp_path / "t"
    )
    assert args[-4:] == ["--session", "s1", "--restart", "--synthetic"]
    with pytest.raises(RagError, match="Cannot start"):
        launcher.start("s1", False, False)


def test_the_container_builds_the_api_use_case(tmp_path: Path) -> None:
    """The composition root wires store, transcripts and launcher."""
    settings = SkillsSettings(analysis_dir=str(tmp_path / "a"))
    store = FileSystemTranscriptStore(str(tmp_path / "t"))
    requests = SkillsContainer(settings).analysis_requests(
        store, str(tmp_path / "t")
    )
    assert isinstance(requests.launcher, SubprocessAnalysisLauncher)
    assert requests.launcher.analysis_dir == tmp_path / "a"
