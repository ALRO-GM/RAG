"""The pipeline settings on the RAG side: the JSON file, ``GET/PUT
/settings`` and the consumers that read them when a job starts (the
analysis launcher's flags, the live reasoner's effort, the CLI)."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.adapters.outbound.json_settings_store import JsonSettingsStore
from src.adapters.outbound.subprocess_analysis import (
    SubprocessAnalysisLauncher,
    option_flags,
)
from src.application.answering import AnswerService
from src.application.pipeline_settings import PipelineSettings
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.application.skills.session_analysis import (
    AnalysisRequests,
    AnalysisResults,
    StartLimits,
)
from src.adapters.outbound.fs_analysis_store import FileSystemAnalysisStore
from src.config import SkillsSettings
from src.domain.errors import RagError
from src.domain.pipeline_settings import (
    PRESETS,
    RunChoices,
    SettingsDocument,
)
from src.domain.skills.claude_models import ClaudeModels
from src.domain.text_analysis import tokenize
from src.skills_bootstrap import SkillsContainer
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryAnalysisLock
from tests.integration.test_transcript_ingest import AUTH, TOKEN, example

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)


def schema_validator() -> Draft202012Validator:
    """The ``settings.v1`` contract."""
    schema = json.loads(
        (ROOT / "contracts" / "settings.v1.schema.json").read_text("utf-8")
    )
    return Draft202012Validator(schema)


def test_the_store_writes_a_contract_document_and_reads_it_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data" / "settings.json"
    store = JsonSettingsStore(str(path))
    assert store.load() is None
    saved = store.update(lambda _: SettingsDocument(
        revision=1, updated_at=NOW, preset="cost", choices=PRESETS["cost"],
    ))
    on_disk = json.loads(path.read_text("utf-8"))
    schema_validator().validate(on_disk)
    assert store.load() == saved
    path.write_text("{not json", encoding="utf-8")
    assert store.load() is None, "an unreadable file reads as nothing saved"


def test_a_failed_change_leaves_the_file_as_it_was(tmp_path: Path) -> None:
    store = JsonSettingsStore(str(tmp_path / "settings.json"))
    store.update(lambda _: SettingsDocument(revision=1))

    def refuse(_: Optional[SettingsDocument]) -> SettingsDocument:
        raise RagError("no")

    with pytest.raises(RagError):
        store.update(refuse)
    loaded = store.load()
    assert loaded is not None and loaded.revision == 1


class Launcher:
    """Records what each start would pass to ``skills analyze``."""

    def __init__(self) -> None:
        """Nothing started."""
        self.models: List[Optional[ClaudeModels]] = []
        self.options: List[Optional[RunChoices]] = []

    def start(
        self, session_id: str, restart: bool, synthetic: bool,
        steps: Sequence[str] = (),
        models: Optional[ClaudeModels] = None,
        options: Optional[RunChoices] = None,
    ) -> None:
        """Remember the models and options."""
        self.models.append(models)
        self.options.append(options)

    def running(self, session_id: str) -> bool:
        """Nothing keeps running."""
        return False

    def stop(self, session_id: str) -> bool:
        """Nothing to stop."""
        return False


def api(tmp_path: Path, launcher: Launcher) -> TestClient:
    """The RAG API with the settings and the analysis routes."""
    transcripts = FileSystemTranscriptStore(str(tmp_path / "transcripts"))
    transcripts.put("demo-es", example())
    settings = PipelineSettings(
        JsonSettingsStore(str(tmp_path / "settings.json")), lambda: NOW,
    )
    search = SearchService(
        FileSystemIndexRepository(str(tmp_path / "index")), tokenize,
        FakeEmbedder(),
    )
    routes = TranscriptRoutes(
        IngestTranscript(transcripts), ShowIngestedTranscript(transcripts),
        TOKEN,
        analysis=AnalysisRequests(
            AnalysisResults(FileSystemAnalysisStore(str(tmp_path / "an"))),
            transcripts, launcher, InMemoryAnalysisLock(),
            StartLimits(0.0, 20), settings=settings.current,
        ),
        settings=settings,
    )
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()), routes
    ))


def test_the_routes_need_the_token_and_validate(tmp_path: Path) -> None:
    client = api(tmp_path, Launcher())
    assert client.get("/settings").status_code == 401
    assert client.put("/settings", json={"preset": "cost"}).status_code == 401
    shown = client.get("/settings", headers=AUTH).json()
    assert shown["stored"] is False and shown["preset"] == "quality"
    assert {s["id"] for s in shown["catalogue"]["steps"]} == set(
        shown["choices"]
    )
    bad = client.put("/settings", headers=AUTH,
                     json={"choices": {"analysis.judge": "rm -rf /"}})
    assert bad.status_code == 422 and "Invalid option" in bad.json()["detail"]
    assert client.put("/settings", headers=AUTH,
                      content=b"[1]").status_code == 422
    assert client.put("/settings", headers=AUTH,
                      content=b"{x").status_code == 422
    assert client.put("/settings", headers=AUTH,
                      content=b"{" + b" " * 9000 + b"}").status_code == 413
    saved = client.put("/settings", headers=AUTH, json={"preset": "cost"})
    assert saved.status_code == 200 and saved.json()["revision"] == 1
    stale = client.put("/settings", headers=AUTH,
                       json={"choices": {"live.asr": "cpu"}, "revision": 0})
    assert stale.status_code == 409
    on_disk = json.loads((tmp_path / "settings.json").read_text("utf-8"))
    schema_validator().validate(on_disk)


def test_an_analysis_start_reads_the_saved_settings(tmp_path: Path) -> None:
    launcher = Launcher()
    client = api(tmp_path, launcher)
    assert client.post("/analysis/sessions/demo-es", headers=AUTH,
                       json={}).status_code == 202
    assert launcher.models == [None] and launcher.options == [None]
    client.put("/settings", headers=AUTH, json={"preset": "cost"})
    client.put("/settings", headers=AUTH,
               json={"choices": {"analysis.act_tagger": "combined"}})
    assert client.post("/analysis/sessions/demo-es", headers=AUTH, json={
        "claude_models": {"synthesis": "opus"},
    }).status_code == 202
    models = launcher.models[-1]
    assert models is not None and models.chosen() == {
        "confirmation": "haiku", "assessment": "haiku", "synthesis": "opus",
    }, "the start's own choice wins over the saved one"
    assert launcher.options[-1] == RunChoices(
        act_tagger="combined", local_preference=("Qwen/Qwen3-4B",)
    )


def test_the_launcher_passes_only_allow_listed_flags(tmp_path: Path) -> None:
    launcher = SubprocessAnalysisLauncher(
        tmp_path, str(tmp_path / "t"), str(tmp_path / "a"),
        python=sys.executable,
    )
    command = launcher.command("demo-es", False, False, options=RunChoices(
        act_tagger="rules", local_preference=("Qwen/Qwen3-4B",),
    ))
    assert '--act_tagger="rules"' in command
    assert '--local_preference="Qwen/Qwen3-4B"' in command
    assert command.index('--act_tagger="rules"') < command.index("analyze")
    assert option_flags(None) == [] and option_flags(RunChoices()) == []
    with pytest.raises(RagError):
        option_flags(RunChoices(act_tagger="--evil"))
    with pytest.raises(RagError):
        option_flags(RunChoices(local_preference=("x; rm",)))


def test_the_skills_cli_takes_the_tagger_and_local_preference() -> None:
    from src.adapters.inbound.cli import RagCLI

    skills = RagCLI().skills(act_tagger="combined",
                             local_preference="Qwen/Qwen3-4B")
    settings = skills._app().settings
    assert settings.act_tagger == "combined"
    assert settings.local_preference == ("Qwen/Qwen3-4B",)


def test_the_live_effort_follows_the_saved_settings(tmp_path: Path) -> None:
    container = SkillsContainer(SkillsSettings(
        settings_path=str(tmp_path / "settings.json"),
        judge_state_path=str(tmp_path / "judge.json"),
    ))
    assert container.live_effort() == "low"
    first = container.warm_model("haiku")
    container.pipeline_settings.change(
        {"choices": {"live.suggester_effort": "high"}}
    )
    assert container.live_effort() == "high"
    second = container.warm_model("haiku")
    assert second is not first
    command = second._judge.command("system")
    assert command[command.index("--effort") + 1] == "high"
    assert container.warm_model("haiku") is second
    judge = json.loads((tmp_path / "judge.json").read_text("utf-8"))
    assert judge["mode"] == "claude_code", "the judge record follows"


def test_uploads_are_read_with_the_saved_ocr_and_captions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import bootstrap
    from src.application.knowledge.chosen_extractor import ChosenExtractor
    from src.adapters.outbound.document_extractors import (
        DocumentExtractors,
    )
    from src.bootstrap import Container
    from src.config import Settings

    monkeypatch.setattr(bootstrap, "docling_available", lambda: True)
    plain = Container(Settings())
    assert plain.upload_reader is plain.document_extractors
    saved = {"knowledge.ocr": "on", "knowledge.captions": "claude"}
    linked = Container(Settings())
    linked.pipeline_choices = lambda: saved
    reader = linked.upload_reader
    assert isinstance(reader, ChosenExtractor)
    chosen = reader.current()
    assert isinstance(chosen, DocumentExtractors)
    assert type(chosen.captioner).__name__ == "ClaudeCodeCaptioner"
    assert chosen.layout is linked.document_extractors.layout
    saved["knowledge.ocr"] = "off"
    off = reader.current()
    assert isinstance(off, DocumentExtractors) and off.layout is None
