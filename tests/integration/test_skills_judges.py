"""Judge adapters: Claude Code (fake subprocess), local judge, state file,
and the ``/analysis/judge`` routes of the RAG."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge, short
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.adapters.outbound.json_judge_state import JsonJudgeStateStore
from src.adapters.outbound.local_judge import LocalJudge, cuda_free_gb
from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.application.skills.judges import JudgeSelection
from src.domain.errors import (
    CorruptedDataError,
    ModelAuthError,
    ModelError,
    QuotaExhaustedError,
)
from src.domain.runtime import ModelOption, RuntimePlan
from src.domain.skills.judging import JudgeState
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator

TOKEN = "t" * 40


class FakeRunner:
    """Record ``claude`` calls and answer with a scripted process."""

    def __init__(
        self,
        stdout: str = "",
        returncode: int = 0,
        stderr: str = "",
        error: Optional[BaseException] = None,
    ) -> None:
        """Script the answer (or the exception) of every call."""
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self, command: Sequence[str], stdin: str, cwd: str,
        env: Dict[str, str], timeout: float,
    ) -> "subprocess.CompletedProcess[str]":
        """Record the call; check the working directory is empty."""
        self.calls.append({
            "command": list(command), "stdin": stdin, "cwd": cwd,
            "env": dict(env), "timeout": timeout,
            "listing": os.listdir(cwd),
        })
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(
            list(command), self.returncode, self.stdout, self.stderr
        )


def result(text: str = '{"act": "respuesta"}', **extra: Any) -> str:
    """Return ``claude -p --output-format json`` output."""
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": text, "api_error_status": None, **extra,
    })


ENVIRON = {
    "PATH": "/usr/bin", "HOME": "/home/x", "LANG": "es_ES.UTF-8",
    "CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": "s",
    "ANTHROPIC_API_KEY": "secret", "RAG_TRANSCRIPTS_TOKEN": TOKEN,
}


def judge(runner: FakeRunner, **kwargs: Any) -> ClaudeCodeJudge:
    """A Claude Code judge over a fake runner and a fixed environment."""
    return ClaudeCodeJudge(environ=ENVIRON, runner=runner, **kwargs)


def test_claude_code_runs_isolated_with_the_text_on_stdin() -> None:
    runner = FakeRunner(result())
    model = judge(runner, model="sonnet", timeout=30, effort="low")
    assert model.complete("SYSTEM", "Frase privada", 50) == (
        '{"act": "respuesta"}'
    )
    call = runner.calls[0]
    command = call["command"]
    for flag in ("-p", "--safe-mode", "--strict-mcp-config",
                 "--disable-slash-commands", "--no-session-persistence"):
        assert flag in command
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--system-prompt") + 1] == "SYSTEM"
    assert command[command.index("--effort") + 1] == "low"
    assert "--bare" not in command and "--mcp-config" not in command
    assert "Frase privada" not in " ".join(command)
    assert call["stdin"] == "Frase privada"
    assert call["env"] == {
        "PATH": "/usr/bin", "HOME": "/home/x", "LANG": "es_ES.UTF-8",
    }
    assert call["listing"] == [] and not Path(call["cwd"]).exists()
    assert call["timeout"] == 30
    assert model.model_name == "claude-code:sonnet"
    assert "--effort" not in judge(runner).command("s")


@pytest.mark.parametrize("message, status, resets", [
    ("Claude AI usage limit reached|1763215200", None, "1763215200"),
    ("You've hit your session limit · resets 3pm (Europe/Madrid)", None,
     "3pm (Europe/Madrid)"),
    ("API Error: 429 Too Many Requests", 429, ""),
])
def test_quota_refusals_are_told_apart(
    message: str, status: Optional[int], resets: str
) -> None:
    runner = FakeRunner(result(message, is_error=True,
                               api_error_status=status), returncode=1)
    with pytest.raises(QuotaExhaustedError) as caught:
        judge(runner).complete("s", "u", 5)
    assert caught.value.resets == resets


@pytest.mark.parametrize("output, code", [
    (result("Invalid API key · Please run /login", is_error=True), 1),
    (result("expired", is_error=True, api_error_status=401), 1),
    ("", 1),
])
def test_sign_in_failures_are_told_apart(output: str, code: int) -> None:
    runner = FakeRunner(output, code, stderr="Not logged in · run /login")
    with pytest.raises(ModelAuthError, match="log in"):
        judge(runner).complete("s", "u", 5)


def test_other_failures_are_model_errors_without_the_text() -> None:
    refused = result(
        "API Error: the safeguards flagged this message\nRequest ID: r",
        is_error=True,
    )
    with pytest.raises(ModelError, match="safeguards") as caught:
        judge(FakeRunner(refused, 1)).complete("s", "texto privado", 5)
    assert not isinstance(caught.value, (QuotaExhaustedError,
                                         ModelAuthError))
    assert "texto privado" not in str(caught.value)
    with pytest.raises(ModelError, match="no JSON result"):
        judge(FakeRunner("not json", 2, "boom")).complete("s", "u", 5)
    with pytest.raises(ModelError, match="empty reply"):
        judge(FakeRunner(result("  "))).complete("s", "u", 5)
    with pytest.raises(ModelError, match="failed"):
        judge(FakeRunner(result("ok"), 3)).complete("s", "u", 5)
    with pytest.raises(ModelError, match="not installed"):
        judge(FakeRunner(error=FileNotFoundError())).complete("s", "u", 5)
    late = subprocess.TimeoutExpired(["claude"], 1)
    with pytest.raises(ModelError, match="within 180 s"):
        judge(FakeRunner(error=late)).complete("s", "u", 5)
    assert short("a\nb", 1) == "a" and short("  ") == ""


def test_the_default_runner_runs_a_program(tmp_path: Path) -> None:
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        "echo '{\"is_error\": false, \"result\": \"hola\"}'\n"
    )
    script.chmod(0o755)
    model = ClaudeCodeJudge(executable=str(script), environ={})
    assert model.complete("s", "u", 5) == "hola"


class FakeLocalModel:
    """A built local model: records its options and releases."""

    built: List[Dict[str, Any]] = []

    def __init__(
        self, name: str, plan: Optional[RuntimePlan] = None, bits: int = 16
    ) -> None:
        """Record what was asked for."""
        self.model_name = name
        self.bits = bits
        self.released = False
        FakeLocalModel.built.append({"name": name, "bits": bits})

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer with the model's name."""
        return self.model_name

    def release(self) -> None:
        """Mark the model released."""
        self.released = True


LADDER = [
    ModelOption(name="small", params_billion=1.7, bits=4),
    ModelOption(name="medium", params_billion=4.0, bits=4),
    ModelOption(name="large", params_billion=8.2, bits=4),
]
GPU = RuntimePlan(device="cuda", threads=4)


def test_local_judge_takes_the_largest_model_the_free_memory_holds() -> None:
    free = [6.0]
    local = LocalJudge(LADDER, GPU, "cpu-model", free_gb=lambda: free[0],
                       build=FakeLocalModel)
    assert local.model_name == "large@4bit"
    assert local.complete("s", "u", 5) == "large"
    assert "6.0 GB free" in local.choice_reason
    model = local._model
    local.release()
    assert isinstance(model, FakeLocalModel) and model.released
    free[0] = 3.2
    assert local.model_name == "medium@4bit"
    free[0] = 0.5
    local.release()
    with pytest.raises(ModelError, match="smallest local judge"):
        local.complete("s", "u", 5)


def test_local_judge_on_cpu_or_fixed_uses_the_given_model() -> None:
    cpu = LocalJudge(LADDER, RuntimePlan(threads=4), "cpu-model",
                     build=FakeLocalModel)
    assert cpu.model_name == "cpu-model" and cpu.choice_reason == "CPU plan"
    fixed = LocalJudge(LADDER, GPU, "cpu-model", fixed="medium",
                       build=FakeLocalModel)
    assert fixed.model_name == "medium@4bit"
    LocalJudge(LADDER, GPU, "x", build=FakeLocalModel).release()
    assert cuda_free_gb() is None or cuda_free_gb() is not None


def test_state_file_round_trip_and_damage(tmp_path: Path) -> None:
    store = JsonJudgeStateStore(str(tmp_path / "skills" / "judge.json"))
    assert store.load() == JudgeState()
    store.save(JudgeState(mode="local", quota_resets="3pm"))
    assert store.load().mode == "local"
    assert not list((tmp_path / "skills").glob("*.part"))
    store.path.write_text("{broken", encoding="utf-8")
    assert store.load() == JudgeState()
    blocked = JsonJudgeStateStore(str(tmp_path / "skills" / "judge.json" /
                                      "x.json"))
    with pytest.raises(CorruptedDataError):
        blocked.save(JudgeState())
    store.path.unlink()
    store.path.mkdir()
    with pytest.raises(CorruptedDataError):
        store.load()


def client_with_judge(folder: Path) -> TestClient:
    """A RAG API with the transcript and judge routes."""
    search = SearchService(
        FileSystemIndexRepository(str(folder / "index")), tokenize,
        FakeEmbedder(),
    )
    store = FileSystemTranscriptStore(str(folder / "transcripts"))
    selection = JudgeSelection(
        JsonJudgeStateStore(str(folder / "judge.json")),
        lambda: __import__("datetime").datetime(2026, 11, 15),
    )
    routes = TranscriptRoutes(
        IngestTranscript(store), ShowIngestedTranscript(store), TOKEN,
        judge=selection,
    )
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()), routes
    ))


def test_judge_routes_need_the_token_and_change_the_choice(
    tmp_path: Path,
) -> None:
    client = client_with_judge(tmp_path)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/analysis/judge").status_code == 401
    assert client.put("/analysis/judge", json={"mode": "local"}).status_code \
        == 401
    shown = client.get("/analysis/judge", headers=auth).json()
    assert (shown["mode"], shown["chosen"], shown["claude_quota"]) == (
        "claude_code", False, "unknown",
    )
    changed = client.put(
        "/analysis/judge", json={"mode": "both"}, headers=auth
    )
    assert changed.status_code == 200 and changed.json()["mode"] == "both"
    assert json.loads((tmp_path / "judge.json").read_text())["mode"] == "both"
    wrong = client.put("/analysis/judge", json={"mode": "api"}, headers=auth)
    assert wrong.status_code == 422 and "Unknown judge" in wrong.text
    assert client.put(
        "/analysis/judge", json={"nope": 1}, headers=auth
    ).status_code == 422
    (tmp_path / "judge.json").unlink()
    (tmp_path / "judge.json").mkdir()
    assert client.get("/analysis/judge", headers=auth).status_code == 500
    assert client.put(
        "/analysis/judge", json={"mode": "local"}, headers=auth
    ).status_code == 500
