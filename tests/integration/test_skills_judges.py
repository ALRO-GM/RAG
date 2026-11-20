"""Judge adapters: Claude Code (fake subprocess), local judge, state file,
and the ``/analysis/judge`` routes of the RAG."""

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
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
from src.adapters.outbound.local_judge import (
    LocalJudge,
    cuda_free_gb,
    quantisation_installed,
)
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


@pytest.mark.parametrize("message, resets, resets_at", [
    ("Claude AI usage limit reached|1763215200",
     "2025-11-15T14:00:00+00:00",
     datetime(2025, 11, 15, 14, 0, tzinfo=timezone.utc)),
    ("You've hit your session limit · resets 3pm (Europe/Madrid)",
     "3pm (Europe/Madrid)", None),
    ("5-hour limit reached ∙ resets 7pm", "7pm", None),
    ("You're out of extra usage", "", None),
])
def test_only_the_subscription_limit_counts_as_quota(
    message: str, resets: str, resets_at: Optional[datetime]
) -> None:
    runner = FakeRunner(result(message, is_error=True), returncode=1)
    with pytest.raises(QuotaExhaustedError) as caught:
        judge(runner).complete("s", "u", 5)
    assert caught.value.resets == resets
    assert caught.value.resets_at == resets_at
    assert len(runner.calls) == 1, "a quota refusal is not retried"


@pytest.mark.parametrize("message, status", [
    ("API Error: 429 Too Many Requests", 429),
    ("API Error: rate limit of 50 requests per minute exceeded", None),
    ("API Error: Overloaded", 529),
    ("API Error: 500 Internal server error", 500),
])
def test_transient_failures_are_retried_then_plain_errors(
    message: str, status: Optional[int]
) -> None:
    waits: List[float] = []
    runner = FakeRunner(result(message, is_error=True,
                               api_error_status=status), returncode=1)
    with pytest.raises(ModelError) as caught:
        judge(runner, retries=2, backoff=1.0,
              sleep=waits.append).complete("s", "u", 5)
    assert not isinstance(caught.value, (QuotaExhaustedError,
                                         ModelAuthError))
    assert len(runner.calls) == 3 and waits == [1.0, 2.0]


def test_a_transient_failure_then_an_answer() -> None:
    class Flaky(FakeRunner):
        """Overloaded once, then fine."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            """Change the answer after the first call."""
            done = super().__call__(*args, **kwargs)
            self.stdout = result("bien")
            self.returncode = 0
            return done

    runner = Flaky(result("Overloaded", is_error=True), returncode=1)
    assert judge(runner, sleep=lambda _: None).complete("s", "u", 5) == "bien"


@pytest.mark.parametrize("output, code, stderr", [
    (result("Invalid API key · Please run /login", is_error=True), 1, ""),
    (result("expired", is_error=True, api_error_status=401), 1, ""),
    (result("OAuth token has expired", is_error=True), 1, ""),
    ("", 1, "Not logged in · Please run /login"),
])
def test_sign_in_failures_are_told_apart(
    output: str, code: int, stderr: str
) -> None:
    runner = FakeRunner(output, code, stderr=stderr)
    with pytest.raises(ModelAuthError, match="log in"):
        judge(runner).complete("s", "u", 5)


@pytest.mark.parametrize("message", [
    "credentials service timeout",
    "Request failed: authentication proxy unreachable",
])
def test_other_messages_naming_credentials_are_not_sign_in(
    message: str
) -> None:
    runner = FakeRunner(result(message, is_error=True), returncode=1)
    with pytest.raises(ModelError) as caught:
        judge(runner).complete("s", "u", 5)
    assert not isinstance(caught.value, (ModelAuthError,
                                         QuotaExhaustedError))


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
    with pytest.raises(ModelError, match="not installed"):
        judge(FakeRunner(error=FileNotFoundError())).complete("s", "u", 5)
    late = subprocess.TimeoutExpired(["claude"], 1)
    with pytest.raises(ModelError, match="within 180 s"):
        judge(FakeRunner(error=late)).complete("s", "u", 5)
    assert short("a\nb", 1) == "a" and short("  ") == ""


def test_a_reply_never_reaches_an_error_message() -> None:
    reply = result("Luis dijo algo privado sobre Marta")
    with pytest.raises(ModelError) as caught:
        judge(FakeRunner(reply, 3)).complete("s", "u", 5)
    assert str(caught.value) == "Claude Code exited with code 3."


def test_the_default_runner_runs_a_program(tmp_path: Path) -> None:
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        "echo '{\"is_error\": false, \"result\": \"hola\"}'\n"
    )
    script.chmod(0o755)
    model = ClaudeCodeJudge(executable=str(script), environ={})
    assert model.complete("s", "u", 5) == "hola"


def alive(pid: int) -> bool:
    """Tell whether process ``pid`` still runs (zombies count as gone)."""
    try:
        status = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return False
    return status != "Z"


def test_a_time_out_kills_the_helpers_too(tmp_path: Path) -> None:
    helper = tmp_path / "helper.pid"
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f"sleep 60 &\necho $! > {helper}\n"
        "sleep 60\n"
    )
    script.chmod(0o755)
    model = ClaudeCodeJudge(executable=str(script), environ={
        "PATH": "/usr/bin:/bin",
    }, timeout=1.0)
    with pytest.raises(ModelError, match="within 1 s"):
        model.complete("s", "u", 5)
    pid = int(helper.read_text())
    deadline = time.monotonic() + 5
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid), "the helper outlived the time-out"


def test_helpers_left_behind_by_a_finished_call_are_killed(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "helper.pid"
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        f"sleep 60 > /dev/null 2>&1 &\necho $! > {helper}\n"
        "echo '{\"is_error\": false, \"result\": \"hola\"}'\n"
    )
    script.chmod(0o755)
    model = ClaudeCodeJudge(executable=str(script), environ={
        "PATH": "/usr/bin:/bin",
    })
    assert model.complete("s", "u", 5) == "hola"
    pid = int(helper.read_text())
    deadline = time.monotonic() + 5
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


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


def local(**kwargs: Any) -> LocalJudge:
    """A local judge over fake models, with quantisation installed."""
    options: Dict[str, Any] = {
        "build": FakeLocalModel, "quantisable": lambda: True,
        "policy": "largest_fitting", "total_gb": lambda: 8.0,
    }
    options.update(kwargs)
    return LocalJudge(LADDER, GPU, "cpu-model", **options)


def test_local_judge_takes_the_largest_model_the_free_memory_holds() -> None:
    free = [6.0]
    judge_ = local(free_gb=lambda: free[0])
    assert judge_.model_name == "large@4bit"
    assert judge_.complete("s", "u", 5) == "large"
    assert "6.0 GB free" in judge_.choice_reason
    model = judge_._model
    judge_.release()
    assert isinstance(model, FakeLocalModel) and model.released
    free[0] = 3.2
    assert judge_.model_name == "medium@4bit"
    free[0] = 0.5
    judge_.release()
    with pytest.raises(ModelError, match="smallest local judge"):
        judge_.complete("s", "u", 5)


def test_local_judge_on_cpu_or_forced_uses_the_given_model() -> None:
    cpu = LocalJudge(LADDER, RuntimePlan(threads=4), "cpu-model",
                     build=FakeLocalModel, policy="largest_fitting")
    assert cpu.model_name == "cpu-model" and cpu.choice_reason == "CPU plan"
    assert local(fixed="medium").model_name == "medium@4bit"
    local().release()
    assert cuda_free_gb() is None or cuda_free_gb() is not None
    assert isinstance(quantisation_installed(), bool)


def test_without_bitsandbytes_the_judge_runs_in_half_precision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    judge_ = local(quantisable=lambda: False, free_gb=lambda: 9.0)
    with caplog.at_level(logging.WARNING):
        assert judge_.model_name == "small", "4 B at 16 bit does not fit"
    assert "half precision" in caplog.text
    assert local(quantisable=lambda: False, fixed="small",
                 free_gb=lambda: 9.0).model_name == "small"
    with pytest.raises(ModelError, match="smallest"):
        local(quantisable=lambda: False, free_gb=lambda: 2.0).model_name


def test_preferred_policy_takes_the_first_that_fits_and_never_smaller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    preferred: Dict[str, Any] = {
        "policy": "preferred", "preference": ("large", "medium"),
    }
    assert local(free_gb=lambda: 6.0, **preferred).model_name == (
        "large@4bit"
    )
    assert local(free_gb=lambda: 3.5, **preferred).model_name == (
        "medium@4bit"
    )
    free = iter([1.0, 1.0, 1.5, 3.5])
    now = [0.0]

    def sleep(seconds: float) -> None:
        """Advance the fake clock."""
        now[0] += seconds

    waiting = local(free_gb=lambda: next(free), sleep=sleep,
                    clock=lambda: now[0], poll_seconds=10, **preferred)
    with caplog.at_level(logging.WARNING):
        assert waiting.model_name == "medium@4bit", "waited, not small"
    assert "waiting for 3.0 GB" in caplog.text
    late = local(free_gb=lambda: 2.0, sleep=sleep, clock=lambda: now[0],
                 poll_seconds=10, wait_seconds=30, **preferred)
    with pytest.raises(ModelError, match="No smaller model is used"):
        late.complete("s", "u", 5)
    assert all(b["name"] != "small" for b in FakeLocalModel.built[-3:])
    with pytest.raises(ModelError, match="must be in model_ladder"):
        local(policy="preferred", preference=("other",)).complete("s", "u", 5)
    no_bnb = local(quantisable=lambda: False, free_gb=lambda: 2.0,
                   **preferred)
    with pytest.raises(ModelError, match="group local-judge"):
        no_bnb.complete("s", "u", 5)
    assert local(policy="preferred", fixed="small").model_name == (
        "small@4bit"
    ), "an explicit --local_model is still honoured"


def test_preferred_policy_uses_the_measured_8b_peak() -> None:
    qwen = [
        ModelOption(name="Qwen/Qwen3-4B", params_billion=4.0, bits=4),
        ModelOption(name="Qwen/Qwen3-8B", params_billion=8.2, bits=4),
    ]

    def judge_with(free: float) -> LocalJudge:
        """A preferred judge over the real names with ``free`` GB free."""
        return LocalJudge(
            qwen, GPU, "cpu-model", build=FakeLocalModel,
            quantisable=lambda: True, total_gb=lambda: 8.0,
            free_gb=lambda: free,
            preference=("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"),
        )

    assert judge_with(6.0).model_name == "Qwen/Qwen3-4B@4bit", (
        "5.3 GB estimated, 6.31 GB measured: 6 GB free is not enough"
    )
    assert judge_with(6.8).model_name == "Qwen/Qwen3-8B@4bit"
    assert judge_with(3.4).model_name == "Qwen/Qwen3-4B@4bit"


def test_the_default_preference_is_8b_then_4b() -> None:
    from src.config import SkillsSettings

    settings = SkillsSettings()
    assert settings.local_policy == "preferred"
    assert settings.local_preference == ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
    assert settings.local_bits == 4
    names = [name for name, _ in settings.model_ladder]
    assert all(name in names for name in settings.local_preference)


def test_fixed_policy_waits_for_memory_instead_of_downgrading() -> None:
    free = iter([1.0, 2.0, 6.0])
    now = [0.0]
    waits: List[float] = []

    def sleep(seconds: float) -> None:
        """Advance the fake clock."""
        waits.append(seconds)
        now[0] += seconds

    judge_ = local(policy="fixed", fixed="large", free_gb=lambda: next(free),
                   sleep=sleep, clock=lambda: now[0], poll_seconds=10,
                   wait_seconds=100)
    assert judge_.model_name == "large@4bit"
    assert waits == [10, 10]
    late = local(policy="fixed", fixed="large", free_gb=lambda: 1.0,
                 sleep=sleep, clock=lambda: now[0], poll_seconds=10,
                 wait_seconds=25)
    with pytest.raises(ModelError, match="Waited 25 s"):
        late.complete("s", "u", 5)
    with pytest.raises(ModelError, match="needs the model to pin"):
        local(policy="fixed").complete("s", "u", 5)
    other = local(policy="fixed", fixed="other/model")
    assert other.model_name == "other/model"
    assert "size unknown" in other.choice_reason


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


def test_two_writers_never_lose_each_others_fields(tmp_path: Path) -> None:
    path = str(tmp_path / "judge.json")
    web, job = JsonJudgeStateStore(path), JsonJudgeStateStore(path)
    rounds = 60

    def choose() -> None:
        """The web keeps choosing a judge."""
        for n in range(rounds):
            mode = "local" if n % 2 else "both"
            web.update(lambda s: s.model_copy(update={"mode": mode}))

    def record() -> None:
        """A job keeps recording answers."""
        for n in range(rounds):
            job.update(lambda s: s.model_copy(
                update={"quota_resets": f"r{n}"}
            ))

    threads = [threading.Thread(target=choose),
               threading.Thread(target=record)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    final = JsonJudgeStateStore(path).load()
    assert final.mode == "local"
    assert final.quota_resets == f"r{rounds - 1}"


def test_judge_state_path_is_relative_to_the_repository(
    tmp_path: Path,
) -> None:
    from src.skills_bootstrap import REPOSITORY_ROOT, judge_state_file

    assert judge_state_file("data/skills/judge.json") == (
        REPOSITORY_ROOT / "data/skills/judge.json"
    )
    assert judge_state_file(str(tmp_path / "x.json")) == tmp_path / "x.json"


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


def test_claude_code_reports_the_tokens_of_each_call() -> None:
    usage = {
        "input_tokens": 3, "output_tokens": 9,
        "cache_creation_input_tokens": 1000, "cache_read_input_tokens": 50,
    }
    model = judge(FakeRunner(result(usage=usage, total_cost_usd=0.0123)))
    assert model.last_usage is None
    model.complete("S", "u", 10)
    assert model.last_usage is not None
    assert model.last_usage.input_tokens == 3
    assert model.last_usage.output_tokens == 9
    assert model.last_usage.cache_creation_input_tokens == 1000
    assert model.last_usage.cache_read_input_tokens == 50
    assert model.last_usage.cost_usd == 0.0123 and model.last_usage.calls == 1


def test_claude_code_without_usage_or_after_a_failure_has_none() -> None:
    model = judge(FakeRunner(result()))
    model.complete("S", "u", 10)
    assert model.last_usage is None
    bad = judge(FakeRunner(result(usage={"input_tokens": "x"})))
    bad.complete("S", "u", 10)
    assert bad.last_usage is not None and bad.last_usage.input_tokens == 0
    broken = judge(FakeRunner("not json", returncode=1), retries=0)
    with pytest.raises(Exception):
        broken.complete("S", "u", 10)
    assert broken.last_usage is None
