"""Opencode judge: command line, environment and recorded event streams."""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.adapters.outbound.opencode_judge import (
    DEFAULT_OPENCODE_MODEL,
    OpencodeJudge,
)
from src.domain.errors import (
    ModelAuthError,
    ModelError,
    QuotaExhaustedError,
    TransientModelError,
)

FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "opencode"
)


def recorded(name: str) -> str:
    """Return a real ``opencode 2.3.2 run --format json`` stream."""
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return handle.read()


def line(kind: str, **fields: Any) -> str:
    """Return one raw event line of the 2.x stream."""
    return json.dumps({"type": kind, **fields})


ENVIRON = {
    "PATH": "/usr/bin", "HOME": "/home/x", "LANG": "es_ES.UTF-8",
    "ANTHROPIC_API_KEY": "secret", "OPENCODE_API_KEY": "secret",
}


class FakeRunner:
    """Record calls and answer with a scripted process."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        error: Optional[BaseException] = None,
    ) -> None:
        """Script the answer (or the exception) of every call."""
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self, command: Sequence[str], stdin: str, cwd: str,
        env: Dict[str, str], timeout: float,
    ) -> "subprocess.CompletedProcess[str]":
        """Record the call and the file the transcript was written to."""
        index = list(command).index("--file") + 1
        path = str(command[index])
        self.calls.append({
            "command": list(command), "stdin": stdin, "cwd": cwd,
            "env": dict(env), "timeout": timeout,
            "prompt_file": os.path.isabs(path),
            "prompt": (
                open(path, encoding="utf-8").read() if os.path.isfile(path)
                else ""
            ),
            "listing": os.listdir(cwd),
        })
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(
            list(command), self.returncode, self.stdout, self.stderr
        )


def answer(text: str = '{"rating": 4}') -> str:
    """Return the events of one answer in the opencode 2.x shape."""
    return "\n".join([
        line("step_start", part={"type": "step-start"}),
        line("text", part={"type": "text", "text": text}),
        line("step_finish", part={"type": "step-finish", "cost": 0}),
    ])


def judge(runner: FakeRunner, **kwargs: Any) -> OpencodeJudge:
    """An opencode judge over a fake runner and a fixed environment."""
    return OpencodeJudge(environ=ENVIRON, runner=runner, **kwargs)


def test_command_uses_the_current_cli_and_selected_model() -> None:
    """The prompt travels in a file; only flags opencode 2.x has are used."""
    runner = FakeRunner(answer())
    model = judge(runner, model="opencode/beam-free", timeout=45)
    assert model.complete("RUBRIC", "Frase privada", 50) == '{"rating": 4}'
    call = runner.calls[0]
    command = call["command"]
    for flag in ("--standalone", "--archived", "--format", "--model",
                 "--title", "--file"):
        assert flag in command
    for gone in ("--pure", "--dir", "--agent", "--print-logs"):
        assert gone not in command
    assert command[command.index("--format") + 1] == "json"
    assert command[command.index("--model") + 1] == "opencode/beam-free"
    assert command[-1] == "Answer using the attached file"
    assert "Frase privada" not in " ".join(command)
    assert "RUBRIC" not in " ".join(command)
    assert call["prompt"].startswith("RUBRIC\n")
    assert call["prompt"].endswith("Frase privada")
    assert call["prompt_file"] is True
    assert call["listing"] == ["prompt.txt"]
    assert call["stdin"] == ""
    assert call["timeout"] == 45
    assert call["env"]["HOME"] == "/home/x"
    assert call["env"]["TMPDIR"] == call["cwd"]
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "OPENCODE_API_KEY" not in call["env"]
    assert model.model_name == "opencode:opencode/beam-free"


def test_default_model_is_configurable() -> None:
    """``--opencode_model`` is the only setting needed to pick a model."""
    runner = FakeRunner(answer())
    model = judge(runner)
    assert model.complete("s", "u", 10) == '{"rating": 4}'
    command = runner.calls[0]["command"]
    assert command[command.index("--model") + 1] == DEFAULT_OPENCODE_MODEL
    assert model.model_name == f"opencode:{DEFAULT_OPENCODE_MODEL}"


@pytest.mark.parametrize("message, expected", [
    ("stream error: Free usage limit reached", QuotaExhaustedError),
    ("not logged in, please sign in", ModelAuthError),
    ("AI_APICallError: Overloaded", TransientModelError),
    ("AI_APICallError: Internal server error", TransientModelError),
    ("unknown failure", ModelError),
])
def test_errors_are_translated(message: str, expected: type) -> None:
    """The judge reports clean errors, never a traceback or the transcript."""
    runner = FakeRunner(stderr=message, returncode=1)
    with pytest.raises(expected):
        judge(runner, retries=0).complete("s", "u", 10)
    assert len(runner.calls) == 1


def test_transient_error_is_retried_but_quota_is_not() -> None:
    """Only transient failures get another try, with backoff."""
    runner = FakeRunner(stderr="Overloaded", returncode=1)
    model = judge(runner, retries=2)
    model._sleep = lambda _: None
    with pytest.raises(TransientModelError):
        model.complete("s", "u", 10)
    assert len(runner.calls) == 3

    quota = FakeRunner(line("error", error={
        "type": "provider.usage-limit",
        "message": "Free usage limit reached",
    }))
    model = judge(quota, retries=2)
    model._sleep = lambda _: None
    with pytest.raises(QuotaExhaustedError):
        model.complete("s", "u", 10)
    assert len(quota.calls) == 1


def test_empty_answer_times_out_and_missing_command_are_reported() -> None:
    """An empty reply and a missing binary do not hang or leak text."""
    runner = FakeRunner()
    with pytest.raises(ModelError, match="no answer"):
        judge(runner).complete("s", "u", 10)

    missing = FakeRunner(error=FileNotFoundError("opencode"))
    with pytest.raises(ModelError, match="not installed"):
        judge(missing, retries=0).complete("s", "u", 10)

    timeout = FakeRunner(
        error=subprocess.TimeoutExpired(["opencode"], 1)
    )
    with pytest.raises(TransientModelError, match="did not answer"):
        judge(timeout, retries=0, timeout=1).complete("s", "u", 10)


def test_recorded_answer_stream_is_parsed() -> None:
    """Contract: a real opencode 2.3.2 stream gives text and tokens."""
    model = judge(FakeRunner(recorded("answer.2.3.2.jsonl")))
    assert model.complete("S", "u", 10) == '{"ok": true}'
    usage = model.last_usage
    assert usage is not None and usage.calls == 1
    assert (usage.input_tokens, usage.output_tokens) == (11803, 98)
    assert usage.cache_read_input_tokens == 1152
    assert usage.cache_creation_input_tokens == 0


def test_recorded_error_stream_is_an_error_not_an_answer() -> None:
    """Contract: a real ``error`` event is raised with its message."""
    model = judge(FakeRunner(recorded("error.2.3.2.jsonl"), returncode=1),
                  retries=0)
    with pytest.raises(ModelError, match="Model unavailable"):
        model.complete("S", "u", 10)


def test_usage_sums_steps_and_skips_noise() -> None:
    """Several steps add up; non-JSON and untokened events are ignored."""
    lines = [
        line("step_finish", part={"cost": 0.25, "tokens": {
            "input": 11, "output": 4, "reasoning": 2,
            "cache": {"read": 30, "write": 8}}}),
        line("step_finish", part={"cost": 0, "tokens": {"input": 1}}),
        line("step_finish", part={"reason": "stop"}),
        line("text", part={"type": "text", "text": "{}"}),
        "not json",
    ]
    model = judge(FakeRunner(stdout="\n".join(lines)))
    model.complete("S", "u", 10)
    usage = model.last_usage
    assert usage is not None and usage.calls == 1
    assert (usage.input_tokens, usage.output_tokens) == (12, 6)
    assert (usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens) == (30, 8)
    assert usage.cost_usd == 0.25


def test_without_token_events_there_is_no_usage() -> None:
    """An answer without ``step_finish`` reports no usage."""
    model = judge(FakeRunner(stdout=line(
        "text", part={"type": "text", "text": "{}"})))
    model.complete("S", "u", 10)
    assert model.last_usage is None


def test_error_message_shapes() -> None:
    """Error objects of other shapes still give a readable message."""
    for payload in ("plain", {"name": "Boom"}, {"data": {"message": "deep"}},
                    {"x": 1}):
        model = judge(FakeRunner(line("error", error=payload)), retries=0)
        with pytest.raises(ModelError):
            model.complete("S", "u", 10)
