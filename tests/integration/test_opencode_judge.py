"""Opencode judge: inline agent, model choice and raw event parsing."""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence

import pytest

from src.adapters.outbound.opencode_judge import (
    DEFAULT_OPENCODE_MODEL,
    OpencodeJudge,
    event,
)
from src.domain.errors import (
    ModelAuthError,
    ModelError,
    QuotaExhaustedError,
    TransientModelError,
)

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


def answer(text: str = '{"rating": 4}', message: str = "m1") -> str:
    """Return the raw events of one streamed assistant answer."""
    part = {
        "id": "p1", "sessionID": "s1", "messageID": message,
        "type": "text", "text": text, "time": {"start": 1, "end": 2},
    }
    return "\n".join([
        event("message.updated", sessionID="s1", info={
            "id": "title_msg", "role": "assistant", "agent": "title",
        }),
        event("message.part.updated", sessionID="s1", part={
            "id": "pt", "sessionID": "s1", "messageID": "title_msg",
            "type": "text", "text": "New session - private words",
        }),
        event("message.updated", sessionID="s1", info={
            "id": message, "role": "assistant", "agent": "ragjudge",
        }),
        event("message.part.updated", sessionID="s1", part={
            **part, "text": text[:5], "time": {"start": 1},
        }),
        event("message.part.delta", sessionID="s1", messageID=message,
              partID="p1", field="text", delta=text[5:]),
        event("message.part.updated", sessionID="s1", part=part),
    ])


def judge(runner: FakeRunner, **kwargs: Any) -> OpencodeJudge:
    """An opencode judge over a fake runner and a fixed environment."""
    return OpencodeJudge(environ=ENVIRON, runner=runner, **kwargs)


def test_isolated_agent_uses_the_system_prompt_and_selected_model() -> None:
    """Tools are denied, the rubric is the agent prompt and text is parsed."""
    runner = FakeRunner(answer())
    model = judge(runner, model="opencode/beam-free", timeout=45)
    assert model.complete("RUBRIC", "Frase privada", 50) == '{"rating": 4}'
    call = runner.calls[0]
    command = call["command"]
    for flag in ("--pure", "--print-logs", "--format", "--agent", "--model",
                 "--title", "--dir", "--file"):
        assert flag in command
    assert command[command.index("--format") + 1] == "json"
    assert command[command.index("--agent") + 1] == "ragjudge"
    assert command[command.index("--model") + 1] == "opencode/beam-free"
    assert command[command.index("--log-level") + 1] == "ERROR"
    assert command[-1] == "Answer using the attached file"
    assert "Frase privada" not in " ".join(command)
    assert "RUBRIC" not in " ".join(command)
    assert call["prompt"] == "Frase privada"
    assert call["prompt_file"] is True
    assert call["listing"] == ["prompt.txt"]
    assert call["stdin"] == ""
    assert call["timeout"] == 45
    config = json.loads(call["env"]["OPENCODE_CONFIG_CONTENT"])
    agent = config["agent"]["ragjudge"]
    assert agent["prompt"].startswith("RUBRIC\n")
    assert agent["permission"] == {"*": "deny"}
    assert agent["steps"] == 1
    assert call["env"]["HOME"] == call["cwd"]
    assert call["env"]["OPENCODE_DB"] == os.path.join(call["cwd"], "judge.db")
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert "OPENCODE_API_KEY" not in call["env"]
    assert model.model_name == "opencode:opencode/beam-free"


def test_default_model_and_custom_agent_are_configurable() -> None:
    """``--opencode_model`` is the only setting needed to pick a model."""
    runner = FakeRunner(answer())
    model = judge(runner, agent="custom")
    assert model.complete("s", "u", 10) == '{"rating": 4}'
    command = runner.calls[0]["command"]
    assert command[command.index("--model") + 1] == DEFAULT_OPENCODE_MODEL
    assert command[command.index("--agent") + 1] == "custom"
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

    quota = FakeRunner(event("session.error", sessionID="s1", error={
        "name": "UnknownError",
        "data": {"message": "Free usage limit reached"},
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
