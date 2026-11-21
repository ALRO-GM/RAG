"""Warm Claude Code backend: protocol, restarts, caps, errors, privacy.

The process is faked (a scripted :class:`StreamProcess`) except for one
test that runs a tiny stand-in program through the real subprocess
transport.
"""

import json
import logging
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge
from src.adapters.outbound.claude_code_stream import (
    ClaudeCodeStream,
    SubprocessStream,
)
from src.domain.errors import (
    ModelAuthError,
    ModelError,
    ModelTimeoutError,
    QuotaExhaustedError,
)

SECRET = "texto-secreto-del-cliente"


def delta() -> str:
    """Return a partial-message text delta event."""
    return json.dumps({"type": "stream_event", "event": {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "x"},
    }})


def done(text: str = "respuesta", tokens: int = 100, **extra: Any) -> str:
    """Return a ``result`` event."""
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": False,
        "result": text, "api_error_status": None,
        "usage": {"input_tokens": tokens, "output_tokens": 7,
                  "cache_read_input_tokens": 0}, **extra,
    })


class ScriptedProcess:
    """A fake process: each sent line is answered by the next script."""

    def __init__(self, scripts: Sequence[Sequence[Optional[str]]]):
        """Keep the output lines to give after each message."""
        self.scripts = [list(x) for x in scripts]
        self.sent: List[str] = []
        self.pending: List[Optional[str]] = []
        self.closed = False

    def send(self, line: str) -> None:
        """Record the message and queue its answer."""
        self.sent.append(line)
        self.pending = list(self.scripts.pop(0)) if self.scripts else [""]

    def receive(self, timeout: float) -> Optional[str]:
        """Give the next scripted line (``None`` is a time-out)."""
        return self.pending.pop(0) if self.pending else None

    def close(self) -> None:
        """Record the stop."""
        self.closed = True


class Spawns:
    """A spawner handing out scripted processes and recording starts."""

    def __init__(self, *processes: ScriptedProcess):
        """Serve ``processes`` in order."""
        self.processes = list(processes)
        self.started: List[Dict[str, Any]] = []

    def __call__(
        self, command: List[str], env: Mapping[str, str]
    ) -> ScriptedProcess:
        """Record the start and return the next process."""
        process = self.processes.pop(0)
        self.started.append({"command": command, "env": dict(env)})
        return process


def backend(spawns: Spawns, **options: Any) -> ClaudeCodeStream:
    """Return a stream backend over ``spawns`` with a fixed environment."""
    return ClaudeCodeStream(
        environ={"PATH": "/bin", "HOME": "/h", "SECRET": "no"},
        spawner=spawns, **options,
    )


def test_one_process_serves_consecutive_calls() -> None:
    process = ScriptedProcess([[delta(), done("uno")], [delta(), done("dos")]])
    spawns = Spawns(process)
    model = backend(spawns)
    assert model.complete("sys", "hola", 100) == "uno"
    assert model.complete("sys", "otra", 100) == "dos"
    assert len(spawns.started) == 1 and len(process.sent) == 2
    message = json.loads(process.sent[0])
    assert message["message"] == {"role": "user", "content": "hola"}
    assert model.model_name == "claude-code:haiku"


def test_command_is_isolated_streaming_and_capped() -> None:
    spawns = Spawns(ScriptedProcess([[done()]]))
    backend(spawns, thinking_allowance=0).complete("sys prompt", "u", 90)
    command = spawns.started[0]["command"]
    for flag in ("--safe-mode", "--strict-mcp-config",
                 "--disable-slash-commands", "--no-session-persistence",
                 "--input-format", "--include-partial-messages"):
        assert flag in command
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert command[command.index("--model") + 1] == "haiku"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--system-prompt") + 1] == "sys prompt"
    env = spawns.started[0]["env"]
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "90"
    assert "SECRET" not in env and env["HOME"] == "/h"
    assert "u" not in command[command.index("--system-prompt") + 2:]


def test_latency_breakdown_and_usage_are_reported() -> None:
    ticks = iter([0.0, 0.01, 0.02, 0.03, 0.30, 0.31, 0.52])
    model = ClaudeCodeStream(
        spawner=Spawns(ScriptedProcess([[delta(), done()]])),
        clock=lambda: next(ticks), environ={},
    )
    model.complete("s", "u", 50)
    latency = model.last_latency
    assert latency is not None
    assert latency.process_start_ms == 10.0
    assert latency.first_token_ms == 280.0
    assert latency.total_ms == 510.0
    assert model.last_usage is not None
    assert model.last_usage.output_tokens == 7


def test_system_prompt_or_cap_change_restarts() -> None:
    first = ScriptedProcess([[done()]])
    second = ScriptedProcess([[done()]])
    third = ScriptedProcess([[done()]])
    model = backend(Spawns(first, second, third))
    model.complete("a", "u", 50)
    model.complete("b", "u", 50)
    model.complete("b", "u", 60)
    assert first.closed and second.closed and not third.closed
    assert model.restarts == 2


def test_process_is_windowed_by_turns() -> None:
    scripts = [[done()] for _ in range(4)]
    one, two = ScriptedProcess(scripts[:2]), ScriptedProcess(scripts[2:])
    spawns = Spawns(one, two)
    model = backend(spawns, max_turns=2)
    for _ in range(3):
        model.complete("s", "u", 50)
    assert len(spawns.started) == 2 and one.closed


def test_process_is_windowed_by_context_size() -> None:
    one = ScriptedProcess([[done(tokens=5000)]])
    two = ScriptedProcess([[done()]])
    spawns = Spawns(one, two)
    model = backend(spawns, max_context_tokens=1000)
    model.complete("s", "u", 50)
    model.complete("s", "u", 50)
    assert len(spawns.started) == 2


def test_timeout_kills_the_process_and_next_call_restarts() -> None:
    stuck = ScriptedProcess([[delta()]])
    fresh = ScriptedProcess([[done("ok")]])
    model = backend(Spawns(stuck, fresh), timeout=1.0)
    with pytest.raises(ModelTimeoutError):
        model.complete("s", "u", 50)
    assert stuck.closed
    assert model.complete("s", "u", 50) == "ok"


def test_exit_without_result_is_an_error() -> None:
    model = backend(Spawns(ScriptedProcess([[""]])))
    with pytest.raises(ModelError):
        model.complete("s", "u", 50)


def test_garbage_lines_are_skipped() -> None:
    model = backend(Spawns(ScriptedProcess([["not json", "[1]", done("ok")]])))
    assert model.complete("s", "u", 50) == "ok"


@pytest.mark.parametrize("message,status,error", [
    ("Claude AI usage limit reached|1763215200", None, QuotaExhaustedError),
    ("Invalid API key", 401, ModelAuthError),
])
def test_quota_and_auth_errors_are_typed(
    message: str, status: Optional[int], error: type
) -> None:
    bad = json.dumps({"type": "result", "is_error": True, "result": message,
                      "api_error_status": status})
    model = backend(Spawns(ScriptedProcess([[bad]])))
    with pytest.raises(error):
        model.complete("s", "u", 50)


def test_transcript_text_never_reaches_the_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad = json.dumps({"type": "result", "is_error": True,
                      "result": "boom", "api_error_status": None})
    model = backend(Spawns(ScriptedProcess([[bad]])))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ModelError) as info:
            model.complete("s", SECRET, 50)
    assert SECRET not in caplog.text and SECRET not in str(info.value)


def test_cold_judge_reports_latency_and_timeout_type() -> None:
    out = json.dumps({"type": "result", "is_error": False, "result": "ok",
                      "duration_ms": 400, "usage": {"output_tokens": 3}})

    def runner(*args: Any) -> Any:
        import subprocess
        return subprocess.CompletedProcess([], 0, out, "")

    judge = ClaudeCodeJudge(model="haiku", runner=runner)
    assert judge.complete("s", "u", 10) == "ok"
    latency = judge.last_latency
    assert latency is not None and latency.first_token_ms is None
    assert latency.total_ms >= latency.process_start_ms >= 0

    def slow(*args: Any) -> Any:
        import subprocess
        raise subprocess.TimeoutExpired("claude", 1)

    with pytest.raises(ModelTimeoutError):
        ClaudeCodeJudge(runner=slow).complete("s", "u", 10)


def test_subprocess_transport_round_trip(tmp_path: Path) -> None:
    """The real transport against a stand-in that echoes a result."""
    program = tmp_path / "fake-claude"
    program.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, os\n"
        "for line in sys.stdin:\n"
        "    cap = os.environ.get('CLAUDE_CODE_MAX_OUTPUT_TOKENS')\n"
        "    print(json.dumps({'type': 'result', 'is_error': False,\n"
        "        'result': 'cap=' + str(cap), 'usage': {}}), flush=True)\n"
    )
    program.chmod(program.stat().st_mode | stat.S_IEXEC)
    model = ClaudeCodeStream(
        executable=str(program), timeout=10.0, thinking_allowance=0,
        environ={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    try:
        assert model.complete("s", "hola", 33) == "cap=33"
        assert model.complete("s", "otra", 33) == "cap=33"
        assert model.restarts == 0
    finally:
        model.close()


def test_missing_executable_is_a_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError):
        SubprocessStream([str(tmp_path / "absent")], {})


def test_output_cap_error_is_typed() -> None:
    from src.domain.errors import OutputTooLongError
    bad = json.dumps({"type": "result", "is_error": True, "result": (
        "API Error: Claude's response exceeded the 120 output token "
        "maximum. To configure this behavior, set the "
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
    ), "api_error_status": None})
    model = backend(Spawns(ScriptedProcess([[bad]])))
    with pytest.raises(OutputTooLongError):
        model.complete("s", "u", 120)


def test_process_cap_leaves_room_for_hidden_thinking() -> None:
    spawns = Spawns(ScriptedProcess([[done()]]))
    backend(spawns).complete("s", "u", 200)
    env = spawns.started[0]["env"]
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "500"


def test_deadline_shortens_the_wait_and_kills_the_process() -> None:
    seen: List[float] = []
    stuck = ScriptedProcess([[]])
    original = stuck.receive

    def receive(timeout: float) -> Optional[str]:
        seen.append(timeout)
        return original(timeout)

    stuck.receive = receive  # type: ignore[method-assign]
    model = backend(Spawns(stuck), timeout=10.0)
    with pytest.raises(ModelTimeoutError, match="0.05 s"):
        model.complete_within("s", "u", 50, 0.05)
    assert stuck.closed and seen[0] <= 0.05


def test_deadline_cannot_extend_the_backend_timeout() -> None:
    stuck = ScriptedProcess([[]])
    model = backend(Spawns(stuck), timeout=1.0)
    with pytest.raises(ModelTimeoutError, match="1 s"):
        model.complete_within("s", "u", 50, 60.0)
