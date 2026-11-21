"""Warm Claude Code backend: one persistent ``claude -p`` process.

A cold ``claude -p`` call costs a process start every time: median
1.93 s, p90 5.56 s for Haiku. One process kept alive with
``--input-format stream-json --output-format stream-json`` answers in
a median of 0.83 s (``docs/live/HAIKU_LATENCY.md``), which the live
reasoner's 1.8 s budget needs. This adapter keeps that process:

- the isolation flags are the cold judge's
  (:class:`~src.adapters.outbound.claude_code_judge.ClaudeCodeJudge`):
  no settings, tools, MCP, skills or session file; empty working
  directory; reduced environment; the prompt goes through stdin;
- every request is self-contained (the window is in the prompt), so
  the process's own history is dead weight: it is restarted after
  ``max_turns`` calls or when a reply reports more than
  ``max_context_tokens`` of input, and whenever the system prompt or
  the output cap changes (both are fixed at process start);
- the output cap is ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` of the process,
  plus ``thinking_allowance``: Haiku's hidden thinking counts against
  that cap (183 of 286 output tokens in one measured call) and cut the
  replies of ``suggester.support@1`` while the visible JSON was ~100
  tokens (outliers of 384-797 tokens caused the 4-6 s tails);
- one call at a time (a lock); a time-out kills the process;
- errors are the cold judge's (quota, sign-in, transient), read from
  the same ``result`` object. Transcript text is never logged.

It reports the latency breakdown of the last call in
:attr:`ClaudeCodeStream.last_latency`.
"""

import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, List, Mapping, Optional, Protocol, Tuple

from src.adapters.outbound.claude_code_judge import (
    ClaudeCodeJudge,
    LIVE_CLAUDE_MODEL,
    _kill_group,
)
from src.domain.errors import ModelError, ModelTimeoutError
from src.domain.reasoning.roles import Latency
from src.domain.skills.usage import Usage

STREAM_FLAGS = (
    "--input-format", "stream-json", "--verbose",
    "--include-partial-messages",
)
"""What turns the cold command into the persistent one."""


class StreamProcess(Protocol):
    """A running ``claude`` process speaking stream-json."""

    def send(self, line: str) -> None:
        """Write one line to its standard input."""
        ...

    def receive(self, timeout: float) -> Optional[str]:
        """Return the next output line; ``None`` on time-out, ``""`` at EOF."""
        ...

    def close(self) -> None:
        """Stop it and everything it started."""
        ...


Spawner = Callable[[List[str], Mapping[str, str]], StreamProcess]


class SubprocessStream:
    """A :class:`StreamProcess` over ``subprocess.Popen``.

    A reader thread moves output lines to a queue, so a read can time
    out. The process starts in a new session in an empty directory that
    is removed on :meth:`close`.
    """

    def __init__(self, command: List[str], env: Mapping[str, str]):
        """Start ``command`` with ``env`` in a fresh empty directory."""
        self._dir = tempfile.mkdtemp(prefix="reasoner-")
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=self._dir, env=dict(env),
                text=True, start_new_session=True,
            )
        except FileNotFoundError:
            shutil.rmtree(self._dir, ignore_errors=True)
            raise ModelError(
                f"Claude Code ({command[0]}) is not installed."
            )
        self._lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Move stdout lines to the queue; ``""`` marks the end."""
        stdout = self._process.stdout
        assert stdout is not None
        for line in stdout:
            self._lines.put(line)
        self._lines.put("")

    def send(self, line: str) -> None:
        """Write ``line`` (plus a newline) to the process."""
        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.write(line + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError):
            raise ModelError("Claude Code stopped before the request.")

    def receive(self, timeout: float) -> Optional[str]:
        """Return the next line, or ``None`` after ``timeout`` seconds."""
        try:
            return self._lines.get(timeout=max(timeout, 0.0))
        except queue.Empty:
            return None

    def close(self) -> None:
        """Kill the process group and remove the directory."""
        _kill_group(self._process)
        shutil.rmtree(self._dir, ignore_errors=True)


def _spawn(command: List[str], env: Mapping[str, str]) -> StreamProcess:
    """Start a real process."""
    return SubprocessStream(command, env)


def _is_text_delta(event: Mapping[str, Any]) -> bool:
    """Tell whether a stream event carries the first text of the reply."""
    if event.get("type") != "stream_event":
        return False
    inner = event.get("event")
    if not isinstance(inner, dict):
        return False
    delta = inner.get("delta")
    return (
        inner.get("type") == "content_block_delta"
        and isinstance(delta, dict) and delta.get("type") == "text_delta"
    )


class ClaudeCodeStream:
    """Answer system + user messages with one warm ``claude`` process.

    Args:
        model: Claude Code alias or id (``haiku`` for the live reasoner).
        executable: The ``claude`` program.
        timeout: Seconds one call may take before the process is killed.
        max_turns: Restart the process after this many calls.
        max_context_tokens: Restart when a reply reports more input
            tokens than this (input plus both caches).
        thinking_allowance: Output tokens added to the process cap for
            the model's hidden thinking, which the cap also counts.
        environ: Where ``PATH``, ``HOME`` and the locale are read from.
        spawner: Starts the process (replaced in tests).
        clock: Monotonic seconds (replaced in tests).
    """

    def __init__(
        self,
        model: str = LIVE_CLAUDE_MODEL,
        executable: str = "claude",
        timeout: float = 10.0,
        max_turns: int = 20,
        max_context_tokens: int = 12000,
        thinking_allowance: int = 300,
        environ: Optional[Mapping[str, str]] = None,
        spawner: Spawner = _spawn,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """Keep the options; the process starts on the first call."""
        self._judge = ClaudeCodeJudge(
            model=model, executable=executable, environ=environ,
        )
        self.timeout = timeout
        self.thinking_allowance = max(thinking_allowance, 0)
        self.max_turns = max(max_turns, 1)
        self.max_context_tokens = max_context_tokens
        self._spawner = spawner
        self._clock = clock
        self._lock = threading.Lock()
        self._process: Optional[StreamProcess] = None
        self._key: Optional[Tuple[str, int]] = None
        self._turns = 0
        self._context = 0
        self.restarts = 0
        self.last_usage: Optional[Usage] = None
        self.last_latency: Optional[Latency] = None

    @property
    def model_name(self) -> str:
        """Return the identifier (``claude-code:<model>``)."""
        return self._judge.model_name

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return Claude's reply to ``user`` under ``system``.

        ``max_tokens`` caps the output of the process (fixed when it
        starts, so a different cap restarts it).
        """
        return self.complete_within(system, user, max_tokens, None)

    def complete_within(
        self, system: str, user: str, max_tokens: int,
        deadline_s: Optional[float],
    ) -> str:
        """Like :meth:`complete`, giving up after ``deadline_s`` seconds.

        The caller's deadline (``constraints.deadline_ms``) can only
        shorten the backend's own time-out. When it passes, the
        process is killed (the call is cancelled) and
        :class:`~src.domain.errors.ModelTimeoutError` is raised.
        """
        with self._lock:
            self.last_usage = None
            self.last_latency = None
            try:
                return self._call(system, user, max_tokens, deadline_s)
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        """Stop the process, if any (the next call starts a new one)."""
        process, self._process, self._key = self._process, None, None
        if process is not None:
            process.close()

    def _needs_restart(self, key: Tuple[str, int]) -> bool:
        """Tell whether the process must be (re)started for ``key``."""
        return (
            self._process is None or self._key != key
            or self._turns >= self.max_turns
            or self._context > self.max_context_tokens
        )

    def _start(self, system: str, cap: int) -> None:
        """Start a fresh process for ``system`` with an output cap."""
        self.close()
        command = self._judge.command(
            system, output_format="stream-json", extra=STREAM_FLAGS
        )
        env = dict(self._judge.environment())
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
            max(cap, 1) + self.thinking_allowance
        )
        self._process = self._spawner(command, env)
        self._key = (system, cap)
        self._turns = 0
        self._context = 0

    def _call(
        self, system: str, user: str, cap: int,
        deadline_s: Optional[float] = None,
    ) -> str:
        """Send one message and read events until its ``result``."""
        began = self._clock()
        spawned = 0.0
        if self._needs_restart((system, cap)):
            if self._process is not None:
                self.restarts += 1
            self._start(system, cap)
            spawned = (self._clock() - began) * 1000.0
        process = self._process
        assert process is not None
        sent = self._clock()
        process.send(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": user},
        }))
        allowed = self.timeout if deadline_s is None else min(
            self.timeout, deadline_s
        )
        deadline = sent + allowed
        first_ms: Optional[float] = None
        while True:
            line = process.receive(deadline - self._clock())
            if line is None:
                raise ModelTimeoutError(
                    f"Claude Code did not answer within {allowed:g} s."
                )
            if line == "":
                raise ModelError("Claude Code stopped without an answer.")
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if first_ms is None and _is_text_delta(event):
                first_ms = (self._clock() - sent) * 1000.0
            if event.get("type") == "result":
                break
        total = (self._clock() - sent) * 1000.0
        text = self._judge.parse(0, json.dumps(event), "")
        self.last_usage = self._judge.last_usage
        self._turns += 1
        if self.last_usage is not None:
            self._context = (
                self.last_usage.input_tokens
                + self.last_usage.cache_read_input_tokens
                + self.last_usage.cache_creation_input_tokens
            )
        self.last_latency = Latency(
            process_start_ms=round(spawned, 1),
            first_token_ms=None if first_ms is None else round(first_ms, 1),
            total_ms=round(total + spawned, 1),
        )
        return text
