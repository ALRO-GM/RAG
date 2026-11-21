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
- one call at a time (a lock); a deadline **never kills** a process
  (that made the next calls cold-start inside their own deadline and
  time out in bursts): the in-flight turn is marked abandoned, its
  late answer is drained and dropped, and the next call goes to a
  second warm process. A pool of ``pool_size`` (default 2) processes
  is kept: a spare is started in the background after each call and by
  :meth:`ClaudeCodeStream.warm` (service start); a process retired by
  the turn or context window is replaced before it is stopped. Other
  errors (exit, protocol) do stop the process;
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
from dataclasses import dataclass
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


def _is_result(line: str) -> bool:
    """Tell whether an output line is a ``result`` event."""
    try:
        event = json.loads(line)
    except ValueError:
        return False
    return isinstance(event, dict) and event.get("type") == "result"


@dataclass
class _Worker:
    """One pooled process and what is known of its state."""

    process: StreamProcess
    key: Tuple[str, int]
    turns: int = 0
    context: int = 0
    abandoned: bool = False
    """A turn was given up on: its answer is still to come (and dropped)."""


def _in_thread(task: Callable[[], None]) -> None:
    """Run ``task`` on a daemon thread (the default background runner)."""
    threading.Thread(target=task, daemon=True).start()


class ClaudeCodeStream:
    """Answer system + user messages with a pool of warm ``claude`` processes.

    Args:
        model: Claude Code alias or id (``haiku`` for the live reasoner).
        executable: The ``claude`` program.
        timeout: Seconds one call may take before it is given up.
        max_turns: Retire a process after this many calls.
        max_context_tokens: Retire when a reply reports more input
            tokens than this (input plus both caches).
        thinking_allowance: Output tokens added to the process cap for
            the model's hidden thinking, which the cap also counts.
        environ: Where ``PATH``, ``HOME`` and the locale are read from.
        spawner: Starts the process (replaced in tests).
        clock: Monotonic seconds (replaced in tests).
        pool_size: Healthy processes to keep (1 = no spare).
        background: Runs a task off the request path (replaced in tests
            by a synchronous runner).
        effort: Claude Code's ``--effort`` level of the processes, or
            ``None`` for its default.
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
        pool_size: int = 2,
        background: Callable[[Callable[[], None]], None] = _in_thread,
        effort: Optional[str] = None,
    ):
        """Keep the options; processes start on the first call or warm."""
        self._judge = ClaudeCodeJudge(
            model=model, executable=executable, environ=environ,
            effort=effort,
        )
        self.timeout = timeout
        self.thinking_allowance = max(thinking_allowance, 0)
        self.max_turns = max(max_turns, 1)
        self.max_context_tokens = max_context_tokens
        self.pool_size = max(pool_size, 1)
        self._spawner = spawner
        self._clock = clock
        self._background = background
        self._lock = threading.Lock()
        self._pool_lock = threading.Lock()
        self._workers: List[_Worker] = []
        self._spawning = 0
        self._started = False
        self._closed = False
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
        starts, so a different cap uses other processes).
        """
        return self.complete_within(system, user, max_tokens, None)

    def complete_within(
        self, system: str, user: str, max_tokens: int,
        deadline_s: Optional[float],
    ) -> str:
        """Like :meth:`complete`, giving up after ``deadline_s`` seconds.

        The caller's deadline (``constraints.deadline_ms``) can only
        shorten the backend's own time-out. When it passes,
        :class:`~src.domain.errors.ModelTimeoutError` is raised and the
        process is left running: its late answer is dropped.
        """
        key = (system, max_tokens)
        with self._lock:
            self.last_usage = None
            self.last_latency = None
            self._closed = False
            worker, spawned = self._acquire(key)
            try:
                return self._turn(worker, user, deadline_s, spawned)
            except ModelTimeoutError:
                worker.abandoned = True
                self._cap_abandoned()
                raise
            except Exception:
                self._drop(worker)
                raise
            finally:
                self._after_turn(worker, key)

    def warm(self, system: str, max_tokens: int) -> None:
        """Start the pool for ``(system, max_tokens)`` off the request path.

        Called at service start so that the first call finds a warm
        process.
        """
        key = (system, max_tokens)
        with self._pool_lock:
            self._closed = False
        self._background(lambda: self._fill(key))

    def close(self) -> None:
        """Stop every process (the next call starts a new one)."""
        with self._pool_lock:
            self._closed = True
            workers, self._workers = self._workers, []
        for worker in workers:
            worker.process.close()

    def _retired(self, worker: _Worker) -> bool:
        """Tell whether ``worker`` has used up its turn or context window."""
        return (worker.turns >= self.max_turns
                or worker.context > self.max_context_tokens)

    def _drop(self, worker: _Worker) -> None:
        """Remove ``worker`` from the pool and stop it."""
        with self._pool_lock:
            if worker in self._workers:
                self._workers.remove(worker)
        worker.process.close()

    def _cap_abandoned(self) -> None:
        """Stop the oldest abandoned processes beyond ``pool_size``."""
        with self._pool_lock:
            stale = [w for w in self._workers if w.abandoned]
        for worker in stale[:-self.pool_size]:
            self._drop(worker)

    def _reap(self, key: Tuple[str, int]) -> None:
        """Drop other-key processes; drain the late answers of abandoned ones.

        A drained turn is over (the process is ready again); a process
        that ended is removed.
        """
        with self._pool_lock:
            workers = list(self._workers)
        for worker in workers:
            if worker.key != key:
                self._drop(worker)
            elif worker.abandoned:
                while True:
                    line = worker.process.receive(0.0)
                    if line is None:
                        break
                    if line == "":
                        self._drop(worker)
                        break
                    if _is_result(line):
                        worker.abandoned = False
                        break

    def _acquire(self, key: Tuple[str, int]) -> Tuple[_Worker, float]:
        """Return a ready process for ``key`` and the ms spent starting it."""
        self._reap(key)
        with self._pool_lock:
            for worker in self._workers:
                if not worker.abandoned and not self._retired(worker):
                    return worker, 0.0
        began = self._clock()
        worker = self._start(key)
        if self._started:
            self.restarts += 1
        self._started = True
        return worker, (self._clock() - began) * 1000.0

    def _start(self, key: Tuple[str, int], prime: bool = False) -> _Worker:
        """Start one process for ``key`` and add it to the pool.

        A ``claude`` process does its start-up work on its first message
        (measured: a freshly spawned one did not answer its first real
        request within 1.8 s), so a process started off the request
        path is primed with a throw-away turn before it joins the pool.
        """
        system, cap = key
        command = self._judge.command(
            system, output_format="stream-json", extra=STREAM_FLAGS
        )
        env = dict(self._judge.environment())
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
            max(cap, 1) + self.thinking_allowance
        )
        worker = _Worker(self._spawner(command, env), key)
        if prime:
            try:
                self._prime(worker)
            except ModelError:
                worker.process.close()
                raise
        with self._pool_lock:
            self._workers.append(worker)
        return worker

    def _prime(self, worker: _Worker) -> None:
        """Run a throw-away turn so the process has done its start-up."""
        worker.process.send(json.dumps({
            "type": "user", "message": {"role": "user", "content": "ping"},
        }))
        deadline = self._clock() + self.timeout
        while True:
            line = worker.process.receive(deadline - self._clock())
            if line is None:
                raise ModelError("Claude Code did not start in time.")
            if line == "":
                raise ModelError("Claude Code stopped while starting.")
            if _is_result(line):
                return

    def _after_turn(self, used: _Worker, key: Tuple[str, int]) -> None:
        """Retire ``used`` if worn out, keep a spare; both off the path.

        The replacement is started before the old process is stopped.
        """
        def task() -> None:
            self._fill(key)
            with self._pool_lock:
                worn = used in self._workers and self._retired(used)
            if worn:
                self._drop(used)

        self._background(task)

    def _fill(self, key: Tuple[str, int]) -> None:
        """Start processes until ``pool_size`` healthy ones exist."""
        while True:
            with self._pool_lock:
                healthy = sum(
                    1 for w in self._workers
                    if w.key == key and not w.abandoned
                    and not self._retired(w)
                )
                if (self._closed or healthy + self._spawning
                        >= self.pool_size):
                    return
                self._spawning += 1
            try:
                self._start(key, prime=True)
            except ModelError:
                return
            finally:
                with self._pool_lock:
                    self._spawning -= 1

    def _turn(
        self, worker: _Worker, user: str, deadline_s: Optional[float],
        spawned: float,
    ) -> str:
        """Send one message and read events until its ``result``."""
        process = worker.process
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
        worker.turns += 1
        if self.last_usage is not None:
            worker.context = (
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
