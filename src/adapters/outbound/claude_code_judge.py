"""Chat model adapter: Claude through the Claude Code command line.

The owner's Claude subscription is used through ``claude -p`` (print
mode) instead of the paid API. Each call runs one isolated, one-turn
session (``docs/skills/DESIGN.md`` §4.1 records how this was checked):

- ``--safe-mode``: no CLAUDE.md, skills, plugins, hooks, MCP servers,
  custom agents or output styles; authentication works normally;
- ``--setting-sources ""``: no user, project or local settings file;
- ``--strict-mcp-config`` with no ``--mcp-config``: no MCP server;
- ``--tools ""`` and ``--disable-slash-commands``: no tool, no skill;
- ``--system-prompt``: our prompt replaces Claude Code's own;
- ``--no-session-persistence``: nothing is written to the session list;
- the working directory is a new, empty temporary directory, removed
  afterwards, so no project file is near;
- the environment is reduced to ``PATH``, ``HOME``, ``LANG`` and
  ``LC_ALL``: variables of a Claude Code session that started the
  server never reach the judge;
- the user message goes through standard input, never the command
  line (which other local users can read in the process list).

``--bare`` is not used: it isolates as well, but then only an API key
is accepted, and the subscription signs in with OAuth.

The reply is the ``--output-format json`` result: an object whose
``result`` is the model's text. Quota (usage limit) and sign-in
failures raise their own errors, so the caller can fall back to the
local judge on the first and tell the owner to log in on the second.
Transcript text is never logged nor put in an error message.
"""

import json
import os
import re
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Callable, List, Mapping, Optional, Sequence

from src.domain.errors import (
    ModelAuthError,
    ModelError,
    QuotaExhaustedError,
    TransientModelError,
)
from src.domain.skills.usage import Usage

DEFAULT_CLAUDE_MODEL = "opus"
"""Claude Code's alias of the best model the subscription offers."""

KEPT_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL")
"""The only environment variables handed to ``claude``."""

QUOTA_PATTERN = re.compile(
    r"claude ai usage limit reached|"
    r"hit your (?:session |weekly |opus |sonnet |5-hour )?limit|"
    r"(?:5-hour|session|weekly|opus|sonnet) limit reached|"
    r"out of extra usage",
    re.IGNORECASE,
)
"""The subscription's usage-limit messages (and nothing else).

A per-minute API rate limit (``429``) is not one of them: it is
transient (:data:`TRANSIENT_PATTERN`) and must not send the next hours
to the local judge.
"""
TRANSIENT_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|internal server error|"
    r"service unavailable|bad gateway|gateway time-?out",
    re.IGNORECASE,
)
AUTH_PATTERN = re.compile(
    r"not logged in|please run /login|invalid api key|"
    r"oauth token (?:has )?expired|login (?:has )?expired",
    re.IGNORECASE,
)
RESET_EPOCH = re.compile(r"\|(\d{9,11})\b")
RESET_TEXT = re.compile(r"resets?\s+(?:at\s+)?([^.\n·]{1,60})", re.IGNORECASE)

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _count(value: Any) -> int:
    """Return ``value`` as a token count (0 when it is not a number)."""
    return value if isinstance(value, int) and value > 0 else 0


def usage_of(data: Mapping[str, Any]) -> Optional[Usage]:
    """Return the tokens ``claude -p`` reported, or ``None`` if none.

    Reads the ``usage`` object and ``total_cost_usd`` of the JSON
    result; nothing else is asked of the model.
    """
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    cost = data.get("total_cost_usd")
    return Usage(
        calls=1,
        input_tokens=_count(raw.get("input_tokens")),
        output_tokens=_count(raw.get("output_tokens")),
        cache_creation_input_tokens=_count(
            raw.get("cache_creation_input_tokens")
        ),
        cache_read_input_tokens=_count(raw.get("cache_read_input_tokens")),
        cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0,
    )


def _kill_group(process: "subprocess.Popen[str]") -> None:
    """Kill ``process`` and every process it started (its session)."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run(
    command: Sequence[str],
    stdin: str,
    cwd: str,
    env: Mapping[str, str],
    timeout: float,
) -> "subprocess.CompletedProcess[str]":
    """Run ``command`` with ``stdin`` and capture its output.

    The program starts in a new session, so on a time-out (or any other
    exception) the whole group is killed, helpers included, and once it
    exits no helper of it is left behind.
    """
    process = subprocess.Popen(
        list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=cwd, env=dict(env), text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    finally:
        _kill_group(process)
    return subprocess.CompletedProcess(
        list(command), process.returncode, stdout, stderr
    )


def short(text: str, limit: int = 200) -> str:
    """Return the first line of ``text``, cut to ``limit`` characters."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit]


def reset_time(message: str) -> "tuple[str, Optional[datetime]]":
    """Return when a usage-limit message says the quota comes back.

    ``("", None)`` when it does not say; the moment as a UTC datetime
    when it gives a Unix timestamp (``…|1763215200``), else ``None``.
    """
    epoch = RESET_EPOCH.search(message)
    if epoch:
        moment = datetime.fromtimestamp(int(epoch.group(1)), timezone.utc)
        return moment.isoformat(), moment
    when = RESET_TEXT.search(message)
    return (when.group(1).strip() if when else ""), None


class ClaudeCodeJudge:
    """Answer one system + user message with ``claude -p``.

    Args:
        model: Claude Code model alias or full name (``opus`` is the
            best model of the subscription).
        executable: The ``claude`` program (looked up on ``PATH``).
        timeout: Seconds one call may take.
        effort: Claude Code's ``--effort`` level, or ``None`` for its
            default.
        environ: Where ``PATH``, ``HOME`` and the locale are read from.
        runner: Runs the command (replaced in tests).
        retries: Further attempts after a transient failure (overload,
            rate limit, ``5xx``); quota and sign-in failures are final.
        backoff: Seconds before the first retry, doubled after each.
        sleep: Waits between attempts (replaced in tests).
    """

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        executable: str = "claude",
        timeout: float = 180.0,
        effort: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        runner: Runner = _run,
        retries: int = 2,
        backoff: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Keep the options; nothing runs until :meth:`complete`."""
        self.retries = max(retries, 0)
        self.backoff = backoff
        self._sleep = sleep
        self.model = model
        self.executable = executable
        self.timeout = timeout
        self.effort = effort
        self._environ = os.environ if environ is None else environ
        self._runner = runner
        self.last_usage: Optional[Usage] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the judge (``claude-code:<model>``)."""
        return f"claude-code:{self.model}"

    def command(self, system: str) -> List[str]:
        """Return the command line of one isolated call."""
        command = [
            self.executable, "-p",
            "--safe-mode",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--tools", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format", "json",
            "--model", self.model,
            "--system-prompt", system,
        ]
        if self.effort:
            command += ["--effort", self.effort]
        return command

    def environment(self) -> dict[str, str]:
        """Return the reduced environment the judge runs with."""
        return {
            name: self._environ[name]
            for name in KEPT_ENVIRONMENT if name in self._environ
        }

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return Claude's reply to ``user`` under ``system``.

        ``max_tokens`` is not passed on: Claude Code has no such
        option, and the prompts ask for one short JSON object. A
        transient failure is retried ``retries`` times; it never counts
        as the quota being used up.
        """
        delay = self.backoff
        for attempt in range(self.retries + 1):
            try:
                return self._once(system, user)
            except TransientModelError:
                if attempt == self.retries:
                    raise
                self._sleep(delay)
                delay *= 2
        raise ModelError("unreachable")  # pragma: no cover

    def _once(self, system: str, user: str) -> str:
        """Make one call."""
        self.last_usage = None
        with tempfile.TemporaryDirectory(prefix="judge-") as empty:
            try:
                done = self._runner(
                    self.command(system), user, empty, self.environment(),
                    self.timeout,
                )
            except FileNotFoundError:
                raise ModelError(
                    f"Claude Code ({self.executable}) is not installed."
                )
            except subprocess.TimeoutExpired:
                raise ModelError(
                    f"Claude Code did not answer within {self.timeout:g} s."
                )
        return self.parse(done.returncode, done.stdout, done.stderr)

    def parse(self, returncode: int, stdout: str, stderr: str) -> str:
        """Return the model's text from ``claude -p`` output, or raise.

        The tokens the output reports are kept in :attr:`last_usage`.

        Errors carry Claude Code's own error message at most, never the
        text of a reply.
        """
        data: Any = None
        try:
            data = json.loads(stdout) if stdout.strip() else None
        except ValueError:
            data = None
        if not isinstance(data, dict):
            self._fail(stderr, None)
            raise ModelError(
                f"Claude Code gave no JSON result (exit {returncode})."
            )
        text = data.get("result")
        self.last_usage = usage_of(data)
        status = data.get("api_error_status")
        if data.get("is_error"):
            message = str(text or stderr)
            self._fail(message, status)
            raise ModelError(
                f"Claude Code failed (exit {returncode}): {short(message)}"
            )
        if returncode != 0:
            self._fail(stderr, status)
            raise ModelError(f"Claude Code exited with code {returncode}.")
        if not isinstance(text, str) or not text.strip():
            raise ModelError("Claude Code returned an empty reply.")
        return text.strip()

    def _fail(self, message: str, status: Any) -> None:
        """Raise the specific error a failure message points to, if any."""
        if QUOTA_PATTERN.search(message):
            resets, resets_at = reset_time(message)
            raise QuotaExhaustedError(
                "Claude's usage limit is reached"
                + (f" (resets {resets})" if resets else "") + ".",
                resets, resets_at,
            )
        if status == 401 or AUTH_PATTERN.search(message):
            raise ModelAuthError(
                "Claude Code is not signed in or its sign-in expired; run "
                "`claude` once and log in with the subscription."
            )
        if (
            status == 429 or (isinstance(status, int) and status >= 500)
            or TRANSIENT_PATTERN.search(message)
        ):
            raise TransientModelError(
                f"Claude Code failed for a passing reason "
                f"(status {status}): {short(message)}"
            )
