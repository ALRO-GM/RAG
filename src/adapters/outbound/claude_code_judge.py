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
import subprocess
import tempfile
from typing import Any, Callable, List, Mapping, Optional, Sequence

from src.domain.errors import ModelAuthError, ModelError, QuotaExhaustedError

DEFAULT_CLAUDE_MODEL = "opus"
"""Claude Code's alias of the best model the subscription offers."""

KEPT_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL")
"""The only environment variables handed to ``claude``."""

QUOTA_PATTERN = re.compile(
    r"usage limit|limit reached|hit your (?:\w+ )?limit|"
    r"out of (?:extra )?usage|quota|rate.?limit|credit balance is too low",
    re.IGNORECASE,
)
AUTH_PATTERN = re.compile(
    r"/login|not logged in|log ?in again|invalid api key|"
    r"authenticat|oauth token|unauthori[sz]ed|credential",
    re.IGNORECASE,
)
RESET_EPOCH = re.compile(r"\|(\d{9,11})\b")
RESET_TEXT = re.compile(r"resets?\s+(?:at\s+)?([^.\n]{1,60})", re.IGNORECASE)

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _run(
    command: Sequence[str],
    stdin: str,
    cwd: str,
    env: Mapping[str, str],
    timeout: float,
) -> "subprocess.CompletedProcess[str]":
    """Run ``command`` with ``stdin`` and capture its output."""
    return subprocess.run(
        list(command), input=stdin, cwd=cwd, env=dict(env),
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def short(text: str, limit: int = 200) -> str:
    """Return the first line of ``text``, cut to ``limit`` characters."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line[:limit]


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
    """

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        executable: str = "claude",
        timeout: float = 180.0,
        effort: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
        runner: Runner = _run,
    ):
        """Keep the options; nothing runs until :meth:`complete`."""
        self.model = model
        self.executable = executable
        self.timeout = timeout
        self.effort = effort
        self._environ = os.environ if environ is None else environ
        self._runner = runner

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
        option, and the prompts ask for one short JSON object.
        """
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
        """Return the model's text from ``claude -p`` output, or raise."""
        data: Any = None
        try:
            data = json.loads(stdout) if stdout.strip() else None
        except ValueError:
            data = None
        if not isinstance(data, dict):
            self._fail(returncode, stderr or stdout, None)
            raise ModelError(
                f"Claude Code gave no JSON result (exit {returncode})."
            )
        text = data.get("result")
        status = data.get("api_error_status")
        if data.get("is_error") or returncode != 0:
            self._fail(returncode, str(text or stderr), status)
            raise ModelError(
                f"Claude Code failed (exit {returncode}): "
                f"{short(str(text or stderr))}"
            )
        if not isinstance(text, str) or not text.strip():
            raise ModelError("Claude Code returned an empty reply.")
        return text.strip()

    def _fail(self, returncode: int, message: str, status: Any) -> None:
        """Raise the specific error a failure message points to, if any."""
        if status == 429 or QUOTA_PATTERN.search(message):
            epoch = RESET_EPOCH.search(message)
            when = RESET_TEXT.search(message)
            resets = (
                epoch.group(1) if epoch
                else when.group(1).strip() if when else ""
            )
            raise QuotaExhaustedError(
                "Claude's usage limit is reached"
                + (f" (resets {resets})" if resets else "") + ".",
                resets,
            )
        if status in (401, 403) or AUTH_PATTERN.search(message):
            raise ModelAuthError(
                "Claude Code is not signed in or its sign-in expired; run "
                "`claude` once and log in with the subscription."
            )
