"""Judge through the opencode command line (any of its models).

A throw-away inline agent carries the assessment prompt and denies every
tool; the transcript is attached as a file in an empty directory, so it
never appears in the process arguments. The prompt is read from
``opencode run --format json``'s raw events, not from the terminal.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.adapters.outbound.claude_code_judge import _run as run_command, short
from src.domain.errors import (
    ModelAuthError,
    ModelError,
    QuotaExhaustedError,
    TransientModelError,
)

DEFAULT_OPENCODE_MODEL = "opencode/deepseek-v4.1-flash-free"
"""Model used when ``--judge opencode`` names none."""

DEFAULT_OPENCODE_AGENT = "ragjudge"
"""Inline agent that carries the assessment prompt."""

DEFAULT_OPENCODE_TIMEOUT = 180

KEPT_ENVIRONMENT = ("LANG", "LC_ALL")
"""Locale variables copied from the parent process."""

QUOTA_PATTERN = re.compile(
    r"free usage limit reached|usage limit reached|quota|"
    r"insufficient (?:credits|balance)",
    re.IGNORECASE,
)
TRANSIENT_PATTERN = re.compile(
    r"overloaded|rate.?limit|too many requests|internal server error|"
    r"service unavailable|bad gateway|gateway time-?out|network|fetch failed",
    re.IGNORECASE,
)
AUTH_PATTERN = re.compile(
    r"not logged in|sign in|authentication|invalid api key|unauthori[sz]ed",
    re.IGNORECASE,
)

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class OpencodeJudge:
    """An opencode model answering as a soft-skills judge."""

    def __init__(
        self,
        model: str = DEFAULT_OPENCODE_MODEL,
        agent: str = DEFAULT_OPENCODE_AGENT,
        timeout: float = DEFAULT_OPENCODE_TIMEOUT,
        executable: str = "opencode",
        retries: int = 1,
        backoff: float = 1.0,
        runner: Optional[Runner] = None,
        environ: Optional[Mapping[str, str]] = None,
    ):
        """Wire the command, the model and the sandbox environment.

        Args:
            model: Any ``provider/model`` of ``opencode models``.
            agent: Name of the throw-away inline agent.
            timeout: Seconds allowed for one reply.
            executable: Name or path of the opencode command.
            retries: Extra tries after a transient failure.
            backoff: Seconds before the first retry (doubled afterwards).
            runner: Command runner, replaced in tests.
            environ: Parent environment, replaced in tests.
        """
        self.model = model
        self.agent = agent
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._sleep = time.sleep
        self._runner: Runner = runner or run_command
        self._environ = dict(environ if environ is not None else os.environ)
        self.executable = shutil.which(executable) or executable

    @property
    def model_name(self) -> str:
        """Return the model that answered, including its provider."""
        return f"opencode:{self.model}"

    def settings(self, system: str) -> Dict[str, Any]:
        """Return the inline V1 config: one agent, no tools, no sharing.

        opencode 1.x reads V1 keys (``agent``, ``permission``); the V2
        ``agents``/``permissions`` spelling is rejected by this version.
        The caller's system prompt becomes the agent's prompt.
        """
        return {
            "agent": {
                self.agent: {
                    "description": "Soft-skills judge",
                    "mode": "primary",
                    "prompt": (
                        f"{system}\nAnswer only with the JSON object asked "
                        "for. Never use tools."
                    ),
                    "permission": {"*": "deny"},
                    "steps": 1,
                },
            },
            "share": "disabled",
        }

    def command(self, cwd: str, prompt_file: str) -> List[str]:
        """Return the command for one question."""
        return [
            self.executable, "run", "--pure", "--format", "json",
            "--agent", self.agent, "--model", self.model,
            "--title", "judge", "--print-logs", "--log-level", "ERROR",
            "--dir", cwd, "--file", prompt_file,
            "Answer using the attached file",
        ]

    def environment(self, cwd: str, system: str) -> Dict[str, str]:
        """Return the isolated environment of the judge process."""
        env = {
            name: self._environ[name]
            for name in KEPT_ENVIRONMENT if name in self._environ
        }
        env.update({
            "HOME": cwd,
            "PATH": self._environ.get("PATH", "/usr/bin:/bin"),
            "TMPDIR": cwd,
            "OPENCODE_CONFIG_CONTENT": json.dumps(self.settings(system)),
            "OPENCODE_DB": os.path.join(cwd, "judge.db"),
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        })
        return env

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return opencode's reply under a private inline system prompt.

        The caller's ``system`` narrows the judge's task; ``max_tokens``
        is not a CLI option and the prompts already ask for short JSON.
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
        """Make one call with the question in a temporary file."""
        with tempfile.TemporaryDirectory(prefix="judge-opencode-") as cwd:
            prompt_file = os.path.join(cwd, "prompt.txt")
            with open(prompt_file, "w", encoding="utf-8") as handle:
                handle.write(user)
            try:
                done = self._runner(
                    self.command(cwd, prompt_file), "", cwd,
                    self.environment(cwd, system), self.timeout,
                )
            except FileNotFoundError:
                raise ModelError("opencode is not installed.")
            except subprocess.TimeoutExpired:
                raise TransientModelError(
                    f"opencode did not answer within {self.timeout:g} s."
                )
        return self.parse(done.returncode, done.stdout, done.stderr)

    def parse(self, returncode: int, stdout: str, stderr: str) -> str:
        """Return the assistant's text from raw JSON events, or raise."""
        text, error = reply(stdout, stderr)
        if error or returncode != 0:
            self._fail(error or short(stderr) or short(stdout), returncode)
        if not text:
            raise ModelError("opencode gave no answer.")
        return text

    def _fail(self, detail: str, returncode: int) -> None:
        """Raise the error that best describes the failure."""
        message = detail or f"opencode failed (exit {returncode})."
        if QUOTA_PATTERN.search(message):
            raise QuotaExhaustedError(f"opencode: {short(message)}")
        if AUTH_PATTERN.search(message):
            raise ModelAuthError(f"opencode: {short(message)}")
        if TRANSIENT_PATTERN.search(message) or returncode < 0:
            raise TransientModelError(f"opencode: {short(message)}")
        raise ModelError(f"opencode: {short(message)}")


def _error_message(value: Any) -> str:
    """Return the human part of an opencode error object."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return str(value)
    data = value.get("data")
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, str):
        return message
    for key in ("message", "name", "error"):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
    return json.dumps(value, ensure_ascii=False)


def reply(stdout: str, stderr: str) -> Tuple[str, str]:
    """Return ``(text, error)`` from ``opencode run --format json``.

    Text parts are keyed by part id: opencode resends each part with its
    full text as it streams, so appending every update would repeat it.
    The session's title agent is ignored; only assistant messages count.
    """
    parts: Dict[str, str] = {}
    order: List[str] = []
    agents: Dict[str, str] = {}
    error = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        props = event.get("properties") or {}
        if not isinstance(props, dict):
            continue
        if kind == "session.error" and not error:
            error = _error_message(props.get("error"))
        if kind == "message.updated":
            info = props.get("info") or {}
            if isinstance(info, dict):
                message_id = info.get("id")
                if isinstance(message_id, str):
                    agents[message_id] = str(info.get("agent") or "")
                if info.get("error"):
                    error = error or _error_message(info["error"])
        if kind == "message.part.delta":
            part_id = props.get("partID")
            if props.get("field") == "text" and isinstance(part_id, str):
                delta = props.get("delta")
                if isinstance(delta, str):
                    accumulated = parts.get(part_id, "") + delta
                    _remember(parts, order, part_id, accumulated)
        if kind != "message.part.updated":
            continue
        part = props.get("part") or {}
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        part_id = part.get("id")
        text = part.get("text")
        if not isinstance(part_id, str) or not isinstance(text, str):
            continue
        message_id = part.get("messageID")
        if agents.get(str(message_id)) == "title":
            continue
        _remember(parts, order, part_id, text)
    if not error:
        log = short(stderr)
        if log and "stream error" in log.lower():
            error = log
    return "".join(parts[key] for key in order), error


def _remember(parts: Dict[str, str], order: List[str], key: str,
              text: str) -> None:
    """Keep the longest text seen for one part, in first-seen order."""
    if key not in parts:
        order.append(key)
    if len(text) >= len(parts.get(key, "")):
        parts[key] = text


def event(event_type: str, **properties: Any) -> str:
    """Return one raw JSON event line (used by tests)."""
    return json.dumps({"type": event_type, "properties": properties})
