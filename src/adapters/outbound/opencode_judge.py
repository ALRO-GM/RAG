"""Judge through the opencode command line (any of its models).

Written for opencode 2.x (``run --standalone``): the assessment prompt and
the transcript are attached as one file in an empty directory, so they
never appear in the process arguments, and the reply is read from
``opencode run --format json``'s raw events (``text``, ``step_finish``,
``error``), not from the terminal.
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
from src.domain.skills.usage import Usage

DEFAULT_OPENCODE_MODEL = "opencode/deepseek-v4.1-flash-free"
"""Model used when ``--judge opencode`` names none."""

DEFAULT_OPENCODE_AGENT = "ragjudge"
"""Kept for configuration compatibility; opencode 2.x has no inline agent."""

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
            agent: Unused (opencode 2.x has no inline agents); kept so
                existing settings still load.
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
        self.last_usage: Optional[Usage] = None

    @property
    def model_name(self) -> str:
        """Return the model that answered, including its provider."""
        return f"opencode:{self.model}"

    def message(self, system: str, user: str) -> str:
        """Return the attached file's text: the task, then the transcript.

        opencode 2.x has no inline agent, so the caller's system prompt
        travels in the same file as the question, never in the process
        arguments.
        """
        return (
            f"{system}\n\nAnswer only with the JSON object asked for. "
            f"Never use tools.\n\n{user}"
        )

    def command(self, prompt_file: str) -> List[str]:
        """Return the command for one question (opencode 2.x)."""
        return [
            self.executable, "run", "--standalone", "--archived",
            "--format", "json", "--model", self.model,
            "--title", "judge", "--file", prompt_file,
            "Answer using the attached file",
        ]

    def environment(self, cwd: str) -> Dict[str, str]:
        """Return the environment of the judge process.

        ``HOME`` is kept: with a private one opencode loses its
        credentials and the anonymous free-usage limit applies at once.
        """
        env = {
            name: self._environ[name]
            for name in KEPT_ENVIRONMENT if name in self._environ
        }
        env["PATH"] = self._environ.get("PATH", "/usr/bin:/bin")
        env["TMPDIR"] = cwd
        if "HOME" in self._environ:
            env["HOME"] = self._environ["HOME"]
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
        self.last_usage = None
        with tempfile.TemporaryDirectory(prefix="judge-opencode-") as cwd:
            prompt_file = os.path.join(cwd, "prompt.txt")
            with open(prompt_file, "w", encoding="utf-8") as handle:
                handle.write(self.message(system, user))
            try:
                done = self._runner(
                    self.command(prompt_file), "", cwd,
                    self.environment(cwd), self.timeout,
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
        self.last_usage = usage_of(stdout)
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


def _events(stdout: str) -> List[Dict[str, Any]]:
    """Return the JSON object lines of ``--format json`` output."""
    found: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            found.append(item)
    return found


def usage_of(stdout: str) -> Optional[Usage]:
    """Return the tokens of the ``step_finish`` events.

    Each ``step_finish`` part carries ``tokens`` (``input``, ``output``,
    ``reasoning`` and ``cache.read``/``cache.write``) and ``cost``; the
    steps are summed. ``None`` when no event reports tokens.
    """
    total = Usage(calls=1)
    found = False
    for item in _events(stdout):
        part = item.get("part")
        if (
            item.get("type") != "step_finish" or not isinstance(part, dict)
            or not isinstance(part.get("tokens"), dict)
        ):
            continue
        found = True
        tokens = part["tokens"]
        cache = tokens.get("cache")
        cache = cache if isinstance(cache, dict) else {}
        cost = part.get("cost")
        total = total.plus(Usage(
            input_tokens=_count(tokens.get("input")),
            output_tokens=(
                _count(tokens.get("output")) + _count(tokens.get("reasoning"))
            ),
            cache_creation_input_tokens=_count(cache.get("write")),
            cache_read_input_tokens=_count(cache.get("read")),
            cost_usd=float(cost) if isinstance(cost, (int, float)) else 0.0,
        ))
    return total if found else None


def _count(value: Any) -> int:
    """Return ``value`` as a token count (0 when it is not a number)."""
    return value if isinstance(value, int) and value > 0 else 0


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

    Every ``text`` event carries one finished text part; they are joined
    in order. An ``error`` event gives the error message.
    """
    texts: List[str] = []
    error = ""
    for item in _events(stdout):
        if item.get("type") == "error" and not error:
            error = _error_message(item.get("error"))
        part = item.get("part")
        if item.get("type") == "text" and isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    if not error:
        log = short(stderr)
        if log and "stream error" in log.lower():
            error = log
    return "".join(texts), error
