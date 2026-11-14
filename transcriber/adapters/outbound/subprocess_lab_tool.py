"""Run a lab model as a subprocess in its own environment.

Each family of models has an environment outside the repository
(``<envs_dir>/<family>``, made by ``make lab-env``) and a runner script
(``lab/runners/<family>.py``) that follows the runner contract of
``docs/transcriber/LAB.md`` §3:

    <env>/bin/python lab/runners/<family>.py <stage> --model <id>
        --in <wav> --out <dir> --params <json>

The runner writes ``<dir>/result.json`` (and a denoiser
``<dir>/audio.wav``). The process is started from an argument list (no
shell), in its own process group, with a minimal environment: a few
variables are passed through (cache locations, the GPU choice, offline
switches) and the Hugging Face token is added only for a gated model,
read from ``HF_TOKEN`` or the standard token file; it never appears on
the command line or in a message. Output goes to ``<dir>/runner.log``.
A run longer than ``timeout`` seconds, or cancelled, is killed with its
whole process group.

:class:`FileSystemLabEnvironment` tells whether a family's environment
is ready (its interpreter, its runner and the marker ``make lab-env``
leaves after the runner's self-test), whether a token is present and
whether a model's weights are in the Hugging Face cache.
:class:`JsonCatalogue` reads the catalogue file.
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import ValidationError

from transcriber.application.lab_ports import ToolRequest
from transcriber.domain.errors import LabStepError, StorageError
from transcriber.domain.lab import (
    Catalogue,
    Segment,
    StageOutcome,
)
from transcriber.domain.models import Word

READY_MARKER = ".ragputin-lab-ready"
"""File ``make lab-env`` leaves in an environment once its self-test passed."""

PASSED_VARIABLES = (
    "HOME", "LANG", "LC_ALL", "TMPDIR", "HF_HOME", "HF_HUB_CACHE",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "XDG_CACHE_HOME",
    "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_HOME",
)
"""Variables of the service's environment a runner also gets."""

DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"

EXIT_CODES: Dict[int, str] = {
    1: "failed", 2: "bad_input", 3: "needs_token", 4: "out_of_memory",
    5: "unsupported_language", 6: "not_installed",
}
"""Runner exit code → error code, when it left no ``result.json``."""

POLL_SECONDS = 0.5
MAX_MESSAGE = 600


def token_file(environ: Mapping[str, str]) -> Path:
    """Return where ``huggingface-cli login`` keeps the token."""
    home = environ.get("HF_HOME")
    if home:
        return Path(home) / "token"
    cache = environ.get("XDG_CACHE_HOME") or str(
        Path(environ.get("HOME", "~")).expanduser() / ".cache"
    )
    return Path(cache) / "huggingface" / "token"


def read_token(environ: Mapping[str, str]) -> Optional[str]:
    """Return the Hugging Face token, or ``None`` when there is none."""
    value = environ.get("HF_TOKEN", "").strip()
    if value:
        return value
    try:
        value = token_file(environ).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def hub_dir(environ: Mapping[str, str]) -> Path:
    """Return the Hugging Face hub cache directory."""
    if environ.get("HF_HUB_CACHE"):
        return Path(environ["HF_HUB_CACHE"])
    if environ.get("HF_HOME"):
        return Path(environ["HF_HOME"]) / "hub"
    cache = environ.get("XDG_CACHE_HOME") or str(
        Path(environ.get("HOME", "~")).expanduser() / ".cache"
    )
    return Path(cache) / "huggingface" / "hub"


class FileSystemLabEnvironment:
    """What is installed for the lab on this machine (see the module)."""

    def __init__(
        self, envs_dir: Path, runners_dir: Path,
        environ: Mapping[str, str],
    ) -> None:
        """Find environments in ``envs_dir`` and runners in ``runners_dir``."""
        self._envs = Path(envs_dir).expanduser()
        self._runners = Path(runners_dir)
        self._environ = environ

    def interpreter(self, family: str) -> Path:
        """Return the Python of ``family``'s environment."""
        return self._envs / family / "bin" / "python"

    def runner(self, family: str) -> Path:
        """Return the runner script of ``family``."""
        return self._runners / f"{family}.py"

    def family_problem(self, family: str) -> Optional[str]:
        """Return why ``family`` cannot run, or ``None`` when it can."""
        if not self.runner(family).is_file():
            return f"no runner lab/runners/{family}.py"
        if not self.interpreter(family).exists():
            return (
                f"not installed: run make lab-env FAMILY={family}"
            )
        if not (self._envs / family / READY_MARKER).is_file():
            return (
                f"not verified: its self-test has not passed (make"
                f" lab-env FAMILY={family})"
            )
        return None

    def token_present(self) -> bool:
        """Tell whether a Hugging Face token is available."""
        return read_token(self._environ) is not None

    def weights_cached(self, weight: str) -> Optional[bool]:
        """Tell whether the repo ``weight`` is in the hub cache."""
        if "://" in weight or weight.count("/") != 1:
            return None
        folder = hub_dir(self._environ) / (
            "models--" + weight.replace("/", "--")
        ) / "snapshots"
        try:
            return any(folder.iterdir())
        except OSError:
            return False


class SubprocessLabTool:
    """Run one stage with the family's runner (see the module)."""

    def __init__(
        self,
        environment: FileSystemLabEnvironment,
        environ: Mapping[str, str],
        timeout: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Use ``environment``'s interpreters and runners."""
        self._environment = environment
        self._environ = environ
        self._timeout = timeout
        self._clock = clock

    def release(self) -> None:
        """Nothing is held in this process."""

    def command(self, request: ToolRequest) -> List[str]:
        """Return the argument list that runs ``request``."""
        family = request.entry.family
        return [
            str(self._environment.interpreter(family)),
            str(self._environment.runner(family)),
            request.stage,
            "--model", request.entry.id,
            "--in", str(request.audio),
            "--out", str(request.workspace),
            "--params", json.dumps(dict(request.params), sort_keys=True),
        ]

    def environment(self, gated: bool) -> Dict[str, str]:
        """Return the runner's environment (token only when ``gated``)."""
        env = {"PATH": DEFAULT_PATH, "PYTHONNOUSERSITE": "1",
               "PYTHONUNBUFFERED": "1"}
        for name in PASSED_VARIABLES:
            if name in self._environ:
                env[name] = self._environ[name]
        env.setdefault("LANG", "C.UTF-8")
        if gated:
            token = read_token(self._environ)
            if token is not None:
                env["HF_TOKEN"] = token
        return env

    def run(
        self, request: ToolRequest, cancelled: Callable[[], bool]
    ) -> StageOutcome:
        """Run the runner and return its outcome (see the module)."""
        family = request.entry.family
        problem = self._environment.family_problem(family)
        if problem is not None:
            raise LabStepError("not_installed", f"{family}: {problem}")
        env = self.environment(request.entry.gated)
        token = env.get("HF_TOKEN")
        log = request.workspace / "runner.log"
        started = self._clock()
        with open(log, "wb") as output:
            process = subprocess.Popen(
                self.command(request), stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, env=env,
                cwd=str(request.workspace), start_new_session=True,
            )
            try:
                code = self._wait(process, started, cancelled)
            finally:
                if process.poll() is None:
                    _kill(process)
        return self._outcome(request, code, log, token)

    def _wait(
        self, process: "subprocess.Popen[bytes]", started: float,
        cancelled: Callable[[], bool],
    ) -> int:
        """Wait for the process; kill it when cancelled or too slow."""
        while True:
            try:
                return process.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            if cancelled():
                _kill(process)
                raise LabStepError("cancelled", "The run was cancelled.")
            if self._clock() - started > self._timeout:
                _kill(process)
                raise LabStepError(
                    "timeout",
                    f"The runner took longer than {self._timeout:.0f} s.",
                )

    def _outcome(
        self, request: ToolRequest, code: int, log: Path,
        token: Optional[str],
    ) -> StageOutcome:
        """Read ``result.json``; raise :class:`LabStepError` on failure."""
        path = request.workspace / "result.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise LabStepError(
                EXIT_CODES.get(code, "failed"),
                _clean(
                    f"The runner exited with code {code} and no result:"
                    f" {_tail(log)}", token, request,
                ),
            ) from None
        if not isinstance(data, dict):
            raise LabStepError("failed", "The runner's result is not JSON.")
        if not data.get("ok"):
            error = data.get("error") or {}
            raise LabStepError(
                str(error.get("code") or EXIT_CODES.get(code, "failed")),
                _clean(str(error.get("message") or ""), token, request),
            )
        if code != 0:
            raise LabStepError(
                "failed", f"The runner wrote a result but exited {code}."
            )
        return parse_result(data, request)


def parse_result(
    data: Mapping[str, Any], request: ToolRequest
) -> StageOutcome:
    """Return a runner's successful ``result.json`` as an outcome."""
    if data.get("stage") != request.stage or data.get("model") != (
        request.entry.id
    ):
        raise LabStepError(
            "failed", "The runner answered for another stage or model."
        )
    try:
        return StageOutcome(
            stage=request.stage, model=request.entry.id,
            model_version=_text(data.get("model_version")),
            device=_text(data.get("device")),
            seconds=data.get("seconds"),
            load_seconds=data.get("load_seconds"),
            peak_gpu_mb=data.get("peak_gpu_mb"),
            audio_seconds=data.get("audio_seconds"),
            notes=[str(n)[:500] for n in data.get("notes") or []][:20],
            segments=[
                Segment(start=float(s["start"]), end=float(s["end"]),
                        speaker=str(s["speaker"]))
                for s in data.get("segments") or []
            ],
            overlap=[
                (float(a), float(b)) for a, b in data.get("overlap") or []
            ],
            words=[
                Word(text=str(w["text"]), start=float(w["start"]),
                     end=max(float(w["end"]), float(w["start"])),
                     probability=_probability(w.get("probability")))
                for w in data.get("words") or []
            ],
            language=_text(data.get("language")),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise LabStepError(
            "failed", f"The runner's result is malformed: {exc}"[:600]
        ) from exc


def _text(value: Any) -> Optional[str]:
    """Return ``value`` as a short string, or ``None``."""
    return None if value is None else str(value)[:200]


def _probability(value: Any) -> Optional[float]:
    """Return a probability clamped to [0, 1], or ``None``."""
    if value is None:
        return None
    return min(max(float(value), 0.0), 1.0)


def _kill(process: "subprocess.Popen[bytes]") -> None:
    """Kill the process and its group, and reap it."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()


def _tail(log: Path, lines: int = 3) -> str:
    """Return the last lines of the runner's log."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " | ".join(text.strip().splitlines()[-lines:])


def _clean(text: str, token: Optional[str], request: ToolRequest) -> str:
    """Return ``text`` without the token or the private paths, shortened."""
    if token:
        text = text.replace(token, "<token>")
    for path in (request.workspace, request.audio):
        text = text.replace(str(path), f"<{path.name}>")
    return text[:MAX_MESSAGE]


class JsonCatalogue:
    """The catalogue read from a JSON file (``lab/catalogue.json``)."""

    def __init__(self, path: Path) -> None:
        """Read ``path`` when asked (its last version is kept)."""
        self._path = Path(path)
        self._cached: Optional[Catalogue] = None
        self._stamp: Optional[float] = None

    def load(self) -> Catalogue:
        """Return the catalogue, read again when the file changed."""
        try:
            stamp = self._path.stat().st_mtime
            if self._cached is None or stamp != self._stamp:
                text = self._path.read_text(encoding="utf-8")
                self._cached = Catalogue.model_validate_json(text)
                self._stamp = stamp
        except OSError as exc:
            raise StorageError(
                f"Cannot read the lab catalogue {self._path.name}."
            ) from exc
        except ValidationError as exc:
            raise StorageError(
                f"The lab catalogue is invalid: {exc.errors()[0]['msg']}"
            ) from exc
        return self._cached


class OsProcessTable:
    """Process ids of this machine (``os.kill(pid, 0)``)."""

    def current(self) -> int:
        """Return this process's id."""
        return os.getpid()

    def alive(self, pid: int) -> bool:
        """Tell whether process ``pid`` exists."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
