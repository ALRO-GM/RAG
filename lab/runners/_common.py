"""Shared helpers of the model-lab runners (contract ``lab.result.v1``).

Every runner is a standalone script run by its family's interpreter::

    runner.py <stage> --model ID --in WAV --out DIR [--params JSON]
              [--device cuda|cpu]
    runner.py --selftest

It always writes ``DIR/result.json`` (atomically), success or failure,
and exits with the code of :data:`EXIT_CODES`. Only the standard
library, numpy and soundfile are used here, so every family can import
it; scipy is used for resampling when present.
"""

import argparse
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SCHEMA = "lab.result.v1"
STAGES = ("denoise", "diarize", "recognise")
RATE = 16000
EXIT_CODES = {
    "failed": 1,
    "unknown_model": 2,
    "bad_input": 2,
    "needs_token": 3,
    "out_of_memory": 4,
    "unsupported_language": 5,
    "not_installed": 6,
}
"""Error code of ``result.json`` → exit code of the runner."""


class RunnerError(Exception):
    """A failure the runner reports with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        """Keep the code (a key of :data:`EXIT_CODES`) and message."""
        super().__init__(message)
        self.code = code
        self.message = message


class Context:
    """What a stage function gets: input, output, params, device."""

    def __init__(
        self, stage: str, model: str, source: Path, out: Path,
        params: Dict[str, Any], device: str,
    ) -> None:
        """Keep the arguments of the run."""
        self.stage = stage
        self.model = model
        self.source = source
        self.out = out
        self.params = params
        self.device = device
        self.notes: List[str] = []
        self.load_seconds = 0.0
        self.model_version: Optional[str] = None
        self._started = time.monotonic()

    def loaded(self) -> None:
        """Mark the end of model loading (the rest is processing)."""
        self.load_seconds = time.monotonic() - self._started
        self._started = time.monotonic()

    def processing_seconds(self) -> float:
        """Return the seconds since :meth:`loaded` (or the start)."""
        return time.monotonic() - self._started

    def param(self, name: str, default: Any = None) -> Any:
        """Return parameter ``name`` (``default`` when absent)."""
        return self.params.get(name, default)

    def ignore_unknown(self, known: Tuple[str, ...]) -> None:
        """Add a note for every parameter not in ``known``."""
        for key in sorted(self.params):
            if key not in known:
                self.notes.append(f"parameter {key!r} ignored")


Handler = Callable[[Context], Dict[str, Any]]
"""A stage of one model: returns the stage's result fields."""


def version_of(package: str) -> Optional[str]:
    """Return the installed version of ``package``, or ``None``."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def read_audio(path: Path, rate: int = RATE) -> np.ndarray:
    """Return the WAV at ``path`` as mono float32 at ``rate``."""
    import soundfile

    try:
        data, source_rate = soundfile.read(
            str(path), dtype="float32", always_2d=True
        )
    except Exception as exc:  # unreadable input
        raise RunnerError("bad_input", f"cannot read the input: {exc}")
    mono = data.mean(axis=1).astype(np.float32)
    return resample(mono, int(source_rate), rate)


def resample(samples: np.ndarray, source: int, target: int) -> np.ndarray:
    """Return ``samples`` resampled from ``source`` to ``target`` Hz."""
    if source == target:
        return samples.astype(np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(source, target)
        out = resample_poly(samples, target // g, source // g)
    except ImportError:  # pragma: no cover - every env has scipy today
        n = int(round(len(samples) * target / source))
        grid = np.linspace(0, len(samples) - 1, n)
        out = np.interp(grid, np.arange(len(samples)), samples)
    return np.asarray(out, dtype=np.float32)


def pieces(samples: Any, piece: int, rate: int, run: Any) -> Any:
    """Apply ``run`` to overlapping pieces and cross-fade them (1 s)."""
    fade = rate
    out = np.zeros(len(samples), dtype=np.float32)
    weight = np.zeros(len(samples), dtype=np.float32)
    start = 0
    while start < len(samples):
        end = min(start + piece + fade, len(samples))
        chunk = samples[start:end]
        if len(chunk) < rate // 10:  # a tail too short to enhance
            out[start:end] += chunk
            weight[start:end] += 1.0
            break
        enhanced = run(chunk)
        enhanced = np.pad(enhanced, (0, len(chunk) - len(enhanced)))
        ramp = np.ones(len(chunk), dtype=np.float32)
        edge = min(fade, len(chunk))
        if start > 0:
            ramp[:edge] = np.linspace(0.0, 1.0, edge)
        if end < len(samples):
            ramp[-edge:] = np.minimum(ramp[-edge:],
                                      np.linspace(1.0, 0.0, edge))
        out[start:end] += enhanced * ramp
        weight[start:end] += ramp
        if end >= len(samples):
            break
        start += piece
    return out / np.maximum(weight, 1e-6)


def write_audio(
    path: Path, samples: np.ndarray, rate: int, length: int
) -> None:
    """Write ``samples`` at ``rate`` as 16 kHz mono PCM16 of ``length``.

    The audio is resampled to 16 kHz and padded or trimmed to exactly
    ``length`` samples (the input's), so times stay comparable.
    """
    import soundfile

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    mono = resample(mono, rate, RATE)
    if mono.shape[0] < length:
        mono = np.pad(mono, (0, length - mono.shape[0]))
    mono = np.clip(mono[:length], -1.0, 1.0)
    temporary = path.with_name(f".{path.name}.tmp")
    soundfile.write(
        str(temporary), mono, RATE, subtype="PCM_16", format="WAV"
    )
    os.replace(temporary, path)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    """Write ``value`` as JSON to ``path`` atomically."""
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def hf_token() -> Optional[str]:
    """Return the Hugging Face token (``HF_TOKEN``, then the hub's)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def require_token(url: str) -> str:
    """Return the token, or fail with ``needs_token`` naming ``url``."""
    token = hf_token()
    if not token:
        raise RunnerError(
            "needs_token",
            f"needs token: accept the licence at {url} and log in"
            " (huggingface-cli login)",
        )
    return token


def hf_revision(repo: str) -> str:
    """Return ``repo@<commit>`` of the cached snapshot, or ``repo``."""
    try:
        from huggingface_hub import scan_cache_dir

        for item in scan_cache_dir().repos:
            if item.repo_id == repo:
                revisions = sorted(
                    item.revisions, key=lambda r: r.last_modified
                )
                if revisions:
                    return f"{repo}@{revisions[-1].commit_hash[:12]}"
    except Exception:
        pass
    return repo


def peak_gpu_mb(sampled: Optional[float] = None) -> Optional[float]:
    """Return the peak GPU memory of this process in MiB, if known.

    The larger of torch's peak reserved memory and ``sampled`` (the
    largest use ``nvidia-smi`` reported for this process while the
    stage ran, which also sees libraries that do not allocate through
    torch, such as CTranslate2 or ONNX Runtime).
    """
    found: List[float] = [] if sampled is None else [sampled]
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            reserved = torch.cuda.max_memory_reserved() / 2 ** 20
            if reserved > 0:
                found.append(reserved)
    except Exception:
        pass
    return round(max(found), 1) if found else None


class GpuSampler:
    """Poll ``nvidia-smi`` for this process's GPU memory in a thread."""

    def __init__(self, every: float = 0.5) -> None:
        """Prepare to poll every ``every`` seconds."""
        import threading

        self.every = every
        self.peak: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self) -> "GpuSampler":
        """Start polling (does nothing without ``nvidia-smi``)."""
        import shutil

        if shutil.which("nvidia-smi"):
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop polling."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(5)

    def _poll(self) -> None:
        """Record the largest memory use of this process."""
        import subprocess

        me = str(os.getpid())
        while not self._stop.is_set():
            try:
                text = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
            except Exception:
                return
            for line in text.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == me:
                    try:
                        value = float(parts[1])
                    except ValueError:
                        continue
                    self.peak = max(self.peak or 0.0, value)
            self._stop.wait(self.every)


def default_device() -> str:
    """Return ``cuda`` when torch sees a GPU, else ``cpu``."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def classify(exc: BaseException) -> RunnerError:
    """Return the :class:`RunnerError` an unexpected exception means."""
    if isinstance(exc, RunnerError):
        return exc
    text = f"{type(exc).__name__}: {exc}"
    if (
        "OutOfMemory" in type(exc).__name__ or "out of memory" in text
        or isinstance(exc, MemoryError)
    ):
        return RunnerError("out_of_memory", text[:500])
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return RunnerError("not_installed", text[:500])
    return RunnerError("failed", text[:500])


def selftest(family: str, imports: Callable[[], Dict[str, str]]) -> int:
    """Print the self-test line of ``family``; return the exit code.

    ``imports`` imports the family's libraries (no weights, no network)
    and returns ``package -> version``.
    """
    report: Dict[str, Any] = {
        "ok": True, "family": family,
        "python": platform.python_version(),
    }
    try:
        report["packages"] = imports()
        import torch

        report["cuda"] = bool(torch.cuda.is_available())
    except BaseException as exc:
        report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


def parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the runner's arguments."""
    parser = argparse.ArgumentParser(description="Model-lab runner.")
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--model")
    parser.add_argument("--in", dest="source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args(argv)


def main(
    family: str,
    handlers: Dict[Tuple[str, str], Handler],
    imports: Callable[[], Dict[str, str]],
    argv: Optional[List[str]] = None,
) -> int:
    """Run one stage of one model, or the self-test; return exit code.

    ``handlers`` maps ``(stage, model id)`` to the function that runs
    it. ``result.json`` is written whatever happens.
    """
    args = parse(argv)
    if args.selftest:
        return selftest(family, imports)
    if not (args.stage and args.model and args.source and args.out):
        print("stage, --model, --in and --out are needed", file=sys.stderr)
        return EXIT_CODES["bad_input"]
    args.source, args.out = args.source.resolve(), args.out.resolve()
    result: Dict[str, Any] = {
        "schema": SCHEMA, "stage": args.stage, "model": args.model,
    }
    try:
        if not args.out.is_dir():
            raise RunnerError("bad_input", "--out must be a directory")
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as exc:
            raise RunnerError("bad_input", f"--params: {exc}")
        if not isinstance(params, dict):
            raise RunnerError("bad_input", "--params must be an object")
        handler = handlers.get((args.stage, args.model))
        if handler is None:
            raise RunnerError(
                "unknown_model",
                f"{family} has no {args.stage} model {args.model!r}",
            )
        device = args.device or default_device()
        context = Context(
            args.stage, args.model, args.source, args.out, params, device
        )
        audio_seconds = _duration(args.source)
        with GpuSampler() as sampler:
            fields = handler(context)
        result.update(
            ok=True,
            model_version=context.model_version or args.model,
            device=context.device,
            seconds=round(context.processing_seconds(), 3),
            load_seconds=round(context.load_seconds, 3),
            peak_gpu_mb=peak_gpu_mb(sampler.peak),
            audio_seconds=audio_seconds,
            notes=context.notes,
            **fields,
        )
        code = 0
    except BaseException as exc:
        error = classify(exc)
        if error.code == "failed":
            traceback.print_exc()
        result.update(ok=False, error={
            "code": error.code, "message": error.message,
        })
        code = EXIT_CODES.get(error.code, 1)
    if args.out.is_dir():
        write_json(args.out / "result.json", result)
    return code


def _duration(path: Path) -> float:
    """Return the length of the WAV at ``path`` in seconds."""
    import soundfile

    try:
        info = soundfile.info(str(path))
    except Exception as exc:
        raise RunnerError("bad_input", f"cannot read the input: {exc}")
    return round(float(info.frames) / float(info.samplerate), 3)


def segments_sorted(
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return non-empty segments rounded to ms and sorted by start."""
    kept = [
        {"start": round(float(s["start"]), 3),
         "end": round(float(s["end"]), 3),
         "speaker": str(s["speaker"])}
        for s in segments if float(s["end"]) > float(s["start"])
    ]
    return sorted(kept, key=lambda s: (s["start"], s["end"], s["speaker"]))


def overlap_spans(
    segments: List[Dict[str, Any]], step: float = 0.01
) -> List[List[float]]:
    """Return the ``[start, end]`` spans where two segments overlap."""
    if not segments:
        return []
    end = max(s["end"] for s in segments)
    count = np.zeros(int(np.ceil(end / step)) + 1, dtype=np.int32)
    for s in segments:
        count[int(round(s["start"] / step)):int(round(s["end"] / step))] += 1
    spans: List[List[float]] = []
    on = count >= 2
    index = 0
    while index < len(on):
        if on[index]:
            first = index
            while index < len(on) and on[index]:
                index += 1
            spans.append([round(first * step, 3), round(index * step, 3)])
        index += 1
    return spans


def spans_to_segments(
    activity: np.ndarray, step: float, labels: List[str],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Turn ``frames x speakers`` activity into speaker segments."""
    segments: List[Dict[str, Any]] = []
    on = activity >= threshold
    for column, label in enumerate(labels):
        index = 0
        frames = on[:, column]
        while index < len(frames):
            if frames[index]:
                first = index
                while index < len(frames) and frames[index]:
                    index += 1
                segments.append({
                    "start": first * step, "end": index * step,
                    "speaker": label,
                })
            index += 1
    return segments_sorted(segments)


def language_param(context: Context) -> Optional[str]:
    """Return the ``language`` parameter (``None`` for auto)."""
    value = context.param("language")
    if value in (None, "", "auto"):
        return None
    return str(value).split("-", 1)[0].lower()
