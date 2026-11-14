"""Lab runner of the ``ctranslate2`` family (faster-whisper).

Model:

``faster-whisper-large-v3-turbo`` (recognise)
    ``openai/whisper-large-v3-turbo`` (MIT) converted locally to
    CTranslate2 with int8 weights and float16 compute
    (``ct2-transformers-converter``, run by ``make lab-env
    FAMILY=ctranslate2`` into ``~/.local/share/ragputin/lab-models/
    faster-whisper-large-v3-turbo-int8``), decoded by faster-whisper
    with word timestamps (beam 5, no VAD filter by default).

CTranslate2 needs cuBLAS 12 and cuDNN 9: they are loaded from the
project's ``nvidia`` packages (visible through the layered path) before
CTranslate2 starts. See ``_common`` for the contract.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

# The script's directory must not shadow a package of the same name
# (nemo, pyannote...): drop it from the path, load the helpers by file.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
_SPEC = importlib.util.spec_from_file_location("_common", HERE / "_common.py")
assert _SPEC is not None and _SPEC.loader is not None
_common = importlib.util.module_from_spec(_SPEC)
sys.modules["_common"] = _common
_SPEC.loader.exec_module(_common)
Context = _common.Context
RunnerError = _common.RunnerError

MODELS = Path.home() / ".local" / "share" / "ragputin" / "lab-models"
CONVERTED = {
    "faster-whisper-large-v3-turbo": (
        "faster-whisper-large-v3-turbo-int8", "int8_float16",
        "openai/whisper-large-v3-turbo",
    ),
}


def preload_cuda() -> None:
    """Load cuBLAS 12 and cuDNN 9 from the ``nvidia`` pip packages."""
    import ctypes
    import importlib.util as util

    spec = util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    roots = [Path(p) for p in spec.submodule_search_locations]
    for pattern in ("cublas/lib/libcublasLt.so.*", "cublas/lib/libcublas.so.*",
                    "cudnn/lib/libcudnn*.so.*"):
        for root in roots:
            for library in sorted(root.glob(pattern)):
                try:
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def recognise(context: Context) -> Dict[str, Any]:
    """Recognise with faster-whisper on the converted model."""
    context.ignore_unknown(("language", "beam_size", "vad_filter"))
    directory, compute, source = CONVERTED[context.model]
    path = MODELS / directory
    if not (path / "model.bin").is_file():
        raise RunnerError(
            "not_installed",
            f"the converted model is missing: run make lab-env"
            f" FAMILY=ctranslate2 (converts {source})",
        )
    if context.device == "cuda":
        preload_cuda()
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(path), device=context.device,
        compute_type=compute if context.device == "cuda" else "int8",
    )
    context.model_version = (
        f"{source} ct2:{compute} faster-whisper=="
        f"{_common.version_of('faster-whisper')}"
    )
    samples = _common.read_audio(context.source)
    context.loaded()
    language = _common.language_param(context)
    if language is not None and language not in model.supported_languages:
        raise RunnerError(
            "unsupported_language", f"Whisper does not know {language!r}"
        )
    segments, info = model.transcribe(
        samples, language=language, word_timestamps=True,
        beam_size=int(context.param("beam_size", 5)),
        vad_filter=bool(context.param("vad_filter", False)),
    )
    words: List[Dict[str, Any]] = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append({
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3), "text": text,
                    "probability": round(float(word.probability), 4),
                })
    context.notes.append(
        f"detected {info.language} ({info.language_probability:.2f})"
        if language is None else f"language forced to {language}"
    )
    return {"words": words, "language": info.language}


HANDLERS = {("recognise", model): recognise for model in CONVERTED}


def imports() -> Dict[str, str]:
    """Import faster-whisper and CTranslate2 (no weights)."""
    import ctranslate2  # noqa: F401
    from faster_whisper import WhisperModel  # noqa: F401

    return {p: str(_common.version_of(p)) for p in (
        "faster-whisper", "ctranslate2", "torch",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("ctranslate2", HANDLERS, imports))
