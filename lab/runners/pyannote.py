"""Lab runner of the ``pyannote`` family (pyannote.audio 4).

Models (both gated on Hugging Face: the owner accepts the conditions
and logs in; the token is read from ``HF_TOKEN`` or the hub's login):

``pyannote-3.1`` (diarize)
    ``pyannote/speaker-diarization-3.1`` (MIT): powerset segmentation
    3.0, WeSpeaker embeddings, agglomerative clustering.
``pyannote-community-1`` (diarize)
    ``pyannote/speaker-diarization-community-1`` (CC BY 4.0): the
    newer open pipeline (VBx clustering), with an exclusive output.

Audio is passed in memory (``{"waveform", "sample_rate"}``), so
torchcodec never decodes a file. See ``_common`` for the contract.
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

PIPELINES = {
    "pyannote-3.1": "pyannote/speaker-diarization-3.1",
    "pyannote-community-1": "pyannote/speaker-diarization-community-1",
}


def diarize(context: Context) -> Dict[str, Any]:
    """Diarize with one of the pyannote pipelines (needs the token)."""
    repo = PIPELINES[context.model]
    url = f"https://huggingface.co/{repo}"
    token = _common.require_token(url)
    context.ignore_unknown(
        ("num_speakers", "min_speakers", "max_speakers", "exclusive")
    )
    import torch
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(repo, token=token)
    except Exception as exc:
        name = type(exc).__name__
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            raise RunnerError(
                "needs_token",
                f"needs token: accept the licence at {url} (and its"
                " segmentation model's) and log in (huggingface-cli login)",
            )
        raise
    if pipeline is None:
        raise RunnerError(
            "needs_token",
            f"needs token: accept the licence at {url} and log in"
            " (huggingface-cli login)",
        )
    pipeline.to(torch.device(context.device))
    context.model_version = (
        f"{_common.hf_revision(repo)} pyannote.audio=="
        f"{_common.version_of('pyannote.audio')}"
    )
    samples = _common.read_audio(context.source)
    context.loaded()
    options = {
        key: int(context.param(key)) for key in (
            "num_speakers", "min_speakers", "max_speakers"
        ) if context.param(key) is not None
    }
    waveform = torch.from_numpy(samples).unsqueeze(0)
    output = pipeline(
        {"waveform": waveform, "sample_rate": _common.RATE}, **options
    )
    annotation = getattr(output, "speaker_diarization", output)
    segments: List[Dict[str, Any]] = [
        {"start": turn.start, "end": turn.end, "speaker": str(label)}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    segments = _common.segments_sorted(segments)
    result: Dict[str, Any] = {
        "segments": segments, "overlap": _common.overlap_spans(segments),
    }
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        context.notes.append(
            "segments: the pipeline's speaker_diarization (overlap kept);"
            " its exclusive_speaker_diarization is in 'exclusive'"
        )
        result["exclusive"] = _common.segments_sorted([
            {"start": turn.start, "end": turn.end, "speaker": str(label)}
            for turn, _, label in exclusive.itertracks(yield_label=True)
        ])
    return result


HANDLERS = {("diarize", model): diarize for model in PIPELINES}


def imports() -> Dict[str, str]:
    """Import pyannote.audio's pipeline class (no weights)."""
    from pyannote.audio import Pipeline  # noqa: F401

    return {p: str(_common.version_of(p)) for p in (
        "pyannote.audio", "torch", "torchaudio", "lightning",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("pyannote", HANDLERS, imports))
