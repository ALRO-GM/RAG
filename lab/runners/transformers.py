"""Lab runner of the ``transformers`` family (models in transformers).

Model:

``nemotron-3-diarization`` (diarize)
    ``nvidia/Nemotron-3-Diarization`` (OpenMDW-1.1, commercial use
    allowed by its card): up to eight speakers, frame logits every
    10 ms, run with the processor's chunked streaming inference (the
    model card's loop: no limit on the recording's length, bounded
    memory) in ``streaming_mode`` (default: the mode with the longest
    latency the processor offers). Segments come from the processor's
    ``extract_speaker_dict`` at ``threshold``; overlap is native.

It uses the project's transformers (5.20.2 has the model type
``nemotron3_diarization``) through the layered path; the environment
adds librosa for the feature extractor. NeMo is not needed. See
``_common`` for the contract.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

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

NEMOTRON = "nvidia/Nemotron-3-Diarization"


def nemotron(context: Context) -> Dict[str, Any]:
    """Diarize with Nemotron-3-Diarization, chunk by chunk."""
    import torch
    from transformers import (
        AutoModelForAudioFrameClassification,
        AutoProcessor,
    )

    context.ignore_unknown(
        ("threshold", "streaming_mode", "num_speakers", "max_speakers")
    )
    if context.param("num_speakers") is not None:
        context.notes.append("num_speakers is not used (up to 8 speakers)")
    processor = AutoProcessor.from_pretrained(NEMOTRON)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        NEMOTRON
    ).to(context.device).eval()
    modes = list(processor.streaming_modes)
    mode = context.param("streaming_mode") or max(
        modes, key=lambda m: _latency(processor, m)
    )
    processor.set_streaming_mode(mode)
    context.model_version = (
        f"{_common.hf_revision(NEMOTRON)} transformers=="
        f"{_common.version_of('transformers')}"
    )
    samples = _common.read_audio(context.source)
    context.loaded()
    rate = processor.feature_extractor.sampling_rate
    logits = []
    cache = None
    with torch.inference_mode():
        for inputs in _chunks(processor, samples, rate):
            inputs = inputs.to(model.device, dtype=model.dtype)
            outputs = model(**inputs, speaker_cache=cache)
            logits.append(outputs.logits)
            cache = outputs.speaker_cache
    joined = torch.cat(logits, dim=1)
    threshold = float(context.param("threshold", 0.5))
    found = processor.extract_speaker_dict(joined, threshold=threshold)[0]
    segments = _common.segments_sorted([
        {"start": s["Start"], "end": s["End"],
         "speaker": f"spk{s['Speaker']}"}
        for s in found
    ])
    context.notes.append(
        f"streaming_mode {mode} ({processor.streaming_latency_ms} ms),"
        f" threshold {threshold}"
    )
    return {"segments": segments,
            "overlap": _common.overlap_spans(segments)}


def _latency(processor: Any, mode: str) -> float:
    """Return the latency of ``mode`` in ms (0 when unknown)."""
    current = processor.streaming_mode
    try:
        processor.set_streaming_mode(mode)
        return float(processor.streaming_latency_ms)
    except Exception:
        return 0.0
    finally:
        processor.set_streaming_mode(current)


def _chunks(processor: Any, audio: Any, rate: int) -> Iterator[Any]:
    """Yield the processor inputs of each chunk (model card's loop)."""
    yield processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=rate, is_streaming=True, is_first_audio_chunk=True,
    )
    frame = processor.num_mel_frames_per_step
    start = processor.audio_chunk_start(frame)
    while start + processor.num_samples_per_audio_chunk <= audio.shape[0]:
        end = start + processor.num_samples_per_audio_chunk
        yield processor(
            audio[start:end], sampling_rate=rate, is_streaming=True,
            is_first_audio_chunk=False,
        )
        frame += processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(frame)
    yield processor(
        audio[start:], sampling_rate=rate, is_streaming=True,
        is_first_audio_chunk=False, is_last_audio_chunk=True,
    )


HANDLERS = {("diarize", "nemotron-3-diarization"): nemotron}


def imports() -> Dict[str, str]:
    """Import the transformers classes and librosa (no weights)."""
    import librosa  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForAudioFrameClassification,
        AutoProcessor,
    )
    from transformers.models.nemotron3_diarization import (  # noqa: F401
        modeling_nemotron3_diarization,
    )

    return {p: str(_common.version_of(p)) for p in (
        "transformers", "librosa", "torch",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("transformers", HANDLERS, imports))
