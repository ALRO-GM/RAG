"""Lab runner of the ``nemo`` family (NVIDIA NeMo 3.0.0).

Models:

``sortformer-streaming-v2`` (diarize)
    ``nvidia/diar_streaming_sortformer_4spk-v2``: frame posteriors of up
    to four speakers every 80 ms, run with the model card's offline
    ("very high latency") streaming setting. A frame is a speaker's
    when its posterior is at least ``threshold`` (0.5); overlapping
    segments are kept and the overlap stretches are reported.
``nemo-clustering`` (diarize)
    NeMo's clustering diarizer: MarbleNet VAD, TitaNet-L embeddings at
    several scales, spectral clustering (``num_speakers`` when given).
``parakeet-tdt-0.6b-v3`` (recognise)
    ``nvidia/parakeet-tdt-0.6b-v3``: 25 European languages (Spanish
    included), the language is detected (no language argument), word
    timestamps from the TDT decoder.

Run with the ``nemo`` environment's interpreter; see ``_common``.
"""

import importlib.util
import json
import sys
import tempfile
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

SORTFORMER = "nvidia/diar_streaming_sortformer_4spk-v2"
PARAKEET = "nvidia/parakeet-tdt-0.6b-v3"
TITANET = "titanet_large"
VAD = "vad_multilingual_marblenet"
PARAKEET_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el",
    "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es",
    "sv", "ru", "uk",
)
"""The 25 languages of the parakeet-tdt-0.6b-v3 model card."""


def _quiet() -> None:
    """Keep NeMo's logging to warnings."""
    import logging

    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel(logging.WARNING)


def sortformer(context: Context) -> Dict[str, Any]:
    """Diarize with streaming Sortformer v2 (offline setting)."""
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    context.ignore_unknown(("threshold", "num_speakers", "max_speakers"))
    if context.param("num_speakers") is not None:
        context.notes.append("num_speakers is not used (up to 4 speakers)")
    threshold = float(context.param("threshold", 0.5))
    _quiet()
    model = SortformerEncLabelModel.from_pretrained(
        SORTFORMER, map_location=context.device
    ).eval()
    modules = model.sortformer_modules
    modules.chunk_len = 340
    modules.chunk_right_context = 40
    modules.fifo_len = 40
    modules.spkcache_update_period = 300
    context.model_version = _common.hf_revision(SORTFORMER)
    samples = _common.read_audio(context.source)
    context.loaded()
    with torch.inference_mode():
        _, tensors = model.diarize(
            audio=[samples], batch_size=1, sample_rate=_common.RATE,
            include_tensor_outputs=True,
        )
    frames = tensors[0].squeeze(0).float().cpu().numpy()
    step = (len(samples) / _common.RATE) / frames.shape[0]
    labels = [f"spk{n}" for n in range(frames.shape[1])]
    segments = _common.spans_to_segments(frames, step, labels, threshold)
    context.notes.append(f"frame step {step:.3f} s, threshold {threshold}")
    return {"segments": segments,
            "overlap": _common.overlap_spans(segments)}


def clustering(context: Context) -> Dict[str, Any]:
    """Diarize with NeMo's clustering diarizer (VAD + TitaNet-L)."""
    context.ignore_unknown(("num_speakers", "max_speakers", "batch_size"))
    _quiet()
    with tempfile.TemporaryDirectory(prefix="nemo-diar-") as tmp:
        return _clustered(context, Path(tmp))


def _clustered(context: Context, work: Path) -> Dict[str, Any]:
    """Run the clustering diarizer with its files in ``work``."""
    from nemo.collections.asr.models import ClusteringDiarizer
    from omegaconf import OmegaConf

    samples = _common.read_audio(context.source)
    wav = work / "audio.wav"
    _common.write_audio(wav, samples, _common.RATE, len(samples))
    manifest = work / "manifest.json"
    count = context.param("num_speakers")
    manifest.write_text(json.dumps({
        "audio_filepath": str(wav), "offset": 0, "duration": None,
        "label": "infer", "text": "-", "num_speakers": count,
        "rttm_filepath": None, "uem_filepath": None,
    }) + "\n")
    config = OmegaConf.create(_clustering_config(
        str(manifest), str(work / "out"), context.device,
        count is not None, int(context.param("max_speakers") or 8),
        int(context.param("batch_size", 16)),
    ))
    diarizer = ClusteringDiarizer(cfg=config)
    context.model_version = f"nemo:{VAD}+{TITANET}"
    context.loaded()
    diarizer.diarize()
    rttm = work / "out" / "pred_rttms" / "audio.rttm"
    segments: List[Dict[str, Any]] = []
    for line in rttm.read_text().splitlines():
        parts = line.split()
        if len(parts) > 7 and parts[0] == "SPEAKER":
            start, length = float(parts[3]), float(parts[4])
            segments.append({"start": start, "end": start + length,
                             "speaker": parts[7]})
    segments = _common.segments_sorted(segments)
    return {"segments": segments,
            "overlap": _common.overlap_spans(segments)}


def _clustering_config(
    manifest: str, out: str, device: str, oracle: bool, max_speakers: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Return NeMo's ``diar_infer_meeting.yaml`` settings, inline."""
    return {
        "name": "ClusterDiarizer", "num_workers": 0, "sample_rate": 16000,
        "batch_size": batch_size, "device": device, "verbose": False,
        "diarizer": {
            "manifest_filepath": manifest, "out_dir": out,
            "oracle_vad": False, "collar": 0.25, "ignore_overlap": True,
            "vad": {
                "model_path": VAD, "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.63, "shift_length_in_sec": 0.08,
                    "smoothing": False, "overlap": 0.5, "onset": 0.9,
                    "offset": 0.5, "pad_onset": 0, "pad_offset": 0,
                    "min_duration_on": 0, "min_duration_off": 0.6,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": TITANET,
                "parameters": {
                    "window_length_in_sec": [3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
                    "shift_length_in_sec": [1.5, 1.25, 1.0, 0.75, 0.5, 0.25],
                    "multiscale_weights": [1, 1, 1, 1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": oracle,
                    "max_num_speakers": max_speakers,
                    "enhanced_count_thres": 80, "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False, "chunk_cluster_count": 50,
                    "embeddings_per_chunk": 10000,
                },
            },
            "msdd_model": {"model_path": None,
                           "parameters": {"use_speaker_model_from_ckpt": True,
                                          "infer_batch_size": 25,
                                          "sigmoid_threshold": [0.7],
                                          "seq_eval_mode": False,
                                          "split_infer": True,
                                          "diar_window_length": 50,
                                          "overlap_infer_spk_limit": 5}},
            "asr": {"model_path": None,
                    "parameters": {"asr_based_vad": False}},
        },
    }


def _extract_in_memory(context: Context) -> None:
    """Unpack ``.nemo`` archives in ``/dev/shm`` when it has room.

    NeMo unpacks the 2.5 GB archive into a temporary directory while it
    loads; on a nearly full disk that copy is better kept in memory.
    """
    import shutil

    shm = Path("/dev/shm")
    if shm.is_dir() and shutil.disk_usage(shm).free > 6 * 2 ** 30:
        tempfile.tempdir = str(shm)
        context.notes.append("archive unpacked in /dev/shm")


def parakeet(context: Context) -> Dict[str, Any]:
    """Recognise with parakeet-tdt-0.6b-v3 (word timestamps)."""
    import torch
    from nemo.collections.asr.models import ASRModel

    context.ignore_unknown(("language", "local_attention"))
    language = _common.language_param(context)
    if language is not None and language not in PARAKEET_LANGUAGES:
        raise RunnerError(
            "unsupported_language",
            f"parakeet-tdt-0.6b-v3 does not know {language!r}",
        )
    if language is not None:
        context.notes.append("language is detected by the model, not set")
    _quiet()
    _extract_in_memory(context)
    model = ASRModel.from_pretrained(PARAKEET, map_location=context.device)
    model.eval()
    if context.param("local_attention", True):
        # Long audio: limited-context attention, as the card advises
        # for recordings longer than 24 minutes; also bounds memory.
        model.change_attention_model("rel_pos_local_attn", [256, 256])
        model.change_subsampling_conv_chunking_factor(1)
        context.notes.append("local attention [256, 256]")
    context.model_version = _common.hf_revision(PARAKEET)
    samples = _common.read_audio(context.source)
    context.loaded()
    with torch.inference_mode():
        output = model.transcribe(
            [samples], batch_size=1, timestamps=True, verbose=False
        )
    hypothesis = output[0]
    words = [
        {"start": round(float(w["start"]), 3),
         "end": round(float(w["end"]), 3), "text": str(w["word"])}
        for w in hypothesis.timestamp.get("word", [])
    ]
    return {"words": words, "language": None}


HANDLERS = {
    ("diarize", "sortformer-streaming-v2"): sortformer,
    ("diarize", "nemo-clustering"): clustering,
    ("recognise", "parakeet-tdt-0.6b-v3"): parakeet,
}


def imports() -> Dict[str, str]:
    """Import NeMo's ASR collection (no weights, no network)."""
    import nemo.collections.asr  # noqa: F401
    from nemo.collections.asr.models import (  # noqa: F401
        ASRModel,
        ClusteringDiarizer,
        SortformerEncLabelModel,
    )

    return {p: str(_common.version_of(p)) for p in (
        "nemo-toolkit", "torch", "lightning", "lhotse",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("nemo", HANDLERS, imports))
