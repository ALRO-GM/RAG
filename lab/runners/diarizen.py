"""Lab runner of the ``diarizen`` family (BUT Speech@FIT DiariZen).

Model:

``diarizen-wavlm-large-s80-md-v2`` (diarize)
    ``BUT-FIT/diarizen-wavlm-large-s80-md-v2``: a pruned WavLM-Large
    powerset segmentation model with WeSpeaker ResNet34-LM embeddings
    (``pyannote/wespeaker-voxceleb-resnet34-LM``, CC BY 4.0, not gated)
    and VBx clustering, through DiariZen's fork of pyannote.audio 3.1.1.
    **The DiariZen weights are CC BY-NC 4.0: evaluation only, no
    commercial use.** Overlapping speech is native (powerset).

DiariZen reads its input with ``torchaudio.load``, which in torchaudio
2.11 needs FFmpeg through torchcodec; the runner replaces it with a
soundfile reader for the one WAV it is given. The fork also imports
``torchaudio.AudioMetaData`` (a type, removed from torchaudio): an
empty class stands in.

A recording where the segmentation finds no speaker at all (silence,
noise only) makes every local speaker of every chunk inactive; the
fork then builds an array of ``max(cluster) + 1 = -1`` clusters and
fails with "negative dimensions are not allowed". The runner answers
such a recording with no segments and a note instead; recordings with
speech take the pipeline's own path unchanged. See ``_common`` for the
contract.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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

DIARIZEN = {
    "diarizen-wavlm-large-s80-md-v2":
        "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
}
EMBEDDING = "pyannote/wespeaker-voxceleb-resnet34-LM"
NO_SPEECH = "no speech found: every speaker of every chunk is inactive"
"""The note of a recording the segmentation finds no speaker in."""


class NoSpeech(Exception):
    """The segmentation found no active speaker anywhere."""


def guard_no_speech(pipeline: Any) -> None:
    """Make ``pipeline`` raise :class:`NoSpeech` instead of failing.

    The fork's ``reconstruct`` gets the hard clusters (``-2`` for an
    inactive local speaker of a chunk); when no cluster is 0 or more,
    there is no speaker to reconstruct and it would fail (see the
    module doc). Otherwise the original ``reconstruct`` runs as before.
    """
    import numpy as np

    reconstruct: Callable[..., Any] = pipeline.reconstruct

    def guarded(segmentations: Any, hard_clusters: Any, count: Any) -> Any:
        """Refuse a reconstruction with no active speaker."""
        clusters = np.asarray(hard_clusters)
        if clusters.size == 0 or int(clusters.max()) < 0:
            raise NoSpeech()
        return reconstruct(segmentations, hard_clusters, count)

    pipeline.reconstruct = guarded


def _shims() -> None:
    """Make torchaudio look as the fork expects (see the module doc)."""
    import soundfile
    import torch
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            """Stand-in for torchaudio's removed metadata class."""

        torchaudio.AudioMetaData = AudioMetaData

    def load(path: Any, *args: Any, **kwargs: Any) -> Tuple[Any, int]:
        """Read a WAV as ``(channels x samples, rate)``, like torchaudio."""
        data, rate = soundfile.read(str(path), dtype="float32",
                                    always_2d=True)
        return torch.from_numpy(data.T.copy()), int(rate)

    torchaudio.load = load
    # torch >= 2.6 loads checkpoints with ``weights_only``: allow the
    # few plain classes these pyannote-era checkpoints pickle.
    from pyannote.audio.core.task import (
        Problem,
        Resolution,
        Specifications,
    )

    torch.serialization.add_safe_globals([
        torch.torch_version.TorchVersion, Specifications, Problem,
        Resolution,
    ])


def diarize(context: Context) -> Dict[str, Any]:
    """Diarize with DiariZen (VBx clustering)."""
    import tempfile

    context.ignore_unknown(
        ("num_speakers", "min_speakers", "max_speakers", "batch_size")
    )
    _shims()
    from diarizen.pipelines.inference import DiariZenPipeline

    repo = DIARIZEN[context.model]
    pipeline = DiariZenPipeline.from_pretrained(repo)
    guard_no_speech(pipeline)
    batch = int(context.param("batch_size", 8))
    pipeline.segmentation_batch_size = batch
    pipeline.embedding_batch_size = batch
    count = context.param("num_speakers")
    if count is not None:
        pipeline.min_speakers = pipeline.max_speakers = int(count)
    else:
        if context.param("min_speakers") is not None:
            pipeline.min_speakers = int(context.param("min_speakers"))
        if context.param("max_speakers") is not None:
            pipeline.max_speakers = int(context.param("max_speakers"))
    context.model_version = (
        f"{_common.hf_revision(repo)} {_common.hf_revision(EMBEDDING)}"
    )
    samples = _common.read_audio(context.source)
    context.loaded()
    with tempfile.TemporaryDirectory() as work:
        wav = Path(work) / "audio.wav"
        _common.write_audio(wav, samples, _common.RATE, len(samples))
        try:
            annotation = pipeline(str(wav), sess_name="audio")
        except NoSpeech:
            annotation = None
    segments: List[Dict[str, Any]] = []
    if annotation is None:
        context.notes.append(NO_SPEECH)
    else:
        segments = [
            {"start": turn.start, "end": turn.end,
             "speaker": f"spk{label}"}
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
    segments = _common.segments_sorted(segments)
    context.notes.append("weights CC BY-NC 4.0 (non-commercial)")
    return {"segments": segments,
            "overlap": _common.overlap_spans(segments)}


HANDLERS = {("diarize", model): diarize for model in DIARIZEN}


def imports() -> Dict[str, str]:
    """Import DiariZen's pipeline (no weights)."""
    _shims()
    from diarizen.pipelines.inference import (  # noqa: F401
        DiariZenPipeline,
    )

    return {p: str(_common.version_of(p)) for p in (
        "diarizen", "pyannote.audio", "torch", "torchaudio",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("diarizen", HANDLERS, imports))
