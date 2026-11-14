"""Lab runner of the ``speechbrain`` family (SpeechBrain 1.1).

Models:

``sepformer-dns4`` (denoise)
    ``speechbrain/sepformer-dns4-16k-enhancement`` (Apache-2.0): a
    SepFormer trained on DNS-4 for 16 kHz enhancement. Run on 10 s
    pieces with 1 s cross-faded overlaps, so memory does not grow with
    the recording.
``metricgan-plus`` (denoise)
    ``speechbrain/metricgan-plus-voicebank`` (Apache-2.0): spectral
    mask enhancement trained on VoiceBank-DEMAND.
``ecapa-spectral`` (diarize)
    The lab's own simple diarizer (not a SpeechBrain recipe): an energy
    speech mask, 1.5 s windows every 0.75 s inside speech, ECAPA-TDNN
    embeddings (``speechbrain/spkrec-ecapa-voxceleb``, Apache-2.0),
    spectral clustering of their cosine affinities (the number of
    speakers given, else the largest eigengap up to ``max_speakers``),
    each 10 ms speech frame labelled by the nearest window centre. One
    speaker per frame: it never marks overlap.

Audio is handed to the models as tensors (no torchaudio file I/O).
See ``_common`` for the contract.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

SEPFORMER = "speechbrain/sepformer-dns4-16k-enhancement"
METRICGAN = "speechbrain/metricgan-plus-voicebank"
ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
SAVE = Path.home() / ".local" / "share" / "ragputin" / "lab-models"
FRAME = 0.01


def _fetch(cls: Any, repo: str, device: str) -> Any:
    """Load a SpeechBrain pretrained interface from the hub."""
    return cls.from_hparams(
        source=repo, savedir=str(SAVE / "speechbrain" / repo.split("/")[1]),
        run_opts={"device": device},
    )


def matched_level(out: Any, chunk: Any) -> Any:
    """Scale ``out`` to the RMS level of the input ``chunk``.

    SepFormer is trained scale-invariant: its output has an arbitrary
    level (seen: an RMS about 17 times the input's, clipped), and
    MetricGAN+ doubled it. The diarizers and the voice model then hear
    a distorted voice, so each piece is brought back to its input's
    level before the pieces are joined.
    """
    import numpy as np

    have = float(np.sqrt(np.mean(np.square(out)))) if len(out) else 0.0
    want = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
    if have <= 1e-9:
        return out
    return out * (want / have)


def sepformer(context: Context) -> Dict[str, Any]:
    """Enhance with the DNS-4 SepFormer, 10 s at a time."""
    import numpy as np
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    context.ignore_unknown(("piece_seconds",))
    model = _fetch(SepformerSeparation, SEPFORMER, context.device)
    context.model_version = _common.hf_revision(SEPFORMER)
    samples = _common.read_audio(context.source)
    context.loaded()
    piece = int(float(context.param("piece_seconds", 10.0)) * _common.RATE)

    def run(chunk: "np.ndarray") -> "np.ndarray":
        """Enhance one piece."""
        batch = torch.from_numpy(chunk).unsqueeze(0).to(context.device)
        with torch.inference_mode():
            out = model.separate_batch(batch)
        return matched_level(
            out[0, :, 0].float().cpu().numpy()[:len(chunk)], chunk
        )

    enhanced = _common.pieces(samples, piece, _common.RATE, run)
    _common.write_audio(
        context.out / "audio.wav", enhanced, _common.RATE, len(samples)
    )
    return {"audio": "audio.wav", "sample_rate": _common.RATE}


def metricgan(context: Context) -> Dict[str, Any]:
    """Enhance with MetricGAN+ (VoiceBank), 10 s at a time."""
    import numpy as np
    import torch
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    context.ignore_unknown(("piece_seconds",))
    model = _fetch(SpectralMaskEnhancement, METRICGAN, context.device)
    context.model_version = _common.hf_revision(METRICGAN)
    samples = _common.read_audio(context.source)
    context.loaded()
    piece = int(float(context.param("piece_seconds", 10.0)) * _common.RATE)

    def run(chunk: "np.ndarray") -> "np.ndarray":
        """Enhance one piece."""
        batch = torch.from_numpy(chunk).unsqueeze(0).to(context.device)
        with torch.inference_mode():
            out = model.enhance_batch(
                batch, lengths=torch.tensor([1.0], device=context.device)
            )
        return matched_level(out[0].float().cpu().numpy()[:len(chunk)],
                             chunk)

    enhanced = _common.pieces(samples, piece, _common.RATE, run)
    _common.write_audio(
        context.out / "audio.wav", enhanced, _common.RATE, len(samples)
    )
    return {"audio": "audio.wav", "sample_rate": _common.RATE}


def speech_mask(samples: Any, rate: int) -> Any:
    """Return a 10 ms speech mask from frame energy.

    A frame is speech when its level is within 30 dB of the loud
    frames (95th percentile) and at least 12 dB above the quiet ones
    (10th percentile); gaps under 0.3 s are filled and bursts under
    0.2 s dropped.
    """
    import numpy as np

    hop = int(rate * FRAME)
    count = len(samples) // hop
    frames = samples[:count * hop].reshape(count, hop)
    level = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    loud, quiet = np.percentile(level, 95), np.percentile(level, 10)
    mask = (level > loud - 30) & (level > quiet + 12)
    mask = _fill(mask, int(0.3 / FRAME), True)
    return _fill(mask, int(0.2 / FRAME), False)


def _fill(mask: Any, length: int, value: bool) -> Any:
    """Set runs of ``not value`` shorter than ``length`` to ``value``."""
    out = mask.copy()
    index = 0
    while index < len(out):
        if out[index] != value:
            first = index
            while index < len(out) and out[index] != value:
                index += 1
            inner = first > 0 and index < len(out)
            if index - first < length and (inner or not value):
                out[first:index] = value
        else:
            index += 1
    return out


def _windows(mask: Any, length: float, hop: float) -> List[Tuple[int, int]]:
    """Return frame spans of windows inside the speech regions."""
    spans: List[Tuple[int, int]] = []
    size, step = int(length / FRAME), int(hop / FRAME)
    index = 0
    while index < len(mask):
        if mask[index]:
            first = index
            while index < len(mask) and mask[index]:
                index += 1
            if index - first <= size:
                spans.append((first, index))
                continue
            start = first
            while start + size <= index:
                spans.append((start, start + size))
                start += step
            if spans[-1][1] < index:
                spans.append((index - size, index))
        index += 1
    return spans


def spectral_labels(vectors: Any, count: Any, most: int) -> Any:
    """Cluster unit vectors by spectral clustering of their cosines."""
    import numpy as np
    from sklearn.cluster import KMeans

    n = vectors.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int)
    affinity = np.clip(vectors @ vectors.T, 0.0, 1.0)
    keep = max(int(n * 0.2), 2)  # p-pruning: each row's strongest
    pruned = np.zeros_like(affinity)
    order = np.argsort(-affinity, axis=1)[:, :keep]
    rows = np.arange(n)[:, None]
    pruned[rows, order] = affinity[rows, order]
    affinity = (pruned + pruned.T) / 2
    degree = affinity.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-10))
    laplacian = np.eye(n) - inverse[:, None] * affinity * inverse[None, :]
    values, vectors_ = np.linalg.eigh(laplacian)
    if count is None:
        gaps = np.diff(values[:most + 1])
        count = int(np.argmax(gaps[:most]) + 1)
    count = max(1, min(int(count), n))
    embedding = vectors_[:, :count]
    embedding /= np.maximum(
        np.linalg.norm(embedding, axis=1, keepdims=True), 1e-10
    )
    return KMeans(count, n_init=10, random_state=0).fit_predict(embedding)


def ecapa(context: Context) -> Dict[str, Any]:
    """Diarize: energy mask, ECAPA windows, spectral clustering."""
    import numpy as np
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    context.ignore_unknown(("num_speakers", "max_speakers", "window",
                            "hop"))
    model = _fetch(EncoderClassifier, ECAPA, context.device)
    context.model_version = _common.hf_revision(ECAPA)
    samples = _common.read_audio(context.source)
    context.loaded()
    mask = speech_mask(samples, _common.RATE)
    spans = _windows(mask, float(context.param("window", 1.5)),
                     float(context.param("hop", 0.75)))
    if not spans:
        return {"segments": []}
    hop = int(_common.RATE * FRAME)
    width = max(b - a for a, b in spans) * hop
    batch = np.zeros((len(spans), width), dtype=np.float32)
    lengths = np.zeros(len(spans), dtype=np.float32)
    for row, (a, b) in enumerate(spans):
        piece = samples[a * hop:b * hop]
        batch[row, :len(piece)] = piece
        lengths[row] = len(piece) / width
    vectors = []
    with torch.inference_mode():
        for first in range(0, len(spans), 64):
            out = model.encode_batch(
                torch.from_numpy(batch[first:first + 64]).to(context.device),
                torch.from_numpy(lengths[first:first + 64]).to(
                    context.device),
            )
            vectors.append(out.squeeze(1).float().cpu().numpy())
    matrix = np.concatenate(vectors)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True),
                         1e-10)
    labels = spectral_labels(
        matrix, context.param("num_speakers"),
        int(context.param("max_speakers") or 8),
    )
    centres = np.array([(a + b) / 2 for a, b in spans])
    frames = np.flatnonzero(mask)
    nearest = np.abs(frames[:, None] - centres[None, :]).argmin(axis=1)
    frame_label = np.full(len(mask), -1)
    frame_label[frames] = labels[nearest]
    segments: List[Dict[str, Any]] = []
    index = 0
    while index < len(frame_label):
        label = frame_label[index]
        first = index
        while index < len(frame_label) and frame_label[index] == label:
            index += 1
        if label >= 0:
            segments.append({"start": first * FRAME, "end": index * FRAME,
                             "speaker": f"spk{label}"})
    context.notes.append(
        f"{len(spans)} windows, {len(set(labels.tolist()))} speakers"
    )
    return {"segments": _common.segments_sorted(segments)}


HANDLERS = {
    ("denoise", "sepformer-dns4"): sepformer,
    ("denoise", "metricgan-plus"): metricgan,
    ("diarize", "ecapa-spectral"): ecapa,
}


def imports() -> Dict[str, str]:
    """Import the SpeechBrain interfaces used (no weights)."""
    from speechbrain.inference.enhancement import (  # noqa: F401
        SpectralMaskEnhancement,
    )
    from speechbrain.inference.separation import (  # noqa: F401
        SepformerSeparation,
    )
    from speechbrain.inference.speaker import (  # noqa: F401
        EncoderClassifier,
    )
    import sklearn  # noqa: F401

    return {p: str(_common.version_of(p)) for p in (
        "speechbrain", "torch", "torchaudio", "scikit-learn",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("speechbrain", HANDLERS, imports))
