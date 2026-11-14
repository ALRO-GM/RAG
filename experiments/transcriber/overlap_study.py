"""Study identification and overlapped speech on real sessions (TR-14).

Issue #21. The recordings are private: they are read in place from the
transcriber's data directory and every intermediate file (decoded
windows, embeddings, frame activities) goes to ``--cache``, a directory
outside the repository. Only aggregate numbers reach
``experiments/transcriber/overlap.jsonl``: no audio, no text, no names.
Sessions are called ``R1``, ``R2``... in the order of their ids and the
enrolled people ``A``, ``B``... in the order of their ids.

Steps (each reads what the previous ones cached):

``embed``
    Decode each session, rebuild its speech regions and 5 s / 1.5 s
    analysis windows from the stored words exactly as
    :class:`~transcriber.application.transcribe.TranscribeSession`
    does, and embed them with the transcriber's WavLM adapter.
``wavlm-sd``
    Frame-wise two-speaker activity from ``microsoft/wavlm-base-plus-sd``
    (transformers, already installed), in 20 s chunks.
``sortformer``
    Frame-wise four-speaker activity from NVIDIA's streaming Sortformer
    (``nvidia/diar_streaming_sortformer_4spk-v2``). Needs NeMo, which
    is not a dependency of the repository: run this step with an
    interpreter that has it (``docs/transcriber/OVERLAP.md`` §6).
``language``
    The language Whisper detects in every 5 s window (argmax only).
``analyse``
    Write the aggregate rows: window-to-voiceprint similarities,
    separability, the rules that would name both people, overlap per
    candidate, agreement between candidates and with the cheap
    embedding score.

Usage (``--sessions`` and ``--voiceprints`` default to the
transcriber's data directory, which is only read)::

    PYTHONPATH=. python experiments/transcriber/overlap_study.py embed \
        --cache /tmp/tr14
    PYTHONPATH=. python experiments/transcriber/overlap_study.py \
        wavlm-sd --cache /tmp/tr14
    PYTHONPATH=. /path/to/nemo/python \
        experiments/transcriber/overlap_study.py sortformer \
        --cache /tmp/tr14
    PYTHONPATH=. python experiments/transcriber/overlap_study.py \
        language --cache /tmp/tr14
    PYTHONPATH=. python experiments/transcriber/overlap_study.py \
        analyse --cache /tmp/tr14

The script may import :mod:`transcriber`; it never imports ``src``.
"""

import argparse
import importlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from transcriber.application.ports import Audio, RecordedAudio
from transcriber.domain.models import Voiceprint, Word
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.speakers import analysis_windows, speech_regions
from transcriber.domain.voices import (
    FloatArray,
    centre,
    cluster_speakers,
    seed_voices,
    unit_rows,
    voice_centroids,
)

OUTPUT = Path("experiments/transcriber/overlap.jsonl")
SD_MODEL = "microsoft/wavlm-base-plus-sd"
# NeMo Sortformer checkpoints this step can load. Nemotron-3-Diarization
# needs NeMo's development branch, whose lhotse requires Python 3.11:
# it did not run next to the repository's Python 3.10 (OVERLAP.md).
SORTFORMERS = {
    "sortformer-v2": "nvidia/diar_streaming_sortformer_4spk-v2",
}
MODELS = ("wavlm-sd", "sortformer-v2")
LANGUAGE_MODEL = "openai/whisper-large-v3-turbo"
# Every activity is put on one grid before it is compared.
GRID = 0.01
# A frame is active for a speaker at this posterior (both models'
# documented default).
ACTIVE = 0.5
SD_CHUNK = 20.0
# A window or an utterance "falls on overlap" from this share of its
# time on frames a model marks as two voices; an utterance also when
# it holds this many seconds of them.
SHARE = 0.2
SECONDS = 0.5
THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(30)]
FLOORS = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SHARES_AT = (0.8, 0.85, 0.9, 0.95)

Row = Dict[str, Any]
BoolArray = npt.NDArray[np.bool_]


class Session(NamedTuple):
    """One recording: its alias, folder and what the session said."""

    alias: str
    folder: Path
    language: str
    expected: List[str]
    num_speakers: Optional[int]


def sessions_of(root: Path) -> List[Session]:
    """Return the finished sessions under ``root`` as R1, R2, ..."""
    found: List[Session] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = json.loads((folder / "session.json").read_text())
        if meta.get("state") != "done":
            continue
        found.append(Session(
            f"R{len(found) + 1}", folder, str(meta.get("language")),
            list(meta.get("expected_speakers") or []),
            meta.get("num_speakers"),
        ))
    return found


def voiceprints_of(root: Path) -> Tuple[List[Voiceprint], Dict[str, str]]:
    """Return the voiceprints and the alias (A, B, ...) of each id."""
    prints = [
        Voiceprint.model_validate(json.loads(p.read_text())["voiceprint"])
        for p in sorted(root.glob("*.json"))
    ]
    prints.sort(key=lambda v: v.speaker_id)
    return prints, {
        v.speaker_id: chr(ord("A") + i) for i, v in enumerate(prints)
    }


def words_of(session: Session) -> List[Word]:
    """Return the stored (gated, attributed) words of ``session``."""
    raw = json.loads((session.folder / "words.json").read_text())
    return [Word.model_validate(w) for w in raw]


def decode(session: Session) -> Audio:
    """Decode the stored recording with the transcriber's decoder."""
    from transcriber.adapters.outbound.av_audio_decoder import (
        AvAudioDecoder,
    )

    path = session.folder / "audio.flac"
    return AvAudioDecoder().decode(RecordedAudio(path.read_bytes(), "flac"))


def gpu_reset() -> None:
    """Reset torch's peak memory counters when a GPU is in use."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak() -> Optional[float]:
    """Return the peak memory torch reserved since the reset, in GB."""
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_reserved() / 1e9, 2)


# -------------------------------------------------------------- embed


def embed(args: argparse.Namespace) -> None:
    """Rebuild the pipeline's windows and embed them (one npz each)."""
    from transcriber.adapters.outbound.wavlm_voice_encoder import (
        WavLMVoiceEncoder,
    )

    p = TranscriptionParameters()
    prints, alias = voiceprints_of(args.voiceprints)
    args.cache.mkdir(parents=True, exist_ok=True)
    np.save(args.cache / "voiceprints.npy",
            np.array([v.vector for v in prints], dtype=np.float32))
    encoder = WavLMVoiceEncoder(device=args.device)
    meta: List[Row] = []
    for session in sessions_of(args.sessions):
        audio = decode(session)
        words = words_of(session)
        regions = speech_regions(words, p.region_gap)
        coarse_w = analysis_windows(regions, *p.coarse())
        fine_w = analysis_windows(regions, *p.fine())
        gpu_reset()
        began = time.perf_counter()
        coarse = encoder.embed(audio, coarse_w)
        fine = encoder.embed(audio, fine_w)
        seconds = time.perf_counter() - began
        np.savez(
            args.cache / f"{session.alias}.npz",
            coarse=coarse, fine=fine,
            coarse_windows=np.array(coarse_w, dtype=np.float64),
            fine_windows=np.array(fine_w, dtype=np.float64),
            regions=np.array(regions, dtype=np.float64),
        )
        meta.append({
            "alias": session.alias, "duration": audio.duration,
            "language": session.language,
            "expected": [alias[s] for s in session.expected],
            "num_speakers": session.num_speakers,
            "embed_seconds": round(seconds, 2), "gpu_peak_gb": gpu_peak(),
        })
        print(f"{session.alias}: {len(coarse_w)} + {len(fine_w)} windows"
              f" in {seconds:.1f} s", flush=True)
    encoder.release()
    (args.cache / "sessions.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------- overlap models


def on_grid(
    frames: npt.NDArray[np.float32], hop: float, duration: float
) -> npt.NDArray[np.float32]:
    """Resample per-frame activities (frames x speakers) onto ``GRID``."""
    count = int(np.ceil(duration / GRID))
    index = np.minimum(
        (np.arange(count) * GRID / hop).astype(np.int64),
        frames.shape[0] - 1,
    )
    out: npt.NDArray[np.float32] = frames[index].astype(np.float32)
    return out


def wavlm_sd(args: argparse.Namespace) -> None:
    """Frame activities of the WavLM diarization head, 20 s chunks."""
    import torch
    import transformers

    loader: Any = transformers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_reset()
    began = time.perf_counter()
    extractor = loader.AutoFeatureExtractor.from_pretrained(SD_MODEL)
    model = loader.WavLMForAudioFrameClassification.from_pretrained(
        SD_MODEL, torch_dtype=torch.float16 if device == "cuda"
        else torch.float32,
    ).to(device).eval()
    load = time.perf_counter() - began
    timing: List[Row] = []
    for session in sessions_of(args.sessions):
        audio = decode(session)
        rate = audio.sample_rate
        step = int(SD_CHUNK * rate)
        parts: List[npt.NDArray[np.float32]] = []
        began = time.perf_counter()
        for start in range(0, audio.samples.shape[0], step):
            clip = audio.samples[start:start + step]
            inputs = extractor(clip, sampling_rate=rate,
                               return_tensors="pt")
            values = inputs["input_values"].to(device, model.dtype)
            with torch.no_grad():
                logits = model(input_values=values).logits[0]
            parts.append(
                torch.sigmoid(logits.float()).cpu().numpy()
            )
        seconds = time.perf_counter() - began
        frames = np.vstack(parts)
        hop = audio.duration / frames.shape[0]
        np.save(args.cache / f"{session.alias}.wavlm-sd.npy",
                on_grid(frames, hop, audio.duration))
        timing.append({"alias": session.alias, "seconds": seconds,
                       "duration": audio.duration})
        print(f"{session.alias}: {frames.shape} frames in {seconds:.1f} s",
              flush=True)
    save_timing(args.cache, "wavlm-sd", load, timing, gpu_peak())


def sortformer(args: argparse.Namespace) -> None:
    """Frame activities of a NeMo Sortformer model (needs NeMo)."""
    import torch

    name = args.model
    models: Any = importlib.import_module("nemo.collections.asr.models")
    gpu_reset()
    began = time.perf_counter()
    model = models.SortformerEncLabelModel.from_pretrained(
        SORTFORMERS[name], map_location="cuda"
        if torch.cuda.is_available() else "cpu",
    ).eval()
    # The model cards' offline ("very high latency") streaming setting.
    modules = model.sortformer_modules
    modules.chunk_len = 340
    modules.chunk_right_context = 40
    modules.fifo_len = 40
    modules.spkcache_update_period = 300
    load = time.perf_counter() - began
    timing: List[Row] = []
    for number, session in enumerate(sessions_of(args.sessions)):
        audio = decode(session)
        if number == 0:  # warm up once, so the first timing is fair
            model.diarize(audio=[audio.samples[:160000]], batch_size=1,
                          sample_rate=audio.sample_rate)
        began = time.perf_counter()
        _, tensors = model.diarize(
            audio=[audio.samples], batch_size=1,
            sample_rate=audio.sample_rate, include_tensor_outputs=True,
        )
        seconds = time.perf_counter() - began
        frames = tensors[0].squeeze(0).float().cpu().numpy()
        hop = audio.duration / frames.shape[0]
        np.save(args.cache / f"{session.alias}.{name}.npy",
                on_grid(frames, hop, audio.duration))
        timing.append({"alias": session.alias, "seconds": seconds,
                       "duration": audio.duration, "hop": hop})
        print(f"{session.alias}: {frames.shape} frames in {seconds:.1f} s",
              flush=True)
    save_timing(args.cache, name, load, timing, gpu_peak())


def save_timing(
    cache: Path, name: str, load: float, timing: Sequence[Row],
    peak: Optional[float],
) -> None:
    """Keep how long a model took and how much GPU memory it held."""
    (cache / f"timing.{name}.json").write_text(json.dumps({
        "load_seconds": round(load, 1), "gpu_peak_gb": peak,
        "sessions": list(timing),
    }, indent=1))


def language(args: argparse.Namespace) -> None:
    """Detect the language of every 5 s window with Whisper.

    Only the argmax language code is kept per window, to see whether
    the voiceprint similarity depends on the language spoken.
    """
    import torch
    import transformers

    loader: Any = transformers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = loader.WhisperProcessor.from_pretrained(LANGUAGE_MODEL)
    model = loader.WhisperForConditionalGeneration.from_pretrained(
        LANGUAGE_MODEL, torch_dtype=torch.float16 if device == "cuda"
        else torch.float32,
    ).to(device).eval()
    tokenizer = processor.tokenizer
    for session in sessions_of(args.sessions):
        audio = decode(session)
        windows = np.load(args.cache / f"{session.alias}.npz")[
            "coarse_windows"]
        codes: List[str] = []
        for start, end in windows.tolist():
            features = processor(
                audio.cut(start, end), sampling_rate=audio.sample_rate,
                return_tensors="pt",
            ).input_features.to(device, model.dtype)
            with torch.no_grad():
                token = model.detect_language(features)
            codes.append(
                tokenizer.decode(token.tolist()).strip("<|>"))
        (args.cache / f"{session.alias}.language.json").write_text(
            json.dumps(codes))
        print(session.alias, {c: codes.count(c) for c in set(codes)},
              flush=True)


# ------------------------------------------------------------ analyse


class Recording(NamedTuple):
    """Everything cached about one session, ready to compare."""

    alias: str
    duration: float
    coarse: FloatArray
    fine: FloatArray
    coarse_windows: npt.NDArray[np.float64]
    fine_windows: npt.NDArray[np.float64]
    regions: npt.NDArray[np.float64]
    utterances: List[Tuple[float, float]]
    activity: Dict[str, npt.NDArray[np.float32]]
    languages: Optional[List[str]]


def load(cache: Path, sessions: Path) -> List[Recording]:
    """Read the cache (and the utterance times of each transcript)."""
    meta = {r["alias"]: r for r in json.loads(
        (cache / "sessions.json").read_text())}
    out: List[Recording] = []
    for session in sessions_of(sessions):
        alias = session.alias
        arrays = np.load(cache / f"{alias}.npz")
        transcript = json.loads(
            (session.folder / "transcript.json").read_text())
        activity = {
            m: np.load(cache / f"{alias}.{m}.npy") for m in MODELS
            if (cache / f"{alias}.{m}.npy").exists()
        }
        language_file = cache / f"{alias}.language.json"
        out.append(Recording(
            alias, float(meta[alias]["duration"]),
            arrays["coarse"], arrays["fine"], arrays["coarse_windows"],
            arrays["fine_windows"], arrays["regions"],
            [(float(u["start"]), float(u["end"]))
             for u in transcript["utterances"]],
            activity,
            json.loads(language_file.read_text())
            if language_file.exists() else None,
        ))
    return out


def quantiles(values: npt.NDArray[np.float64]) -> List[float]:
    """Return the :data:`QUANTILES` of ``values``, rounded."""
    if values.size == 0:
        return []
    return [round(float(q), 3) for q in np.quantile(values, QUANTILES)]


def auc(positive: npt.NDArray[np.float64],
        negative: npt.NDArray[np.float64]) -> Optional[float]:
    """Chance that a positive scores above a negative (ties count half)."""
    if positive.size == 0 or negative.size == 0:
        return None
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return round(float(greater + ties / 2), 3)


def similarity(vectors: FloatArray, prints: FloatArray) -> FloatArray:
    """Cosine of every (raw, unit) window with every voiceprint."""
    out: FloatArray = unit_rows(vectors) @ unit_rows(prints).T
    return out


def reference(recording: Recording, prints: FloatArray) -> List[int]:
    """Person (voiceprint row) of each 5 s window, without thresholds.

    The two-way clustering of the pipeline (centred long windows, the
    number of people given), each cluster named by the one-to-one
    assignment of its mean to the voiceprints. Not a ground truth: a
    third method the others are compared with.
    """
    count = prints.shape[0]
    labels = cluster_speakers(centre(recording.coarse), count)
    sims = voice_centroids(unit_rows(recording.coarse), labels) @ \
        unit_rows(prints).T
    rows, cols = linear_sum_assignment(sims, maximize=True)
    person = dict(zip(rows.tolist(), cols.tolist()))
    return [person[label] for label in labels]


def expected_rule(
    coarse: FloatArray, prints: FloatArray, floor: float
) -> List[int]:
    """Person of each long window under the expected-speakers rule.

    The windows are clustered into as many voices as people expected
    (centred, as the pipeline does); each voice goes to the expected
    voiceprint of the one-to-one assignment that maximises the total
    cosine, when that cosine reaches ``floor``; -1 is anonymous.
    """
    labels = cluster_speakers(centre(coarse), int(prints.shape[0]))
    sims = voice_centroids(unit_rows(coarse), labels) @ unit_rows(prints).T
    rows, cols = linear_sum_assignment(sims, maximize=True)
    person = {r: c for r, c in zip(rows.tolist(), cols.tolist())
              if sims[r, c] >= floor}
    return [person.get(label, -1) for label in labels]


def seeded_rule(
    coarse: FloatArray, fine: FloatArray, prints: FloatArray,
    threshold: float,
) -> List[int]:
    """Person of each long window under the pipeline's seeding."""
    voiceprints = [
        Voiceprint(speaker_id=f"p{i}", model="m", vector=row.tolist(),
                   consent_version="c",
                   enrolled_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for i, row in enumerate(prints)
    ]
    seeded = seed_voices(coarse, fine, voiceprints, int(prints.shape[0]),
                         threshold)
    ids = {g: int(p[1:]) for g, p in seeded.people.items()}
    return [ids.get(g, -1) for g in seeded.coarse]


def compare(rule: Sequence[int], ref: Sequence[int]) -> Row:
    """Share of windows a rule names as the reference, other, nobody."""
    rule_a, ref_a = np.array(rule), np.array(ref)
    named = rule_a >= 0
    return {
        "same": round(float(np.mean(named & (rule_a == ref_a))), 3),
        "other": round(float(np.mean(named & (rule_a != ref_a))), 3),
        "anonymous": round(float(np.mean(~named)), 3),
        "people_named": sorted({int(x) for x in rule_a[named]}),
    }


def mask(recording: Recording, model: str, voices: int) -> BoolArray:
    """Grid frames where ``model`` hears at least ``voices`` speakers."""
    active = (recording.activity[model] >= ACTIVE).sum(axis=1)
    out: BoolArray = active >= voices
    return out


def span_mask(spans: npt.NDArray[np.float64], count: int) -> BoolArray:
    """Grid frames covered by ``spans`` (seconds)."""
    out = np.zeros(count, dtype=bool)
    for start, end in spans.tolist():
        out[int(start / GRID):int(end / GRID)] = True
    return out


def share_on(spans: Sequence[Tuple[float, float]],
             frames: BoolArray) -> npt.NDArray[np.float64]:
    """Share of each span's frames that are set in ``frames``."""
    out = []
    for start, end in spans:
        first = int(start / GRID)
        last = max(int(end / GRID), first + 1)
        out.append(float(frames[first:last].mean()))
    return np.array(out, dtype=np.float64)


def seconds_on(spans: Sequence[Tuple[float, float]],
               frames: BoolArray) -> npt.NDArray[np.float64]:
    """Seconds of each span on frames set in ``frames``."""
    return np.array([
        float(frames[int(s / GRID):int(e / GRID)].sum()) * GRID
        for s, e in spans
    ], dtype=np.float64)


def identification_rows(
    recordings: Sequence[Recording], prints: FloatArray,
) -> List[Row]:
    """Rows of question (a): who the windows look like, and the rules."""
    names = [chr(ord("A") + i) for i in range(prints.shape[0])]
    rows: List[Row] = [{
        "kind": "voiceprints",
        "cosine": round(float(unit_rows(prints)[0] @ unit_rows(prints)[1]),
                        3),
    }]
    centroids: Dict[Tuple[str, str], FloatArray] = {}
    for rec in recordings:
        ref = np.array(reference(rec, prints))
        sims = similarity(rec.coarse, prints).astype(np.float64)
        for k, name in enumerate(names):
            own, other = ref == k, ref != k
            members = unit_rows(rec.coarse[own])
            centroid = members.mean(axis=0)
            centroids[(rec.alias, name)] = centroid / np.linalg.norm(centroid)
            row: Row = {
                "kind": "window_similarity", "recording": rec.alias,
                "person": name, "windows": int(own.sum()),
            }
            for j, print_name in enumerate(names):
                row[f"to_{print_name}"] = quantiles(sims[own, j])
            row["own_minus_other"] = quantiles(
                sims[own, k] - sims[own][:, [j for j in range(len(names))
                                             if j != k]].max(axis=1))
            row["own_share_at"] = {
                str(t): round(float((sims[own, k] >= t).mean()), 3)
                for t in SHARES_AT}
            row["auc_own_vs_other_windows"] = auc(sims[own, k],
                                                  sims[other, k])
            row["group_to"] = {
                print_name: round(float(
                    centroids[(rec.alias, name)] @ unit_rows(prints)[j]), 3)
                for j, print_name in enumerate(names)}
            rows.append(row)
        if rec.languages is not None:
            codes = np.array(rec.languages)
            for k, name in enumerate(names):
                for code in sorted(set(rec.languages)):
                    chosen = (ref == k) & (codes == code)
                    rows.append({
                        "kind": "language", "recording": rec.alias,
                        "person": name, "language": code,
                        "windows": int(chosen.sum()),
                        "to_own": quantiles(sims[chosen, k]),
                    })
        for threshold in THRESHOLDS:
            rows.append({
                "kind": "seeded", "recording": rec.alias,
                "threshold": threshold,
                **compare(seeded_rule(rec.coarse, rec.fine, prints,
                                      threshold), ref.tolist()),
            })
        for floor in FLOORS:
            rows.append({
                "kind": "expected_rule", "recording": rec.alias,
                "floor": floor,
                **compare(expected_rule(rec.coarse, prints, floor),
                          ref.tolist()),
            })
    rows.extend(session_voiceprint_rows(recordings, prints, centroids))
    keys = sorted(centroids)
    rows.append({
        "kind": "centroids_across_recordings",
        "labels": [f"{r}-{p}" for r, p in keys],
        "cosine": [[round(float(centroids[a] @ centroids[b]), 3)
                    for b in keys] for a in keys],
    })
    return rows


def session_voiceprint_rows(
    recordings: Sequence[Recording], prints: FloatArray,
    centroids: Dict[Tuple[str, str], FloatArray],
) -> List[Row]:
    """What voiceprints taken from another session would have done.

    For each recording and each *other* recording as the source, the
    voiceprints are replaced by the source's reference voices (mean of
    their 5 s windows): how the windows score against them, and whether
    the pipeline's seeding at its default threshold names both people.
    This is what re-enrolling from a conversation on the same phone
    would give, measured on recordings the vectors did not come from.
    """
    names = [chr(ord("A") + i) for i in range(prints.shape[0])]
    rows: List[Row] = []
    for rec in recordings:
        ref = np.array(reference(rec, prints))
        for source in recordings:
            if source.alias == rec.alias:
                continue
            borrowed = np.vstack([centroids[(source.alias, n)]
                                  for n in names]).astype(np.float32)
            sims = similarity(rec.coarse, borrowed).astype(np.float64)
            row: Row = {
                "kind": "session_voiceprints", "recording": rec.alias,
                "source": source.alias,
                "seeded_at_default": compare(seeded_rule(
                    rec.coarse, rec.fine, borrowed, 0.95), ref.tolist()),
            }
            for k, name in enumerate(names):
                own = ref == k
                row[f"{name}_to_own"] = quantiles(sims[own, k])
                row[f"{name}_to_other"] = quantiles(sims[own, 1 - k])
                row[f"{name}_own_share_at"] = {
                    str(t): round(float((sims[own, k] >= t).mean()), 3)
                    for t in SHARES_AT}
            rows.append(row)
    return rows


def overlap_rows(
    recordings: Sequence[Recording], prints: FloatArray, cache: Path,
) -> List[Row]:
    """Rows of question (b): overlap per model, agreement, cheap score."""
    rows: List[Row] = []
    for name in MODELS:
        timing_file = cache / f"timing.{name}.json"
        if timing_file.exists():
            timing = json.loads(timing_file.read_text())
            audio = sum(s["duration"] for s in timing["sessions"])
            spent = sum(s["seconds"] for s in timing["sessions"])
            rows.append({
                "kind": "speed", "model": name,
                "audio_seconds": round(audio, 1),
                "seconds": round(spent, 2),
                "times_real_time": round(audio / spent),
                "load_seconds": timing["load_seconds"],
                "gpu_peak_gb": timing["gpu_peak_gb"],
            })
    for rec in recordings:
        models = [m for m in MODELS if m in rec.activity]
        count = min(a.shape[0] for a in rec.activity.values())
        regions = span_mask(rec.regions, count)
        two = {m: mask(rec, m, 2)[:count] for m in models}
        one = {m: mask(rec, m, 1)[:count] for m in models}
        for m in models:
            speakers = (rec.activity[m] >= ACTIVE).mean(axis=0)
            rows.append({
                "kind": "overlap", "recording": rec.alias, "model": m,
                "speech_seconds": round(float(one[m].sum()) * GRID, 1),
                "overlap_seconds": round(float(two[m].sum()) * GRID, 1),
                "share_of_own_speech": round(
                    float(two[m].sum() / max(one[m].sum(), 1)), 3),
                "region_seconds": round(float(regions.sum()) * GRID, 1),
                "share_of_regions": round(
                    float((two[m] & regions).sum() / regions.sum()), 3),
                "speakers_over_1pct": int((speakers > 0.01).sum()),
                "speech_agrees_with_regions": round(float(
                    (one[m] == regions).mean()), 3),
            })
        both = np.logical_and.reduce([two[m] for m in models])
        either = np.logical_or.reduce([two[m] for m in models])
        if len(models) == 2:
            first, second = models
            rows.append({
                "kind": "agreement", "recording": rec.alias,
                "models": models,
                "iou": round(float(both.sum() / max(either.sum(), 1)), 3),
                f"share_of_{first}_in_{second}": round(
                    float(both.sum() / max(two[first].sum(), 1)), 3),
                f"share_of_{second}_in_{first}": round(
                    float(both.sum() / max(two[second].sum(), 1)), 3),
                "both_seconds": round(float(both.sum()) * GRID, 1),
            })
        utterances = rec.utterances
        for label, frames in [*two.items(), ("both", both)]:
            shares = share_on(utterances, frames)
            secs = seconds_on(utterances, frames)
            rows.append({
                "kind": "utterances", "recording": rec.alias,
                "model": label, "utterances": len(utterances),
                f"share_with_{SHARE}_of_time": round(
                    float((shares >= SHARE).mean()), 3),
                f"share_with_{SECONDS}_s": round(
                    float((secs >= SECONDS).mean()), 3),
            })
        rows.extend(cheap_rows(rec, prints, two))
        if "sortformer-v2" in rec.activity:
            rows.append(diarization_agreement(rec, prints))
    return rows


def cheap_rows(
    rec: Recording, prints: FloatArray, two: Dict[str, BoolArray],
) -> List[Row]:
    """How the embedding-only "two voices" scores follow each model.

    For every 1.5 s window: ``voiceprints_min``, the lower of its
    cosines to the two voiceprints (high when it looks like both);
    ``voiceprints_close``, one minus their difference; ``centres_min``,
    the lower of its cosines to the two reference voices, centred on
    the recording as the pipeline does.
    """
    sims = similarity(rec.fine, prints).astype(np.float64)
    mean: FloatArray = rec.coarse.mean(axis=0).astype(np.float32)
    ref = reference(rec, prints)
    centres = voice_centroids(centre(rec.coarse, mean), ref)
    to_centres = (centre(rec.fine, mean) @ centres.T).astype(np.float64)
    scores = {
        "voiceprints_min": sims.min(axis=1),
        "voiceprints_close": 1.0 - np.abs(sims[:, 0] - sims[:, 1]),
        "centres_min": to_centres.min(axis=1),
    }
    windows = [(float(s), float(e)) for s, e in rec.fine_windows.tolist()]
    rows: List[Row] = []
    for model, frames in two.items():
        positive = share_on(windows, frames) >= SHARE
        for name, score in scores.items():
            rows.append({
                "kind": "cheap_score", "recording": rec.alias,
                "model": model, "score": name,
                "windows": len(windows),
                "overlap_windows": int(positive.sum()),
                "auc": auc(score[positive], score[~positive]),
            })
    return rows


def diarization_agreement(rec: Recording, prints: FloatArray) -> Row:
    """Share of 5 s windows where Sortformer's main voice is the reference.

    Each window takes the Sortformer speaker with the most activity in
    it; speakers are mapped to the reference people one to one.
    """
    activity = rec.activity["sortformer-v2"]
    ref = reference(rec, prints)
    main = [int(activity[int(s / GRID):int(e / GRID)].sum(axis=0).argmax())
            for s, e in rec.coarse_windows.tolist()]
    table = np.zeros((activity.shape[1], prints.shape[0]))
    for m, r in zip(main, ref):
        table[m, r] += 1
    rows, cols = linear_sum_assignment(table, maximize=True)
    return {
        "kind": "sortformer_vs_reference", "recording": rec.alias,
        "windows": len(ref),
        "agreement": round(float(table[rows, cols].sum() / len(ref)), 3),
    }


def analyse(args: argparse.Namespace) -> None:
    """Write the aggregate rows to ``--output`` (rewritten)."""
    prints: FloatArray = np.load(args.cache / "voiceprints.npy")
    recordings = load(args.cache, args.sessions)
    meta = {r["alias"]: r for r in json.loads(
        (args.cache / "sessions.json").read_text())}
    rows = [{"kind": "recording", "recording": r.alias,
             "duration": round(r.duration, 1),
             "coarse_windows": int(r.coarse.shape[0]),
             "fine_windows": int(r.fine.shape[0]),
             "utterances": len(r.utterances),
             "embed_seconds": meta[r.alias]["embed_seconds"],
             "embed_gpu_peak_gb": meta[r.alias]["gpu_peak_gb"]}
            for r in recordings]
    rows += identification_rows(recordings, prints)
    rows += overlap_rows(recordings, prints, args.cache)
    args.output.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(f"{len(rows)} rows to {args.output}")


# --------------------------------------------------------------- main


def main() -> None:
    """Parse the command line and run one step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=["embed", "wavlm-sd", "sortformer", "language",
                         "analyse"])
    parser.add_argument("--sessions", type=Path,
                        default=Path("data/transcriber/sessions"))
    parser.add_argument("--voiceprints", type=Path,
                        default=Path("data/transcriber/voiceprints"))
    parser.add_argument("--cache", type=Path, required=True,
                        help="private directory outside the repository")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=sorted(SORTFORMERS),
                        default="sortformer-v2")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    steps = {"embed": embed, "wavlm-sd": wavlm_sd,
             "sortformer": sortformer, "language": language,
             "analyse": analyse}
    steps[args.step](args)


if __name__ == "__main__":
    main()
