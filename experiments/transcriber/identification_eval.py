"""Measure speaker identification with enrolment (ticket TR-05).

Synthetic recordings are built from Multilingual LibriSpeech Spanish
(public read speech with transcripts and reader ids), as in
``experiments/audio/meeting_eval.py``: utterances of several readers
concatenated with 0.2-1.0 s of digital silence between them. The
``dev`` split (20 readers) is used to choose rules and thresholds; the
``test`` split (20 other readers) is held out and only reported.

Per split the plan holds, without reusing any utterance:

* one enrolment sample per reader: about 30 s of *other* utterances of
  that reader, cut at 30.0 s;
* two-reader enrolment samples (a second reader takes 50%, 33% or 20%
  of the 30 s), which the sample rules should reject;
* one-voice recordings (6 utterances of one reader), two-voice
  recordings (8 alternating turns of two readers) and four-reader
  meetings (16 turns, never the same reader twice in a row).

``prepare`` downloads the two parquet files on demand (about 315 MB)
into ``--data-dir`` (ignored by git), recognises every recording with
the transcriber's Whisper adapter and embeds its windows with the
transcriber's WavLM adapter, caching the results so that an interrupted
run resumes. ``analyse`` reads the cache and writes the rows of
``experiments/transcriber/*.jsonl`` (the analysis files are rewritten;
``recognition.jsonl`` keeps one row per recording, written once).

Usage::

    PYTHONPATH=. python experiments/transcriber/identification_eval.py \
        prepare
    PYTHONPATH=. python experiments/transcriber/identification_eval.py \
        analyse

The script may import :mod:`transcriber`; it never imports ``src``.
"""

import argparse
import io
import json
import random
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
from scipy.optimize import linear_sum_assignment

from transcriber.adapters.outbound.av_audio_decoder import (
    DEFAULT_SAMPLE_RATE,
    resample,
    to_mono,
)
from transcriber.application.ports import Audio
from transcriber.domain.activity import heard_words, speech_activity
from transcriber.domain.enrolment import SampleRules, enrolment_vector
from transcriber.domain.errors import SampleRejectedError
from transcriber.domain.models import Voiceprint, Word
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.speakers import (
    SpeakerSpan,
    analysis_windows,
    assign_speakers,
    smooth_speakers,
    spans_from_windows,
    speaker_label,
    speech_regions,
    speech_seconds,
)
from transcriber.domain.voices import (
    FloatArray,
    Normalisation,
    centre,
    centre_on_session,
    cluster_speakers,
    group_voices,
    match_voices,
    seed_voices,
    unit_rows,
    unit_vectors,
    voice_centroids,
)

DATASET = (
    "https://huggingface.co/datasets/facebook/multilingual_librispeech/"
    "resolve/main/spanish/{split}-00000-of-00001.parquet"
)
SPLITS = ("dev", "test")
LANGUAGE = "es"
ENROL_SECONDS = 30.0
SECOND_READER_SHARES = (0.5, 1 / 3, 0.2)
# Coarse below 0.85, fine above: raw cosines of different readers'
# voices are mostly between 0.7 and 0.97.
THRESHOLDS = [round(float(x), 2) for x in np.arange(-0.2, 0.851, 0.05)] + [
    round(float(x), 2) for x in np.arange(0.86, 0.991, 0.01)]
OUTPUT = Path("experiments/transcriber")
GROUP_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
OLD_SINGLE_VOICE_THRESHOLD = 0.9

Row = Dict[str, Any]


# --------------------------------------------------------------- data


def ensure_parquet(data_dir: Path, split: str) -> Path:
    """Return the local parquet of ``split``, downloading it if needed."""
    target = data_dir / "mls" / f"spanish-{split}.parquet"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print(f"downloading {split} split to {target}", flush=True)
    urllib.request.urlretrieve(DATASET.format(split=split), partial)
    partial.rename(target)
    return target


class Corpus:
    """The utterances of one split: audio bytes, reader and transcript."""

    def __init__(self, path: Path) -> None:
        """Load the parquet table into memory."""
        import pyarrow.parquet as pq

        table = pq.read_table(
            path, columns=["audio", "transcript", "speaker_id",
                           "audio_duration"],
        )
        self.audio: List[bytes] = [
            a["bytes"] for a in table.column("audio").to_pylist()
        ]
        self.text: List[str] = table.column("transcript").to_pylist()
        self.reader: List[str] = table.column("speaker_id").to_pylist()
        self.duration: List[float] = table.column(
            "audio_duration"
        ).to_pylist()

    def samples(self, row: int) -> FloatArray:
        """Decode utterance ``row`` to mono 16 kHz samples."""
        import soundfile

        data, rate = soundfile.read(
            io.BytesIO(self.audio[row]), dtype="float32", always_2d=True
        )
        return resample(to_mono(data), int(rate), DEFAULT_SAMPLE_RATE)


# --------------------------------------------------------------- plan


def make_plan(corpus: Corpus, split: str, seed: int, meetings: int,
              ones: int, twos: int, two_reader_samples: int) -> Row:
    """Draw every recording of a split; no utterance is used twice."""
    rng = random.Random(f"{seed}-{split}")
    readers = sorted(set(corpus.reader))
    pools: Dict[str, List[int]] = {r: [] for r in readers}
    for row, reader in enumerate(corpus.reader):
        pools[reader].append(row)
    for pool in pools.values():
        rng.shuffle(pool)

    def piece(reader: str, gap: Optional[float] = None) -> Row:
        """Take the next unused utterance of ``reader``."""
        return {"reader": reader, "row": pools[reader].pop(),
                "gap": round(rng.uniform(0.2, 1.0), 3) if gap is None
                else gap}

    def enough(reader: str, seconds: float) -> List[Row]:
        """Take utterances of ``reader`` until ``seconds`` are covered."""
        taken: List[Row] = []
        while sum(corpus.duration[p["row"]] + p["gap"]
                  for p in taken) < seconds:
            taken.append(piece(reader, 0.3))
        return taken

    recordings: List[Row] = []
    for reader in readers:
        recordings.append({
            "id": f"{split}-enrol-{reader}", "kind": "enrol",
            "readers": [reader], "limit": ENROL_SECONDS,
            "pieces": enough(reader, ENROL_SECONDS),
        })
    for number in range(two_reader_samples):
        first, second = rng.sample(readers, 2)
        share = SECOND_READER_SHARES[number % len(SECOND_READER_SHARES)]
        head = ENROL_SECONDS * (1 - share)
        pieces = enough(first, head)
        # Cut the first reader at ``head`` seconds: the second reader's
        # utterance starts there.
        pieces[-1]["take"] = round(
            head - sum(corpus.duration[p["row"]] + p["gap"]
                       for p in pieces[:-1]) - pieces[-1]["gap"], 3)
        pieces += enough(second, ENROL_SECONDS - head)
        recordings.append({
            "id": f"{split}-enroltwo-{number:02d}", "kind": "enroltwo",
            "readers": [first, second], "limit": ENROL_SECONDS,
            "second_share": round(share, 3), "pieces": pieces,
        })
    for number in range(ones):
        reader = readers[number % len(readers)]
        recordings.append({
            "id": f"{split}-one-{number:02d}", "kind": "one",
            "readers": [reader], "pieces": [piece(reader) for _ in range(6)],
        })
    for number in range(twos):
        pair = rng.sample(readers, 2)
        recordings.append({
            "id": f"{split}-two-{number:02d}", "kind": "two",
            "readers": pair,
            "pieces": [piece(pair[turn % 2]) for turn in range(8)],
        })
    for number in range(meetings):
        people = rng.sample(readers, 4)
        pieces, previous = [], ""
        for _ in range(16):
            reader = rng.choice([r for r in people if r != previous])
            previous = reader
            pieces.append(piece(reader))
        absent = rng.choice([r for r in readers if r not in people])
        unenrolled_one = rng.choice(people)
        unenrolled_two = rng.sample(people, 2)
        recordings.append({
            "id": f"{split}-meeting-{number:02d}", "kind": "meeting",
            "readers": people, "pieces": pieces, "absent": absent,
            "unenrolled_one": unenrolled_one,
            "unenrolled_two": unenrolled_two,
        })
    return {"split": split, "seed": seed, "readers": readers,
            "recordings": recordings}


class Built(NamedTuple):
    """The audio of a planned recording and its ground truth."""

    audio: Audio
    spans: List[SpeakerSpan]
    reference: str


def build(corpus: Corpus, recording: Row) -> Built:
    """Concatenate the pieces of ``recording`` (gap, then utterance)."""
    rate = DEFAULT_SAMPLE_RATE
    parts: List[FloatArray] = []
    spans: List[SpeakerSpan] = []
    texts: List[str] = []
    cursor = 0.0
    for item in recording["pieces"]:
        samples = corpus.samples(item["row"])
        if "take" in item:
            samples = samples[:max(int(item["take"] * rate), 0)]
        gap = np.zeros(int(item["gap"] * rate), dtype=np.float32)
        start = cursor + gap.size / rate
        cursor = start + samples.size / rate
        parts += [gap, samples]
        spans.append(SpeakerSpan(
            speaker=item["reader"], start=start, end=cursor))
        texts.append(corpus.text[item["row"]])
    samples_all = np.concatenate(parts).astype(np.float32)
    limit = recording.get("limit")
    if limit is not None:
        samples_all = samples_all[:int(limit * rate)]
        end = samples_all.size / rate
        spans = [s.model_copy(update={"end": min(s.end, end)})
                 for s in spans if s.start < end]
    return Built(Audio(samples_all, rate), spans, " ".join(texts))


# ------------------------------------------------------------ prepare


def words_path(cache: Path, rec_id: str) -> Path:
    """Return where the recognised words of ``rec_id`` are cached."""
    return cache / f"{rec_id}.words.json"


def vectors_path(cache: Path, rec_id: str) -> Path:
    """Return where the window embeddings of ``rec_id`` are cached."""
    return cache / f"{rec_id}.npz"


def read_jsonl(path: Path) -> List[Row]:
    """Return the rows of a JSON-lines file (none if it is missing)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Row]) -> int:
    """Rewrite ``path`` with ``rows``; return how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def windows_of(words: Sequence[Word], kind: str,
               p: TranscriptionParameters) -> Tuple[
                   List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return the long and short analysis windows the pipeline would use.

    Enrolment samples follow ``EnrolSpeaker`` (regions from every
    recognised word, long windows only); the other recordings follow
    ``TranscribeSession`` (words already gated by speech activity).
    """
    regions = speech_regions(words, p.region_gap)
    coarse = analysis_windows(regions, *p.coarse())
    fine = [] if kind.startswith("enrol") else analysis_windows(
        regions, *p.fine())
    return coarse, fine


def gated(words: Sequence[Word], audio: Audio,
          p: TranscriptionParameters) -> List[Word]:
    """Apply the speech-activity gate of ``TranscribeSession``."""
    activity = speech_activity(audio.samples, audio.sample_rate,
                               p.activity_frame, p.activity_threshold)
    return heard_words([w for w in words if w.text.strip()], activity)


def prepare(args: argparse.Namespace) -> None:
    """Download, plan, recognise and embed every recording (resumable)."""
    import torch

    from transcriber.adapters.outbound.wavlm_voice_encoder import (
        WavLMVoiceEncoder,
    )
    from transcriber.adapters.outbound.whisper_recogniser import (
        WhisperRecogniser,
    )

    p = TranscriptionParameters()
    recognition = OUTPUT / "recognition.jsonl"
    done = {r["id"] for r in read_jsonl(recognition)}
    for split in SPLITS:
        corpus = Corpus(ensure_parquet(args.data_dir, split))
        cache = args.data_dir / "cache" / split
        cache.mkdir(parents=True, exist_ok=True)
        plan_file = cache / "plan.json"
        if plan_file.exists():
            plan = json.loads(plan_file.read_text())
        else:
            plan = make_plan(corpus, split, args.seed, args.meetings,
                             args.ones, args.twos, args.two_reader_samples)
            plan_file.write_text(json.dumps(plan))
        recogniser = WhisperRecogniser(batch_size=args.asr_batch,
                                       device=args.device)
        for recording in plan["recordings"]:
            path = words_path(cache, recording["id"])
            if path.exists():
                continue
            built = build(corpus, recording)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            speech = recogniser.recognise(built.audio, LANGUAGE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            words = [w for w in speech.words if w.text.strip()]
            path.write_text(json.dumps(
                [[w.text, w.start, w.end] for w in words]))
            if recording["id"] in done:
                continue
            row: Row = {
                "id": recording["id"], "split": split,
                "kind": recording["kind"],
                "duration": round(built.audio.duration, 2),
                "recogniser": recogniser.model_name,
                "batch_size": args.asr_batch,
                "asr_seconds": round(seconds, 2),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            if torch.cuda.is_available():
                row["gpu_peak_allocated_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1e9, 2)
                row["gpu_peak_reserved_gb"] = round(
                    torch.cuda.max_memory_reserved() / 1e9, 2)
            with recognition.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"{recording['id']}: {row['duration']:.0f} s audio,"
                  f" {seconds:.1f} s ASR", flush=True)
        recogniser.release()
        encoder = WavLMVoiceEncoder(device=args.device)
        for recording in plan["recordings"]:
            path = vectors_path(cache, recording["id"])
            if path.exists():
                continue
            built = build(corpus, recording)
            words = load_words(cache, recording["id"])
            if recording["kind"] != "enrol" and \
                    recording["kind"] != "enroltwo":
                words = gated(words, built.audio, p)
            coarse_w, fine_w = windows_of(words, recording["kind"], p)
            began = time.perf_counter()
            coarse = encoder.embed(built.audio, coarse_w)
            fine = encoder.embed(built.audio, fine_w)
            seconds = time.perf_counter() - began
            np.savez(path, coarse=coarse, fine=fine,
                     coarse_windows=np.array(coarse_w, dtype=np.float64),
                     fine_windows=np.array(fine_w, dtype=np.float64),
                     seconds=np.array(seconds))
        encoder.release()


def load_words(cache: Path, rec_id: str) -> List[Word]:
    """Return the cached recognised words of ``rec_id``."""
    raw = json.loads(words_path(cache, rec_id).read_text())
    return [Word(text=t, start=s, end=e) for t, s, e in raw]


# ------------------------------------------------------------ analyse


class Recording(NamedTuple):
    """A planned recording with its cached words, windows and vectors."""

    plan: Row
    built: Built
    words: List[Word]
    coarse: FloatArray
    fine: FloatArray
    coarse_windows: List[Tuple[float, float]]
    fine_windows: List[Tuple[float, float]]
    embed_seconds: float


def load_split(data_dir: Path, split: str) -> Tuple[Row, List[Recording]]:
    """Load the plan and every cached recording of ``split``."""
    corpus = Corpus(ensure_parquet(data_dir, split))
    cache = data_dir / "cache" / split
    plan = json.loads((cache / "plan.json").read_text())
    p = TranscriptionParameters()
    out: List[Recording] = []
    for recording in plan["recordings"]:
        built = build(corpus, recording)
        words = load_words(cache, recording["id"])
        if not recording["kind"].startswith("enrol"):
            words = gated(words, built.audio, p)
        stored = np.load(vectors_path(cache, recording["id"]))

        def pairs(name: str) -> List[Tuple[float, float]]:
            """Return the windows stored under ``name``."""
            return [(float(a), float(b)) for a, b in stored[name].reshape(
                -1, 2).tolist()]

        out.append(Recording(
            recording, built, words,
            stored["coarse"].astype(np.float32),
            stored["fine"].astype(np.float32),
            pairs("coarse_windows"), pairs("fine_windows"),
            float(stored["seconds"]),
        ))
    return plan, out


def window_truth(windows: Sequence[Tuple[float, float]],
                 spans: Sequence[SpeakerSpan]) -> List[Optional[str]]:
    """Return the reader of each window, ``None`` when it is mixed.

    A window is pure when at least 90% of its overlap with speech
    belongs to one reader.
    """
    labels: List[Optional[str]] = []
    for start, end in windows:
        share: Dict[str, float] = defaultdict(float)
        for span in spans:
            share[span.speaker] += max(
                0.0, min(end, span.end) - max(start, span.start))
        total = sum(share.values())
        if total <= 0:
            labels.append(None)
            continue
        best = max(share, key=lambda k: share[k])
        labels.append(best if share[best] / total >= 0.9 else None)
    return labels


def word_truth(words: Sequence[Word],
               spans: Sequence[SpeakerSpan]) -> List[str]:
    """Return the reader of each word (span at its middle, else nearest)."""
    out: List[str] = []
    for word in words:
        middle = (word.start + word.end) / 2
        inside = [s for s in spans if s.start <= middle <= s.end]
        if inside:
            out.append(inside[0].speaker)
        else:
            out.append(min(spans, key=lambda s: min(
                abs(middle - s.start), abs(middle - s.end))).speaker)
    return out


def quantiles(values: Sequence[float]) -> Row:
    """Summarise ``values``: count, mean and quantiles."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    q = np.quantile(array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"n": int(array.size), "mean": round(float(array.mean()), 4),
            **{f"p{k}": round(float(v), 4) for k, v in zip(
                (1, 5, 25, 50, 75, 95, 99), q)}}


def separation(same: Sequence[float], other: Sequence[float]) -> Row:
    """Return the AUC and equal error rate of same against other."""
    a = np.asarray(same, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {}
    joined = np.concatenate([a, b])
    ranks = joined.argsort().argsort().astype(np.float64) + 1
    auc = (ranks[:a.size].sum() - a.size * (a.size + 1) / 2) / (
        a.size * b.size)
    cuts = np.unique(joined)
    miss = np.searchsorted(np.sort(a), cuts, side="left") / a.size
    false = 1.0 - np.searchsorted(np.sort(b), cuts, side="left") / b.size
    at = int(np.argmin(np.abs(miss - false)))
    eer = float((miss[at] + false[at]) / 2)
    eer_at = float(cuts[at])
    return {"auc": round(float(auc), 4), "eer": round(eer, 4),
            "eer_threshold": round(eer_at, 4)}


def global_mean(recordings: Sequence[Recording]) -> FloatArray:
    """Return the mean raw unit long-window vector of ``recordings``."""
    rows = [unit_rows(r.coarse) for r in recordings
            if r.plan["kind"] == "meeting" and r.coarse.shape[0]]
    mean: FloatArray = np.vstack(rows).mean(axis=0).astype(np.float32)
    return mean


# Q1 ---------------------------------------------------------------


def window_rows(split: str, recordings: Sequence[Recording],
                other_mean: FloatArray) -> List[Row]:
    """Cosines between windows of the same and of different readers."""
    rows: List[Row] = []
    meetings = [r for r in recordings if r.plan["kind"] == "meeting"]
    for scale in ("coarse", "fine"):
        for norm in ("raw", "recording_mean", "global_mean"):
            same: List[float] = []
            other: List[float] = []
            for rec in meetings:
                vectors = rec.coarse if scale == "coarse" else rec.fine
                windows = (rec.coarse_windows if scale == "coarse"
                           else rec.fine_windows)
                if norm == "raw":
                    unit = unit_rows(vectors)
                elif norm == "recording_mean":
                    unit = centre(vectors, rec.coarse.mean(axis=0))
                else:
                    unit = centre(vectors, other_mean)
                truth = window_truth(windows, rec.built.spans)
                keep = [i for i, t in enumerate(truth) if t is not None]
                sims = unit[keep] @ unit[keep].T
                for a in range(len(keep)):
                    for b in range(a + 1, len(keep)):
                        if truth[keep[a]] == truth[keep[b]]:
                            same.append(float(sims[a, b]))
                        else:
                            other.append(float(sims[a, b]))
            rows.append({
                "split": split, "question": "window_cosines",
                "windows": "5s" if scale == "coarse" else "1.5s",
                "normalisation": norm, "meetings": len(meetings),
                "same_reader": quantiles(same),
                "different_reader": quantiles(other),
                **separation(same, other),
            })
    return rows


def single_stats(vectors: FloatArray) -> Row:
    """Statistics that may tell one voice from several (raw vectors)."""
    unit = unit_rows(vectors).astype(np.float64)
    mean = unit.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-12)
    to_mean = unit @ mean
    pair = unit @ unit.T
    upper = pair[np.triu_indices(unit.shape[0], k=1)]
    split2 = 1.0
    if unit.shape[0] >= 2:
        labels = cluster_speakers(unit.astype(np.float32), num_speakers=2)
        cents = voice_centroids(unit.astype(np.float32), labels)
        if cents.shape[0] == 2:
            split2 = float(cents[0] @ cents[1])
    return {
        "min_to_mean": round(float(to_mean.min()), 4),
        "p10_to_mean": round(float(np.quantile(to_mean, 0.1)), 4),
        "mean_pairwise": round(float(upper.mean()) if upper.size else 1.0,
                               4),
        "split2_centroid_cosine": round(split2, 4),
        "windows": int(unit.shape[0]),
    }


SINGLE_STATS = ("min_to_mean", "p10_to_mean", "mean_pairwise",
                "split2_centroid_cosine")


def single_rows(split: str, recordings: Sequence[Recording]) -> List[Row]:
    """One row per recording: voices present and the statistics."""
    rows: List[Row] = []
    for rec in recordings:
        kind = rec.plan["kind"]
        if kind == "enroltwo" or rec.coarse.shape[0] < 2:
            continue
        voices = len(set(rec.plan["readers"]))
        rows.append({
            "split": split, "question": "single_voice_statistics",
            "id": rec.plan["id"], "kind": kind, "voices": voices,
            "duration": round(rec.built.audio.duration, 1),
            **single_stats(rec.coarse),
        })
    return rows


def single_rule_rows(rows: Sequence[Row]) -> List[Row]:
    """Error of 'one voice when statistic >= t' per voice count."""
    out: List[Row] = []
    for split in SPLITS:
        mine = [r for r in rows if r["split"] == split]
        for stat in SINGLE_STATS:
            for t in [round(x, 3) for x in np.arange(0.5, 1.0001, 0.005)]:
                row: Row = {"split": split, "question": "single_voice_rule",
                            "statistic": stat, "threshold": t}
                for voices in (1, 2, 4):
                    group = [r for r in mine if r["voices"] == voices]
                    if voices == 1:
                        wrong = sum(r[stat] < t for r in group)
                    else:
                        wrong = sum(r[stat] >= t for r in group)
                    row[f"errors_{voices}"] = int(wrong)
                    row[f"count_{voices}"] = len(group)
                out.append(row)
    return out


# Q3 ---------------------------------------------------------------


def enrolment_inputs(rec: Recording, p: TranscriptionParameters) -> Tuple[
        List[Tuple[float, float]], float]:
    """Return the regions and duration ``EnrolSpeaker`` would pass."""
    return (speech_regions(rec.words, p.region_gap),
            rec.built.audio.duration)


def try_enrol(rec: Recording, rules: SampleRules,
              p: TranscriptionParameters) -> Tuple[Optional[str],
                                                   Optional[FloatArray]]:
    """Return ``(rejection reason, voiceprint vector)`` of a sample."""
    regions, duration = enrolment_inputs(rec, p)
    try:
        vector = enrolment_vector(rec.coarse, rec.coarse_windows, regions,
                                  duration, rules)
    except SampleRejectedError as exc:
        return exc.reason, None
    return None, np.asarray(vector, dtype=np.float32)


def enrolment_rows(split: str, recordings: Sequence[Recording],
                   p: TranscriptionParameters) -> List[Row]:
    """Accept or reject each sample over a grid of sample rules."""
    rows: List[Row] = []
    samples = [r for r in recordings if r.plan["kind"].startswith("enrol")]
    for rec in samples:
        regions, duration = enrolment_inputs(rec, p)
        rows.append({
            "split": split, "question": "enrolment_sample",
            "id": rec.plan["id"], "kind": rec.plan["kind"],
            "second_share": rec.plan.get("second_share", 0.0),
            "duration": round(duration, 2),
            "speech_seconds": round(speech_seconds(regions), 2),
            "windows": int(rec.coarse.shape[0]),
            **single_stats(rec.coarse),
        })
    for same in [round(float(x), 2) for x in np.arange(0.3, 0.951, 0.05)]:
        for share in (0.0, 0.1, 0.2, 0.3):
            rules = SampleRules(same_voice_threshold=same,
                                max_other_voice_share=share)
            outcome: Dict[str, Counter[str]] = defaultdict(Counter)
            for rec in samples:
                reason, _ = try_enrol(rec, rules, p)
                key = rec.plan["kind"]
                if key == "enroltwo":
                    key = f"enroltwo_{rec.plan['second_share']:.2f}"
                outcome[key][reason or "accepted"] += 1
            rows.append({
                "split": split, "question": "sample_rules",
                "same_voice_threshold": same,
                "max_other_voice_share": share,
                "outcomes": {k: dict(v) for k, v in sorted(
                    outcome.items())},
            })
    return rows


# Q2 ---------------------------------------------------------------


class Labelled(NamedTuple):
    """A recording's words with the diarizer's labels, and its groups."""

    words: List[Word]
    vectors: FloatArray


def diarize(rec: Recording, count: Optional[int],
            p: TranscriptionParameters,
            single: Callable[[FloatArray], bool]) -> Labelled:
    """Run the pipeline's voice stage (``TranscribeSession``) on ``rec``.

    ``single`` is the single-voice rule applied when ``count`` is
    unknown; a recording it accepts is grouped as one voice.
    """
    if count is None and single(rec.coarse):
        count = 1
    groups = group_voices(rec.coarse, rec.fine, count, p.group_threshold,
                          p.max_speakers, p.centre_windows, None)
    if rec.fine_windows:
        spans = spans_from_windows(rec.fine_windows, groups.fine)
    else:
        spans = spans_from_windows(rec.coarse_windows, groups.coarse)
    words = assign_speakers(rec.words, spans)
    if p.smoothing_run:
        words = smooth_speakers(words, p.smoothing_run)
    return Labelled(words, groups.vectors)


def name_words(labelled: Labelled, prints: Dict[str, FloatArray],
               threshold: float, normalisation: Normalisation,
               taken: Set[str]) -> List[str]:
    """Return the final speaker of each word (cluster then match)."""
    order = list(dict.fromkeys(w.speaker or "" for w in labelled.words))
    rows = {speaker_label(n): n for n in range(labelled.vectors.shape[0])}
    voiceprints = voiceprints_of(prints)
    matched = match_voices(
        labelled.vectors[[rows[label] for label in order]], voiceprints,
        threshold, normalisation, taken=taken,
    )
    names = dict(zip(order, [m.speaker for m in matched]))
    return [names[w.speaker or ""] for w in labelled.words]


def voiceprints_of(prints: Dict[str, FloatArray]) -> List[Voiceprint]:
    """Wrap the study's enrolment vectors as domain voiceprints."""
    return [
        Voiceprint(speaker_id=k, model="m", vector=[float(x) for x in v],
                   consent_version="1",
                   enrolled_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for k, v in prints.items()
    ]


def guided_words(rec: Recording, prints: Dict[str, FloatArray],
                 threshold: float, normalisation: Normalisation,
                 count: Optional[int], p: TranscriptionParameters,
                 single: Callable[[FloatArray], bool],
                 taken: Set[str]) -> List[str]:
    """Return the final speaker of each word, voiceprints as seeds.

    The voices come from the pipeline's own
    :func:`~transcriber.domain.voices.seed_voices` (the method this
    study measured; it was a private copy here before): each long window
    takes the enrolled person it is most similar to when the cosine
    (after ``normalisation``) reaches ``threshold``, the windows nobody
    claims are clustered into anonymous voices, and the short windows
    take the nearest group.
    """
    seeded = seed_voices(
        rec.coarse, rec.fine, voiceprints_of(prints), count, threshold,
        normalisation, p.group_threshold, p.max_speakers, single,
    )
    if rec.fine_windows:
        spans = spans_from_windows(rec.fine_windows, seeded.fine)
    else:
        spans = spans_from_windows(rec.coarse_windows, seeded.coarse)
    words = assign_speakers(rec.words, spans)
    if p.smoothing_run:
        words = smooth_speakers(words, p.smoothing_run)
    order = list(dict.fromkeys(w.speaker or "" for w in words))
    final: Dict[str, str] = {}
    anon_number = 1
    for label in order:
        number = int(label.split("_")[1]) - 1
        if number in seeded.people:
            final[label] = seeded.people[number]
        else:
            while f"S{anon_number}" in taken:
                anon_number += 1
            final[label] = f"S{anon_number}"
            anon_number += 1
    return [final[w.speaker or ""] for w in words]


def score(names: Sequence[str], truth: Sequence[str],
          enrolled: Set[str], present: Set[str]) -> Row:
    """Count what happened to each word and each reader of a meeting."""
    anonymous = sorted({n for n in names if n not in enrolled})
    strangers = sorted(present - enrolled)
    mapping: Dict[str, str] = {}
    if anonymous and strangers:
        counts = np.zeros((len(anonymous), len(strangers)))
        for n, t in zip(names, truth):
            if n in anonymous and t in strangers:
                counts[anonymous.index(n), strangers.index(t)] += 1
        r, c = linear_sum_assignment(-counts)
        mapping = {anonymous[a]: strangers[b] for a, b in zip(r, c)}
    right = impostor = missed = wrong = absent = 0
    enrolled_words = stranger_words = 0
    per_reader: Dict[str, Counter[str]] = defaultdict(Counter)
    for n, t in zip(names, truth):
        per_reader[t][n] += 1
        if n in enrolled and n not in present:
            absent += 1
        if t in enrolled:
            enrolled_words += 1
            if n == t:
                right += 1
            elif n in enrolled:
                wrong += 1
            else:
                missed += 1
        else:
            stranger_words += 1
            if n in enrolled:
                impostor += 1
            elif mapping.get(n) == t:
                right += 1
    impostor_readers = sum(
        1 for t in strangers
        if per_reader[t] and per_reader[t].most_common(1)[0][0] in enrolled)
    missed_readers = sum(
        1 for t in present & enrolled
        if per_reader[t] and per_reader[t].most_common(1)[0][0]
        not in enrolled)
    wrong_readers = sum(
        1 for t in present & enrolled
        if per_reader[t] and per_reader[t].most_common(1)[0][0] in enrolled
        and per_reader[t].most_common(1)[0][0] != t)
    return {
        "words": len(names), "right": right,
        "enrolled_words": enrolled_words, "stranger_words": stranger_words,
        "impostor_words": impostor, "missed_words": missed,
        "wrong_id_words": wrong, "absent_named_words": absent,
        "strangers": len(strangers), "impostor_readers": impostor_readers,
        "enrolled_present": len(present & enrolled),
        "missed_readers": missed_readers, "wrong_readers": wrong_readers,
        "labels": len(set(names)),
    }


CONDITIONS = ("all_enrolled", "one_unenrolled", "two_unenrolled",
              "enrolled_absent")


def enrolled_set(meeting: Row, condition: str) -> Set[str]:
    """Return the readers with a voiceprint under ``condition``."""
    people = set(meeting["readers"])
    if condition == "one_unenrolled":
        return people - {meeting["unenrolled_one"]}
    if condition == "two_unenrolled":
        return people - set(meeting["unenrolled_two"])
    if condition == "enrolled_absent":
        return people | {meeting["absent"]}
    return people


def centred_on(mean: FloatArray) -> Normalisation:
    """Return the normalisation that centres both sides on ``mean``."""
    def normalise(session: FloatArray, enrolled: FloatArray) -> Tuple[
            FloatArray, FloatArray]:
        """Subtract the fixed mean and re-normalise."""
        return centre(session, mean), centre(enrolled, mean)
    return normalise


def centre_on_enrolled(session: FloatArray,
                       enrolled: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Centre both sides on the mean of the candidate voiceprints."""
    if enrolled.shape[0] < 3:
        return unit_vectors(session, enrolled)
    mean: FloatArray = enrolled.mean(axis=0).astype(np.float32)
    return centre(session, mean), centre(enrolled, mean)


def normalisations(rec: Recording, other_mean: FloatArray,
                   pool_mean: FloatArray) -> Dict[str, Normalisation]:
    """Return every normalisation compared, by name."""
    return {
        "unit": unit_vectors,
        "centre_on_session": centre_on_session,
        "recording_mean": centred_on(
            rec.coarse.mean(axis=0).astype(np.float32)),
        "global_mean_other_split": centred_on(other_mean),
        "pool_mean_other_readers": centred_on(pool_mean),
        "centre_on_enrolled": centre_on_enrolled,
    }


def identification_rows(
    split: str, recordings: Sequence[Recording],
    prints: Dict[str, FloatArray], other_mean: FloatArray,
    p: TranscriptionParameters, single: Callable[[FloatArray], bool],
    single_name: str, thresholds: Sequence[float],
) -> List[Row]:
    """Sweep conditions, methods, normalisations, counts and thresholds."""
    meetings = [r for r in recordings if r.plan["kind"] == "meeting"]
    totals: Dict[Tuple[str, str, str, str, float], List[Row]] = \
        defaultdict(list)
    for rec in meetings:
        truth = word_truth(rec.words, rec.built.spans)
        present = set(rec.plan["readers"])
        others = [k for k in prints if k not in present]
        pool_mean: FloatArray = np.vstack(
            [prints[k] for k in others]).mean(axis=0).astype(np.float32)
        norms = normalisations(rec, other_mean, pool_mean)
        diarized = {
            "known": diarize(rec, len(present), p, single),
            "estimated": diarize(rec, None, p, single),
        }
        for condition in CONDITIONS:
            enrolled = enrolled_set(rec.plan, condition)
            candidates = {k: prints[k] for k in sorted(enrolled)}
            taken = set(prints)
            for count_name, count in (("known", len(present)),
                                      ("estimated", None)):
                for norm_name, norm in norms.items():
                    for t in thresholds:
                        names = name_words(diarized[count_name], candidates,
                                           t, norm, taken)
                        totals[(condition, "cluster_then_match", norm_name,
                                count_name, t)].append(
                            score(names, truth, enrolled, present))
                        names = guided_words(rec, candidates, t, norm,
                                             count, p, single, taken)
                        totals[(condition, "voiceprints_as_seeds",
                                norm_name, count_name, t)].append(
                            score(names, truth, enrolled, present))
        print(f"  identification {rec.plan['id']} done", flush=True)
    rows: List[Row] = []
    for (condition, method, norm_name, count_name, t), scores in sorted(
            totals.items()):
        summed = {k: sum(s[k] for s in scores) for k in scores[0]}
        rows.append({
            "split": split, "question": "identification",
            "condition": condition, "method": method,
            "normalisation": norm_name, "speakers": count_name,
            "single_voice_rule": single_name, "threshold": t,
            "meetings": len(scores), **summed,
            "right_share": round(summed["right"] / summed["words"], 4),
            "right_share_worst_meeting": round(min(
                s["right"] / s["words"] for s in scores), 4),
            "impostor_share": round(
                summed["impostor_words"] / summed["stranger_words"], 4)
            if summed["stranger_words"] else None,
            "missed_share": round(
                summed["missed_words"] / summed["enrolled_words"], 4),
            "wrong_id_share": round(
                summed["wrong_id_words"] / summed["enrolled_words"], 4),
        })
    return rows


def diarization_rows(split: str, recordings: Sequence[Recording],
                     p: TranscriptionParameters,
                     rules: Dict[str, Callable[[FloatArray], bool]]
                     ) -> List[Row]:
    """Anonymous diarization: words right and voices found per rule."""
    rows: List[Row] = []
    for kind in ("one", "two", "meeting"):
        group = [r for r in recordings if r.plan["kind"] == kind]
        for rule_name, rule in rules.items():
            settings = [("known", p.group_threshold)] + [
                ("estimated", g) for g in GROUP_THRESHOLDS]
            for count_name, group_threshold in settings:
                right = total = 0
                found: List[int] = []
                q = p.model_copy(update={"group_threshold": group_threshold})
                for rec in group:
                    count = (len(set(rec.plan["readers"]))
                             if count_name == "known" else None)
                    labelled = diarize(rec, count, q, rule)
                    truth = word_truth(rec.words, rec.built.spans)
                    labels = [w.speaker or "" for w in labelled.words]
                    found.append(len(set(labels)))
                    right += best_mapping_right(labels, truth)
                    total += len(labels)
                rows.append({
                    "split": split, "question": "diarization",
                    "kind": kind, "single_voice_rule": rule_name,
                    "speakers": count_name,
                    "group_threshold": group_threshold,
                    "recordings": len(group),
                    "words": total, "right": right,
                    "right_share": round(right / total, 4),
                    "voices_found": found,
                })
    return rows


def best_mapping_right(labels: Sequence[str], truth: Sequence[str]) -> int:
    """Return the words right under the best one-to-one label mapping."""
    a = sorted(set(labels))
    b = sorted(set(truth))
    counts = np.zeros((len(a), len(b)))
    for x, y in zip(labels, truth):
        counts[a.index(x), b.index(y)] += 1
    r, c = linear_sum_assignment(-counts)
    return int(counts[r, c].sum())


# Q1 bis: voiceprints against meeting voices ----------------------------


def match_cosine_rows(split: str, recordings: Sequence[Recording],
                      prints: Dict[str, FloatArray],
                      other_mean: FloatArray) -> List[Row]:
    """Cosine between a reader's true group and voiceprints, per norm."""
    rows: List[Row] = []
    meetings = [r for r in recordings if r.plan["kind"] == "meeting"]
    gathered: Dict[str, Tuple[List[float], List[float]]] = defaultdict(
        lambda: ([], []))
    for rec in meetings:
        present = set(rec.plan["readers"])
        truth = window_truth(rec.coarse_windows, rec.built.spans)
        readers = sorted(present)
        groups = np.vstack([
            unit_rows(rec.coarse)[[i for i, t in enumerate(truth)
                                   if t == r]].mean(axis=0)
            for r in readers]).astype(np.float32)
        groups = unit_rows(groups)
        others = [k for k in prints if k not in present]
        pool_mean: FloatArray = np.vstack(
            [prints[k] for k in others]).mean(axis=0).astype(np.float32)
        ids = sorted(prints)
        matrix = np.vstack([prints[k] for k in ids]).astype(np.float32)
        for name, norm in normalisations(rec, other_mean,
                                         pool_mean).items():
            if name == "centre_on_enrolled":
                continue  # depends on the candidates; see identification
            session, enrolled = norm(groups, matrix)
            sims = session @ enrolled.T
            for g, reader in enumerate(readers):
                for k, pid in enumerate(ids):
                    target = gathered[name][0 if pid == reader else 1]
                    target.append(float(sims[g, k]))
    for name, (same, other) in gathered.items():
        rows.append({
            "split": split, "question": "group_to_voiceprint_cosines",
            "normalisation": name, "meetings": len(meetings),
            "same_reader": quantiles(same),
            "different_reader": quantiles(other),
            **separation(same, other),
        })
    return rows


# Q4 / Q5 ------------------------------------------------------------

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalise_words(text: str) -> List[str]:
    """Lower-case words without punctuation or accents (as ``src``)."""
    folded = "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
    return _TOKEN.findall(folded)


_EDITS: Dict[Tuple[str, str], Tuple[int, int]] = {}


def edit_counts(reference: str, hypothesis: str) -> Tuple[int, int]:
    """Return ``(word edits, reference words)`` (memoised)."""
    key = (reference, hypothesis)
    if key not in _EDITS:
        _EDITS[key] = _edit_counts(reference, hypothesis)
    return _EDITS[key]


def _edit_counts(reference: str, hypothesis: str) -> Tuple[int, int]:
    """Return ``(word edits, reference words)``."""
    ref, hyp = normalise_words(reference), normalise_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, heard in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (expected != heard))
        previous = current
    return previous[-1], len(ref)


def activity_rows(split: str, recordings: Sequence[Recording],
                  data_dir: Path) -> List[Row]:
    """Words the speech-activity gate keeps and drops, per level."""
    cache = data_dir / "cache" / split
    rows: List[Row] = []
    group = [r for r in recordings
             if r.plan["kind"] in ("meeting", "one", "two")]
    raw = {r.plan["id"]: load_words(cache, r.plan["id"]) for r in group}
    for frame in (0.01, 0.025, 0.05):
        for level in (0.0, 0.001, 0.003, 0.01, 0.02, 0.05):
            counts: Counter[str] = Counter()
            edits = reference = 0
            for rec in group:
                words = raw[rec.plan["id"]]
                activity = speech_activity(
                    rec.built.audio.samples, rec.built.audio.sample_rate,
                    frame, level)
                kept = heard_words(words, activity)
                kept_ids = {id(w) for w in kept}
                for w in words:
                    over = any(min(w.end, s.end) - max(w.start, s.start) > 0
                               for s in rec.built.spans)
                    status = "kept" if id(w) in kept_ids else "dropped"
                    counts[f"{'speech' if over else 'silence'}_{status}"] \
                        += 1
                e, n = edit_counts(rec.built.reference,
                                   " ".join(w.text for w in kept))
                edits += e
                reference += n
            rows.append({
                "split": split, "question": "activity_gate",
                "frame": frame, "threshold": level,
                "recordings": len(group), **dict(counts),
                "wer": round(edits / reference, 4),
            })
    return rows


def recognition_rows(split: str, recordings: Sequence[Recording]
                     ) -> List[Row]:
    """WER of the meetings (gate applied) and voice-encoder speed."""
    meetings = [r for r in recordings if r.plan["kind"] == "meeting"]
    edits = reference = 0
    per: List[float] = []
    audio = embed = 0.0
    for rec in meetings:
        e, n = edit_counts(rec.built.reference,
                           " ".join(w.text for w in rec.words))
        edits += e
        reference += n
        per.append(e / n)
        audio += rec.built.audio.duration
        embed += rec.embed_seconds
    timed = [r for r in read_jsonl(OUTPUT / "recognition.jsonl")
             if r["split"] == split and r["kind"] == "meeting"]
    asr = sum(r["asr_seconds"] for r in timed)
    return [{
        "split": split, "question": "recognition_meetings",
        "meetings": len(meetings), "reference_words": reference,
        "wer": round(edits / reference, 4),
        "wer_mean_of_meetings": round(float(np.mean(per)), 4),
        "wer_min": round(min(per), 4), "wer_max": round(max(per), 4),
        "audio_seconds": round(audio, 1),
        "voice_embedding_seconds": round(embed, 2),
        "asr_seconds": round(asr, 1),
        "asr_times_real_time": round(audio / asr, 1),
        "asr_gpu_peak_allocated_gb": max(
            r.get("gpu_peak_allocated_gb", 0.0) for r in timed),
        "asr_gpu_peak_reserved_gb": max(
            r.get("gpu_peak_reserved_gb", 0.0) for r in timed),
        "asr_batch_size": timed[0]["batch_size"],
    }]


# ------------------------------------------------------------- driver


def single_rule(statistic: str, threshold: Optional[float]
                ) -> Callable[[FloatArray], bool]:
    """Return 'one voice when statistic >= threshold' (None: never)."""
    def rule(vectors: FloatArray) -> bool:
        """Apply the rule to raw long-window vectors."""
        if threshold is None or vectors.shape[0] == 0:
            return False
        if vectors.shape[0] == 1:
            return True
        return bool(single_stats(vectors)[statistic] >= threshold)
    return rule


def analyse(args: argparse.Namespace) -> None:
    """Compute every analysis from the cache and write the rows."""
    p = TranscriptionParameters()
    loaded = {s: load_split(args.data_dir, s) for s in SPLITS}
    means = {s: global_mean(loaded[s][1]) for s in SPLITS}
    other = {"dev": means["test"], "test": means["dev"]}
    what = set(args.what.split(","))
    # The rule before TR-05: every window at 0.9 or more from the mean.
    current = single_rule("min_to_mean", OLD_SINGLE_VOICE_THRESHOLD)
    chosen = single_rule(args.single_statistic, args.single_threshold)
    chosen_name = f"{args.single_statistic}>={args.single_threshold}"
    current_name = f"min_to_mean>={OLD_SINGLE_VOICE_THRESHOLD}"
    rules = SampleRules(same_voice_threshold=args.same_voice_threshold,
                        max_other_voice_share=args.max_other_voice_share)
    if "windows" in what:
        rows: List[Row] = []
        for s in SPLITS:
            rows += window_rows(s, loaded[s][1], other[s])
        stats: List[Row] = []
        for s in SPLITS:
            stats += single_rows(s, loaded[s][1])
        print(write_jsonl(OUTPUT / "windows.jsonl", rows), "window rows")
        print(write_jsonl(OUTPUT / "single_voice.jsonl",
                          stats + single_rule_rows(stats)),
              "single-voice rows")
    if "enrolment" in what:
        rows = []
        for s in SPLITS:
            rows += enrolment_rows(s, loaded[s][1], p)
        print(write_jsonl(OUTPUT / "enrolment.jsonl", rows),
              "enrolment rows")
    if "activity" in what:
        rows = []
        for s in SPLITS:
            rows += activity_rows(s, loaded[s][1], args.data_dir)
            rows += recognition_rows(s, loaded[s][1])
        print(write_jsonl(OUTPUT / "activity.jsonl", rows),
              "activity rows")
    if "diarization" in what:
        rows = []
        for s in SPLITS:
            rows += diarization_rows(s, loaded[s][1], p, {
                current_name: current, chosen_name: chosen})
        print(write_jsonl(OUTPUT / "diarization.jsonl", rows),
              "diarization rows")
    if "identification" in what:
        rows = []
        for s in SPLITS:
            prints: Dict[str, FloatArray] = {}
            rejected: List[str] = []
            for rec in loaded[s][1]:
                if rec.plan["kind"] != "enrol":
                    continue
                reason, vector = try_enrol(rec, rules, p)
                if vector is None:
                    rejected.append(f"{rec.plan['id']}: {reason}")
                else:
                    prints[rec.plan["readers"][0]] = vector
            if rejected:
                raise SystemExit(f"genuine samples rejected: {rejected}")
            rows += match_cosine_rows(s, loaded[s][1], prints, other[s])
            rows += identification_rows(
                s, loaded[s][1], prints, other[s], p, chosen, chosen_name,
                THRESHOLDS)
        print(write_jsonl(OUTPUT / "identification.jsonl", rows),
              "identification rows")


def main() -> None:
    """Parse the command line and run ``prepare`` or ``analyse``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "analyse"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/tr05"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meetings", type=int, default=20)
    parser.add_argument("--ones", type=int, default=20)
    parser.add_argument("--twos", type=int, default=20)
    parser.add_argument("--two-reader-samples", type=int, default=18)
    parser.add_argument("--asr-batch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--what",
        default="windows,enrolment,activity,diarization,identification")
    parser.add_argument("--single-statistic", default="p10_to_mean")
    parser.add_argument("--single-threshold", type=float, default=0.95)
    parser.add_argument("--same-voice-threshold", type=float,
                        default=0.85)
    parser.add_argument("--max-other-voice-share", type=float,
                        default=0.1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        analyse(args)


if __name__ == "__main__":
    main()
