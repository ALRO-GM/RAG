"""Pause-aligned voice pieces against the fixed grid (ticket TR-19).

Today the short voice windows lie on a fixed grid (1.5 s every 0.75 s)
and a sentence is cut at a change of speaker only when the windows
clearly support it (:func:`~transcriber.domain.certainty.
keep_sentences_whole`). This study asks whether cutting the audio at
pauses and word gaps, so that each piece is likely one voice, finds
quick exchanges better. Three ways are compared, the long pass (who is
there, voiceprints as seeds) being the same for all:

* ``grid``: the current pipeline (fixed grid, whole-sentence rule);
* ``pieces``: the speech regions cut at pauses of the activity gate
  and at gaps between recognised words; each piece is embedded and
  scored against the voices of the long pass; pieces shorter than a
  minimum take a neighbour's voice or the grid's;
* ``hybrid``: the grid, with each change of speaker moved to the
  nearest pause or word gap, and runs inside a sentence kept only with a
  minimum of evidence.

Material: the TR-05 recordings (``identification_eval.py``; two-voice
recordings and four-reader meetings, long turns) and new *quick
exchanges* built from utterances of the same public read-speech set
that TR-05 did not use: two readers, long turns alternating with blocks
of ten short turns of 0.5-3 s cut out of real utterances at gaps
between the words Whisper found in them, with 0-0.4 s of silence
between turns (``quickgap``) or none (``quicknogap``, the same pieces).
The truth of every word is known by construction.

Commands (``--data-dir`` as in TR-05; ``prepare`` of
``identification_eval.py`` must have run first)::

    PYTHONPATH=. python experiments/transcriber/pause_segments.py prepare
    PYTHONPATH=. python experiments/transcriber/pause_segments.py analyse
    PYTHONPATH=. python experiments/transcriber/pause_segments.py cost
    PYTHONPATH=. python experiments/transcriber/pause_segments.py real \\
        --sessions data/transcriber --work /tmp/tr19-real

``analyse`` embeds the pieces it needs on the GPU the first time (cached
per recording) and writes ``experiments/transcriber/pause_segments.jsonl``.
``real`` copies the private sessions to a scratch directory that is
deleted on exit and prints aggregate counts only. The script may import
:mod:`transcriber`; it never imports ``src``.
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
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

import identification_eval as tr05
from transcriber.adapters.outbound.av_audio_decoder import DEFAULT_SAMPLE_RATE
from transcriber.application.ports import Audio
from transcriber.domain.activity import heard_words, speech_activity
from transcriber.domain.certainty import (
    VoiceEvidence,
    WindowIndex,
    _neighbour,
    _Rule,
    _runs,
    _sentences,
    assess_utterances,
    support,
)
from transcriber.domain.enrolment import SampleRules
from transcriber.domain.languages import recognition_language
from transcriber.domain.models import Word
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.seams import resolve_seams
from transcriber.domain.sentences import build_utterances
from transcriber.domain.speakers import (
    SpeakerSpan,
    analysis_windows,
    smooth_speakers,
    spans_from_windows,
    speaker_label,
    speech_regions,
    turns_from_words,
)
from transcriber.domain.voices import (
    FloatArray,
    centre,
    group_voices,
    is_single_voice,
    seed_voices,
    unit_rows,
    unit_vectors,
    voice_centroids,
)

Row = Dict[str, Any]
Window = Tuple[float, float]
OUTPUT = Path("experiments/transcriber/pause_segments.jsonl")
QUICK_RECORDINGS = 20
QUICK_BLOCK = 10
PIECE_RANGE = (0.5, 3.0)
GAP_RANGE = (0.0, 0.4)
SOURCES_PER_READER = 3
# Pieces shorter than this are not embedded: alone, a clip of 0.30 s
# makes the x-vector head of WavLM fail (its convolution is longer than
# the frames left) and one of 0.32 s gives a NaN vector; 0.34 s was the
# shortest that gave a vector (checked on the GPU for this study). A
# vector with NaN is treated as no vector.
SHORTEST_EMBEDDED = 0.35
LENGTHS = (0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TOLERANCE = 0.5
# A length is "usable" when a piece of it goes to the right one of two
# voices at least this often (fixed before any measurement).
USABLE_ACCURACY = 0.9
KINDS = ("two", "meeting", "quickgap", "quicknogap")
QUICK_KINDS = ("quickgap", "quicknogap")
CONDITIONS = ("all_enrolled", "one_unenrolled", "two_unenrolled",
              "enrolled_absent")


# ----------------------------------------------------------- building


def cut_candidates(samples: FloatArray, words: Sequence[Word],
                   rate: int) -> List[float]:
    """Return one cut time per gap between consecutive words.

    The cut is the quietest 10 ms frame between the end of a word and
    the start of the next (0.05 s of slack on each side, never past the
    middle of either word), so a piece never ends inside a sound.
    """
    frame = int(0.01 * rate)
    cuts: List[float] = []
    for a, b in zip(words, words[1:]):
        lo = max(a.end - 0.05, (a.start + a.end) / 2)
        hi = min(b.start + 0.05, (b.start + b.end) / 2)
        if hi - lo < 0.01:
            cuts.append(round((a.end + b.start) / 2, 3))
            continue
        first, last = int(lo * rate), int(hi * rate)
        chunk = samples[first:last]
        count = max(chunk.size // frame, 1)
        energy = [float(np.mean(np.square(chunk[i * frame:(i + 1) * frame])))
                  if chunk[i * frame:(i + 1) * frame].size else math.inf
                  for i in range(count)]
        best = int(np.argmin(energy))
        cuts.append(round((first + best * frame + frame / 2) / rate, 3))
    return sorted(set(c for c in cuts if c > 0))


def short_pieces(duration: float, cuts: Sequence[float],
                 rng: random.Random) -> List[Window]:
    """Cut an utterance at ``cuts`` into pieces of about 0.5-3 s.

    Each piece aims at a length drawn uniformly from
    :data:`PIECE_RANGE` and ends at the cut closest to it (never
    shorter than the range's minimum when a later cut allows).
    """
    out: List[Window] = []
    start = 0.0
    bounds = list(cuts) + [duration]
    while True:
        target = rng.uniform(*PIECE_RANGE)
        later = [c for c in bounds if c - start >= PIECE_RANGE[0]]
        if not later:
            break
        end = min(later, key=lambda c: abs(c - start - target))
        if end - start > PIECE_RANGE[1] + 1.0:
            break
        out.append((round(start, 3), round(end, 3)))
        start = end
        if end >= duration:
            break
    return out


def make_quick_plan(corpus: tr05.Corpus, used: Set[int], readers: List[str],
                    split: str, seed: int,
                    source_words: Callable[[List[int]], Dict[int, List[Word]]]
                    ) -> Row:
    """Draw the quick-exchange recordings of ``split`` (unused utterances).

    Each recording: a long turn of reader A, ten short turns alternating
    B, A, ..., a long turn of B, ten short turns alternating A, B, ...
    The short turns come, in order, from up to
    :data:`SOURCES_PER_READER` utterances per reader, cut at word gaps.
    """
    rng = random.Random(f"tr19-{seed}-{split}")
    pools: Dict[str, List[int]] = {r: [] for r in readers}
    for row, reader in enumerate(corpus.reader):
        if row not in used:
            pools[reader].append(row)
    for pool in pools.values():
        rng.shuffle(pool)
    drafts: List[Tuple[List[str], Dict[str, int], Dict[str, List[int]]]] = []
    for _ in range(QUICK_RECORDINGS):
        pair = rng.sample([r for r in readers
                           if len(pools[r]) > SOURCES_PER_READER], 2)
        longs = {r: pools[r].pop() for r in pair}
        sources = {r: [pools[r].pop() for _ in range(SOURCES_PER_READER)]
                   for r in pair}
        drafts.append((pair, longs, sources))
    words = source_words(sorted({row for _, _, s in drafts
                                 for rows in s.values() for row in rows}))
    recordings: List[Row] = []
    for number, (pair, longs, sources) in enumerate(drafts):
        queue: Dict[str, List[Row]] = {r: [] for r in pair}
        for reader in pair:
            for row in sources[reader]:
                samples = corpus.samples(row)
                duration = samples.size / DEFAULT_SAMPLE_RATE
                cuts = cut_candidates(samples, words[row],
                                      DEFAULT_SAMPLE_RATE)
                for start, end in short_pieces(duration, cuts, rng):
                    queue[reader].append({"reader": reader, "row": row,
                                          "start": start, "end": end,
                                          "quick": True})
        a, b = pair
        items: List[Row] = []

        def long_turn(reader: str) -> Row:
            """Return a whole utterance of ``reader`` as one long turn."""
            row = longs[reader]
            return {"reader": reader, "row": row, "start": 0.0,
                    "end": round(corpus.duration[row], 3), "quick": False}

        items.append(long_turn(a))
        for turn in range(QUICK_BLOCK):
            items.append(queue[(b, a)[turn % 2]].pop(0))
        items.append(long_turn(b))
        for turn in range(QUICK_BLOCK):
            items.append(queue[(a, b)[turn % 2]].pop(0))
        for n, item in enumerate(items):
            item["gap"] = 0.3 if n == 0 else round(rng.uniform(*GAP_RANGE), 3)
        recordings.append({"id": f"{split}-quick-{number:02d}",
                           "readers": pair, "pieces": items,
                           "absent": rng.choice([r for r in readers
                                                 if r not in pair]),
                           "unenrolled_one": rng.choice(pair)})
    return {"split": split, "seed": seed, "recordings": recordings}


class Mix(NamedTuple):
    """The audio of a recording, its truth spans and which are quick."""

    audio: Audio
    spans: List[SpeakerSpan]
    quick: List[bool]


def build_quick(corpus: tr05.Corpus, recording: Row, gaps: bool) -> Mix:
    """Concatenate the turns of a quick recording (gaps kept or not)."""
    rate = DEFAULT_SAMPLE_RATE
    parts: List[FloatArray] = []
    spans: List[SpeakerSpan] = []
    quick: List[bool] = []
    cursor = 0.0
    for n, item in enumerate(recording["pieces"]):
        samples = corpus.samples(item["row"])
        samples = samples[int(item["start"] * rate):int(item["end"] * rate)]
        gap_seconds = item["gap"] if gaps or n == 0 else 0.0
        gap = np.zeros(int(gap_seconds * rate), dtype=np.float32)
        start = cursor + gap.size / rate
        cursor = start + samples.size / rate
        parts += [gap, samples]
        spans.append(SpeakerSpan(speaker=item["reader"], start=start,
                                 end=cursor))
        quick.append(bool(item["quick"]))
    return Mix(Audio(np.concatenate(parts).astype(np.float32), rate),
               spans, quick)


def quick_cache(data_dir: Path, split: str) -> Path:
    """Return the directory of this study's cache for ``split``."""
    return data_dir / "tr19" / split


def prepare(args: argparse.Namespace) -> None:
    """Plan, build and recognise the quick exchanges (resumable)."""
    from transcriber.adapters.outbound.whisper_recogniser import (
        WhisperRecogniser,
    )

    recogniser = WhisperRecogniser(batch_size=args.asr_batch,
                                   device=args.device)
    for split in tr05.SPLITS:
        corpus = tr05.Corpus(tr05.ensure_parquet(args.data_dir, split))
        cache = quick_cache(args.data_dir, split)
        cache.mkdir(parents=True, exist_ok=True)
        plan_file = cache / "plan.json"
        if not plan_file.exists():
            old = json.loads((args.data_dir / "cache" / split /
                              "plan.json").read_text())
            used = {p["row"] for r in old["recordings"]
                    for p in r["pieces"]}

            def source_words(rows: List[int]) -> Dict[int, List[Word]]:
                """Recognise each source utterance alone (cached)."""
                stored = cache / "sources.json"
                known: Dict[str, Any] = (json.loads(stored.read_text())
                                         if stored.exists() else {})
                for row in rows:
                    if str(row) in known:
                        continue
                    audio = Audio(corpus.samples(row), DEFAULT_SAMPLE_RATE)
                    speech = recogniser.recognise(audio, tr05.LANGUAGE)
                    known[str(row)] = [[w.text, w.start, w.end]
                                       for w in speech.words
                                       if w.text.strip()]
                    if len(known) % 20 == 0:
                        stored.write_text(json.dumps(known))
                stored.write_text(json.dumps(known))
                return {row: [Word(text=t, start=s, end=e)
                              for t, s, e in known[str(row)]]
                        for row in rows}

            plan = make_quick_plan(corpus, used, old["readers"], split,
                                   args.seed, source_words)
            plan_file.write_text(json.dumps(plan))
        plan = json.loads(plan_file.read_text())
        for recording in plan["recordings"]:
            for gaps in (True, False):
                rid = quick_id(recording["id"], gaps)
                path = tr05.words_path(cache, rid)
                if path.exists():
                    continue
                mix = build_quick(corpus, recording, gaps)
                began = time.perf_counter()
                speech = recogniser.recognise(mix.audio, tr05.LANGUAGE)
                path.write_text(json.dumps(
                    [[w.text, w.start, w.end] for w in speech.words
                     if w.text.strip()]))
                print(f"{rid}: {mix.audio.duration:.0f} s,"
                      f" {time.perf_counter() - began:.1f} s ASR",
                      flush=True)
    recogniser.release()


def quick_id(base: str, gaps: bool) -> str:
    """Return the id of the gap or no-gap version of a quick recording."""
    return base.replace("-quick-", "-quickgap-" if gaps else "-quicknogap-")


# ---------------------------------------------------------- embedding


def embed_waiting(encoder: Any, audio: Audio,
                  windows: Sequence[Window]) -> FloatArray:
    """Embed ``windows``; when the shared GPU is full, wait and retry."""
    import torch

    for _ in range(120):
        try:
            rows: FloatArray = encoder.embed(audio, windows)
            return rows
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("GPU full, waiting 30 s", flush=True)
            time.sleep(30)
    raise SystemExit("the GPU stayed full for an hour")


class Embedder:
    """Window embeddings of one recording, computed once and cached."""

    encoder: Any = None
    seconds = 0.0

    def __init__(self, path: Path, audio: Audio) -> None:
        """Load the cached vectors of the recording at ``path``."""
        self.path = path
        self.audio = audio
        self.vectors: Dict[Window, FloatArray] = {}
        self.dirty = False
        if path.exists():
            stored = np.load(path)
            for (a, b), v in zip(stored["windows"].tolist(),
                                 stored["vectors"]):
                self.vectors[(round(a, 3), round(b, 3))] = v

    def get(self, windows: Sequence[Window]) -> FloatArray:
        """Return one row per window, embedding the ones not cached.

        Missing windows are embedded in order of length, so a batch
        pads little (WavLM pads a batch to its longest clip).
        """
        keys = [(round(a, 3), round(b, 3)) for a, b in windows]
        missing = sorted({k for k in keys if k not in self.vectors},
                         key=lambda k: (k[1] - k[0], k[0]))
        if missing:
            if Embedder.encoder is None:
                from transcriber.adapters.outbound.wavlm_voice_encoder \
                    import WavLMVoiceEncoder
                Embedder.encoder = WavLMVoiceEncoder(device="cuda")
            began = time.perf_counter()
            rows = embed_waiting(Embedder.encoder, self.audio, missing)
            Embedder.seconds += time.perf_counter() - began
            for k, v in zip(missing, rows):
                self.vectors[k] = np.asarray(v, dtype=np.float32)
            self.dirty = True
        if not keys:
            return np.zeros((0, 768), dtype=np.float32)
        return np.vstack([self.vectors[k] for k in keys]).astype(np.float32)

    def save(self) -> None:
        """Write the cache when something new was embedded."""
        if not self.dirty:
            return
        keys = sorted(self.vectors)
        np.savez(self.path, windows=np.array(keys, dtype=np.float64),
                 vectors=np.vstack([self.vectors[k] for k in keys]))
        self.dirty = False


# ------------------------------------------------------------ loading


class Rec(NamedTuple):
    """A recording ready for the voice stage of the pipeline."""

    rid: str
    kind: str
    readers: List[str]
    plan: Row
    audio: Audio
    spans: List[SpeakerSpan]
    quick: List[bool]
    words: List[Word]
    coarse_windows: List[Window]
    fine_windows: List[Window]
    coarse: FloatArray
    fine: FloatArray
    store: Embedder


def pipeline_words(raw: Sequence[Word], audio: Audio,
                   p: TranscriptionParameters) -> List[Word]:
    """Seams resolved and words over silence dropped, as the pipeline."""
    activity = speech_activity(audio.samples, audio.sample_rate,
                               p.activity_frame, p.activity_threshold)
    return heard_words([w for w in resolve_seams(raw) if w.text.strip()],
                       activity)


def make_rec(rid: str, kind: str, plan: Row, audio: Audio,
             spans: List[SpeakerSpan], quick: List[bool],
             raw: Sequence[Word], store_path: Path,
             p: TranscriptionParameters) -> Rec:
    """Run the pipeline up to the voice windows and embed them."""
    words = pipeline_words(raw, audio, p)
    regions = speech_regions(words, p.region_gap)
    coarse_w = analysis_windows(regions, *p.coarse())
    fine_w = analysis_windows(regions, *p.fine())
    store = Embedder(store_path, audio)
    return Rec(rid, kind, list(plan["readers"]), plan, audio, spans, quick,
               words, coarse_w, fine_w, store.get(coarse_w),
               store.get(fine_w), store)


def load_recordings(data_dir: Path, split: str,
                    p: TranscriptionParameters, kinds: Sequence[str]
                    ) -> Tuple[List[Rec], Dict[str, FloatArray]]:
    """Return the recordings of ``kinds`` and the voiceprints of ``split``."""
    corpus = tr05.Corpus(tr05.ensure_parquet(data_dir, split))
    old_cache = data_dir / "cache" / split
    cache = quick_cache(data_dir, split)
    old_plan = json.loads((old_cache / "plan.json").read_text())
    prints: Dict[str, FloatArray] = {}
    rules = SampleRules(same_voice_threshold=0.85,
                        max_other_voice_share=0.1)
    out: List[Rec] = []
    for recording in old_plan["recordings"]:
        kind = recording["kind"]
        if kind not in ("enrol", "two", "meeting"):
            continue
        if kind != "enrol" and kind not in kinds:
            continue
        built = tr05.build(corpus, recording)
        raw = tr05.load_words(old_cache, recording["id"])
        if kind == "enrol":
            stored = np.load(tr05.vectors_path(old_cache, recording["id"]))
            sample = tr05.Recording(
                recording, built, raw, stored["coarse"].astype(np.float32),
                stored["fine"].astype(np.float32),
                [(float(a), float(b)) for a, b in
                 stored["coarse_windows"].reshape(-1, 2).tolist()],
                [], 0.0, raw)
            reason, vector = tr05.try_enrol(sample, rules, p)
            if vector is None:
                raise SystemExit(f"sample rejected: {reason}")
            prints[recording["readers"][0]] = vector
            continue
        if kind == "two":
            rng = random.Random(recording["id"])
            others = [r for r in old_plan["readers"]
                      if r not in recording["readers"]]
            recording = {**recording, "absent": rng.choice(others),
                         "unenrolled_one": rng.choice(recording["readers"])}
        out.append(make_rec(
            recording["id"], kind, recording, built.audio, built.spans,
            [False] * len(built.spans), raw,
            cache / f"{recording['id']}.vectors.npz", p))
    if any(k in QUICK_KINDS for k in kinds):
        plan = json.loads((cache / "plan.json").read_text())
        for recording in plan["recordings"]:
            for gaps in (True, False):
                kind = "quickgap" if gaps else "quicknogap"
                if kind not in kinds:
                    continue
                rid = quick_id(recording["id"], gaps)
                mix = build_quick(corpus, recording, gaps)
                out.append(make_rec(
                    rid, kind, recording, mix.audio, mix.spans, mix.quick,
                    tr05.load_words(cache, rid),
                    cache / f"{rid}.vectors.npz", p))
    for rec in out:
        rec.store.save()
    return out, prints


def enrolled_of(rec: Rec, condition: str) -> Set[str]:
    """Return the readers with a voiceprint under ``condition``."""
    people = set(rec.readers)
    if condition == "one_unenrolled":
        return people - {rec.plan["unenrolled_one"]}
    if condition == "two_unenrolled":
        if rec.kind == "meeting":
            return people - set(rec.plan["unenrolled_two"])
        return set()
    if condition == "enrolled_absent":
        return people | {rec.plan["absent"]}
    return people


# ---------------------------------------------------------- long pass


class LongPass(NamedTuple):
    """Who is there (the long windows), shared by every method.

    ``mean`` and ``centroids`` are the space the short windows are
    scored in (vectors centred on the long windows' mean); ``names``
    maps each diarizer label to the final speaker (an enrolled id or an
    anonymous ``S1``...).
    """

    fine: List[int]
    scores: Optional[FloatArray]
    mean: FloatArray
    centroids: FloatArray
    people: Dict[int, str]


def long_pass(rec: Rec, prints: Dict[str, FloatArray], count: Optional[int],
              p: TranscriptionParameters) -> LongPass:
    """Run the pipeline's grouping (seeded when there are voiceprints)."""
    def one_voice(vectors: FloatArray) -> bool:
        """The pipeline's single-voice rule."""
        return is_single_voice(vectors, p.single_voice_threshold,
                               p.single_voice_quantile)

    mean: FloatArray = rec.coarse.mean(axis=0).astype(np.float32)
    if prints:
        seeded = seed_voices(
            rec.coarse, rec.fine, tr05.voiceprints_of(prints), count,
            p.match_threshold, unit_vectors, p.group_threshold,
            p.max_speakers, one_voice)
        labels, fine, scores = seeded.coarse, seeded.fine, seeded.fine_scores
        people = seeded.people
    else:
        groups = group_voices(
            rec.coarse, rec.fine, count, p.group_threshold, p.max_speakers,
            p.centre_windows, p.single_voice_threshold,
            p.single_voice_quantile)
        labels, fine, scores = groups.coarse, groups.fine, groups.fine_scores
        people = {}
    centroids = voice_centroids(centre(rec.coarse, mean), labels)
    return LongPass(fine, scores, mean, centroids, people)


def piece_scores(vectors: FloatArray, lp: LongPass) -> FloatArray:
    """Cosines of pieces with the long pass's voices (its space)."""
    if vectors.shape[0] == 0:
        return np.zeros((0, lp.centroids.shape[0]), dtype=np.float32)
    out: FloatArray = (centre(vectors, lp.mean) @ lp.centroids.T).astype(
        np.float32)
    return out


def final_names(labels: Sequence[str], lp: LongPass,
                taken: Set[str]) -> Dict[str, str]:
    """Map diarizer labels to names as the TR-05 study does."""
    final: Dict[str, str] = {}
    anon = 1
    for label in dict.fromkeys(labels):
        number = int(label.split("_")[1]) - 1
        if number in lp.people:
            final[label] = lp.people[number]
        else:
            while f"S{anon}" in taken:
                anon += 1
            final[label] = f"S{anon}"
            anon += 1
    return final


# ------------------------------------------------------------ methods


def fast_assign(words: Sequence[Word], spans: Sequence[SpeakerSpan]
                ) -> List[Word]:
    """:func:`~transcriber.domain.speakers.assign_speakers`, vectorised.

    Same rule (most overlap, then nearest edge, then the first span);
    checked against the domain function in ``analyse``.
    """
    if not spans:
        return [w.model_copy(update={"speaker": "SPEAKER_00"})
                for w in words]
    ws = np.array([w.start for w in words])[:, None]
    we = np.array([w.end for w in words])[:, None]
    ss = np.array([s.start for s in spans])[None, :]
    se = np.array([s.end for s in spans])[None, :]
    overlap = np.minimum(we, se) - np.maximum(ws, ss)
    middle = (ws + we) / 2
    distance = np.minimum(np.abs(middle - ss), np.abs(middle - se))
    best = overlap.max(axis=1, keepdims=True)
    masked = np.where(overlap == best, distance, np.inf)
    chosen = masked.argmin(axis=1)
    return [w.model_copy(update={"speaker": spans[int(c)].speaker})
            for w, c in zip(words, chosen)]


def keep_runs(words: Sequence[Word], evidence: Optional[VoiceEvidence],
              votes: int, margin: float, p: TranscriptionParameters
              ) -> Tuple[List[Word], List[Window], int]:
    """:func:`keep_sentences_whole` with the minimum evidence as a knob.

    A run of a sentence keeps its speaker (and the sentence is cut
    there) when its margin is positive and either at least ``votes``
    windows prefer it or the margin reaches ``margin``; otherwise it
    goes to a neighbour, within the default bounds. ``votes`` 2 and
    ``margin`` 0.1 are the pipeline's rule. Also returns how many
    sentences end up holding more than one speaker.
    """
    out = list(words)
    moved: List[Window] = []
    if evidence is None:
        return out, moved, 0
    rule = _Rule(evidence, WindowIndex(evidence.windows), margin,
                 p.whole_sentence_max_move, p.whole_sentence_max_moved)
    split = 0
    for first, end in _sentences(out, p.turn_gap):
        spent = 0.0
        while True:
            runs = _runs(out, first, end)
            if len(runs) < 2:
                break
            order = sorted(range(len(runs)), key=lambda k: (
                out[runs[k][-1]].end - out[runs[k][0]].start))
            done = False
            for k in order:
                run = runs[k]
                start, stop = out[run[0]].start, out[run[-1]].end
                speaker = out[run[0]].speaker
                if speaker is None or stop - start > rule.max_move \
                        or spent + (stop - start) > rule.max_moved:
                    continue
                own = support(evidence, start, stop, speaker, rule.windows)
                weak = (own is None or own.margin <= 0
                        or (own.votes < votes and own.margin < margin))
                if not weak:
                    continue
                target = _neighbour(out, runs, k, rule, start, stop)
                if target is None or target == speaker:
                    continue
                for n in run:
                    out[n] = out[n].model_copy(update={"speaker": target})
                moved.append((start, stop))
                spent += stop - start
                done = True
                break
            if not done:
                break
        if len(_runs(out, first, end)) > 1:
            split += 1
    return out, moved, split


class Outcome(NamedTuple):
    """The words with diarizer labels and what the method saw."""

    words: List[Word]
    evidence: Optional[VoiceEvidence]
    moved: List[Window]
    split_sentences: int
    pieces: List[float]


def grid(rec: Rec, lp: LongPass, p: TranscriptionParameters,
         votes: int = 2, margin: float = 0.1,
         snap: Optional[Callable[[Rec], List[float]]] = None,
         distance: float = 0.0) -> Outcome:
    """The current pipeline; with ``snap``, the hybrid.

    ``snap`` returns the cut candidates (pauses, word gaps); each change
    of speaker between grid spans moves to the nearest candidate within
    ``distance`` seconds.
    """
    spans = spans_from_windows(rec.fine_windows, lp.fine)
    evidence = None if lp.scores is None else VoiceEvidence(
        rec.fine_windows, lp.scores,
        [speaker_label(n) for n in range(lp.scores.shape[1])])
    if snap is not None and len(spans) > 1:
        cuts = np.array(snap(rec))
        moved_spans = list(spans)
        for n in range(len(spans) - 1):
            left, right = moved_spans[n], moved_spans[n + 1]
            edge = left.end
            if cuts.size == 0:
                break
            if right.start - left.end > 0.01:
                continue  # the change already lies in a pause
            near = cuts[np.abs(cuts - edge) <= distance]
            near = near[(near > left.start) & (near < right.end)]
            if near.size == 0:
                continue
            new = float(near[np.argmin(np.abs(near - edge))])
            moved_spans[n] = left.model_copy(update={"end": new})
            moved_spans[n + 1] = right.model_copy(update={"start": new})
        spans = moved_spans
    words = fast_assign(rec.words, spans)
    if p.smoothing_run:
        words = smooth_speakers(words, p.smoothing_run)
    words, moved, split = keep_runs(words, evidence, votes, margin, p)
    return Outcome(words, evidence, moved, split, [])


class PieceSettings(NamedTuple):
    """How ``pieces`` cuts and decides (see :func:`pieces`)."""

    level: Optional[float]
    pause: float
    word_gap: Optional[float]
    minimum: float
    short: str
    longest: Optional[float]
    whole: bool

    def name(self) -> str:
        """Return a short readable name of the settings."""
        return (f"level={self.level},pause={self.pause},"
                f"word_gap={self.word_gap},min={self.minimum},"
                f"short={self.short},longest={self.longest},"
                f"whole={self.whole}")


_PAUSES: Dict[Tuple[str, float, float], List[float]] = {}


def pause_cuts(rec: Rec, level: float, pause: float) -> List[float]:
    """Return the middles of the stretches below ``level`` of ``pause`` s.

    Frames of 25 ms, as the pipeline's activity gate.
    """
    key = (rec.rid, level, pause)
    if key in _PAUSES:
        return _PAUSES[key]
    activity = speech_activity(rec.audio.samples, rec.audio.sample_rate,
                               0.025, level)
    quiet = ~activity.active
    cuts: List[float] = []
    n = 0
    frames = int(quiet.shape[0])
    while n < frames:
        if not quiet[n]:
            n += 1
            continue
        m = n
        while m < frames and quiet[m]:
            m += 1
        if (m - n) * activity.frame >= pause:
            cuts.append(round((n + m) / 2 * activity.frame, 3))
        n = m
    _PAUSES[key] = cuts
    return cuts


def gap_cuts(words: Sequence[Word], gap: float) -> List[float]:
    """Return the middles of the gaps of at least ``gap`` s between words."""
    return [round((a.end + b.start) / 2, 3) for a, b in zip(words, words[1:])
            if b.start - a.end >= gap]


def candidate_cuts(rec: Rec, level: Optional[float], pause: float,
                   word_gap: Optional[float]) -> List[float]:
    """Return the cut candidates of the activity gate and word gaps."""
    cuts: List[float] = []
    if level is not None:
        cuts += pause_cuts(rec, level, pause)
    if word_gap is not None:
        cuts += gap_cuts(rec.words, word_gap)
    return sorted(set(cuts))


def cut_regions(regions: Sequence[Window], cuts: Sequence[float],
                longest: Optional[float]) -> List[Window]:
    """Cut the speech regions at ``cuts``; split pieces over ``longest``."""
    out: List[Window] = []
    array = np.array(cuts)
    for start, end in regions:
        inner = array[(array > start + 0.01) & (array < end - 0.01)]
        edges = [start] + [float(c) for c in inner] + [end]
        for a, b in zip(edges, edges[1:]):
            if longest is not None and b - a > longest:
                parts = int(math.ceil((b - a) / longest))
                step = (b - a) / parts
                out += [(round(a + i * step, 3), round(a + (i + 1) * step, 3))
                        for i in range(parts)]
            else:
                out.append((round(a, 3), round(b, 3)))
    return out


def pieces(rec: Rec, lp: LongPass, p: TranscriptionParameters,
           s: PieceSettings) -> Outcome:
    """Pause-aligned pieces scored against the long pass's voices.

    Pieces of at least ``s.minimum`` seconds take their nearest voice.
    A shorter piece takes (``short="merge"``) the voice of the
    neighbouring long piece of its region that its own vector is closer
    to (its own nearest voice when it has no such neighbour or is too
    short to embed), or (``short="grid"``) is left to the fixed grid.
    """
    if lp.scores is None:
        return grid(rec, lp, p)
    regions = speech_regions(rec.words, p.region_gap)
    cuts = candidate_cuts(rec, s.level, s.pause, s.word_gap)
    cut = cut_regions(regions, cuts, s.longest)
    embeddable = [w for w in cut if w[1] - w[0] >= SHORTEST_EMBEDDED]
    vectors = rec.store.get(embeddable)
    finite = np.asarray(~np.isnan(vectors).any(axis=1), dtype=bool)
    embeddable = [w for w, ok in zip(embeddable, finite.tolist()) if ok]
    vectors = vectors[finite]
    row_of = {w: n for n, w in enumerate(embeddable)}
    scores = piece_scores(vectors, lp)
    labels: List[Optional[int]] = []
    for w in cut:
        if w[1] - w[0] >= s.minimum and w in row_of:
            labels.append(int(np.argmax(scores[row_of[w]])))
        else:
            labels.append(None)
    region_of = [next(i for i, r in enumerate(regions)
                      if r[0] - 1e-3 <= w[0] and w[1] <= r[1] + 1e-3)
                 for w in cut]
    centred = centre(vectors, lp.mean) if vectors.shape[0] else vectors
    final: List[Optional[int]] = list(labels)
    for i, w in enumerate(cut):
        if labels[i] is not None or s.short != "merge":
            continue
        sides = []
        for j in (range(i - 1, -1, -1), range(i + 1, len(cut))):
            for k in j:
                if region_of[k] != region_of[i]:
                    break
                if labels[k] is not None:
                    sides.append(k)
                    break
        if w in row_of and sides:
            mine = centred[row_of[w]]
            best = max(sides, key=lambda k: float(
                mine @ centred[row_of[cut[k]]]))
            final[i] = labels[best]
        elif sides:
            final[i] = labels[max(sides, key=lambda k: cut[k][1] - cut[k][0])]
        elif w in row_of:
            final[i] = int(np.argmax(scores[row_of[w]]))
    long_rows = [row_of[w] for w, lab in zip(cut, labels) if lab is not None]
    evidence = VoiceEvidence(
        [embeddable[r] for r in long_rows], scores[long_rows],
        [speaker_label(n) for n in range(lp.centroids.shape[0])],
    ) if long_rows else None
    decided = [SpeakerSpan(speaker=speaker_label(lab), start=w[0], end=w[1])
               for w, lab in zip(cut, final) if lab is not None]
    words = fast_assign(rec.words, decided)
    if s.short == "grid":
        fallback = fast_assign(
            rec.words, spans_from_windows(rec.fine_windows, lp.fine))
        open_spans = [w for w, lab in zip(cut, final) if lab is None]
        words = [f if any(a <= (x.start + x.end) / 2 <= b
                          for a, b in open_spans) else x
                 for x, f in zip(words, fallback)]
    if p.smoothing_run:
        words = smooth_speakers(words, p.smoothing_run)
    moved: List[Window] = []
    split = 0
    if s.whole:
        words, moved, split = keep_runs(words, evidence, 2, 0.1, p)
    else:
        for first, end in _sentences(words, p.turn_gap):
            split += len(_runs(words, first, end)) > 1
    return Outcome(words, evidence, moved, split,
                   [w[1] - w[0] for w in cut])


# ------------------------------------------------------------ scoring


def right_flags(names: Sequence[str], truth: Sequence[str],
                enrolled: Set[str], present: Set[str]) -> List[bool]:
    """Whether each word is right, by the rule of ``tr05.score``."""
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
    return [n == t if t in enrolled else
            (n not in enrolled and mapping.get(n) == t)
            for n, t in zip(names, truth)]


def span_of_words(words: Sequence[Word],
                  spans: Sequence[SpeakerSpan]) -> List[int]:
    """Return the truth span of each word (as ``tr05.word_truth``)."""
    out: List[int] = []
    for word in words:
        middle = (word.start + word.end) / 2
        inside = [i for i, s in enumerate(spans)
                  if s.start <= middle <= s.end]
        if inside:
            out.append(inside[0])
        else:
            out.append(min(range(len(spans)), key=lambda i: min(
                abs(middle - spans[i].start), abs(middle - spans[i].end))))
    return out


def matched_changes(truth: Sequence[float], found: Sequence[float],
                    tolerance: float = TOLERANCE) -> int:
    """Pair truth and found change points within ``tolerance``, greedily."""
    pairs = sorted((abs(t - f), i, j) for i, t in enumerate(truth)
                   for j, f in enumerate(found) if abs(t - f) <= tolerance)
    used_t: Set[int] = set()
    used_f: Set[int] = set()
    for _, i, j in pairs:
        if i not in used_t and j not in used_f:
            used_t.add(i)
            used_f.add(j)
    return len(used_t)


def truth_changes(spans: Sequence[SpeakerSpan]) -> List[float]:
    """Return the times where the true speaker changes."""
    return [(a.end + b.start) / 2 for a, b in zip(spans, spans[1:])
            if a.speaker != b.speaker]


def found_changes(words: Sequence[Word], names: Sequence[str]) -> List[float]:
    """Return the times between consecutive words of different names."""
    return [(a.end + b.start) / 2
            for a, b, x, y in zip(words, words[1:], names, names[1:])
            if x != y]


SHORT_BINS = LENGTHS


def tally(rec: Rec, outcome: Outcome, lp: LongPass, enrolled: Set[str],
          taken: Set[str]) -> Row:
    """Count words right (all, quick), change points and short errors."""
    labels = [w.speaker or "" for w in outcome.words]
    names_of = final_names(labels, lp, taken)
    names = [names_of[x] for x in labels]
    which = span_of_words(outcome.words, rec.spans)
    truth = [rec.spans[i].speaker for i in which]
    flags = right_flags(names, truth, enrolled, set(rec.readers))
    quick = [rec.quick[i] for i in which]
    lengths = [rec.spans[i].end - rec.spans[i].start for i in which]
    real = truth_changes(rec.spans)
    found = found_changes(outcome.words, names)
    row: Row = {
        "words": len(flags), "right": sum(flags),
        "quick_words": sum(quick),
        "quick_right": sum(f for f, q in zip(flags, quick) if q),
        "truth_changes": len(real), "found_changes": len(found),
        "matched_changes": matched_changes(real, found),
        "minutes": rec.audio.duration / 60,
        "wrong": len(flags) - sum(flags),
        "split_sentences": outcome.split_sentences,
    }
    for edge in SHORT_BINS:
        row[f"wrong_under_{edge}"] = sum(
            1 for f, n in zip(flags, lengths) if not f and n < edge)
    return row


def summarise(tallies: Sequence[Row]) -> Row:
    """Pool tallies into shares and rates."""
    total = {k: sum(t[k] for t in tallies) for k in tallies[0]}
    found = total["found_changes"]
    return {
        **{k: (round(v, 3) if isinstance(v, float) else v)
           for k, v in total.items()},
        "right_share": round(total["right"] / max(total["words"], 1), 4),
        "quick_right_share": round(
            total["quick_right"] / total["quick_words"], 4)
        if total["quick_words"] else None,
        "change_precision": round(total["matched_changes"] / found, 4)
        if found else None,
        "change_recall": round(
            total["matched_changes"] / total["truth_changes"], 4)
        if total["truth_changes"] else None,
        "false_changes_per_minute": round(
            (found - total["matched_changes"]) / total["minutes"], 3),
    }


Method = Callable[[Rec, LongPass], Outcome]


class Bench:
    """The recordings of a split with their long passes, cached."""

    def __init__(self, recs: List[Rec], prints: Dict[str, FloatArray],
                 p: TranscriptionParameters, counts: Sequence[str]) -> None:
        """Run the long pass of every recording, condition and count."""
        self.recs = recs
        self.p = p
        self.taken = set(prints)
        self.runs: List[Tuple[Rec, str, str, LongPass, Set[str]]] = []
        for rec in recs:
            for condition in CONDITIONS:
                enrolled = enrolled_of(rec, condition)
                candidates = {k: prints[k] for k in sorted(enrolled)}
                for count_name in counts:
                    count = len(rec.readers) if count_name == "given" \
                        else None
                    self.runs.append((
                        rec, condition, count_name,
                        long_pass(rec, candidates, count, p), enrolled))

    def evaluate(self, method: Method) -> Dict[Tuple[str, str], Row]:
        """Pool ``method`` per kind and count of speakers."""
        tallies: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
        for rec, _, count_name, lp, enrolled in self.runs:
            outcome = method(rec, lp)
            tallies[(rec.kind, count_name)].append(
                tally(rec, outcome, lp, enrolled, self.taken))
        return {k: summarise(v) for k, v in tallies.items()}


def save_stores(recs: Iterable[Rec]) -> None:
    """Write every recording's new embeddings to its cache."""
    for rec in recs:
        rec.store.save()


# -------------------------------------------------------- reliability


def reliability_rows(split: str, recs: Sequence[Rec],
                     prints: Dict[str, FloatArray]) -> List[Row]:
    """How a voice vector's reliability falls with the piece's length.

    In each four-reader meeting, pieces of each length of
    :data:`LENGTHS` are cut at random inside each turn's recognised
    speech (three per turn and length). Reported per length: cosines of
    pairs of pieces of the same reader against different readers (raw,
    and centred on the meeting's mean as the pipeline scores them),
    how often a piece's nearest true voice of the meeting (centroid of
    its pure long windows, centred) is its reader among the four and
    against each other reader alone (two voices), and the cosine with
    the reader's voiceprint against the others' (raw).
    """
    rows: List[Row] = []
    by_length: Dict[float, Dict[str, List[float]]] = {
        L: defaultdict(list) for L in LENGTHS}
    for rec in recs:
        if rec.kind != "meeting":
            continue
        rng = random.Random(f"reliability-{rec.rid}")
        mean: FloatArray = rec.coarse.mean(axis=0).astype(np.float32)
        truth = tr05.window_truth(rec.coarse_windows, rec.spans)
        readers = sorted(set(rec.readers))
        cents = np.vstack([unit_rows(centre(rec.coarse[[
            i for i, t in enumerate(truth) if t == r]], mean).mean(
                axis=0, keepdims=True)) for r in readers])
        prints_m = unit_rows(np.vstack([prints[r] for r in readers]))
        for L in LENGTHS:
            windows: List[Window] = []
            owners: List[int] = []
            for span in rec.spans:
                inside = [w for w in rec.words
                          if span.start <= (w.start + w.end) / 2 <= span.end]
                if not inside:
                    continue
                lo, hi = inside[0].start, inside[-1].end
                if hi - lo < L:
                    continue
                for _ in range(3):
                    a = rng.uniform(lo, hi - L)
                    windows.append((round(a, 3), round(a + L, 3)))
                    owners.append(readers.index(span.speaker))
            vectors = rec.store.get(windows)
            finite = np.asarray(~np.isnan(vectors).any(axis=1), dtype=bool)
            vectors = vectors[finite]
            owners = [o for o, ok in zip(owners, finite.tolist()) if ok]
            raw = unit_rows(vectors)
            centred = centre(vectors, mean)
            stats = by_length[L]
            pair_raw = raw @ raw.T
            pair_c = centred @ centred.T
            own = np.array(owners)
            same = own[:, None] == own[None, :]
            upper = np.triu(np.ones_like(same, dtype=bool), k=1)
            stats["same_raw"] += pair_raw[same & upper].tolist()
            stats["other_raw"] += pair_raw[~same & upper].tolist()
            stats["same_centred"] += pair_c[same & upper].tolist()
            stats["other_centred"] += pair_c[~same & upper].tolist()
            to_cent = centred @ cents.T
            stats["nearest_of_four"] += (
                to_cent.argmax(axis=1) == own).astype(float).tolist()
            for i, o in enumerate(owners):
                for k in range(len(readers)):
                    if k != o:
                        stats["nearest_of_two"].append(
                            float(to_cent[i, o] > to_cent[i, k]))
            to_print = raw @ prints_m.T
            for i, o in enumerate(owners):
                stats["print_same"].append(float(to_print[i, o]))
                stats["print_other"] += [float(to_print[i, k])
                                         for k in range(len(readers))
                                         if k != o]
    for L in LENGTHS:
        st = by_length[L]
        raw_sep = tr05.separation(st["same_raw"], st["other_raw"])
        cen_sep = tr05.separation(st["same_centred"], st["other_centred"])
        prt_sep = tr05.separation(st["print_same"], st["print_other"])
        rows.append({
            "split": split, "question": "reliability", "length": L,
            "pieces": len(st["nearest_of_four"]),
            "same_raw_p50": round(float(np.median(st["same_raw"])), 4),
            "other_raw_p50": round(float(np.median(st["other_raw"])), 4),
            "raw_auc": raw_sep.get("auc"), "raw_eer": raw_sep.get("eer"),
            "same_centred_p50": round(float(np.median(
                st["same_centred"])), 4),
            "other_centred_p50": round(float(np.median(
                st["other_centred"])), 4),
            "centred_auc": cen_sep.get("auc"),
            "centred_eer": cen_sep.get("eer"),
            "nearest_of_four": round(float(np.mean(
                st["nearest_of_four"])), 4),
            "nearest_of_two": round(float(np.mean(st["nearest_of_two"])), 4),
            "voiceprint_auc": prt_sep.get("auc"),
            "voiceprint_eer": prt_sep.get("eer"),
        })
    return rows


def gap_rows(split: str, recs: Sequence[Rec]) -> List[Row]:
    """How often the recogniser leaves a gap between consecutive words."""
    rows: List[Row] = []
    for kind in KINDS:
        gaps = np.array([b.start - a.end for rec in recs if rec.kind == kind
                         for a, b in zip(rec.words, rec.words[1:])])
        rows.append({
            "split": split, "question": "word_gaps", "kind": kind,
            "pairs": int(gaps.size),
            **{f"share_at_least_{g}": round(float((gaps >= g).mean()), 4)
               for g in (0.01, 0.05, 0.1, 0.2, 0.3)},
        })
    return rows


def usable_length(rows: Sequence[Row], split: str) -> Optional[float]:
    """Shortest length at or above which two-voice accuracy stays usable."""
    mine = sorted((r for r in rows if r["split"] == split
                   and r["question"] == "reliability"),
                  key=lambda r: r["length"])
    for n, r in enumerate(mine):
        if all(x["nearest_of_two"] >= USABLE_ACCURACY for x in mine[n:]):
            return float(r["length"])
    return None


# ------------------------------------------------------------- sweeps


def piece_sweep() -> List[PieceSettings]:
    """Return the settings of ``pieces`` tried on ``dev``, stage one.

    Where to cut, with the rest fixed (minimum 0.5 s, short pieces
    merged, pieces split above 3 s, whole-sentence rule on).
    """
    out: List[PieceSettings] = []
    levels: List[Optional[float]] = [None, 0.003, 0.01, 0.03]
    gaps: List[Optional[float]] = [None, 0.05, 0.1]
    for level in levels:
        for pause in ((0.1, 0.2, 0.3) if level is not None else (0.0,)):
            for gap in gaps:
                if level is None and gap is None:
                    continue
                out.append(PieceSettings(level, pause, gap, 0.5, "merge",
                                         3.0, True))
    return out


def stage_two(best: PieceSettings) -> List[PieceSettings]:
    """Return the settings tried around the best cuts, stage two."""
    out: List[PieceSettings] = []
    for minimum in (0.35, 0.5, 0.75, 1.0):
        for short in ("merge", "grid"):
            for longest in (1.5, 3.0, None):
                for whole in (True, False):
                    out.append(best._replace(minimum=minimum, short=short,
                                             longest=longest, whole=whole))
    return out


def hybrid_sweep() -> List[Row]:
    """Return the hybrid settings tried on ``dev``."""
    out: List[Row] = []
    for source in ({"level": None, "pause": 0.0, "word_gap": 0.05},
                   {"level": None, "pause": 0.0, "word_gap": 0.1},
                   {"level": None, "pause": 0.0, "word_gap": 0.2},
                   {"level": 0.01, "pause": 0.1, "word_gap": None},
                   {"level": 0.01, "pause": 0.2, "word_gap": None},
                   {"level": 0.03, "pause": 0.1, "word_gap": 0.1}):
        for distance in (0.25, 0.5, 0.75):
            out.append({**source, "distance": distance, "votes": 2,
                        "margin": 0.1})
    for votes in (1, 2, 3, 4):
        for margin in (0.05, 0.1, 0.2, 0.3):
            out.append({"level": None, "pause": 0.0, "word_gap": None,
                        "distance": 0.0, "votes": votes, "margin": margin})
    return out


def hybrid_method(cfg: Row, p: TranscriptionParameters) -> Method:
    """Return the hybrid with the settings ``cfg``."""
    def cuts(rec: Rec) -> List[float]:
        """The cut candidates of the settings."""
        return candidate_cuts(rec, cfg["level"], cfg["pause"],
                              cfg["word_gap"])

    snap = cuts if cfg["level"] is not None or cfg["word_gap"] is not None \
        else None

    def method(rec: Rec, lp: LongPass) -> Outcome:
        """The hybrid."""
        return grid(rec, lp, p, cfg["votes"], cfg["margin"], snap,
                    cfg["distance"])
    return method


def rows_of(split: str, method: str, settings: Row,
            result: Dict[Tuple[str, str], Row]) -> List[Row]:
    """Turn an evaluation into rows."""
    return [{"split": split, "question": "method", "method": method,
             "settings": settings, "kind": kind, "speakers": count,
             **summary}
            for (kind, count), summary in sorted(result.items())]


def dev_score(result: Dict[Tuple[str, str], Row]) -> float:
    """Mean words-right share over the quick kinds (the choice on dev)."""
    shares = [v["right_share"] for (k, _), v in result.items()
              if k in QUICK_KINDS]
    return float(np.mean(shares))


def check_assign(recs: Sequence[Rec], p: TranscriptionParameters) -> None:
    """Make sure :func:`fast_assign` equals the domain function."""
    from transcriber.domain.speakers import assign_speakers

    for rec in recs[:12]:
        labels = [n % 3 for n in range(len(rec.fine_windows))]
        spans = spans_from_windows(rec.fine_windows, labels)
        slow = [w.speaker for w in assign_speakers(rec.words, spans)]
        fast = [w.speaker for w in fast_assign(rec.words, spans)]
        if slow != fast:
            raise SystemExit(f"fast_assign differs on {rec.rid}")


def analyse(args: argparse.Namespace) -> None:
    """Sweep on dev, report the chosen settings on both splits."""
    p = TranscriptionParameters()
    rows: List[Row] = []
    benches: Dict[str, Bench] = {}
    for split in tr05.SPLITS:
        recs, prints = load_recordings(args.data_dir, split, p, KINDS)
        check_assign(recs, p)
        rows += reliability_rows(split, recs, prints)
        rows += gap_rows(split, recs)
        save_stores(recs)
        benches[split] = Bench(recs, prints, p, ("given", "estimated"))
        print(f"{split}: {len(recs)} recordings loaded", flush=True)
    dev = benches["dev"]
    base = {"method": "grid"}

    def baseline(rec: Rec, lp: LongPass) -> Outcome:
        """The current pipeline."""
        return grid(rec, lp, p)

    results: Dict[str, Tuple[Row, Dict[Tuple[str, str], Row]]] = {}
    results["grid"] = (base, dev.evaluate(baseline))
    rows += rows_of("dev", "grid", base, results["grid"][1])
    print(f"grid: {dev_score(results['grid'][1]):.4f}", flush=True)
    best_cut: Optional[Tuple[float, PieceSettings]] = None
    for s in piece_sweep():
        def method(rec: Rec, lp: LongPass, s: PieceSettings = s) -> Outcome:
            """``pieces`` with ``s``."""
            return pieces(rec, lp, p, s)
        result = dev.evaluate(method)
        save_stores(dev.recs)
        rows += rows_of("dev", "pieces", s._asdict(), result)
        score = dev_score(result)
        print(f"pieces {s.name()}: {score:.4f}", flush=True)
        if best_cut is None or score > best_cut[0]:
            best_cut = (score, s)
    assert best_cut is not None
    best_pieces = best_cut
    for s in stage_two(best_cut[1]):
        if s == best_cut[1]:
            continue

        def method2(rec: Rec, lp: LongPass, s: PieceSettings = s
                    ) -> Outcome:
            """``pieces`` with ``s``."""
            return pieces(rec, lp, p, s)
        result = dev.evaluate(method2)
        save_stores(dev.recs)
        rows += rows_of("dev", "pieces", s._asdict(), result)
        score = dev_score(result)
        print(f"pieces {s.name()}: {score:.4f}", flush=True)
        if score > best_pieces[0]:
            best_pieces = (score, s)
    best_hybrid: Optional[Tuple[float, Row]] = None
    for cfg in hybrid_sweep():
        result = dev.evaluate(hybrid_method(cfg, p))
        rows += rows_of("dev", "hybrid", cfg, result)
        score = dev_score(result)
        print(f"hybrid {cfg}: {score:.4f}", flush=True)
        if best_hybrid is None or score > best_hybrid[0]:
            best_hybrid = (score, cfg)
    assert best_hybrid is not None
    chosen_pieces = best_pieces[1]
    chosen_hybrid = best_hybrid[1]
    test = benches["test"]

    def on_test(rec: Rec, lp: LongPass) -> Outcome:
        """``pieces`` with the settings chosen on dev."""
        return pieces(rec, lp, p, chosen_pieces)

    rows += rows_of("test", "grid", base, test.evaluate(baseline))
    rows += rows_of("test", "pieces", chosen_pieces._asdict(),
                    test.evaluate(on_test))
    save_stores(test.recs)
    rows += rows_of("test", "hybrid", chosen_hybrid,
                    test.evaluate(hybrid_method(chosen_hybrid, p)))
    rows.append({"question": "chosen", "pieces": chosen_pieces._asdict(),
                 "pieces_dev_score": round(best_pieces[0], 4),
                 "hybrid": chosen_hybrid,
                 "hybrid_dev_score": round(best_hybrid[0], 4),
                 "grid_dev_score": round(dev_score(results["grid"][1]), 4),
                 "usable_length_dev": usable_length(rows, "dev"),
                 "usable_length_test": usable_length(rows, "test")})
    print(tr05.write_jsonl(args.output, rows), "rows")


# --------------------------------------------------------------- cost


def cost(args: argparse.Namespace) -> None:
    """Time the embedding of the grid and of the chosen pieces (dev).

    A fresh encoder embeds, per recording, the grid's long and short
    windows and then the chosen pieces, without any cache; prints and
    appends a row with the GPU seconds per audio minute of each.
    """
    import torch

    from transcriber.adapters.outbound.wavlm_voice_encoder import (
        WavLMVoiceEncoder,
    )

    p = TranscriptionParameters()
    chosen = [r for r in tr05.read_jsonl(args.output)
              if r.get("question") == "chosen"][-1]
    s = PieceSettings(**chosen["pieces"])
    recs, _ = load_recordings(args.data_dir, "dev", p, KINDS)
    encoder = WavLMVoiceEncoder(device="cuda")
    encoder.embed(recs[0].audio, recs[0].fine_windows[:8])  # warm up
    spent = {"coarse": 0.0, "fine": 0.0, "pieces": 0.0}
    minutes = 0.0
    counts = {"fine": 0, "pieces": 0}
    for rec in recs:
        regions = speech_regions(rec.words, p.region_gap)
        cut = [w for w in cut_regions(regions, candidate_cuts(
            rec, s.level, s.pause, s.word_gap), s.longest)
            if w[1] - w[0] >= SHORTEST_EMBEDDED]
        cut.sort(key=lambda w: w[1] - w[0])
        for name, windows in (("coarse", rec.coarse_windows),
                              ("fine", rec.fine_windows), ("pieces", cut)):
            torch.cuda.synchronize()
            began = time.perf_counter()
            encoder.embed(rec.audio, windows)
            torch.cuda.synchronize()
            spent[name] += time.perf_counter() - began
        counts["fine"] += len(rec.fine_windows)
        counts["pieces"] += len(cut)
        minutes += rec.audio.duration / 60
    row = {"split": "dev", "question": "cost", "audio_minutes":
           round(minutes, 2),
           **{f"{k}_seconds_per_minute": round(v / minutes, 3)
              for k, v in spent.items()},
           "fine_windows": counts["fine"], "pieces": counts["pieces"],
           "device": torch.cuda.get_device_name(0)}
    print(row)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


# --------------------------------------------------------------- real


def real(args: argparse.Namespace) -> None:
    """Aggregate changes on private recordings (no ground truth).

    Copies the first three finished sessions and the voiceprints to a
    scratch directory (deleted on exit), recognises and embeds each
    recording as the pipeline does, groups the voices with the session's
    options (the expected-speakers rule clusters into the listed number
    of people) and prints, per recording ``R1``... and per method, the
    number of turns and sentences, the share of sentences doubtful and
    overlap, the sentences holding more than one speaker and the
    lengths of the pieces. Nothing else is printed or written.
    """
    from real_sessions import scratch_copy

    from transcriber.adapters.outbound.av_audio_decoder import (
        AvAudioDecoder,
    )
    from transcriber.adapters.outbound.fs_session_repository import (
        FileSystemSessionRepository,
    )
    from transcriber.adapters.outbound.whisper_recogniser import (
        WhisperRecogniser,
    )

    p = TranscriptionParameters()
    chosen = [r for r in tr05.read_jsonl(args.output)
              if r.get("question") == "chosen"][-1]
    settings = PieceSettings(**chosen["pieces"])
    hybrid = chosen["hybrid"]
    lines: List[Row] = []
    with scratch_copy(args.sessions, args.work, 3) as ids:
        repo = FileSystemSessionRepository(args.work)
        decoder = AvAudioDecoder()
        recogniser = WhisperRecogniser(batch_size=2, device="cuda")
        for number, sid in enumerate(ids, start=1):
            session = repo.get(sid)
            assert session is not None
            audio = decoder.decode(repo.recording(sid))
            speech = recogniser.recognise(
                audio, recognition_language(session.language))
            work_dir = args.work / "vectors"
            work_dir.mkdir(parents=True, exist_ok=True)
            count = session.num_speakers
            rec = make_rec(
                f"R{number}", "real", {"readers": []}, audio, [], [],
                speech.words, work_dir / f"R{number}.npz", p)
            groups = group_voices(
                rec.coarse, rec.fine, count, p.group_threshold,
                p.max_speakers, p.centre_windows, p.single_voice_threshold,
                p.single_voice_quantile)
            mean: FloatArray = rec.coarse.mean(axis=0).astype(np.float32)
            lp = LongPass(groups.fine, groups.fine_scores, mean,
                          voice_centroids(centre(rec.coarse, mean),
                                          groups.coarse), {})
            methods: Dict[str, Method] = {
                "grid": lambda r, lp: grid(r, lp, p),
                "pieces": lambda r, lp: pieces(r, lp, p, settings),
                "hybrid": hybrid_method(hybrid, p),
            }
            for name, method in methods.items():
                outcome = method(rec, lp)
                lines.append({"recording": f"R{number}", "method": name,
                              **real_summary(outcome, p)})
        recogniser.release()
    for line in lines:
        print(json.dumps(line))


def real_summary(outcome: Outcome, p: TranscriptionParameters) -> Row:
    """Aggregate counts of one method on one real recording."""
    turns = turns_from_words(outcome.words, p.turn_gap)
    utterances = build_utterances(turns)
    assessed = assess_utterances(utterances, outcome.evidence,
                                 p.doubt_margin, outcome.moved)
    lengths = np.array(outcome.pieces) if outcome.pieces else None
    row: Row = {
        "turns": len(turns), "sentences": len(assessed),
        "doubtful": sum(bool(u.doubtful) for u in assessed),
        "overlap": sum(bool(u.overlap) for u in assessed),
        "split_sentences": outcome.split_sentences,
        "moved_stretches": len(outcome.moved),
    }
    if lengths is not None:
        row["pieces"] = int(lengths.size)
        for q in (0.1, 0.5, 0.9):
            row[f"piece_p{int(q * 100)}"] = round(float(np.quantile(
                lengths, q)), 2)
        for edge in (0.5, 1.0, 1.5, 3.0):
            row[f"pieces_under_{edge}"] = int((lengths < edge).sum())
    return row


def main() -> None:
    """Parse the command line and run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=("prepare", "analyse", "cost", "real"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/tr05"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--asr-batch", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sessions", type=Path,
                        default=Path("data/transcriber"))
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "analyse":
        analyse(args)
    elif args.command == "cost":
        cost(args)
    else:
        if args.work is None:
            sys.exit("--work is required (a new scratch directory)")
        real(args)


if __name__ == "__main__":
    main()
