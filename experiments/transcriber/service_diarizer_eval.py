"""Measure the transcription with a lab diarizer (TR-18, issue #68).

Two measurements of the service's own path (:class:`~transcriber.
application.transcribe.TranscribeSession` with the
:class:`~transcriber.adapters.outbound.lab_runner_diarizer.
LabRunnerDiarizer` the service builds from its settings), written as
rows to ``experiments/transcriber/service_diarizer.jsonl``:

``corpus``
    The corpus sessions the lab imported (AMI meetings, CALLHOME calls;
    ``docs/transcriber/LAB.md`` §13). Each session's original audio is
    transcribed as the lab's built-in step does (a scratch session, no
    one enrolled, the real Whisper and WavLM models) but with the
    diarizer the settings' rule picks; the turns of the attributed
    words are scored against the corpus reference with the lab's
    scoring (:func:`~transcriber.domain.lab_reference.score_track`),
    and so are the diarizer's own segments (``--builtin`` runs the
    built-in grouping instead, for comparison). ``--calls-speakers`` is the
    number of speakers a CALLHOME session is started with (2: a phone
    call, which picks Nemotron-3); AMI meetings are started without one.
    A thread samples ``nvidia-smi`` every 0.5 s for the card's peak.

``identification``
    The synthetic meetings of ``IDENTIFICATION.md`` (cache of
    ``identification_eval.py prepare``). For each meeting the rule's
    diarizer runs on the meeting's audio (count given: 4, and
    estimated), the labels are named exactly as the transcription does
    (:func:`transcriber.domain.diarization.segment_voices`, then
    ``match_voices`` at the match threshold, unit vectors) and the
    words on the right id are counted with that study's ``score`` under
    its four enrolment conditions; the current default (voiceprints as
    seeds of the built-in grouping, unit, 0.95) is recomputed from the
    same cache for comparison.

``summary`` writes ``experiments/transcriber/results/
diarizer_switch.json`` (read by the study page) from the rows.

Usage::

    PYTHONPATH=. python experiments/transcriber/service_diarizer_eval.py \\
        corpus --data-dir <dir holding lab/<session>/...>
    PYTHONPATH=. python experiments/transcriber/service_diarizer_eval.py \\
        identification --data-dir data/tr05

The script may import :mod:`transcriber`; it never imports ``src``.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from transcriber.adapters.outbound.fs_lab_repository import (
    FileSystemLabRepository,
)
from transcriber.application.lab_builtin import (
    SCRATCH_ID,
    _GivenAudio,
    _NoLocks,
    _NoVoiceprints,
    _ScratchSessions,
    turn_segments,
)
from transcriber.application.ports import Audio, Diarizer
from transcriber.application.transcribe import (
    TranscribeSession,
    TranscriptionOptions,
)
from transcriber.bootstrap import build_diarizer, build_models
from transcriber.config import load_settings
from transcriber.domain.comparison import Track
from transcriber.domain.diarization import (
    BUILTIN_RULE,
    Diarization,
    DiarizerRule,
    label_order,
    label_windows,
    segment_spans,
    segment_voices,
)
from transcriber.domain.lab_reference import score_track
from transcriber.domain.models import Session, SessionState, Word
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.speakers import assign_speakers, smooth_speakers
from transcriber.domain.voices import FloatArray, unit_vectors

OUTPUT = Path("experiments/transcriber/service_diarizer.jsonl")
Row = Dict[str, Any]


class GpuPeak:
    """Sample the card's used memory (``nvidia-smi``) in a thread."""

    def __init__(self, every: float = 0.5) -> None:
        """Start sampling every ``every`` seconds."""
        self.peak = 0
        self.first: Optional[int] = None
        self._stop = threading.Event()
        self._every = every
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _used(self) -> Optional[int]:
        """Return the MiB used on GPU 0, or ``None`` without a GPU."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            return int(out.strip().splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError,
                IndexError):
            return None

    def _loop(self) -> None:
        """Keep the largest sample until stopped."""
        while not self._stop.is_set():
            used = self._used()
            if used is not None:
                if self.first is None:
                    self.first = used
                self.peak = max(self.peak, used)
            self._stop.wait(self._every)

    def stop(self) -> int:
        """Stop sampling; return the peak in MiB."""
        self._stop.set()
        self._thread.join()
        return self.peak


class Recording:
    """A diarizer that keeps what it returned (to score it too)."""

    def __init__(self, inner: Diarizer) -> None:
        """Wrap ``inner``."""
        self.inner = inner
        self.last: Optional[Diarization] = None
        self.seconds = 0.0

    def problem(self, model: str) -> Optional[str]:
        """Ask the wrapped diarizer."""
        return self.inner.problem(model)

    def diarize(
        self, model: str, audio: Audio, num_speakers: Optional[int],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Diarization:
        """Diarize and keep the result and the time it took."""
        began = time.perf_counter()
        self.last = self.inner.diarize(model, audio, num_speakers, cancelled)
        self.seconds = time.perf_counter() - began
        return self.last


def torch_peak(reset: bool = False) -> Optional[int]:
    """Return this process's peak CUDA allocation in MiB (or reset it)."""
    import torch

    if not torch.cuda.is_available():
        return None
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return None
    return int(torch.cuda.max_memory_allocated() / 2**20)


def corpus(args: argparse.Namespace) -> None:
    """Transcribe each corpus session with the rule's diarizer; score."""
    settings = load_settings(os.environ).model_copy(update={
        "data_dir": args.data_dir,
    })
    if args.builtin:
        settings = settings.model_copy(update={"diarizer": BUILTIN_RULE})
    recogniser, encoder = build_models(settings)
    built, rule = build_diarizer(settings, os.environ, recogniser.release)
    diarizer: Optional[Recording] = None
    if built is not None:
        diarizer = Recording(built)
    elif not args.builtin:
        raise SystemExit("The settings give no lab diarizer.")
    repo = FileSystemLabRepository(args.data_dir)
    names = sorted(
        p.name for p in (args.data_dir / "lab").iterdir()
        if p.name.startswith(("ami-", "callhome-"))
    )
    if args.sessions:
        names = [n for n in names if n in set(args.sessions.split(","))]
    p = settings.transcription
    rows: List[Row] = []
    for name in names:
        reference = repo.reference(name)
        if reference is None or reference.excluded() is not None:
            print(f"{name}: skipped ({reference and reference.excluded()})")
            continue
        key = repo.original_key(name)
        assert key is not None
        audio = repo.load_audio(name, key)
        count = args.calls_speakers if name.startswith("callhome-") \
            else None
        scratch = _ScratchSessions(Session(
            id=SCRATCH_ID, language=reference.language,
            state=SessionState.QUEUED,
        ))
        transcribe = TranscribeSession(
            scratch, _NoVoiceprints(), _GivenAudio(audio), recogniser,
            encoder, _NoLocks(), p, None, diarizer, rule,
        )
        if diarizer is not None:
            diarizer.last = None
        torch_peak(reset=True)
        gpu = GpuPeak()
        began = time.perf_counter()
        transcribe(SCRATCH_ID, TranscriptionOptions(num_speakers=count))
        seconds = time.perf_counter() - began
        in_process = torch_peak()
        recogniser.release()
        encoder.release()
        peak = gpu.stop()
        transcript = scratch.transcript(SCRATCH_ID)
        assert transcript is not None
        words = scratch.words(SCRATCH_ID)
        service = score_track(reference, Track(
            "service", turn_segments(words, p.turn_gap), "service"))
        row: Row = {
            "question": "corpus", "session": name,
            "corpus": reference.corpus, "language": reference.language,
            "duration": round(audio.duration, 1),
            "num_speakers": count,
            "diarizer": transcript.pipeline.diarizer,
            "diarizer_licence": transcript.pipeline.diarizer_licence,
            "rule": rule.model_dump(mode="json"),
            "der": service.der, "missed": service.missed,
            "false_alarm": service.false_alarm,
            "confusion": service.confusion,
            "reference_speech": service.reference_speech,
            "reference_speakers": service.reference_speakers,
            "speakers": service.hypothesis_speakers,
            "speaker_count_error": service.speaker_count_error,
            "overlap_precision": service.overlap_precision,
            "overlap_recall": service.overlap_recall,
            "doubtful_share": round(
                sum(1 for u in transcript.utterances if u.doubtful)
                / max(len(transcript.utterances), 1), 4),
            "overlap_share": round(
                sum(1 for u in transcript.utterances if u.overlap)
                / max(len(transcript.utterances), 1), 4),
            "seconds": round(seconds, 1),
            "gpu_used_mib_at_start": gpu.first,
            "gpu_used_mib_peak": peak,
            "service_torch_peak_mib": in_process,
        }
        if diarizer is not None and diarizer.last is not None:
            raw = score_track(reference, Track(
                "diarizer", diarizer.last.segments, "diarizer"))
            row.update({
                "diarizer_der": raw.der,
                "diarizer_speakers": raw.hypothesis_speakers,
                "diarizer_peak_gpu_mb": diarizer.last.peak_gpu_mb,
                "diarizer_seconds": round(diarizer.seconds, 1),
            })
        rows.append(row)
        print(json.dumps(row), flush=True)
    append(rows)


# ------------------------------------------------------- identification


def study() -> Any:
    """Import ``identification_eval.py`` (next to this script)."""
    path = Path(__file__).resolve().parent / "identification_eval.py"
    spec = importlib.util.spec_from_file_location("identification_eval",
                                                  path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["identification_eval"] = module
    spec.loader.exec_module(module)
    return module


def label_words(
    words: Sequence[Word], found: Diarization, audio: Audio,
    encoder: Any, fine_windows: Sequence[Any], fine: FloatArray,
    p: TranscriptionParameters,
) -> "tuple[List[Word], FloatArray]":
    """Label ``words`` from ``found`` as the transcription does.

    The same steps as ``TranscribeSession._segment_voices`` (label
    windows embedded, voices per label, words on the segment they
    overlap most), then the smoothing; ``fine`` are the cached short
    windows' vectors, used only for the certainty and not needed here.
    """
    order = label_order(found.segments)
    windows, owners = label_windows(
        found.segments, order, audio.duration, *p.fine()
    )
    vectors = encoder.embed(audio, windows) if windows else np.zeros(
        (0, fine.shape[1]), dtype=np.float32)
    voices = segment_voices(vectors, owners, len(order), fine,
                            p.centre_windows)
    labelled = assign_speakers(list(words), segment_spans(
        found.segments, order))
    if p.smoothing_run:
        labelled = smooth_speakers(labelled, p.smoothing_run)
    return labelled, voices.vectors


def identification(args: argparse.Namespace) -> None:
    """Words on the right id with the rule's diarizer vs the default."""
    tr05 = study()
    settings = load_settings(os.environ)
    rule: DiarizerRule = settings.diarizer
    _, encoder = build_models(settings)
    built, _ = build_diarizer(settings, os.environ, encoder.release)
    if built is None:
        raise SystemExit("The settings give no lab diarizer.")
    p = TranscriptionParameters()
    single = tr05.single_rule("p10_to_mean", 0.95)
    rules = tr05.SampleRules()
    rows: List[Row] = []
    for split in args.splits.split(","):
        _, recordings = tr05.load_split(args.data_dir, split)
        prints = tr05.enrolled_prints(recordings, rules, p)
        meetings = [r for r in recordings if r.plan["kind"] == "meeting"]
        totals: Dict[str, List[Row]] = {}
        peak = 0.0
        diarizer_seconds = 0.0
        for rec in meetings:
            truth = tr05.word_truth(rec.words, rec.built.spans)
            present: Set[str] = set(rec.plan["readers"])
            for count_name, count in (("known", len(present)),
                                      ("estimated", None)):
                model = rule.model_for(count)
                began = time.perf_counter()
                found = built.diarize(model, rec.built.audio, count)
                diarizer_seconds += time.perf_counter() - began
                peak = max(peak, found.peak_gpu_mb or 0.0)
                labelled, vectors = label_words(
                    rec.words, found, rec.built.audio, encoder,
                    rec.fine_windows, rec.fine, p)
                encoder.release()
                for condition in tr05.CONDITIONS:
                    enrolled = tr05.enrolled_set(rec.plan, condition)
                    candidates = {k: prints[k] for k in sorted(enrolled)}
                    taken = set(prints)
                    names = tr05.name_words(
                        tr05.Labelled(labelled, vectors), candidates,
                        p.match_threshold, unit_vectors, taken)
                    totals.setdefault(
                        f"{model}|{count_name}|{condition}", []
                    ).append(tr05.score(names, truth, enrolled, present))
                    names = tr05.guided_words(
                        rec, candidates, p.match_threshold, unit_vectors,
                        count, p, single, taken)
                    totals.setdefault(
                        f"seeded-builtin|{count_name}|{condition}", []
                    ).append(tr05.score(names, truth, enrolled, present))
            print(f"{split} {rec.plan['id']} done", flush=True)
        for key, scores in sorted(totals.items()):
            method, count_name, condition = key.split("|")
            summed = {k: sum(s[k] for s in scores) for k in scores[0]}
            rows.append({
                "question": "identification", "split": split,
                "method": method, "speakers": count_name,
                "condition": condition, "threshold": p.match_threshold,
                "meetings": len(scores), "words": summed["words"],
                "right": summed["right"],
                "right_share": round(summed["right"] / summed["words"], 4),
                "right_share_worst_meeting": round(min(
                    s["right"] / s["words"] for s in scores), 4),
                "impostor_words": summed["impostor_words"],
                "wrong_id_words": summed["wrong_id_words"],
                "missed_words": summed["missed_words"],
                "labels": summed["labels"],
                "diarizer_peak_gpu_mb": peak or None,
                "diarizer_seconds_split": round(diarizer_seconds, 1),
            })
    for row in rows:
        print(json.dumps(row))
    append(rows)


SUMMARY = Path("experiments/transcriber/results/diarizer_switch.json")
"""The summary the study page reads (``study_data.py``)."""


def _mean(values: Sequence[float]) -> float:
    """Return the mean of ``values`` rounded to 4 digits."""
    return round(sum(values) / len(values), 4)


def summary(args: argparse.Namespace) -> None:
    """Write :data:`SUMMARY` from the rows in :data:`OUTPUT`.

    Per corpus: the mean DER over its sessions of the transcription's
    turns with the rule's diarizer (``der``, ``n`` sessions), the same
    with the built-in grouping, and the diarizer's own segments. Per
    split: the share of words on the right id, mean of the four
    enrolment conditions, for the rule's diarizer and for the default
    before TR-18 (voiceprints as seeds of the built-in grouping).
    """
    rows = [json.loads(line) for line in OUTPUT.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    corpora: Dict[str, Any] = {}
    for name in sorted({r["corpus"] for r in rows
                        if r["question"] == "corpus"}):
        latest: Dict[Any, Row] = {}
        for r in rows:
            if r["question"] == "corpus" and r["corpus"] == name:
                latest[(r["session"], r["diarizer"] == "builtin")] = r
        new = [r for (_, old), r in latest.items() if not old]
        old = [r for (_, before), r in latest.items() if before]
        corpora[name] = {
            "der": _mean([r["der"] for r in new]), "n": len(new),
            "diarizers": sorted({r["diarizer"] for r in new}),
            "builtin_der": _mean([r["der"] for r in old]),
            "builtin_n": len(old),
            "diarizer_only_der": _mean([r["diarizer_der"] for r in new]),
            "speaker_count_error": _mean(
                [r["speaker_count_error"] for r in new]),
            "builtin_speaker_count_error": _mean(
                [r["speaker_count_error"] for r in old]),
            "doubtful_share": _mean([r["doubtful_share"] for r in new]),
            "builtin_doubtful_share": _mean(
                [r["doubtful_share"] for r in old]),
        }
    identification: Dict[str, Any] = {}
    for r in rows:
        if r["question"] != "identification":
            continue
        method = "before" if r["method"] == "seeded-builtin" else "diarizer"
        key = f"{r['split']}_{r['speakers']}_{method}"
        identification.setdefault(key, []).append(r["right_share"])
    data = {
        "issue": 68,
        "measure": "mean DER over the sessions of the transcription's"
                   " turns of attributed words against the corpus"
                   " reference (experiments/transcriber/"
                   "service_diarizer.jsonl)",
        "corpora": corpora,
        "identification_right_share": {
            k: _mean(v) for k, v in sorted(identification.items())
        },
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=1))


def append(rows: Sequence[Row]) -> None:
    """Append ``rows`` to :data:`OUTPUT`."""
    with OUTPUT.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    """Parse the command line and run one measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=("corpus", "identification", "summary"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--sessions", default="")
    parser.add_argument("--calls-speakers", type=int, default=2)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--builtin", action="store_true",
                        help="corpus: the built-in grouping, for comparison")
    args = parser.parse_args()
    if args.command == "summary":
        summary(args)
    elif args.data_dir is None:
        parser.error("--data-dir is required")
    elif args.command == "corpus":
        corpus(args)
    else:
        identification(args)


if __name__ == "__main__":
    main()
