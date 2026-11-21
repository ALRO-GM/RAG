"""Measure the mono live diarization (LV-6): frames DER and final speakers.

Two commands:

``frames WAV OUT.json [--device cuda]``
    Streams ``WAV`` (mono, 16 kHz) through the Sortformer worker with
    the service's adapter, 0.5 s per message as fast as it answers,
    and keeps every frame; prints the frame count and the p50/p90/max
    time per message.
``der FRAMES.json REFERENCE.json``
    Diarization error rate of the frames (a frame is a speaker's at
    0.5) against a reference: a lab ``reference.json`` (``segments``)
    or a Piper ``*.reference.json`` (``utterances``), with the lab's
    comparison (collar 0.25 s, best one-to-one mapping).
``events EVENTS.jsonl REFERENCE.json``
    Speaker of each final of a ``live-file --diarize`` run against the
    reference speaker that overlaps it most: at its first emission and
    after its last revision. Enrolled ids are compared as they are;
    anonymous labels (``S1``...) are mapped one-to-one to reference
    speakers by the largest total overlap. Also the share of finals
    sent provisional, the ``speaker_update`` count and the timings.

Usage::

    PYTHONPATH=. python experiments/live/mono_diarization.py \\
        frames a.wav f.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from transcriber.domain.comparison import Track, compare_tracks
from transcriber.domain.lab import Segment
from transcriber.domain.live import percentile

ROOT = Path(__file__).resolve().parents[2]
NEMO = Path("~/.local/share/ragputin/lab-envs/nemo/bin/python").expanduser()


def frames(wav: Path, out: Path, device: str) -> None:
    """Stream ``wav`` through the worker; save its frames."""
    import soundfile

    from transcriber.adapters.outbound.sortformer_stream_diarizer import (
        SortformerStreamDiarizer,
        worker_command,
    )

    samples, rate = soundfile.read(str(wav), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        raise SystemExit("the audio must be 16 kHz")
    diarizer = SortformerStreamDiarizer(
        worker_command(NEMO, ROOT / "lab/runners/sortformer_stream.py",
                       device),
        os.environ,
    )
    diarizer.start()
    rows: List[List[float]] = []
    times: List[float] = []
    piece = 8000
    try:
        for start in range(0, len(samples), piece):
            began = time.monotonic()
            diarizer.feed(samples[start:start + piece])
            # Chunks of 0.48 s need 0.56 s of right context (and the
            # STFT edge): frames reach at least 1.2 s behind the audio.
            expected = (start + piece) / 16000 - 1.2
            first, got = diarizer.frames(max(expected, 0.0), 120.0)
            rows.extend(got)
            if start:
                times.append((time.monotonic() - began) * 1000)
        diarizer.finish()
        while len(rows) * 0.08 < len(samples) / 16000 - 0.08:
            first, got = diarizer.frames(len(samples) / 16000, 30.0)
            if not got:
                break
            rows.extend(got)
    finally:
        diarizer.close()
    out.write_text(json.dumps({"ready": diarizer.ready, "probs": rows}))
    print(json.dumps({"frames": len(rows), "message_ms": stats(times)}))


def stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Return n, p50, p90 and max of ``values`` (one decimal)."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 1)

    return {
        "n": len(values), "p50": r(percentile(values, 0.5)),
        "p90": r(percentile(values, 0.9)),
        "max": r(max(values) if values else None),
    }


def reference(path: Path) -> Tuple[List[Segment], float]:
    """Return the reference segments and the duration."""
    data = json.loads(path.read_text("utf-8"))
    if "utterances" in data:
        return [
            Segment(start=u["start"], end=u["end"], speaker=u["speaker"])
            for u in data["utterances"]
        ], float(data["seconds"])
    return [Segment(**s) for s in data["segments"]], float(data["duration"])


def hypothesis(
    probs: Sequence[Sequence[float]], step: float = 0.08,
    threshold: float = 0.5,
) -> List[Segment]:
    """Return the segments of each slot (frames at ``threshold`` or more)."""
    out: List[Segment] = []
    if not probs:
        return out
    active = np.asarray(probs) >= threshold
    padded = np.vstack([active, np.zeros((1, active.shape[1]), bool)])
    for slot in range(active.shape[1]):
        run: Optional[int] = None
        for n, on in enumerate(padded[:, slot]):
            if on and run is None:
                run = n
            elif not on and run is not None:
                out.append(Segment(start=run * step, end=n * step,
                                   speaker=f"spk{slot}"))
                run = None
    return out


def der(frames_path: Path, reference_path: Path) -> Dict[str, Any]:
    """Return the DER and its parts (seconds)."""
    probs = json.loads(frames_path.read_text("utf-8"))["probs"]
    truth, duration = reference(reference_path)
    compared = compare_tracks(
        Track("ref", truth, "ref"), Track("hyp", hypothesis(probs), "hyp"),
        duration,
    )
    return {
        "der": round(compared.error_rate or 0.0, 4),
        "missed": round(compared.missed, 1),
        "false_alarm": round(compared.false_alarm, 1),
        "confusion": round(compared.confusion, 1),
        "speech": round(compared.reference_speech, 1),
    }


def _truth(
    segments: Sequence[Segment], start: float, end: float
) -> Optional[str]:
    """Return the reference speaker overlapping ``start``-``end`` most."""
    best, who = 0.0, None
    for s in segments:
        overlap = min(end, s.end) - max(start, s.start)
        if overlap > best:
            best, who = overlap, s.speaker
    return who


def _mapping(
    speakers: Mapping[str, str], truth: Mapping[str, Optional[str]],
    span: Mapping[str, float],
) -> Dict[str, str]:
    """Map anonymous labels one-to-one to reference speakers."""
    known = sorted({t for t in truth.values() if t})
    labels = sorted(set(speakers.values()))
    out = {label: label for label in labels if label in known}
    free = [label for label in labels if label not in known]
    if not free or not known:
        return out
    gain = np.zeros((len(free), len(known)))
    for ident, label in speakers.items():
        who = truth[ident]
        if label in free and who:
            gain[free.index(label), known.index(who)] += span[ident]
    rows, cols = linear_sum_assignment(gain, maximize=True)
    for r, c in zip(rows.tolist(), cols.tolist()):
        out[free[r]] = known[c]
    return out


def events(events_path: Path, reference_path: Path) -> Dict[str, Any]:
    """Return the speaker accuracy, revisions and timings of a run."""
    lines = events_path.read_text("utf-8").splitlines()
    stream = [json.loads(line) for line in lines if line.strip()]
    first: Dict[str, Dict[str, Any]] = {}
    last: Dict[str, Dict[str, Any]] = {}
    timings: List[Dict[str, float]] = []
    updates = 0
    for event in stream:
        if event["type"] not in ("final", "speaker_update"):
            continue
        u = event["utterance"]
        if event["type"] == "final":
            first.setdefault(u["id"], u)
            timings.append(event["timings"])
        else:
            updates += 1
        last[u["id"]] = u
    segments, _ = reference(reference_path)
    ids = [i for i in last if last[i]["text"].strip()]
    truth = {i: _truth(segments, last[i]["start"], last[i]["end"])
             for i in ids}
    span = {i: last[i]["end"] - last[i]["start"] for i in ids}
    scored = [i for i in ids if truth[i]]

    def accuracy(chosen: Mapping[str, Dict[str, Any]]) -> float:
        speakers = {i: chosen[i]["speaker"] for i in ids}
        mapped = _mapping(speakers, truth, span)
        right = sum(mapped.get(speakers[i]) == truth[i] for i in scored)
        return round(right / len(scored), 4) if scored else 0.0

    # One timing per segment: every sentence of a segment shares it.
    per_segment = {
        (t.get("lag_ms"), t.get("asr_ms"), t.get("endpoint_ms")): t
        for t in timings
    }
    return {
        "finals": len(ids),
        "speaker_updates": updates,
        "accuracy_first_emission": accuracy(first),
        "accuracy_last_revision": accuracy(last),
        "provisional_at_emission": round(sum(
            bool(first[i].get("speaker_provisional")) for i in ids
        ) / max(len(ids), 1), 4),
        "speakers": sorted({last[i]["speaker"] for i in ids}),
        "lag_ms": stats([t["lag_ms"] for t in per_segment.values()]),
        "attribution_ms": stats([
            t["attribution_ms"] for t in per_segment.values()
            if "attribution_ms" in t
        ]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run one of the commands (see the module)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    f = commands.add_parser("frames")
    f.add_argument("wav", type=Path)
    f.add_argument("out", type=Path)
    f.add_argument("--device", default="cuda")
    for name in ("der", "events"):
        c = commands.add_parser(name)
        c.add_argument("first", type=Path)
        c.add_argument("reference", type=Path)
    args = parser.parse_args(argv)
    if args.command == "frames":
        frames(args.wav, args.out, args.device)
    elif args.command == "der":
        print(json.dumps(der(args.first, args.reference)))
    else:
        print(json.dumps(events(args.first, args.reference), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
