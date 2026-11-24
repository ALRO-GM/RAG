"""Measure the employee's voice check on real two-speaker speech (#249).

A one-microphone live session starts with a short voice check: the
employee says a sentence or two, and every later sentence is the
employee's or the other person's. This script measures, on the lab's
recordings with references (CALLHOME Spanish calls, AMI single distant
microphone meetings), how well each way of attributing the later
sentences does, per threshold, from the first seconds of one speaker.

Commands (vectors and frames are kept in ``--work``, outside the
repository; only numbers go to the results file):

``embed SESSION... --lab DIR --work DIR``
    Cuts every reference segment and embeds it with the live path's
    voice encoder (WavLM base-plus-sv, on the CPU): one vector of the
    whole segment (up to ``MAX_CLIP`` seconds, one forward pass: what
    the live path does per sentence) and the 1.5 s window vectors of
    the batch path (what the naming of slots and the reference use).
    Records the time of each whole-segment embedding.
``sanity SESSION... --lab DIR --work DIR``
    Checks that the audio follows its reference: the mean RMS (10 ms
    frames) inside the reference's speech over the mean RMS outside
    it. A session below ``MIN_RATIO`` is scored apart as "audio does
    not follow its reference" (three CALLHOME calls have a nearly
    constant level from start to end).
``frames SESSION... --lab DIR --work DIR [--device cpu]``
    Streams the session through the Sortformer worker with the
    service's adapter (``mono_diarization.frames``), for the approaches
    that use the slots.
``score --work DIR [--results FILE]``
    Simulates, in time order, every approach and threshold (see
    :func:`score`) and writes the table.

Material and protocol: per session, each of the two speakers who talk
most is the "employee" in turn. Their voice check is their earliest
clean segments (no overlap, at least 1 s) until ``check`` seconds; the
test sentences are the reference segments that start after the check,
of everybody (CALLHOME: the other person and anyone else on the line;
AMI: only the other of the two, the meeting's other people's segments
are dropped: one microphone, two people in a room). A segment is
"overlapped" when another speaker talks during 20 % or more of it.

Usage::

    PYTHONPATH=. python experiments/live/voice_check.py embed \\
        callhome-data-0002 --lab data/transcriber/lab --work /tmp/vc
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from transcriber.domain.live import percentile
from transcriber.domain.speakers import analysis_windows

FloatArray = npt.NDArray[np.float32]

RATE = 16000
MAX_CLIP = 10.0
"""Longest stretch of a sentence embedded in one forward pass."""

MIN_CLIP = 0.5
"""Shortest sentence that gets a vector at all."""

OVERLAP = 0.2
"""Share of a segment another speaker covers to call it overlapped."""

MIN_RATIO = 1.3
"""Speech over non-speech level below which a session is set apart."""

CHECK_OVERLAP = 0.1
"""Most of a check segment another speaker may cover."""

CHECKS = (3.0, 5.0, 8.0)
"""Voice check lengths compared (seconds of the employee's speech)."""

THRESHOLDS = tuple(round(0.50 + 0.01 * n, 2) for n in range(50))
CENTRED = tuple(round(-0.50 + 0.025 * n, 3) for n in range(41))
PRODUCTION_MATCH = 0.86
"""The ``match`` of ``VoiceCheckSettings``: the variants are scored at it."""

PRODUCTION_SPLIT: Optional[float] = None
"""The ``split`` of ``VoiceCheckSettings`` (off: see the split sweep)."""

WARM = 4
"""Sentences heard after the check before the session mean is used."""
MARGINS = tuple(round(-0.10 + 0.01 * n, 2) for n in range(21))
BUCKETS = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 1e9))
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- data


@dataclass
class Session:
    """One recording's segments, vectors and (maybe) Sortformer frames."""

    name: str
    corpus: str
    starts: FloatArray
    ends: FloatArray
    speakers: List[str]
    overlap: FloatArray
    whole: FloatArray
    windows: FloatArray
    window_segment: npt.NDArray[np.int64]
    probs: Optional[FloatArray] = None
    embed_ms: List[Tuple[float, float]] = field(default_factory=list)
    energy_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        """Sum the window vectors of each segment once."""
        count = len(self.speakers)
        self.window_sum: FloatArray = np.zeros(
            (count, self.windows.shape[1]), dtype=np.float32)
        np.add.at(self.window_sum, self.window_segment, self.windows)
        self.window_count = np.bincount(
            self.window_segment, minlength=count)


def _overlaps(segments: Sequence[Dict[str, Any]]) -> List[float]:
    """Return the share of each segment another speaker talks over."""
    out: List[float] = []
    for s in segments:
        covered: List[Tuple[float, float]] = []
        for o in segments:
            if o is s or o["speaker"] == s["speaker"]:
                continue
            a, b = max(s["start"], o["start"]), min(s["end"], o["end"])
            if b > a:
                covered.append((a, b))
        covered.sort()
        total, reach = 0.0, s["start"]
        for a, b in covered:
            a = max(a, reach)
            if b > a:
                total += b - a
                reach = b
        span = max(s["end"] - s["start"], 1e-6)
        out.append(total / span)
    return out


def _audio(lab: Path, name: str) -> FloatArray:
    """Return the session's 16 kHz mono audio (the lab's working copy)."""
    import soundfile

    key = json.loads((lab / name / "original.json").read_text())["key"]
    samples, rate = soundfile.read(
        str(lab / name / "audio" / f"{key}.wav"), dtype="float32"
    )
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{name}: audio is {rate} Hz, not {RATE}")
    out: FloatArray = samples.astype(np.float32)
    return out


def embed(names: Sequence[str], lab: Path, work: Path) -> None:
    """Embed every reference segment of each session (see the module)."""
    import torch

    from transcriber.adapters.outbound.wavlm_voice_encoder import (
        WavLMVoiceEncoder,
    )
    from transcriber.application.ports import Audio
    from transcriber.config import DEFAULT_VOICE_MODEL

    encoder = WavLMVoiceEncoder(DEFAULT_VOICE_MODEL, device="cpu")
    work.mkdir(parents=True, exist_ok=True)
    for name in names:
        ref = json.loads((lab / name / "reference.json").read_text("utf-8"))
        segments = sorted(ref["segments"], key=lambda s: s["start"])
        audio = Audio(samples=_audio(lab, name), sample_rate=RATE)
        whole: List[FloatArray] = []
        rows: List[FloatArray] = []
        owner: List[int] = []
        timing: List[Tuple[float, float]] = []
        for n, s in enumerate(segments):
            span = s["end"] - s["start"]
            if span >= MIN_CLIP:
                end = s["start"] + min(span, MAX_CLIP)
                began = time.perf_counter()
                try:
                    vector = encoder.embed(audio, [(s["start"], end)])[0]
                except RuntimeError:  # too short for the encoder's kernels
                    vector = np.full(512, np.nan, dtype=np.float32)
                timing.append((span, (time.perf_counter() - began) * 1000))
                whole.append(vector)
            else:
                whole.append(np.full(512, np.nan, dtype=np.float32))
            spans = analysis_windows([(s["start"], s["end"])])
            if spans:
                vectors = encoder.embed(audio, spans)
                rows.extend(vectors)
                owner.extend([n] * len(vectors))
        np.savez_compressed(
            work / f"{name}.npz",
            starts=np.array([s["start"] for s in segments], np.float32),
            ends=np.array([s["end"] for s in segments], np.float32),
            speakers=np.array([s["speaker"] for s in segments]),
            overlap=np.array(_overlaps(segments), np.float32),
            whole=np.vstack(whole).astype(np.float32),
            windows=np.vstack(rows).astype(np.float32),
            window_segment=np.array(owner, np.int64),
            embed_ms=np.array(timing, np.float32).reshape(-1, 2),
            corpus=np.array(ref.get("corpus", "")),
            threads=np.array(torch.get_num_threads()),
        )
        print(json.dumps({"session": name, "segments": len(segments),
                          "windows": len(rows)}), flush=True)


def sanity(names: Sequence[str], lab: Path, work: Path) -> None:
    """Write each session's speech over non-speech level (see the module)."""
    work.mkdir(parents=True, exist_ok=True)
    for name in names:
        ref = json.loads((lab / name / "reference.json").read_text("utf-8"))
        samples = _audio(lab, name)
        level = np.sqrt(np.convolve(
            samples ** 2, np.ones(1600) / 1600, "same"))[::160]
        inside = np.zeros(len(level), bool)
        for seg in ref["segments"]:
            inside[int(seg["start"] * 100):int(seg["end"] * 100)] = True
        ratio = float(level[inside].mean()) / max(
            float(level[~inside].mean()), 1e-9)
        (work / f"{name}.sanity.json").write_text(json.dumps(
            {"energy_ratio": round(ratio, 3)}))
        print(json.dumps({"session": name, "energy_ratio": round(ratio, 3)}))


def frames(names: Sequence[str], lab: Path, work: Path, device: str) -> None:
    """Stream each session through the Sortformer worker; keep frames."""
    sys.path.insert(0, str(Path(__file__).parent))
    import soundfile

    from mono_diarization import frames as stream

    work.mkdir(parents=True, exist_ok=True)
    for name in names:
        key = json.loads((lab / name / "original.json").read_text())["key"]
        wav = lab / name / "audio" / f"{key}.wav"
        info = soundfile.info(str(wav))
        if info.samplerate != RATE or info.channels != 1:
            raise SystemExit(f"{name}: not 16 kHz mono")
        stream(wav, work / f"{name}.frames.json", device)


def load(work: Path, name: str) -> Session:
    """Read one session's vectors (and frames when there are)."""
    data = np.load(work / f"{name}.npz")
    probs: Optional[FloatArray] = None
    path = work / f"{name}.frames.json"
    if path.exists():
        probs = np.asarray(
            json.loads(path.read_text())["probs"], dtype=np.float32
        )
    check = work / f"{name}.sanity.json"
    ratio = (json.loads(check.read_text())["energy_ratio"]
             if check.exists() else None)
    return Session(
        name=name, corpus=str(data["corpus"]), energy_ratio=ratio,
        starts=data["starts"], ends=data["ends"],
        speakers=[str(s) for s in data["speakers"]],
        overlap=data["overlap"], whole=data["whole"],
        windows=data["windows"], window_segment=data["window_segment"],
        probs=probs,
        embed_ms=[(float(a), float(b)) for a, b in data["embed_ms"]],
    )


# ---------------------------------------------------------------- trials


def unit(vector: FloatArray) -> Optional[FloatArray]:
    """Return ``vector`` made unit, ``None`` when it is (nearly) zero."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    out: FloatArray = (vector / norm).astype(np.float32)
    return out


@dataclass
class Trial:
    """One session with one speaker as the employee and a check length.

    ``tests`` are the segment numbers scored, in time order; ``truth``
    is True for the employee's.
    """

    session: Session
    employee: str
    check: float
    check_seconds: float
    reference: FloatArray
    tests: List[int]
    truth: Dict[int, bool]
    check_segments: List[int]


def trials(session: Session, check: float) -> List[Trial]:
    """Return the session's trials (each main speaker as the employee)."""
    talk: Dict[str, float] = {}
    for s, a, b in zip(session.speakers, session.starts, session.ends):
        talk[s] = talk.get(s, 0.0) + float(b - a)
    main = sorted(talk, key=lambda s: -talk[s])[:2]
    pair = session.corpus == "ami"
    out: List[Trial] = []
    for employee in main:
        chosen: List[int] = []
        seconds = 0.0
        for n, who in enumerate(session.speakers):
            span = float(session.ends[n] - session.starts[n])
            if who != employee or session.overlap[n] > CHECK_OVERLAP \
                    or span < 1.0:
                continue
            chosen.append(n)
            seconds += span
            if seconds >= check:
                break
        if seconds < check:
            continue
        rows = session.windows[np.isin(session.window_segment, chosen)]
        total = unit(rows.sum(axis=0)) if len(rows) else None
        if total is None:
            continue
        after = float(session.ends[chosen[-1]])
        tests = [
            n for n in range(len(session.speakers))
            if session.starts[n] >= after and n not in chosen
            and (not pair or session.speakers[n] in main)
        ]
        out.append(Trial(
            session, employee, check, seconds, total, tests,
            {n: session.speakers[n] == employee for n in tests}, chosen,
        ))
    return out


# ---------------------------------------------------------------- approaches

Decision = Dict[int, Optional[bool]]
"""Per test segment: True employee, False other, None no decision."""


def _cos(trial: Trial, n: int) -> Optional[float]:
    """Return the cosine of segment ``n``'s whole vector with the check."""
    vector = trial.session.whole[n]
    if not np.isfinite(vector).all():
        return None
    return float(vector @ trial.reference)


def verify(trial: Trial, tau: float) -> Decision:
    """Each sentence alone: employee when its cosine with the check >= tau.

    A sentence too short for a vector takes the side of the sentence
    before it (the other person at the start).
    """
    out: Decision = {}
    last = False
    for n in trial.tests:
        c = _cos(trial, n)
        side = last if c is None else c >= tau
        out[n] = side
        last = side
    return out


def verify_centred(trial: Trial, tau: float, raw: float) -> Decision:
    """Like :func:`verify`, on vectors centred on the session's mean.

    The mean is that of the check and of every sentence heard so far
    (this one included): with two people it lies between their voices,
    so a centred cosine above ``tau`` (near 0) says "nearer the check
    than the room". The first ``WARM`` sentences use :func:`verify` at
    ``raw``.
    """
    out: Decision = {}
    total = trial.reference.copy()
    count = 1
    seen = 0
    last = False
    for n in trial.tests:
        vector = trial.session.whole[n]
        if not np.isfinite(vector).all():
            out[n] = last
            continue
        total = total + vector
        count += 1
        seen += 1
        if seen <= WARM:
            side = float(vector @ trial.reference) >= raw
        else:
            mean = total / count
            s, e = unit(vector - mean), unit(trial.reference - mean)
            side = s is not None and e is not None and float(s @ e) >= tau
        out[n], last = side, side
    return out


def two_voices(
    trial: Trial, tau: float, margin: float, adapt: bool = False,
) -> Tuple[Decision, Dict[int, float]]:
    """Two references: the check, and the other person learnt as heard.

    Until a sentence has been given to the other person, the rule is
    :func:`verify` at ``tau``; then a sentence is the employee's when
    its cosine with the check minus its cosine with the other person's
    voice (the direction of the sum of the sentences given to them so
    far) is ``margin`` or more. ``adapt`` also adds the employee's
    sentences to the check's voice. Returns the decisions and each
    sentence's score (the difference, or cosine minus ``tau`` before
    the other voice exists).
    """
    out: Decision = {}
    scores: Dict[int, float] = {}
    other: Optional[FloatArray] = None
    mine = trial.reference.copy()
    last = False
    for n in trial.tests:
        vector = trial.session.whole[n]
        if not np.isfinite(vector).all():
            out[n] = last
            continue
        reference = unit(mine)
        assert reference is not None
        c = float(vector @ reference)
        o = unit(other) if other is not None else None
        if o is None:
            score = c - tau
            side = c >= tau
        else:
            score = c - float(vector @ o)
            side = score >= margin
        out[n], scores[n], last = side, score, side
        if not side:
            other = vector.copy() if other is None else other + vector
        elif adapt:
            mine = mine + vector
    return out, scores


def _slots(trial: Trial) -> Dict[int, Optional[int]]:
    """Return the Sortformer slot of every test and check segment."""
    return {n: slot for n, (slot, _) in _picks(trial).items()}


def check_slot(trial: Trial) -> Decision:
    """The slot that talks during the check is the employee; no voices."""
    slots = _slots(trial)
    seen = [slots[n] for n in trial.check_segments if slots[n] is not None]
    mine = max(set(seen), key=seen.count) if seen else None
    return {n: (slots[n] == mine if slots[n] is not None else None)
            for n in trial.tests}


def slot_naming(trial: Trial, tau: float) -> Tuple[Decision, Decision]:
    """Today's naming with the check as the voiceprint.

    A slot's voice is the direction of the sum of the window vectors
    of its sentences so far (all of them: a simplification of the
    "clean" rule); once it holds 4 s it is compared with the check and
    the slot nearest to it at ``tau`` or more is the employee. Returns
    the side at emission and after the last revision (a slot named
    later revises its earlier sentences).
    """
    slots = _slots(trial)
    s = trial.session
    sums: Dict[int, FloatArray] = {}
    seconds: Dict[int, float] = {}
    named: Optional[int] = None
    emitted: Decision = {}
    for n in trial.tests:
        slot = slots[n]
        emitted[n] = None if slot is None else slot == named
        if slot is None:
            continue
        if s.window_count[n]:
            sums[slot] = sums.get(slot, 0) + s.window_sum[n]
            seconds[slot] = seconds.get(slot, 0.0) + float(
                s.ends[n] - s.starts[n])
        best, score = None, -2.0
        for k, total in sums.items():
            voice = unit(total)
            if voice is None or seconds[k] < 4.0:
                continue
            c = float(voice @ trial.reference)
            if c >= tau and c > score:
                best, score = k, c
        named = best
    final: Decision = {
        n: (None if slots[n] is None else slots[n] == named)
        for n in trial.tests
    }
    return emitted, final


def best_slot(
    trial: Trial, tau: float, extra: Optional[float] = None,
    slot_seconds: float = 2.0, purity: float = 0.0,
    check_first: bool = True,
) -> Tuple[Decision, Decision]:
    """The rule chosen for production (#249): one slot per person.

    Each slot's voice is the direction of the sum of the window
    vectors of its sentences so far (those at least ``purity`` its own
    alone), compared once it holds ``slot_seconds``. The slot nearest
    to the check, at ``tau`` or more, is the employee's; with
    ``extra``, any other slot at ``extra`` or more is too (one person
    split in two slots). Until a slot is named that way, the slot that
    spoke the check stands in (``check_first``). Returns the side at
    emission and after the last revision.
    """
    picks = _picks(trial)
    s = trial.session
    weight: Dict[int, float] = {}
    for n in trial.check_segments:
        slot = picks[n][0]
        if slot is not None:
            weight[slot] = weight.get(slot, 0.0) + float(
                s.ends[n] - s.starts[n])
    fallback = max(sorted(weight), key=lambda k: weight[k]) \
        if weight and check_first else None
    sums: Dict[int, FloatArray] = {}
    seconds: Dict[int, float] = {}
    mine: set[int] = set() if fallback is None else {fallback}

    def decide() -> set[int]:
        scores: Dict[int, float] = {}
        for k, total in sums.items():
            voice = unit(total)
            if voice is not None and seconds[k] >= slot_seconds:
                scores[k] = float(voice @ trial.reference)
        best = max(scores, key=lambda k: scores[k]) if scores else None
        if best is None or scores[best] < tau:
            return set() if fallback is None else {fallback}
        out = {best}
        if extra is not None:
            out |= {k for k, c in scores.items() if c >= extra}
        return out

    emitted: Decision = {}
    for n in trial.tests:
        slot, pure = picks[n]
        emitted[n] = None if slot is None else slot in mine
        if slot is not None and pure >= purity and s.window_count[n]:
            sums[slot] = sums.get(slot, 0) + s.window_sum[n]
            seconds[slot] = seconds.get(slot, 0.0) + float(
                s.ends[n] - s.starts[n])
            mine = decide()
    final: Decision = {
        n: None if picks[n][0] is None else picks[n][0] in mine
        for n in trial.tests
    }
    return emitted, final


def shipped(
    trial: Trial, match: float = 0.86, split: Optional[float] = None,
    purity: float = 0.8, first: float = 2.0, growth: float = 2.0,
    cap: float = 30.0,
) -> Tuple[Decision, Decision, Dict[int, bool]]:
    """The rule as it ships (``MonoAttributor`` with ``PartyPlan``).

    As the service does it: a slot's voice gathers only the sentences
    that are at least ``purity`` its own alone (the check's sentences
    included: they are the employee's speech in their slot), it is
    embedded when the slot first holds ``first`` seconds of such
    speech, then each time it holds ``growth`` times more, until
    ``cap`` seconds (``NamingPlan``); only then are the sides decided
    again (:class:`~transcriber.domain.voice_check.PartySides`: the
    slot nearest to the check at ``match`` or more, any other at
    ``split`` or more, the check's slot until then). Returns the side
    at emission, after the last revision, and whether each sentence
    was sent ``speaker_unsure``.
    """
    picks = _picks(trial)
    s = trial.session
    weight: Dict[int, float] = {}
    for n in trial.check_segments:
        slot = picks[n][0]
        if slot is not None:
            weight[slot] = weight.get(slot, 0.0) + float(
                s.ends[n] - s.starts[n])
    check_slot = max(sorted(weight), key=lambda k: weight[k]) \
        if weight else None
    gathered: Dict[int, FloatArray] = {}
    seconds: Dict[int, float] = {}
    due: Dict[int, float] = {}
    voices: Dict[int, FloatArray] = {}

    def feed(n: int) -> None:
        slot, pure = picks[n]
        if slot is None or pure < purity or not s.window_count[n] \
                or seconds.get(slot, 0.0) >= cap:
            return
        gathered[slot] = gathered.get(slot, 0) + s.window_sum[n]
        seconds[slot] = seconds.get(slot, 0.0) + float(
            s.ends[n] - s.starts[n])
        if seconds[slot] >= due.get(slot, first):
            due[slot] = seconds[slot] * growth
            voice = unit(gathered[slot])
            if voice is not None:
                voices[slot] = voice

    def employee() -> set[int]:
        scores = {k: float(v @ trial.reference) for k, v in voices.items()}
        best = max(sorted(scores), key=lambda k: scores[k]) \
            if scores else None
        if best is None or scores[best] < match:
            return set() if check_slot is None else {check_slot}
        out = {best}
        if split is not None:
            out |= {k for k, c in scores.items() if c >= split}
        return out

    for n in trial.check_segments:
        feed(n)
    emitted: Decision = {}
    unsure: Dict[int, bool] = {}
    for n in trial.tests:
        slot = picks[n][0]
        emitted[n] = None if slot is None else slot in employee()
        unsure[n] = slot is not None and slot not in voices \
            and slot != check_slot
        feed(n)
    mine = employee()
    final: Decision = {
        n: None if picks[n][0] is None else picks[n][0] in mine
        for n in trial.tests
    }
    return emitted, final, unsure


_PICKS: Dict[Tuple[str, str, float],
             Dict[int, Tuple[Optional[int], float]]] = {}


def _picks(trial: Trial) -> Dict[int, Tuple[Optional[int], float]]:
    """Return the slot and purity of every test and check segment."""
    key = (trial.session.name, trial.employee, trial.check)
    known = _PICKS.get(key)
    if known is None:
        known = _PICKS[key] = _pick_all(trial)
    return known


def _pick_all(trial: Trial) -> Dict[int, Tuple[Optional[int], float]]:
    """Compute :func:`_picks` (once per trial)."""
    from transcriber.domain.live_diarization import SpeakerFrames, pick_slot

    probs = trial.session.probs
    assert probs is not None
    frames = SpeakerFrames(0.08, probs.shape[1])
    frames.extend(0, probs.tolist())
    out: Dict[int, Tuple[Optional[int], float]] = {}
    for n in trial.tests + trial.check_segments:
        pick = pick_slot(frames, float(trial.session.starts[n]),
                         float(trial.session.ends[n]), 0.5)
        out[n] = (pick.slot, pick.purity)
    return out


def slot_sides(
    trial: Trial, tau: float, slot_seconds: float = 2.0,
    use_check_slot: bool = True, purity: float = 0.8,
    voice_first: bool = False,
) -> Tuple[Decision, Decision]:
    """The production rule: slots take sides from the check (#249).

    The slot that spoke most of the check is the employee's
    (``use_check_slot``); any slot whose voice (the direction of the
    sum of the window vectors of its clean sentences so far, at least
    ``slot_seconds`` of them) has a cosine of ``tau`` or more with the
    check is the employee's too; every other slot is the other
    person's. ``voice_first``: once a slot has its voice, the voice
    decides, the check's slot included (the check's slot only stands
    in until then). Returns the side at emission and after the last
    revision.
    """
    picks = _picks(trial)
    s = trial.session
    weight: Dict[int, float] = {}
    for n in trial.check_segments:
        slot = picks[n][0]
        if slot is not None:
            weight[slot] = weight.get(slot, 0.0) + float(
                s.ends[n] - s.starts[n])
    mine = max(sorted(weight), key=lambda k: weight[k]) \
        if weight and use_check_slot else None
    sums: Dict[int, FloatArray] = {}
    seconds: Dict[int, float] = {}

    def side(slot: Optional[int]) -> Optional[bool]:
        if slot is None:
            return None
        voice = unit(sums[slot]) if slot in sums else None
        heard = voice is not None and seconds.get(slot, 0.0) >= slot_seconds
        if slot == mine and not (voice_first and heard):
            return True
        if voice is None or not heard:
            return False
        return float(voice @ trial.reference) >= tau

    emitted: Decision = {}
    for n in trial.tests:
        slot, pure = picks[n]
        emitted[n] = side(slot)
        if slot is not None and pure >= purity and s.window_count[n]:
            sums[slot] = sums.get(slot, 0) + s.window_sum[n]
            seconds[slot] = seconds.get(slot, 0.0) + float(
                s.ends[n] - s.starts[n])
    final: Decision = {n: side(picks[n][0]) for n in trial.tests}
    return emitted, final


# ---------------------------------------------------------------- scoring


@dataclass
class Tally:
    """Right, wrong and undecided counts, overall and per kind."""

    right: int = 0
    wrong: int = 0
    none: int = 0
    employee_right: int = 0
    employee: int = 0
    other_right: int = 0
    other: int = 0

    def add(self, truth: bool, side: Optional[bool]) -> None:
        """Count one sentence."""
        if truth:
            self.employee += 1
        else:
            self.other += 1
        if side is None:
            self.none += 1
            return
        if side == truth:
            self.right += 1
            if truth:
                self.employee_right += 1
            else:
                self.other_right += 1
        else:
            self.wrong += 1

    def row(self) -> Dict[str, Any]:
        """Return the shares (undecided counts as wrong in accuracy)."""
        n = self.right + self.wrong + self.none
        return {
            "n": n,
            "accuracy": round(self.right / n, 4) if n else None,
            "employee_recall": round(self.employee_right / self.employee, 4)
            if self.employee else None,
            "other_recall": round(self.other_right / self.other, 4)
            if self.other else None,
            "undecided": self.none,
        }


def tally(
    trials_: Sequence[Trial], decide: Any, where: Any = None,
) -> Tally:
    """Count ``decide(trial)``'s decisions over the trials' tests."""
    out = Tally()
    for trial in trials_:
        decided = decide(trial)
        for n in trial.tests:
            if where is not None and not where(trial, n):
                continue
            out.add(trial.truth[n], decided.get(n))
    return out


def _bucket(low: float, high: float) -> Any:
    """Return a filter of the tests whose duration is in [low, high)."""
    def keep(trial: Trial, n: int) -> bool:
        span = float(trial.session.ends[n] - trial.session.starts[n])
        return low <= span < high
    return keep


def _overlapped(flag: bool) -> Any:
    """Return a filter of the overlapped (or clean) tests."""
    def keep(trial: Trial, n: int) -> bool:
        return bool(trial.session.overlap[n] >= OVERLAP) == flag
    return keep


def _detail(trials_: Sequence[Trial], decide: Any) -> Dict[str, Any]:
    """Return the overall row and the rows per duration and overlap."""
    out: Dict[str, Any] = {"all": tally(trials_, decide).row()}
    for low, high in BUCKETS:
        key = f"{low:g}-{high:g}s" if high < 1e8 else f">={low:g}s"
        out[key] = tally(trials_, decide, _bucket(low, high)).row()
    out["clean"] = tally(trials_, decide, _overlapped(False)).row()
    out["overlapped"] = tally(trials_, decide, _overlapped(True)).row()
    return out


def _unsure(
    trials_: Sequence[Trial], tau: float, margin: float, band: float,
) -> Dict[str, Any]:
    """Accuracy of the sure decisions and share of unsure ones."""
    sure = Tally()
    unsure = 0
    total = 0
    for trial in trials_:
        decided, scores = two_voices(trial, tau, margin)
        for n in trial.tests:
            total += 1
            score = scores.get(n)
            base = margin if n in scores and score is not None else 0.0
            if score is None or abs(score - base) < band:
                unsure += 1
                continue
            sure.add(trial.truth[n], decided[n])
    row = sure.row()
    row["unsure_share"] = round(unsure / total, 4) if total else None
    return row


def _sweep(
    groups: Dict[str, List[Trial]], decide: Any, values: Sequence[float],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return the accuracy of each value of a parameter, per group."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for value in values:
        out[f"{value:.2f}"] = {
            group: tally(t, lambda trial: decide(trial, value)).row()[
                "accuracy"]
            for group, t in groups.items()
        }
    return out


def _best(sweep: Dict[str, Dict[str, Optional[float]]], group: str) -> float:
    """Return the value with the best accuracy in ``group``."""
    return float(max(sweep, key=lambda v: sweep[v][group] or 0.0))


def _follows(session: Session) -> bool:
    """Tell whether the session's audio follows its reference."""
    return session.energy_ratio is None or session.energy_ratio >= MIN_RATIO


def oracle(session: Session) -> Optional[float]:
    """Return the share of sentences nearer their own speaker's centroid.

    The centroids are the sums of every sentence of the two main
    speakers (an upper bound for any rule on these vectors, not a
    method: it uses the answer).
    """
    spk = np.array(session.speakers)
    span = session.ends - session.starts
    talk = {x: float(span[spk == x].sum()) for x in set(session.speakers)}
    main = sorted(talk, key=lambda x: -talk[x])[:2]
    ok = np.isfinite(session.whole).all(axis=1) & np.isin(spk, main)
    if len(main) < 2 or not ok.any():
        return None
    centres = [unit(session.whole[ok & (spk == x)].sum(axis=0))
               for x in main]
    if any(c is None for c in centres):
        return None
    sims = np.vstack([session.whole[ok] @ c for c in centres if c is not None])
    guess = np.array(main)[np.argmax(sims, axis=0)]
    return round(float((guess == spk[ok]).mean()), 4)


def score(work: Path) -> Dict[str, Any]:
    """Run every approach on every session of ``work`` (see the module)."""
    sessions = [load(work, p.stem) for p in sorted(work.glob("*.npz"))]
    results: Dict[str, Any] = {
        "sessions": {
            s.name: {"corpus": s.corpus, "segments": len(s.speakers),
                     "speakers": len(set(s.speakers)),
                     "frames": s.probs is not None,
                     "energy_ratio": s.energy_ratio,
                     "follows_reference": _follows(s),
                     "oracle_centroid_accuracy": oracle(s)}
            for s in sessions
        },
        "embed_ms_per_sentence": {},
        "checks": {},
    }
    spans = [ms for s in sessions for _, ms in s.embed_ms]
    by_len = {f"{low:g}-{high:g}s": [
        ms for s in sessions for span, ms in s.embed_ms
        if low <= span < high] for low, high in BUCKETS}
    results["embed_ms_per_sentence"] = {
        "all": {"n": len(spans), "p50": percentile(spans, 0.5),
                "p90": percentile(spans, 0.9)},
        **{k: {"n": len(v), "p50": percentile(v, 0.5),
               "p90": percentile(v, 0.9)} for k, v in by_len.items()},
    }
    for check in CHECKS:
        all_trials = [t for s in sessions for t in trials(s, check)]
        valid = [t for t in all_trials if _follows(t.session)]
        groups = {
            "callhome": [t for t in valid
                         if t.session.corpus == "callhome"],
            "callhome_two": [t for t in valid
                             if t.session.corpus == "callhome"
                             and len(set(t.session.speakers)) == 2],
            "ami_pair": [t for t in valid if t.session.corpus == "ami"],
            "apart": [t for t in all_trials if not _follows(t.session)],
        }
        groups = {k: v for k, v in groups.items() if v}
        entry: Dict[str, Any] = {
            "trials": {k: len(v) for k, v in groups.items()},
            "sentences": {k: sum(len(t.tests) for t in v)
                          for k, v in groups.items()},
            "check_seconds_mean": round(float(np.mean(
                [t.check_seconds for t in all_trials])), 2),
        }
        verify_sweep = _sweep(groups, verify, THRESHOLDS)
        entry["verify_sweep"] = verify_sweep
        tau = {g: _best(verify_sweep, g) for g in groups}
        entry["verify_best"] = tau
        anchor = tau.get("callhome", 0.0)
        entry["verify_at_callhome_best"] = {
            g: _detail(t, lambda trial: verify(trial, anchor))
            for g, t in groups.items()
        }
        centred = _sweep(
            groups, lambda trial, x: verify_centred(trial, x, anchor),
            CENTRED,
        )
        entry["verify_centred_sweep"] = centred
        tc = _best(centred, "callhome") if "callhome" in groups else 0.0
        entry["verify_centred_best"] = {g: _best(centred, g) for g in groups}
        entry["verify_centred_at_callhome_best"] = {
            g: _detail(t, lambda trial: verify_centred(trial, tc, anchor))
            for g, t in groups.items()
        }
        two_sweep = _sweep(
            groups, lambda trial, m: two_voices(trial, anchor, m)[0],
            MARGINS,
        )
        entry["two_voices_sweep"] = two_sweep
        margin = _best(two_sweep, "callhome") if "callhome" in groups \
            else 0.0
        entry["two_voices_best_margin"] = margin
        entry["two_voices_at_callhome_best"] = {
            g: _detail(t, lambda trial: two_voices(trial, anchor, margin)[0])
            for g, t in groups.items()
        }
        entry["two_voices_tau_sweep"] = _sweep(
            groups, lambda trial, x: two_voices(trial, x, margin)[0],
            THRESHOLDS,
        )
        entry["two_voices_adapt"] = {
            g: _detail(t, lambda trial: two_voices(
                trial, anchor, margin, adapt=True)[0])
            for g, t in groups.items()
        }
        entry["two_voices_unsure"] = {
            f"{band:.2f}": {g: _unsure(t, anchor, margin, band)
                            for g, t in groups.items()}
            for band in (0.0, 0.01, 0.02, 0.03, 0.05)
        }
        framed = {g: [t for t in v if t.session.probs is not None]
                  for g, v in groups.items()}
        framed = {g: v for g, v in framed.items() if v}
        if framed:
            entry["framed_trials"] = {k: len(v) for k, v in framed.items()}
            entry["check_slot"] = {
                g: _detail(t, check_slot) for g, t in framed.items()}
            entry["slot_naming_emitted"] = {
                g: _detail(t, lambda trial: slot_naming(trial, anchor)[0])
                for g, t in framed.items()}
            entry["slot_naming_final"] = {
                g: _detail(t, lambda trial: slot_naming(trial, anchor)[1])
                for g, t in framed.items()}
            framed_groups = {g: v for g, v in framed.items()}
            slot_sweep = _sweep(
                framed_groups,
                lambda trial, x: slot_sides(trial, x)[1], THRESHOLDS,
            )
            entry["slot_sides_sweep_final"] = slot_sweep
            st = _best(slot_sweep, "callhome") \
                if "callhome" in framed else anchor
            entry["slot_sides_best"] = {
                g: _best(slot_sweep, g) for g in framed}
            entry["slot_sides_emitted"] = {
                g: _detail(t, lambda trial: slot_sides(trial, st)[0])
                for g, t in framed.items()}
            entry["slot_sides_final"] = {
                g: _detail(t, lambda trial: slot_sides(trial, st)[1])
                for g, t in framed.items()}
            entry["slot_sides_4s_final"] = {
                g: _detail(t, lambda trial: slot_sides(trial, st, 4.0)[1])
                for g, t in framed.items()}
            first_sweep = _sweep(
                framed_groups, lambda trial, x: slot_sides(
                    trial, x, voice_first=True)[1], THRESHOLDS,
            )
            entry["slot_voice_first_sweep_final"] = first_sweep
            vf = _best(first_sweep, "callhome") \
                if "callhome" in framed else anchor
            entry["slot_voice_first_best"] = {
                g: _best(first_sweep, g) for g in framed}
            entry["slot_voice_first_emitted"] = {
                g: _detail(t, lambda trial: slot_sides(
                    trial, vf, voice_first=True)[0])
                for g, t in framed.items()}
            entry["slot_voice_first_final"] = {
                g: _detail(t, lambda trial: slot_sides(
                    trial, vf, voice_first=True)[1])
                for g, t in framed.items()}
            entry["slot_voice_only_final"] = {
                g: _detail(t, lambda trial: slot_sides(
                    trial, st, use_check_slot=False)[1])
                for g, t in framed.items()}
            best_sweep = _sweep(
                framed_groups, lambda trial, x: best_slot(trial, x)[1],
                THRESHOLDS,
            )
            entry["best_slot_sweep_final"] = best_sweep
            bt = _best(best_sweep, "callhome") \
                if "callhome" in framed else anchor
            entry["best_slot_tau_callhome"] = bt
            # The idealised rule (every sentence feeds the voices, which
            # are compared after each one) is scored at the shipped match.
            bt = PRODUCTION_MATCH
            entry["best_slot_tau"] = bt
            entry["best_slot_best"] = {
                g: _best(best_sweep, g) for g in framed}
            variants: Dict[str, Any] = {
                "best_slot": {},
                "best_slot_extra95": {"extra": 0.95},
                "best_slot_extra97": {"extra": 0.97},
                "best_slot_no_check": {"check_first": False},
                "best_slot_clean": {"purity": 0.8},
                "best_slot_4s": {"slot_seconds": 4.0},
            }
            for name, kw in variants.items():
                entry[f"{name}_emitted"] = {
                    g: _detail(t, lambda trial: best_slot(
                        trial, bt, **kw)[0]) for g, t in framed.items()}
                entry[f"{name}_final"] = {
                    g: _detail(t, lambda trial: best_slot(
                        trial, bt, **kw)[1]) for g, t in framed.items()}
            # The shipped configuration (#249 review): every number the
            # docs quote for the chosen rule comes from these entries.
            entry["shipped_sweep_final"] = _sweep(
                framed_groups, lambda trial, x: shipped(trial, x)[1],
                THRESHOLDS,
            )
            entry["shipped_sweep_split97_final"] = _sweep(
                framed_groups, lambda trial, x: shipped(trial, x, 0.97)[1],
                (0.80, 0.85, 0.86, 0.87, 0.89, 0.91, 0.93, 0.97),
            )
            entry["shipped_split_sweep_final"] = _sweep(
                framed_groups, lambda trial, x: shipped(
                    trial, PRODUCTION_MATCH,
                    None if x > 1.0 else x)[1],
                (0.93, 0.95, 0.97, 0.99, 1.01),  # 1.01: off
            )
            entry["shipped_purity_sweep_final"] = _sweep(
                framed_groups, lambda trial, x: shipped(
                    trial, PRODUCTION_MATCH, PRODUCTION_SPLIT, x)[1],
                (0.0, 0.5, 0.8),
            )
            entry["shipped_emitted"] = {
                g: _detail(t, lambda trial: shipped(
                    trial, PRODUCTION_MATCH, PRODUCTION_SPLIT)[0])
                for g, t in framed.items()}
            entry["shipped_final"] = {
                g: _detail(t, lambda trial: shipped(
                    trial, PRODUCTION_MATCH, PRODUCTION_SPLIT)[1])
                for g, t in framed.items()}
            for flag, name in ((False, "sure"), (True, "unsure")):
                out: Dict[str, Any] = {}
                for g, group in framed.items():
                    count = Tally()
                    for trial in group:
                        sent, _, doubt = shipped(
                            trial, PRODUCTION_MATCH, PRODUCTION_SPLIT)
                        for n in trial.tests:
                            if doubt[n] == flag:
                                count.add(trial.truth[n], sent[n])
                    out[g] = count.row()
                entry[f"shipped_emitted_{name}"] = out
            entry["two_voices_same_trials"] = {
                g: _detail(t, lambda trial: two_voices(
                    trial, anchor, margin)[0])
                for g, t in framed.items()}
        results["checks"][f"{check:g}"] = entry
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """Run one of the commands (see the module)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("embed", "frames", "sanity"):
        c = commands.add_parser(name)
        c.add_argument("sessions", nargs="+")
        c.add_argument("--lab", type=Path, required=True)
        c.add_argument("--work", type=Path, required=True)
        if name == "frames":
            c.add_argument("--device", default="cpu")
    s = commands.add_parser("score")
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--results", type=Path,
                   default=Path(__file__).with_name(
                       "voice_check_results.json"))
    args = parser.parse_args(argv)
    if args.command == "embed":
        embed(args.sessions, args.lab, args.work)
    elif args.command == "frames":
        frames(args.sessions, args.lab, args.work, args.device)
    elif args.command == "sanity":
        sanity(args.sessions, args.lab, args.work)
    else:
        results = score(args.work)
        args.results.write_text(json.dumps(results, indent=1) + "\n")
        print(json.dumps({
            k: {"trials": v["trials"], "verify_best": v["verify_best"],
                "two_voices_best_margin": v["two_voices_best_margin"]}
            for k, v in results["checks"].items()
        }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
