"""Re-tune the certainty and overlap flags offline (TR-19, issue #95).

Reads the files ``flag_capture.py`` wrote (the inputs of the flags of
each corpus session on the lab-diarizer path) and recomputes, with the
domain's own functions, the sentences' ``doubtful`` and ``overlap``
flags for any thresholds. Each sentence is then judged against the human
reference:

- **wrong**: the reference speaker who speaks most inside the sentence
  is not the one its diarizer label is matched to (one-to-one
  assignment of labels to reference speakers by shared speech seconds,
  fixed from the service's own transcript, so it does not move with the
  thresholds). A sentence with no reference speech is not scored.
- **reference overlap**: at least ``--ref-overlap`` seconds inside the
  sentence where the reference has two speakers at once.

For a flag it reports n, how many sentences are flagged, precision
(wrong among flagged) and recall (flagged among wrong) with Wilson 95%
intervals over sentences and a bootstrap over sessions.

Thresholds are chosen by leave-one-session-out cross-validation: for
each session the grid point with the best F1 of the flag on the other
sessions is scored on that session; the pooled held-out predictions give
the "after" figures, scored against the default thresholds on the same
sentences.

Usage::

    PYTHONPATH=. python experiments/transcriber/flag_retune.py \\
        --captures <dir> --out experiments/transcriber/results/flag_retune.json

The script may import :mod:`transcriber`; it never imports ``src``.
"""

import argparse
import bisect
import gzip
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from transcriber.domain.certainty import (
    VoiceEvidence,
    add_detected_overlaps,
    assess_utterances,
    keep_sentences_whole,
)
from transcriber.domain.models import Utterance, Word
from transcriber.domain.sentences import build_utterances
from transcriber.domain.speakers import turns_from_words

Row = Dict[str, Any]


class Session:
    """One captured session with its reference and speaker matching."""

    def __init__(self, raw: Row) -> None:
        """Load a capture and fix the label-to-reference matching."""
        self.name: str = raw["session"]
        self.corpus: str = raw["corpus"]
        self.parameters: Row = raw["parameters"]
        self.words = [Word(**w) for w in raw["words"]]
        labels = raw["evidence_labels"]
        self.windows = [(a, b) for a, b in labels["windows"]]
        self.scores = np.array(labels["scores"], dtype=np.float32)
        self.labels: List[Optional[str]] = labels["voices"]
        self.names: List[Optional[str]] = raw["evidence_names"]
        self.detected: List[Tuple[float, float]] = [
            (a, b) for a, b in raw["detected"]
        ]
        self.stored = [Utterance(**u) for u in raw["utterances"]]
        by_speaker: Dict[str, List[Tuple[float, float]]] = {}
        for seg in raw["reference"]:
            by_speaker.setdefault(seg["speaker"], []).append(
                (seg["start"], seg["end"]))
        self.reference = {
            k: sorted(v) for k, v in by_speaker.items()
        }
        self._lefts = {
            k: [a for a, _ in v] for k, v in self.reference.items()
        }
        self.match = self._matching()

    def seconds(self, who: str, start: float, end: float) -> float:
        """Return the seconds reference speaker ``who`` speaks in a span."""
        segs, lefts = self.reference[who], self._lefts[who]
        n = max(bisect.bisect_left(lefts, start) - 1, 0)
        total = 0.0
        while n < len(segs) and segs[n][0] < end:
            total += max(0.0, min(end, segs[n][1]) - max(start, segs[n][0]))
            n += 1
        return total

    def _matching(self) -> Dict[str, str]:
        """Match each transcript speaker to one reference speaker."""
        hyp = sorted({u.speaker for u in self.stored})
        ref = sorted(self.reference)
        table = np.zeros((len(hyp), len(ref)))
        for u in self.stored:
            for j, who in enumerate(ref):
                table[hyp.index(u.speaker), j] += self.seconds(
                    who, u.start, u.end)
        rows, cols = linear_sum_assignment(-table)
        return {hyp[i]: ref[j] for i, j in zip(rows, cols)
                if table[i, j] > 0}

    def dominant(self, start: float, end: float) -> Optional[str]:
        """Return the reference speaker heard most in a span, if any."""
        held = {w: self.seconds(w, start, end) for w in self.reference}
        who = max(held, key=lambda k: held[k])
        return who if held[who] > 0 else None

    def overlap_seconds(self, start: float, end: float) -> float:
        """Return the seconds with two reference speakers at once."""
        points = sorted({start, end} | {
            t for segs in self.reference.values() for a, b in segs
            for t in (a, b) if start < t < end})
        total = 0.0
        for left, right in zip(points, points[1:]):
            mid = (left + right) / 2
            speaking = sum(
                1 for who in self.reference
                if self.seconds(who, mid - 1e-6, mid + 1e-6) > 0)
            if speaking >= 2:
                total += right - left
        return total

    def evidence(self, voices: Sequence[Optional[str]]) -> VoiceEvidence:
        """Return the evidence with ``voices`` as column names."""
        return VoiceEvidence(self.windows, self.scores, list(voices))

    def recompute(
        self, whole: float, doubt: float, overlap_seconds: float,
        source: str = "either",
    ) -> List[Utterance]:
        """Return the sentences with the flags of the given thresholds.

        ``source`` says which overlaps flag a sentence: ``either`` the
        windows' or the diarizer's (the service's rule), ``voices`` only
        the windows', ``diarizer`` only the diarizer's, ``both`` only
        when both agree.
        """
        p = self.parameters
        evidence = self.evidence(self.labels)
        words, moved = keep_sentences_whole(
            self.words, evidence, whole, p["turn_gap"],
            p["whole_sentence_max_move"], p["whole_sentence_max_moved"],
        )
        naming = {a: b for a, b in zip(self.labels, self.names) if a}
        words = [
            w.model_copy(update={"speaker": naming[w.speaker or ""]})
            for w in words
        ]
        utterances = build_utterances(turns_from_words(words, p["turn_gap"]))
        named = self.evidence(self.names)
        utterances = assess_utterances(utterances, named, doubt, moved)
        merged = add_detected_overlaps(
            utterances, self.detected, overlap_seconds)
        if source == "either":
            return merged
        out = []
        for u in merged:
            flags = {
                "voices": u.overlap_source in ("voices", "both"),
                "diarizer": u.overlap_source in ("detector", "both"),
                "both": u.overlap_source == "both",
            }[source]
            out.append(u.model_copy(update={"overlap": flags}))
        return out

    def truth(self, utterances: Sequence[Utterance], ref_overlap: float
              ) -> List[Optional[Tuple[bool, bool]]]:
        """Return ``(wrong, reference overlap)`` per sentence (or None)."""
        out: List[Optional[Tuple[bool, bool]]] = []
        for u in utterances:
            who = self.dominant(u.start, u.end)
            if who is None:
                out.append(None)
                continue
            wrong = self.match.get(u.speaker) != who
            heard = self.overlap_seconds(u.start, u.end) >= ref_overlap
            out.append((wrong, heard))
        return out


def wilson(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Return the Wilson interval of ``hits`` in ``n`` (0..1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


class Counts:
    """Confusion counts of a flag against a truth."""

    def __init__(self) -> None:
        """Start empty."""
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, flag: bool, truth: bool) -> None:
        """Count one sentence."""
        if flag and truth:
            self.tp += 1
        elif flag:
            self.fp += 1
        elif truth:
            self.fn += 1
        else:
            self.tn += 1

    def merge(self, other: "Counts") -> None:
        """Add ``other``'s counts."""
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    @property
    def n(self) -> int:
        """Return the sentences counted."""
        return self.tp + self.fp + self.fn + self.tn

    def precision(self) -> float:
        """Return wrong among flagged (nan when nothing is flagged)."""
        d = self.tp + self.fp
        return self.tp / d if d else math.nan

    def recall(self) -> float:
        """Return flagged among wrong (nan when nothing is wrong)."""
        d = self.tp + self.fn
        return self.tp / d if d else math.nan

    def f1(self) -> float:
        """Return the harmonic mean (0 when undefined)."""
        p, r = self.precision(), self.recall()
        if math.isnan(p) or math.isnan(r) or p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def summary(self) -> Row:
        """Return the figures as a row, with Wilson intervals."""
        flagged = self.tp + self.fp
        wrong = self.tp + self.fn
        return {
            "n": self.n, "wrong": wrong, "flagged": flagged,
            "flagged_share": _r(flagged / self.n if self.n else math.nan),
            "wrong_share": _r(wrong / self.n if self.n else math.nan),
            "precision": _r(self.precision()),
            "precision_ci": [_r(x) for x in wilson(self.tp, flagged)],
            "recall": _r(self.recall()),
            "recall_ci": [_r(x) for x in wilson(self.tp, wrong)],
            "f1": _r(self.f1()),
        }


def _r(x: float) -> Optional[float]:
    """Round to 4 places; ``None`` for nan."""
    return None if math.isnan(x) else round(x, 4)


Grid = Dict[str, Sequence[Any]]

# What the service uses today.
DEFAULT: Dict[str, Any] = {"whole": 0.1, "doubt": 0.1, "overlap_seconds": 0.5}


def counts_for(
    session: Session, point: Dict[str, Any], flag: str, truth: str,
    ref_overlap: float, cache: Dict[Any, Any],
) -> Counts:
    """Count the flag of one grid point of one session."""
    key = (session.name, tuple(sorted(point.items())))
    if key not in cache:
        sentences = session.recompute(**point)
        cache[key] = (sentences, session.truth(sentences, ref_overlap))
    sentences, known = cache[key]
    out = Counts()
    for u, t in zip(sentences, known):
        if t is None:
            continue
        wrong, heard = t
        value = {
            "doubtful": bool(u.doubtful), "overlap": bool(u.overlap),
            "either": bool(u.doubtful) or bool(u.overlap),
        }[flag]
        out.add(value, {"wrong": wrong, "overlap": heard}[truth])
    return out


def points(grid: Grid) -> List[Dict[str, Any]]:
    """Expand a grid into its points."""
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(
        *(grid[k] for k in keys))]


def bootstrap(per_session: Dict[str, Counts], draws: int = 2000,
              seed: int = 95) -> Row:
    """Return session-level bootstrap intervals of precision and recall."""
    rng = random.Random(seed)
    names = sorted(per_session)
    prec: List[float] = []
    rec: List[float] = []
    for _ in range(draws):
        total = Counts()
        for name in (rng.choice(names) for _ in names):
            total.merge(per_session[name])
        if not math.isnan(total.precision()):
            prec.append(total.precision())
        if not math.isnan(total.recall()):
            rec.append(total.recall())

    def interval(x: List[float]) -> List[Optional[float]]:
        """Return the 2.5 and 97.5 percentiles."""
        if not x:
            return [None, None]
        return [_r(float(np.percentile(x, 2.5))),
                _r(float(np.percentile(x, 97.5)))]
    return {"precision_ci": interval(prec), "recall_ci": interval(rec)}


def cross_validate(
    sessions: Sequence[Session], grid: Grid, flag: str, truth: str,
    ref_overlap: float, cache: Dict[Any, Any],
) -> Tuple[Dict[str, Counts], Dict[str, Dict[str, Any]]]:
    """Leave one session out: pick on the rest, score on the one."""
    grid_points = points(grid)
    table = {
        (s.name, n): counts_for(s, p, flag, truth, ref_overlap, cache)
        for s in sessions for n, p in enumerate(grid_points)
    }
    held: Dict[str, Counts] = {}
    chosen: Dict[str, Dict[str, Any]] = {}
    for s in sessions:
        best, best_f = 0, -1.0
        for n in range(len(grid_points)):
            total = Counts()
            for o in sessions:
                if o.name != s.name:
                    total.merge(table[(o.name, n)])
            if total.f1() > best_f:
                best, best_f = n, total.f1()
        held[s.name] = table[(s.name, best)]
        chosen[s.name] = grid_points[best]
    return held, chosen


def pooled(per_session: Dict[str, Counts]) -> Counts:
    """Return the sum of the sessions' counts."""
    total = Counts()
    for c in per_session.values():
        total.merge(c)
    return total


def report(per_session: Dict[str, Counts]) -> Row:
    """Return the pooled figures, per-corpus figures and the bootstrap."""
    out = pooled(per_session).summary()
    out.update({"session_bootstrap": bootstrap(per_session)})
    by_corpus: Dict[str, Counts] = {}
    for name, c in per_session.items():
        by_corpus.setdefault(name.split("-")[0], Counts()).merge(c)
    out["by_corpus"] = {k: v.summary() for k, v in sorted(by_corpus.items())}
    return out


def main() -> None:
    """Evaluate the flags before and after, with cross-validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ref-overlap", type=float, default=0.5)
    args = parser.parse_args()
    sessions = [
        Session(json.load(gzip.open(p, "rt", encoding="utf-8")))
        for p in sorted(args.captures.glob("*.json.gz"))
    ]
    cache: Dict[Any, Any] = {}
    result: Row = {
        "sessions": {s.name: s.corpus for s in sessions},
        "ref_overlap_seconds": args.ref_overlap,
        "default": DEFAULT,
    }
    # Sanity: the recomputation reproduces the service's own flags.
    same = total = 0
    for s in sessions:
        again = s.recompute(**DEFAULT)
        for a, b in zip(again, s.stored):
            total += 1
            same += (a.doubtful, a.overlap) == (b.doubtful, b.overlap)
    result["reproduced_flags"] = {"same": same, "of": total}
    before: Row = {}
    for flag in ("doubtful", "overlap", "either"):
        before[flag] = report({
            s.name: counts_for(s, DEFAULT, flag, "wrong",
                               args.ref_overlap, cache)
            for s in sessions})
    result["before_wrong"] = before
    result["before_overlap_vs_reference"] = report({
        s.name: counts_for(s, DEFAULT, "overlap", "overlap",
                           args.ref_overlap, cache)
        for s in sessions})
    grid: Grid = {
        "whole": (0.0, 0.05, 0.1, 0.2),
        "doubt": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
        "overlap_seconds": (0.5,),
    }
    held, chosen = cross_validate(
        sessions, grid, "doubtful", "wrong", args.ref_overlap, cache)
    result["after_doubtful_cv"] = report(held)
    result["after_doubtful_choices"] = chosen
    # In-sample grid (for reading the trade-off, not for the claim).
    result["grid_doubtful"] = [
        {"point": p, **pooled({
            s.name: counts_for(s, p, "doubtful", "wrong",
                               args.ref_overlap, cache)
            for s in sessions}).summary()}
        for p in points(grid)
    ]
    # Which overlaps should flag a sentence: how each source predicts a
    # reference overlap (what the flag claims) and a wrong attribution.
    sources: Row = {}
    for source in ("either", "voices", "diarizer", "both"):
        point: Dict[str, Any] = {**DEFAULT, "source": source}
        sources[source] = {
            truth: report({
                s.name: counts_for(s, point, "overlap", truth,
                                   args.ref_overlap, cache)
                for s in sessions})
            for truth in ("overlap", "wrong")
        }
    result["overlap_sources"] = sources
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
