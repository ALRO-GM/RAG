"""Score the transcriber on the rendered simulated discussion (#96).

Stand-alone script (nothing of the project is imported)::

    python experiments/skills/score_audio_simulation.py \\
        sim-reunion-verano.reference.json \\
        sim-reunion-verano-audio.transcript.v1.json [--json out.json]

``reference.json`` is what ``render_simulation.py`` wrote: the rendered
start, end, speaker and text of every sentence of the script. The
transcript is the ``transcript.v1`` the transcriber produced for the
rendered WAV. It reports:

* word error over the whole text (both sides lower-cased, without
  punctuation, in time order), with substitutions, deletions and
  insertions;
* for every reference sentence, the transcript utterance that overlaps
  it most in time, and whether its speaker is the same person (speaker
  ids of enrolled voices are the reference's ids); sentences that no
  utterance overlaps are counted as missed;
* the same for the sentences that start over the previous one
  (``overlap`` in the reference) and for the sentences they cut into;
* the share of reference speech time covered by an utterance of the
  right speaker.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

Segment = Mapping[str, Any]
"""One sentence or utterance: ``speaker``, ``start``, ``end``, ``text``."""

WORD = re.compile(r"[^\W_]+", re.UNICODE)
"""A run of letters or digits: a word once punctuation is dropped."""


def words(text: str) -> List[str]:
    """Return the lower-cased words of ``text`` without punctuation."""
    return WORD.findall(unicodedata.normalize("NFC", text).lower())


def edit_counts(reference: Sequence[str],
                hypothesis: Sequence[str]) -> Dict[str, int]:
    """Return the substitutions, deletions and insertions of an alignment.

    The alignment is the one of least edits (Levenshtein over words);
    among alignments with as many edits, the counts of one of them.
    """
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            same = reference[i - 1] == hypothesis[j - 1]
            cost[i][j] = min(
                cost[i - 1][j - 1] + (0 if same else 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = reference[i - 1] == hypothesis[j - 1]
            if cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
                if not same:
                    counts["substitutions"] += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    return counts


def word_error(reference: Sequence[Segment],
               hypothesis: Sequence[Segment]) -> Dict[str, Any]:
    """Return the word error of ``hypothesis`` over the whole text.

    Both sides are read in time order (by start) and joined.
    """
    ref = [w for s in _by_start(reference) for w in words(str(s["text"]))]
    hyp = [w for s in _by_start(hypothesis) for w in words(str(s["text"]))]
    counts = edit_counts(ref, hyp)
    errors = sum(counts.values())
    return {
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        **counts,
        "wer": errors / len(ref) if ref else 0.0,
    }


def overlap(a: Segment, b: Segment) -> float:
    """Return the seconds two segments share (0 if they do not meet)."""
    return max(0.0, min(float(a["end"]), float(b["end"]))
               - max(float(a["start"]), float(b["start"])))


def best_match(sentence: Segment,
               utterances: Sequence[Segment]) -> Optional[Segment]:
    """Return the utterance that overlaps ``sentence`` most, if any does."""
    best: Optional[Segment] = None
    best_seconds = 0.0
    for utterance in utterances:
        seconds = overlap(sentence, utterance)
        if seconds > best_seconds:
            best, best_seconds = utterance, seconds
    return best


def attribution(reference: Sequence[Segment],
                utterances: Sequence[Segment]) -> List[Dict[str, Any]]:
    """Return, per reference sentence, its best utterance and the match.

    ``match`` is ``None`` when no utterance overlaps the sentence.
    """
    rows = []
    for sentence in reference:
        found = best_match(sentence, utterances)
        rows.append({
            "id": sentence.get("id", ""),
            "speaker": sentence["speaker"],
            "overlap": bool(sentence.get("overlap", False)),
            "text": sentence["text"],
            "utterance": None if found is None else found.get("id", ""),
            "utterance_speaker": None if found is None else found["speaker"],
            "utterance_text": None if found is None else found["text"],
            "match": None if found is None
            else found["speaker"] == sentence["speaker"],
        })
    return rows


def share(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Count the right, wrong and missed sentences of ``rows``."""
    right = sum(1 for r in rows if r["match"] is True)
    wrong = sum(1 for r in rows if r["match"] is False)
    missed = sum(1 for r in rows if r["match"] is None)
    return {
        "sentences": len(rows),
        "right": right,
        "wrong": wrong,
        "missed": missed,
        "accuracy": right / len(rows) if rows else 0.0,
    }


def cut_into(reference: Sequence[Segment]) -> List[int]:
    """Return the indexes of the sentences an overlapping one starts over.

    For each sentence marked ``overlap``, the sentence before it in the
    reference order (the one it is mixed over).
    """
    return [i - 1 for i, s in enumerate(reference)
            if s.get("overlap") and i > 0]


def time_attribution(reference: Sequence[Segment],
                     utterances: Sequence[Segment]) -> float:
    """Return the share of reference speech time with the right speaker.

    For every reference sentence, the seconds shared with utterances of
    the same speaker (capped at the sentence's length), over the total
    length of the sentences.
    """
    total = 0.0
    right = 0.0
    for sentence in reference:
        length = float(sentence["end"]) - float(sentence["start"])
        total += length
        same = sum(overlap(sentence, u) for u in utterances
                   if u["speaker"] == sentence["speaker"])
        right += min(same, length)
    return right / total if total else 0.0


def score(reference: Sequence[Segment],
          utterances: Sequence[Segment]) -> Dict[str, Any]:
    """Return every figure of the module docstring for one transcript."""
    rows = attribution(reference, utterances)
    speakers = sorted({str(s["speaker"]) for s in reference})
    interrupted = cut_into(reference)
    return {
        "words": word_error(reference, utterances),
        "utterances": len(utterances),
        "transcript_speakers": sorted({str(u["speaker"])
                                       for u in utterances}),
        "attribution": share(rows),
        "by_speaker": {p: share([r for r in rows if r["speaker"] == p])
                       for p in speakers},
        "overlapping": share([r for r in rows if r["overlap"]]),
        "cut_into": share([rows[i] for i in interrupted]),
        "time_attribution": time_attribution(reference, utterances),
        "sentences": rows,
    }


def _by_start(segments: Sequence[Segment]) -> List[Segment]:
    """Return ``segments`` sorted by start time (stable)."""
    return sorted(segments, key=lambda s: float(s["start"]))


def _summary(result: Mapping[str, Any]) -> List[str]:
    """Return the printed lines of a score."""
    w = result["words"]
    lines = [
        f"words: WER {w['wer']:.3f} ({w['substitutions']} sub,"
        f" {w['deletions']} del, {w['insertions']} ins;"
        f" {w['reference_words']} reference,"
        f" {w['hypothesis_words']} transcribed)",
        f"utterances: {result['utterances']}, speakers"
        f" {', '.join(result['transcript_speakers'])}",
        f"time with the right speaker: {result['time_attribution']:.3f}",
    ]
    labelled: List[Tuple[str, Mapping[str, Any]]] = [
        ("all sentences", result["attribution"]),
        *[(p, s) for p, s in result["by_speaker"].items()],
        ("overlapping starts", result["overlapping"]),
        ("sentences cut into", result["cut_into"]),
    ]
    for label, s in labelled:
        lines.append(
            f"{label}: {s['right']}/{s['sentences']} right speaker"
            f" ({s['accuracy']:.3f}), {s['wrong']} wrong,"
            f" {s['missed']} missed"
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Score a transcript against a rendered reference and print it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--json", type=Path,
                        help="also write every figure and row here")
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    result = score(reference["utterances"], transcript["utterances"])
    for line in _summary(result):
        print(line)
    if args.json is not None:
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
