"""Latency and accuracy of the live recogniser on a Piper-rendered session.

Reads the ``live.event.v1`` JSON lines written by
``python -m transcriber live-file <wav> --out <events.jsonl>`` and:

* the p50, p90 and maximum of ``endpoint_ms``, ``asr_ms`` and
  ``lag_ms`` over the finals (one value per segment: every sentence of
  a segment carries the same timings), and the partials' ``lag_ms``;
* the word error rate of the finals (their text in time order) against
  the batch transcript of the same audio (``transcript.v1``, an
  agreement measure: the batch recogniser is another model, not the
  truth) and, when given, against the text Piper read (the script).

Usage::

    PYTHONPATH=. python experiments/live/live_asr_latency.py EVENTS BATCH \\
        [--reference SCRIPT_REFERENCE] [--out RESULTS.json]

Text is normalised as the transcriber's lab does
(:func:`transcriber.domain.comparison.normalise_text`).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transcriber.domain.comparison import normalise_text, word_error
from transcriber.domain.live import percentile


def stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Return n, p50, p90 and max of ``values`` (ms, one decimal)."""
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 1)

    return {
        "n": len(values), "p50": r(percentile(values, 0.5)),
        "p90": r(percentile(values, 0.9)),
        "max": r(max(values) if values else None),
    }


def tokens(texts: Sequence[str]) -> List[str]:
    """Return the normalised tokens of ``texts`` joined."""
    return normalise_text(" ".join(texts))


def utterance_texts(path: Path) -> List[str]:
    """Return the utterance texts of a JSON file, in time order."""
    data = json.loads(path.read_text("utf-8"))
    utterances = sorted(data["utterances"], key=lambda u: u["start"])
    return [u["text"] for u in utterances]


def measure(
    events: List[Dict[str, Any]], batch: Path, reference: Optional[Path]
) -> Dict[str, Any]:
    """Return the results of one run."""
    finals = [e for e in events if e["type"] == "final"]
    partials = [e for e in events if e["type"] == "partial"]
    segments: Dict[float, Dict[str, float]] = {}
    for event in finals:  # one entry per segment (shared timings)
        segments.setdefault(event["stream_time"], event["timings"])
    timings = list(segments.values())
    live_text = [
        e["utterance"]["text"]
        for e in sorted(finals, key=lambda e: e["utterance"]["start"])
    ]
    hyp = tokens(live_text)
    out: Dict[str, Any] = {
        "finals": len(finals), "segments": len(segments),
        "partials": len(partials),
        "endpoint_ms": stats([t["endpoint_ms"] for t in timings]),
        "asr_ms": stats([t["asr_ms"] for t in timings]),
        "lag_ms": stats([t["lag_ms"] for t in timings]),
        "partial_lag_ms": stats([
            e["timings"]["lag_ms"] for e in partials
            if "lag_ms" in e["timings"]
        ]),
        "live_tokens": len(hyp),
    }
    batch_tokens = tokens(utterance_texts(batch))
    wer = word_error(batch_tokens, hyp)
    out["wer_vs_batch"] = None if wer is None else round(wer, 4)
    out["batch_tokens"] = len(batch_tokens)
    if reference is not None:
        script = tokens(utterance_texts(reference))
        wer = word_error(script, hyp)
        out["wer_vs_script"] = None if wer is None else round(wer, 4)
        batch_wer = word_error(script, batch_tokens)
        out["batch_wer_vs_script"] = (
            None if batch_wer is None else round(batch_wer, 4)
        )
    return out


def main() -> None:
    """Measure one run and print (and write) the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("events", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--note", default="", help="stored with the results")
    args = parser.parse_args()
    events = [
        json.loads(line)
        for line in args.events.read_text("utf-8").splitlines() if line
    ]
    results = measure(events, args.batch, args.reference)
    if args.note:
        results["note"] = args.note
    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", "utf-8")


if __name__ == "__main__":
    main()
