"""Streaming against batch: word error rate and speaker accuracy (LV-13).

Compares the transcript a live session handed over (``live/<id>/
transcript.json``) and the batch pipeline's transcript (``transcriber
import --wait`` of the mono mix) with the Piper script::

    PYTHONPATH=. python experiments/live/stream_vs_batch.py LIVE BATCH \\
        [--reference REFERENCE.json] [--max-seconds N] \\
        [--live-map S1=agente,S2=cliente] --out RESULT.json

* ``wer_live_vs_script``, ``wer_batch_vs_script``: against the text
  Piper read (the reference's sentences in time order); ``wer_live_vs_batch``
  is agreement of the two recognisers. Without ``--reference`` only the
  last one is given (a corpus without text).
* speaker accuracy, per hypothesis sentence: the reference sentence it
  overlaps most in time must have the sentence's speaker. The live
  labels are mapped by ``--live-map`` (channel ``n`` is speaker ``n`` of
  the product); the batch labels get the best one-to-one mapping to the
  reference speakers (the batch diarizer names people anonymously).
  Reported by sentence and weighted by words.

``--max-seconds`` keeps only what starts before that time (a stream
that was cut short).
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transcriber.domain.comparison import normalise_text, word_error


def load(path: Path, max_seconds: Optional[float]) -> List[Dict[str, Any]]:
    """Return the utterances of a transcript or a reference."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return [u for u in document["utterances"]
            if max_seconds is None or u["start"] < max_seconds]


def tokens(utterances: Sequence[Dict[str, Any]]) -> List[str]:
    """Return the normalised words of ``utterances`` in time order."""
    ordered = sorted(utterances, key=lambda u: (u["start"], u["end"]))
    return normalise_text(" ".join(u["text"] for u in ordered))


def rounded(value: Optional[float]) -> Optional[float]:
    """Round a rate to four decimals."""
    return None if value is None else round(value, 4)


def best_reference(
    utterance: Dict[str, Any], reference: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the reference sentence sharing most time with ``utterance``."""
    best, best_share = None, 0.0
    for sentence in reference:
        share = (min(utterance["end"], sentence["end"])
                 - max(utterance["start"], sentence["start"]))
        if share > best_share:
            best, best_share = sentence, share
    return best


def speaker_accuracy(
    hypothesis: Sequence[Dict[str, Any]],
    reference: Sequence[Dict[str, Any]], mapping: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Return the share of sentences (and words) given to the right person.

    Without ``mapping`` the best one-to-one mapping of the hypothesis
    labels onto the reference speakers is taken.
    """
    pairs = []
    for u in hypothesis:
        sentence = best_reference(u, reference)
        if sentence is not None:
            pairs.append((u, sentence["speaker"]))
    labels = sorted({u["speaker"] for u, _ in pairs})
    people = sorted({s for _, s in pairs})

    def score(choice: Dict[str, str]) -> int:
        return sum(choice.get(u["speaker"]) == who for u, who in pairs)

    if mapping is None:
        mapping = max(
            (dict(zip(labels, picked))
             for picked in itertools.permutations(people, len(labels))),
            key=score, default={},
        )
    right = [choice == mapping.get(u["speaker"]) for u, choice in
             ((u, who) for u, who in pairs)]
    words = [len(normalise_text(u["text"])) for u, _ in pairs]
    return {
        "sentences": len(pairs), "mapping": mapping,
        "by_sentence": rounded(sum(right) / len(right)) if right else None,
        "by_words": rounded(
            sum(w for w, ok in zip(words, right) if ok) / sum(words)
        ) if sum(words) else None,
    }


def main() -> None:
    """Compare the two transcripts and write the result."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("live", type=Path)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--live-map", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    live = load(args.live, args.max_seconds)
    batch = load(args.batch, args.max_seconds)
    out: Dict[str, Any] = {
        "live_sentences": len(live), "batch_sentences": len(batch),
        "live_tokens": len(tokens(live)), "batch_tokens": len(tokens(batch)),
        "wer_live_vs_batch": rounded(word_error(tokens(batch), tokens(live))),
        "max_seconds": args.max_seconds,
    }
    if args.reference is not None:
        script = load(args.reference, args.max_seconds)
        out["script_tokens"] = len(tokens(script))
        out["wer_live_vs_script"] = rounded(
            word_error(tokens(script), tokens(live)))
        out["wer_batch_vs_script"] = rounded(
            word_error(tokens(script), tokens(batch)))
        mapping = dict(p.split("=") for p in args.live_map.split(",") if p)
        out["speaker_live"] = speaker_accuracy(live, script, mapping or None)
        out["speaker_batch"] = speaker_accuracy(batch, script, None)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
