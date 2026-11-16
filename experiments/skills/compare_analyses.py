"""Compare two analyses of the same conversation (#96).

Stand-alone script (nothing of the project is imported)::

    python experiments/skills/compare_analyses.py \\
        data/output/skills/analysis/sim-reunion-verano \\
        data/output/skills/analysis/sim-reunion-verano-audio [--json out.json]

Each argument is a directory ``skills analyze`` wrote. Built for the
simulated discussion analysed twice: on its script and on what the
transcriber made of the rendered audio. People are matched by name. It
reports, per person: words, turns, firm levels, provisional levels and
observations; every competency whose firm or provisional level differs;
and how much of the evidence is shared (the same indicator with a quote
of the same words) and how many confirmed tags each run kept. It also
compares the counts of the conversation events.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

WORD = re.compile(r"[^\W_]+", re.UNICODE)
"""A run of letters or digits: a word once punctuation is dropped."""

Evidence = Tuple[str, str]
"""(indicator id, normalised quote) of one piece of evidence."""


def normalise(text: str) -> str:
    """Return ``text`` lower-cased, without punctuation, single-spaced."""
    lowered = unicodedata.normalize("NFC", text).lower()
    return " ".join(WORD.findall(lowered))


def read_run(directory: Path) -> Dict[str, Any]:
    """Return the summary, assessment, confirmation and events of a run."""
    def load(name: str) -> Any:
        """Parse one JSON file of the run (``{}`` when it is missing)."""
        path = directory / name
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    return {
        "summary": load("summary.json"),
        "assessment": load("assessment.json"),
        "confirmation": load("confirmation.json"),
        "events": load("events.json"),
    }


def levels(person: Mapping[str, Any]) -> Dict[str, Tuple[str, Any]]:
    """Return competency → (``firm`` | ``provisional`` | ``none``, level).

    ``person`` is one entry of ``summary.json``'s ``people``.
    """
    out: Dict[str, Tuple[str, Any]] = {}
    estimates = person.get("estimates") or {}
    for competency, rating in (person.get("ratings") or {}).items():
        if rating is not None:
            out[competency] = ("firm", rating)
        elif competency in estimates:
            out[competency] = ("provisional",
                               estimates[competency].get("rating"))
        else:
            out[competency] = ("none", None)
    return out


def evidence(assessment: Mapping[str, Any]) -> Dict[str, Set[Evidence]]:
    """Return participant → the evidence of every competency, normalised."""
    out: Dict[str, Set[Evidence]] = {}
    for card in assessment.get("scorecards", []):
        found = out.setdefault(str(card["participant"]), set())
        for item in card.get("assessments", []):
            for piece in item.get("evidence", []):
                found.add((str(piece.get("indicator_id", "")),
                           normalise(str(piece.get("quote", "")))))
    return out


def confirmed(confirmation: Mapping[str, Any]) -> Dict[str, Set[Evidence]]:
    """Return speaker → (indicator, sentence) of every tag the judge kept.

    A sentence counts when its ``decision`` is ``confirmed``; the
    indicator is the one the decision names.
    """
    out: Dict[str, Set[Evidence]] = {}
    for item in confirmation.get("sentences", []):
        sentence = item["sentence"]
        kept = out.setdefault(str(sentence["speaker"]), set())
        decision = item.get("decision") or {}
        if decision.get("status") == "confirmed":
            kept.add((str(decision.get("indicator_id", "")),
                      normalise(str(sentence["text"]))))
    return out


def event_counts(events: Mapping[str, Any]) -> Dict[str, int]:
    """Return the number of events of each kind."""
    counts: Dict[str, int] = {}
    for event in events.get("events", []):
        kind = str(event.get("type", ""))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def compare(first: Mapping[str, Any],
            second: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the differences between two runs read by :func:`read_run`."""
    people_a = {p["name"]: p for p in first["summary"].get("people", [])}
    people_b = {p["name"]: p for p in second["summary"].get("people", [])}
    evidence_a = evidence(first["assessment"])
    evidence_b = evidence(second["assessment"])
    tags_a = confirmed(first["confirmation"])
    tags_b = confirmed(second["confirmation"])
    people = []
    for name in sorted(set(people_a) | set(people_b)):
        a = people_a.get(name, {})
        b = people_b.get(name, {})
        levels_a = levels(a)
        levels_b = levels(b)
        changes = [
            {"competency": c,
             "first": list(levels_a.get(c, ("none", None))),
             "second": list(levels_b.get(c, ("none", None)))}
            for c in sorted(set(levels_a) | set(levels_b))
            if levels_a.get(c, ("none", None))
            != levels_b.get(c, ("none", None))
        ]
        ev_a = evidence_a.get(name, set())
        ev_b = evidence_b.get(name, set())
        kept_a = tags_a.get(name, set())
        kept_b = tags_b.get(name, set())
        people.append({
            "name": name,
            "counts": {key: [a.get(key), b.get(key)] for key in (
                "words", "rated", "provisional", "observations")},
            "changes": changes,
            "evidence": [len(ev_a), len(ev_b), len(ev_a & ev_b)],
            "evidence_indicators": [
                len({i for i, _ in ev_a}), len({i for i, _ in ev_b}),
                len({i for i, _ in ev_a} & {i for i, _ in ev_b})],
            "confirmed": [len(kept_a), len(kept_b), len(kept_a & kept_b)],
        })
    events_a = event_counts(first["events"])
    events_b = event_counts(second["events"])
    return {
        "people": people,
        "events": {k: [events_a.get(k, 0), events_b.get(k, 0)]
                   for k in sorted(set(events_a) | set(events_b))},
    }


def _summary(result: Mapping[str, Any]) -> List[str]:
    """Return the printed lines of a comparison."""
    lines = []
    for p in result["people"]:
        counts = ", ".join(f"{k} {v[0]}→{v[1]}"
                           for k, v in p["counts"].items())
        ev = p["evidence"]
        ind = p["evidence_indicators"]
        tags = p["confirmed"]
        lines.append(f"{p['name']}: {counts}")
        lines.append(
            f"  evidence {ev[0]}→{ev[1]} ({ev[2]} the same);"
            f" indicators {ind[0]}→{ind[1]} ({ind[2]} shared);"
            f" confirmed tags {tags[0]}→{tags[1]} ({tags[2]} the same)")
        for change in p["changes"]:
            lines.append(
                f"  {change['competency']}: {change['first'][0]}"
                f" {change['first'][1]} → {change['second'][0]}"
                f" {change['second'][1]}")
    events = ", ".join(f"{k} {v[0]}→{v[1]}"
                       for k, v in result["events"].items())
    lines.append(f"events: {events}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compare two analysis directories and print the differences."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--json", type=Path,
                        help="also write the comparison here")
    args = parser.parse_args(argv)
    result = compare(read_run(args.first), read_run(args.second))
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
