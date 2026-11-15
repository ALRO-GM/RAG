"""Build the data file of the study page (``web/study/data.json``).

``python -m transcriber.adapters.inbound.study_data`` reads the
committed experiment outputs and writes the one JSON file the page
``study.html`` draws from, so the page never holds a number of its own:

- ``experiments/transcriber/results/reference_summary_<corpus>.json``:
  the #44 summaries (``lab reference-summary --corpus X --json``), one
  row per denoiser + diarizer, DER and its parts averaged over the
  recordings (``docs/transcriber/LAB.md`` §13.8);
- ``lab/catalogue.json``: each model's licence (non-commercial flag);
- ``experiments/skills/results/judge_local_qwen3-8b-4bit.json``: the
  Qwen3-8B judge (``docs/skills/DESIGN.md`` §4.2);
- ``experiments/skills/results/conversation_events_eval.json``: the
  conversation-event rules against the expected events (#83);
- optional, read when they exist (until then the page says "pending"):
  ``experiments/transcriber/results/diarizer_switch.json`` (#68:
  ``{"corpora": {"ami": {"der": 0.09, "n": 4}, ...}}``) and
  ``experiments/skills/results/act_tagger_*.json`` (#82: one file per
  approach, ``{"approach": "rules", "items": 1534,
  "top1_accuracy": 0.41}``).

A few figures were measured but have no result file (the Qwen3-1.7B and
4B judges, the Claude Code smoke test, the agreement figures of the
owner's recording, the size of the Spanish dialogue-act set). They are
listed in :data:`DOCUMENTED` with the sentence of the document that
states them; the build fails if that sentence is no longer in that
section, so a figure cannot drift from its source unnoticed.

Every figure carries the id of its source, which the data file maps to
the document, the section and its address on GitHub.

Standard library only, like :mod:`.bundle`.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
"""The repository root."""

OUT = Path(__file__).resolve().parent / "web" / "study" / "data.json"
"""Where the page reads its data."""

REPO_URL = "https://github.com/ALRO-GM/RAG/blob/main/"
"""Where the documents are published (the public repository)."""

ISSUES_URL = "https://github.com/ALRO-GM/RAG/issues/"
"""Where the issues the page names live."""

ISSUES = (68, 82, 83)
"""The issues the page links to (work still in progress or measured)."""

SCHEMA = "study.v1"

CORPORA = ("ami", "callhome")
"""The corpora of #44, in the page's order."""

BUILTIN = "builtin"
"""The transcriber's own diarizer: the system the study compares with."""

NO_DENOISER = "none"

SHORT_NAMES = {
    "builtin": "Built-in",
    "pyannote-3.1": "pyannote 3.1",
    "pyannote-community-1": "pyannote community-1",
    "sortformer-streaming-v2": "Sortformer v2",
    "nemotron-3-diarization": "Nemotron-3",
    "diarizen-wavlm-large-s80-md-v2": "DiariZen",
    "ecapa-spectral": "ECAPA",
    "nemo-clustering": "NeMo clustering",
    "deepfilternet3": "DeepFilterNet3",
    "mossformer2-se-48k": "MossFormer2",
}
"""Short model names that fit a phone; the ids stay in the data."""


@dataclass(frozen=True)
class Source:
    """A section of a document that states figures."""

    path: str
    heading: str
    """The heading's start, as written (``"13.8.1 AMI"``)."""


SOURCES: Dict[str, Source] = {
    "lab_r1": Source("docs/transcriber/LAB.md", "11. Results on R1"),
    "lab_r1_denoise": Source("docs/transcriber/LAB.md", "11.5 Effect"),
    "lab_r1_full": Source("docs/transcriber/LAB.md", "11.7 Full run"),
    "lab_reference": Source("docs/transcriber/LAB.md", "13. Scoring"),
    "lab_scores": Source("docs/transcriber/LAB.md", "13.2 Scores"),
    "lab_no_speech": Source("docs/transcriber/LAB.md", "13.6 Recordings"),
    "lab_invalid": Source("docs/transcriber/LAB.md", "13.7 Rows"),
    "lab_results": Source("docs/transcriber/LAB.md", "13.8 Results"),
    "lab_ami": Source("docs/transcriber/LAB.md", "13.8.1 AMI"),
    "lab_callhome": Source("docs/transcriber/LAB.md", "13.8.2 CALLHOME"),
    "lab_denoiser": Source("docs/transcriber/LAB.md", "13.8.3 Effect"),
    "lab_licences": Source("docs/transcriber/LAB.md", "13.8.4 Licence"),
    "lab_findings": Source("docs/transcriber/LAB.md", "13.8.5 Findings"),
    "lab_against_r1": Source("docs/transcriber/LAB.md", "13.8.6 Against"),
    "lab_limits": Source("docs/transcriber/LAB.md", "13.8.7 Limits"),
    "acts_set": Source("examples/skills/dialogue_acts/README.md",
                       "Contents"),
    "judges_local": Source("docs/skills/DESIGN.md", "4.2 Local judge"),
    "judges_both": Source("docs/skills/DESIGN.md", "4.4 Both mode"),
    "events": Source("docs/skills/CONVERSATION_EVENTS.md", "6. Evaluation"),
}
"""Every section a figure of the page comes from."""


@dataclass(frozen=True)
class Documented:
    """A measured figure without a result file, and where it is stated."""

    key: str
    value: float
    source: str
    quote: str
    """Text of the section that states the value (whitespace folded)."""


DOCUMENTED: Tuple[Documented, ...] = (
    Documented("r1.seconds", 388, "lab_r1", "(388 s, two enrolled people"),
    Documented("r1.people", 2, "lab_r1", "(388 s, two enrolled people"),
    Documented("r1.pipelines", 224, "lab_r1_full", "= **224 pipelines**"),
    Documented("r1.mean_agreement", 0.861, "lab_r1",
               "the agreement has a mean of 0.861"),
    Documented("r1.best_agreement", 0.923, "lab_r1_full",
               "`mossformer2-se-48k` with pyannote-3.1 (0.923)"),
    Documented("speakers.ami", 4, "lab_results",
               "| Speakers in the reference | 4 | 2 |"),
    Documented("speakers.callhome", 2, "lab_results",
               "| Speakers in the reference | 4 | 2 |"),
    Documented("invalid.callhome", 1, "lab_results",
               "`0012` (invalid reference, §13.7)"),
    Documented("acts.dialogues", 51, "acts_set",
               "51 dialogues, 1,534 sentences"),
    Documented("acts.sentences", 1534, "acts_set",
               "51 dialogues, 1,534 sentences"),
    Documented("judge.qwen3-1.7b.accuracy", 0.365, "judges_local",
               "| Qwen3-1.7B, 16 bit | 3.54 GB | 4.8 s | 0.365 (3) |"),
    Documented("judge.qwen3-1.7b.confirmed", 40, "judges_local",
               "| 0.365 (3) | 0.16 | 40 / 23 / 1 |"),
    Documented("judge.qwen3-1.7b.rejected", 23, "judges_local",
               "| 0.365 (3) | 0.16 | 40 / 23 / 1 |"),
    Documented("judge.qwen3-4b.accuracy", 0.545, "judges_local",
               "| Qwen3-4B, 4 bit | 2.87 GB | 39.1 s | 0.545 (1) |"),
    Documented("judge.qwen3-4b.confirmed", 63, "judges_local",
               "| 0.545 (1) | 0.32 | 63 / 1 / 0 |"),
    Documented("judge.qwen3-4b.rejected", 1, "judges_local",
               "| 0.545 (1) | 0.32 | 63 / 1 / 0 |"),
    Documented("judge.qwen3-8b.accuracy", 0.62, "judges_local",
               "| Qwen3-8B, 4 bit | 6.31 GB | 19.7 s | 0.620 (0) |"),
    Documented("judge.acts_items", 200, "judges_local",
               "Dialogue acts: 200 sentences"),
    Documented("judge.claude.right", 14, "judges_both",
               "Claude 14 of 20 right (0.70)"),
    Documented("judge.claude.items", 20, "judges_both",
               "Claude 14 of 20 right (0.70)"),
)
"""Measured figures stated only in a document (see the module doc)."""


class StudyDataError(ValueError):
    """The inputs do not hold what the page needs."""


# ---- documents ------------------------------------------------------------


def slug(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def find_section(text: str, heading: str) -> Tuple[str, str]:
    """Return the full heading starting with ``heading`` and its body.

    The body runs to the next heading of the same or a higher level.
    Raise :class:`StudyDataError` when there is no such heading.
    """
    lines = text.splitlines()
    headings: List[Tuple[int, int, str]] = []
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        match = re.match(r"^(#+)\s+(.*)$", line)
        if match and not fenced:
            headings.append((index, len(match.group(1)), match.group(2)))
    for position, (index, level, title) in enumerate(headings):
        if not title.startswith(heading):
            continue
        end = next((i for i, lv, _ in headings[position + 1:] if lv <= level),
                   len(lines))
        return title.strip(), "\n".join(lines[index + 1:end])
    raise StudyDataError(f"No heading starting with {heading!r}.")


def fold(text: str) -> str:
    """Collapse runs of whitespace to one space."""
    return " ".join(text.split())


def resolve_sources(
    documents: Mapping[str, str],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Return the page's source table and each source's folded body."""
    table: Dict[str, Dict[str, str]] = {}
    bodies: Dict[str, str] = {}
    for key, source in SOURCES.items():
        if source.path not in documents:
            raise StudyDataError(f"Missing document {source.path}.")
        heading, body = find_section(documents[source.path], source.heading)
        table[key] = {
            "path": source.path,
            "section": heading,
            "url": f"{REPO_URL}{source.path}#{slug(heading)}",
        }
        bodies[key] = fold(body)
    return table, bodies


def documented_figures(bodies: Mapping[str, str]) -> Dict[str, float]:
    """The figures of :data:`DOCUMENTED`, checked against their section."""
    out: Dict[str, float] = {}
    for item in DOCUMENTED:
        if fold(item.quote) not in bodies[item.source]:
            raise StudyDataError(
                f"{item.key}: {item.quote!r} is no longer in "
                f"{SOURCES[item.source].path} {SOURCES[item.source].heading}."
            )
        out[item.key] = item.value
    return out


# ---- separating: who speaks when -----------------------------------------


def _mean(metrics: Mapping[str, Any], name: str) -> Optional[float]:
    """The mean of one metric of a summary row, if it has one."""
    value = metrics.get(name, {}).get("mean")
    return None if value is None else float(value)


def gain(best_der: float, der: float) -> Dict[str, Optional[float]]:
    """How much better ``best_der`` is than ``der``.

    ``points``: the difference in success (1 − DER), as a fraction;
    ``error_reduction``: the share of ``der`` the best one avoids
    (``None`` when ``der`` is 0).
    """
    return {
        "points": round(der - best_der, 4),
        "error_reduction": (
            round((der - best_der) / der, 4) if der > 0 else None
        ),
    }


def licences(catalogue: Mapping[str, Any]) -> Dict[str, bool]:
    """Model id -> whether its licence allows commercial use."""
    return {
        str(entry["id"]): bool(entry.get("licence_commercial", True))
        for entry in catalogue.get("entries", [])
    }


def corpus_rows(
    summary: Mapping[str, Any], commercial: Mapping[str, bool],
) -> List[Dict[str, Any]]:
    """One row per denoiser + diarizer, best (lowest DER) first."""
    rows: List[Dict[str, Any]] = []
    for model in summary.get("models", []):
        if model.get("kind") != "diarization":
            continue
        denoiser, _, diarizer = str(model["id"]).partition("+")
        metrics = model["metrics"]
        der = _mean(metrics, "der")
        if der is None or not diarizer:
            continue
        rows.append({
            "id": model["id"],
            "denoiser": denoiser,
            "diarizer": diarizer,
            "denoiser_name": SHORT_NAMES.get(denoiser, denoiser),
            "diarizer_name": SHORT_NAMES.get(diarizer, diarizer),
            "non_commercial": not (
                commercial.get(denoiser, True)
                and commercial.get(diarizer, True)
            ),
            "n": int(model.get("recordings", 0)),
            "der": der,
            "der_sd": metrics.get("der", {}).get("sd"),
            "success": round(1 - der, 4),
            "missed": _mean(metrics, "missed_rate"),
            "false_alarm": _mean(metrics, "false_alarm_rate"),
            "confusion": _mean(metrics, "confusion_rate"),
            "jer": _mean(metrics, "jer"),
            "overlap_precision": _mean(metrics, "overlap_precision"),
            "overlap_recall": _mean(metrics, "overlap_recall"),
            "speaker_error": _mean(metrics, "speaker_count_error"),
            "speaker_abs_error": _mean(metrics, "speaker_count_abs_error"),
        })
    rows.sort(key=lambda r: (r["der"], r["id"]))
    if not rows:
        raise StudyDataError("A summary has no diarization rows.")
    best = rows[0]["der"]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["best"] = rank == 1
        row["builtin"] = (
            row["diarizer"] == BUILTIN and row["denoiser"] == NO_DENOISER
        )
        row["gain"] = gain(best, row["der"])
    return rows


def headline(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The best row against the built-in, the next best and the best
    commercial option."""
    best = rows[0]

    def against(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        return {"id": row["id"], "success": row["success"],
                **gain(best["der"], row["der"])}

    builtin = next((r for r in rows if r["builtin"]), None)
    commercial = next((r for r in rows if not r["non_commercial"]), None)
    return {
        "best": best["id"],
        "success": best["success"],
        "builtin": against(builtin),
        "next": against(rows[1] if len(rows) > 1 else None),
        "best_commercial": (
            None if commercial is None or commercial is best
            else against(commercial)
        ),
    }


def denoiser_effect(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Mean success over the diarizers, per denoiser, best first."""
    groups: Dict[str, List[float]] = {}
    for row in rows:
        groups.setdefault(row["denoiser"], []).append(row["der"])
    effect = [
        {"denoiser": name, "denoiser_name": SHORT_NAMES.get(name, name),
         "diarizers": len(ders),
         "success": round(1 - sum(ders) / len(ders), 4)}
        for name, ders in groups.items()
    ]
    return sorted(effect, key=lambda e: (-e["success"], e["denoiser"]))


def corpus_block(
    corpus: str,
    summary: Mapping[str, Any],
    commercial: Mapping[str, bool],
    figures: Mapping[str, float],
) -> Dict[str, Any]:
    """What the page shows for one corpus."""
    rows = corpus_rows(summary, commercial)
    recordings = summary.get("recordings", [])
    licence = sorted({str(r.get("licence")) for r in recordings
                      if r.get("licence")})
    return {
        "id": corpus,
        "source": f"lab_{corpus}",
        "recordings": len(recordings),
        "seconds": round(sum(float(r.get("duration", 0))
                             for r in recordings)),
        "speakers": int(figures[f"speakers.{corpus}"]),
        "licence": ", ".join(licence),
        "non_commercial": bool(summary.get("non_commercial", False)),
        "excluded_no_speech": sum(
            1 for e in summary.get("excluded", [])
            if e.get("reason") == "no speech in the audio"
        ),
        "excluded_invalid": int(figures.get(f"invalid.{corpus}", 0)),
        "combinations": len(rows),
        "headline": headline(rows),
        "denoisers": denoiser_effect(rows),
        "rows": rows,
    }


def switch_block(switch: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The rule of #68: pending until its measured result file exists."""
    if switch is None:
        return {"issue": 68, "status": "pending", "corpora": {}}
    corpora = {
        str(name): {"der": float(v["der"]), "success": round(1 - v["der"], 4),
                    "n": int(v.get("n", 0))}
        for name, v in switch.get("corpora", {}).items()
    }
    return {"issue": 68, "status": "done", "corpora": corpora}


# ---- labelling ------------------------------------------------------------


def judges(
    figures: Mapping[str, float], qwen8b: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """The dialogue-act judges, in the order they were measured."""
    items = int(figures["judge.acts_items"])
    if int(qwen8b["acts_items"]) != items:
        raise StudyDataError("The 8B result file and DESIGN.md §4.2 "
                             "disagree on the number of sentences.")
    if abs(float(qwen8b["acts_accuracy"])
           - figures["judge.qwen3-8b.accuracy"]) > 1e-9:
        raise StudyDataError("The 8B result file and DESIGN.md §4.2 "
                             "disagree on the accuracy.")
    out: List[Dict[str, Any]] = []
    for model, bits, key in (("Qwen3-1.7B", 16, "qwen3-1.7b"),
                             ("Qwen3-4B", 4, "qwen3-4b")):
        out.append({
            "model": model, "bits": bits, "kind": "local",
            "items": items, "accuracy": figures[f"judge.{key}.accuracy"],
            "tags_confirmed": int(figures[f"judge.{key}.confirmed"]),
            "tags_rejected": int(figures[f"judge.{key}.rejected"]),
            "small_sample": False, "source": "judges_local",
        })
    out.append({
        "model": "Qwen3-8B", "bits": int(qwen8b["bits"]), "kind": "local",
        "items": int(qwen8b["acts_items"]),
        "accuracy": float(qwen8b["acts_accuracy"]),
        "tags_confirmed": int(qwen8b["tags_confirmed"]),
        "tags_rejected": int(qwen8b["tags_rejected"]),
        "small_sample": False, "source": "judges_local",
    })
    right = figures["judge.claude.right"]
    total = figures["judge.claude.items"]
    out.append({
        "model": "Claude Code (opus)", "bits": None, "kind": "hosted",
        "items": int(total), "accuracy": round(right / total, 4),
        "tags_confirmed": None, "tags_rejected": None,
        "small_sample": True, "source": "judges_both",
    })
    return out


def tagger_block(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The approaches of #82: pending until their result files exist."""
    approaches: List[Dict[str, Any]] = [
        {"approach": str(r["approach"]), "items": int(r.get("items", 0)),
         "accuracy": float(r["top1_accuracy"])} for r in results
    ]
    approaches.sort(key=lambda a: (-a["accuracy"], a["approach"]))
    return {"issue": 82, "status": "done" if approaches else "pending",
            "approaches": approaches}


def _ratio(num: int, den: int) -> Optional[float]:
    """``num / den`` rounded, or ``None`` when ``den`` is 0."""
    return round(num / den, 4) if den else None


def events_block(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    """Precision and recall of the conversation events (#83)."""
    modes: Dict[str, Any] = {}
    for mode in ("given_links", "inferred_links"):
        rows = [
            {"type": r["type"], "expected": int(r["expected"]),
             "derived": int(r["derived"]), "matched": int(r["matched"]),
             "precision": r["precision"], "recall": r["recall"]}
            for r in evaluation.get(mode, [])
        ]
        matched = sum(r["matched"] for r in rows)
        modes[mode] = {
            "rows": rows,
            "expected": sum(r["expected"] for r in rows),
            "derived": sum(r["derived"] for r in rows),
            "precision": _ratio(matched, sum(r["derived"] for r in rows)),
            "recall": _ratio(matched, sum(r["expected"] for r in rows)),
        }
    return {"issue": 83, "source": "events",
            "dialogues": len(evaluation.get("dialogues", [])), **modes}


# ---- the whole file -------------------------------------------------------


@dataclass(frozen=True)
class StudyInputs:
    """Everything the page's data is built from, already read."""

    summaries: Mapping[str, Mapping[str, Any]]
    catalogue: Mapping[str, Any]
    qwen8b: Mapping[str, Any]
    events: Mapping[str, Any]
    documents: Mapping[str, str]
    switch: Optional[Mapping[str, Any]] = None
    tagger: Sequence[Mapping[str, Any]] = ()


def build_study(inputs: StudyInputs, updated: date) -> Dict[str, Any]:
    """The page's data from its inputs (pure)."""
    sources, bodies = resolve_sources(inputs.documents)
    figures = documented_figures(bodies)
    commercial = licences(inputs.catalogue)
    missing = [c for c in CORPORA if c not in inputs.summaries]
    if missing:
        raise StudyDataError(f"Missing summaries: {', '.join(missing)}.")
    corpora = [corpus_block(c, inputs.summaries[c], commercial, figures)
               for c in CORPORA]
    denoisers = sorted({r["denoiser"] for c in corpora for r in c["rows"]})
    diarizers = sorted({r["diarizer"] for c in corpora for r in c["rows"]})
    return {
        "schema": SCHEMA,
        "updated": updated.isoformat(),
        "sources": sources,
        "issues": {str(n): {"url": f"{ISSUES_URL}{n}"} for n in ISSUES},
        "bench": {
            "own_recording": {
                "seconds": int(figures["r1.seconds"]),
                "people": int(figures["r1.people"]),
                "pipelines": int(figures["r1.pipelines"]),
                "mean_agreement": figures["r1.mean_agreement"],
                "best_agreement": figures["r1.best_agreement"],
                "source": "lab_r1",
            },
            "acts_set": {
                "dialogues": int(figures["acts.dialogues"]),
                "sentences": int(figures["acts.sentences"]),
                "source": "acts_set",
            },
            "denoisers": len(denoisers),
            "diarizers": len(diarizers),
        },
        "separation": {
            "source": "lab_results",
            "corpora": corpora,
            "rule": switch_block(inputs.switch),
        },
        "labelling": {
            "judges": judges(figures, inputs.qwen8b),
            "tagger": tagger_block(inputs.tagger),
            "events": events_block(inputs.events),
        },
    }


def _read_json(path: Path) -> Any:
    """Parse one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(root: Path = ROOT) -> StudyInputs:
    """Read the inputs from the repository at ``root``."""
    transcriber = root / "experiments" / "transcriber" / "results"
    skills = root / "experiments" / "skills" / "results"
    switch_path = transcriber / "diarizer_switch.json"
    return StudyInputs(
        summaries={
            c: _read_json(transcriber / f"reference_summary_{c}.json")
            for c in CORPORA
        },
        catalogue=_read_json(root / "lab" / "catalogue.json"),
        qwen8b=_read_json(skills / "judge_local_qwen3-8b-4bit.json"),
        events=_read_json(skills / "conversation_events_eval.json"),
        documents={
            s.path: (root / s.path).read_text(encoding="utf-8")
            for s in SOURCES.values()
        },
        switch=_read_json(switch_path) if switch_path.is_file() else None,
        tagger=[_read_json(p)
                for p in sorted(skills.glob("act_tagger_*.json"))],
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the data file from the command line; return the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m transcriber.adapters.inbound.study_data",
        description="Build the study page's data from the results.",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--date", type=date.fromisoformat,
                        default=date.today(),
                        help="the 'last updated' date (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    try:
        data = build_study(load_inputs(args.root), args.date)
    except (OSError, ValueError, KeyError) as exc:
        print(f"study_data: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
