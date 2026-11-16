"""Diarization errors per reference speaker and speaker sex on AMI (#86).

Reuses the lab's stored outputs (no model is run)::

    python experiments/transcriber/speaker_groups.py \\
        data/transcriber/lab experiments/transcriber/results/ami_speaker_sex

For every imported ``ami-*`` lab session it takes the newest ``done``
or ``partial`` run, reads each finished diarization step's segments
from ``stages/<key>/outcome.json`` and splits the track's errors by
reference speaker (:func:`transcriber.domain.lab_reference.score_speakers`:
speech, missed, unattributed, with the lab's collar). The speaker's sex
is the first letter of the AMI participant id (``[MF][IET][EDO]nnn``:
https://groups.inf.ed.ac.uk/ami/corpus/participantids.shtml); it is the
corpus's own label, not inferred here.

Writes ``<out>.json`` (one row per session, track and speaker) and
``<out>.fairness.json`` (and ``<out>_mixed.fairness.json``, only
the meetings with both sexes), a ``fairness.results.v1`` document with one
mean table per diarizer on the original audio (unattributed and missed
share of each person's speech, lower is better, each person once) for
``experiments/skills/fairness_audit.py`` to render.
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from transcriber.domain.comparison import Track
from transcriber.domain.lab import Segment
from transcriber.domain.lab_reference import SessionReference, score_speakers

AMI_IDS = "https://groups.inf.ed.ac.uk/ami/corpus/participantids.shtml"
MIN_PEOPLE = 3
"""Smallest group given a ratio here: AMI's test meetings imported in
#44 have 4 women and 8 men, so the project's default of 5 would leave
the women's group described only."""


def sex_of(speaker: str) -> Optional[str]:
    """Return ``F`` or ``M`` from an AMI participant id, else ``None``."""
    first = speaker[:1]
    return first if first in ("F", "M") else None


def newest_run(session: Path) -> Optional[Dict[str, Any]]:
    """Return the newest finished run of a lab session, if any."""
    runs = []
    for path in (session / "runs").glob("*.json"):
        if path.name.endswith(".results.json"):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("state") in ("done", "partial"):
            runs.append(run)
    return max(runs, key=lambda r: str(r.get("created_at")), default=None)


def tracks_of(session: Path, run: Dict[str, Any]) -> List[Track]:
    """Return the run's finished diarizations as comparison tracks."""
    tracks: List[Track] = []
    for step in run.get("steps", []):
        if step.get("stage") != "diarize" or step.get("state") != "done":
            continue
        stored = session / "stages" / str(step["key"]) / "outcome.json"
        result = json.loads(stored.read_text(encoding="utf-8"))
        segments = [
            Segment(start=s["start"], end=s["end"],
                    speaker=f"~{s['speaker']}")
            for s in result.get("segments", []) if s["end"] > s["start"]
        ]
        track_id = f"{step['denoiser']}+{step['model']}"
        tracks.append(Track(track_id, segments, track_id))
    return tracks


def score_lab(lab: Path) -> List[Dict[str, Any]]:
    """Return one row per AMI session, track and reference speaker."""
    rows: List[Dict[str, Any]] = []
    for session in sorted(lab.glob("ami-*")):
        reference_file = session / "reference.json"
        run = newest_run(session)
        if run is None or not reference_file.exists():
            continue
        reference = SessionReference.model_validate_json(
            reference_file.read_text(encoding="utf-8")
        )
        if reference.excluded() is not None:
            continue
        for track in tracks_of(session, run):
            for score in score_speakers(reference, track):
                rows.append({
                    "session": session.name, "run": run["id"],
                    "track": track.id, "sex": sex_of(score.speaker),
                    **score.model_dump(),
                })
    return rows


def mixed_sessions(rows: List[Dict[str, Any]]) -> List[str]:
    """Return the sessions with speakers of both sexes."""
    sexes: Dict[str, set[str]] = {}
    for r in rows:
        if r["sex"] is not None:
            sexes.setdefault(r["session"], set()).add(r["sex"])
    return sorted(s for s, found in sexes.items() if len(found) == 2)


def fairness_document(
    rows: List[Dict[str, Any]], mixed_only: bool = False
) -> Dict[str, Any]:
    """Return the ``fairness.results.v1`` document of the original audio.

    ``mixed_only`` keeps the meetings with both sexes, so that a meeting
    of one sex (and its recording conditions) cannot make the gap.
    """
    if mixed_only:
        keep = set(mixed_sessions(rows))
        rows = [r for r in rows if r["session"] in keep]
    audits: List[Dict[str, Any]] = []
    tracks = sorted({r["track"] for r in rows if r["track"].startswith(
        "none+")})
    for metric, field in (("unattributed", "error_rate"),
                          ("missed", "missed_rate")):
        for track in tracks:
            observations = [
                {"unit": r["speaker"], "group": r["sex"],
                 "value": r[field]}
                for r in rows
                if r["track"] == track and r["sex"] is not None
                and r[field] is not None
            ]
            audits.append({
                "metric": f"{metric}:{track}", "kind": "mean",
                "direction": "lower_is_better", "min_units": MIN_PEOPLE,
                "description": (
                    f"Parte del habla puntuada de cada persona que `{track}` "
                    + ("no dio a la etiqueta emparejada con esa persona "
                       "(perdida o dada a otra)."
                       if metric == "unattributed" else
                       "marcó como silencio (nadie habla).")
                ),
                "observations": observations,
            })
    sessions = sorted({r["session"] for r in rows})
    return {
        "schema_name": "fairness.results.v1",
        "title": "Diarización por sexo del hablante (AMI"
        + (", sólo reuniones mixtas)" if mixed_only else ")"),
        "date": date.today().isoformat(),
        "system": (
            "diarizadores del laboratorio sobre el audio original, salidas "
            "de las ejecuciones de #44 (sin volver a ejecutar); collar "
            "0,25 s, emparejamiento uno a uno de etiquetas del DER"
        ),
        "data": f"AMI sdm, test, CC BY 4.0: {', '.join(sessions)}",
        "groups_source": (
            "sexo en el id de participante de AMI (primera letra M/F), "
            f"metadato del corpus: {AMI_IDS}"
        ),
        "notes": [
            "ES2004b y ES2004c tienen las mismas cuatro personas: cada "
            "persona se promedia sobre sus reuniones y cuenta una vez.",
            f"{len(sessions)} reuniones, "
            f"{len({r['speaker'] for r in rows})} personas: los "
            "intervalos son anchos y dicen más que los cocientes. No es "
            "un contraste de hipótesis.",
        ],
        "audits": audits,
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Score the lab's AMI sessions and write both JSON files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lab", type=Path, help="the lab's data directory")
    parser.add_argument("out", type=Path, help="output path without .json")
    args = parser.parse_args(argv)
    rows = score_lab(args.lab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(
        json.dumps(rows, indent=1) + "\n", encoding="utf-8"
    )
    for suffix, mixed in (("", False), ("_mixed", True)):
        Path(f"{args.out}{suffix}.fairness.json").write_text(
            json.dumps(fairness_document(rows, mixed), indent=1,
                       ensure_ascii=False) + "\n", encoding="utf-8",
        )
    print(f"{len(rows)} rows from "
          f"{len({r['session'] for r in rows})} sessions")


if __name__ == "__main__":
    main()
