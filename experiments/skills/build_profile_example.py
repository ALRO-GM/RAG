"""Build the synthetic multi-session example of ``skills profile`` (#84).

Input (hand-written, committed): ``examples/skills/profile/example.json``
(sessions, who is who, and the level and evidence turns of each
person in each session) and one ``sessions/<id>.txt`` transcript per
session (``[hh:mm:ss] Name: text``). The people and conversations are
invented; no model is called.

Output (committed): ``transcripts/<id>.transcript.v1.json``, the
transcript as the transcriber would publish it with enrolled speakers,
and ``output/<id>.assessment.json``, the assessment as ``skills
assess`` writes it. Each hand-written level goes through the same
evidence rules as a judge's (``decide_rating``): a level the rules
would not keep is an error, so the example never shows something the
analyzer could not produce.

Usage::

    python experiments/skills/build_profile_example.py build
    python experiments/skills/build_profile_example.py check

``check`` exits 1 if ``build`` would change a committed file.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # pragma: no cover - run as a script
    sys.path.insert(0, str(ROOT))

from src.adapters.outbound.json_catalogue_repository import (  # noqa: E402
    JsonCatalogueRepository,
)
from src.domain.skills.assessment import (  # noqa: E402
    CompetencyAssessment,
    Evidence,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.grounding import decide_rating  # noqa: E402
from src.domain.skills.interaction import describe  # noqa: E402
from src.domain.skills.rubric import RubricCatalogue  # noqa: E402
from src.domain.skills.transcript import (  # noqa: E402
    Transcript,
    from_contract,
    parse_text,
)
from src.domain.skills.transcript_contract import read_contract  # noqa: E402

EXAMPLE = ROOT / "examples" / "skills" / "profile"
RULES: Dict[str, float] = {
    "min_incidents": 3, "min_episodes": 2, "episode_gap": 3,
    "min_similarity": 0.3, "min_participant_words": 40, "candidates": 8,
}
"""The default evidence rules of ``SkillsSettings``."""


def contract(
    session: Dict[str, Any], text: str, people: Dict[str, str],
    language: str,
) -> Dict[str, Any]:
    """Return a ``transcript.v1`` document for one hand-written session."""
    parsed = parse_text(session["id"], text)
    names = parsed.speakers()
    unknown = [n for n in names if n not in people]
    if unknown:
        raise ValueError(f"{session['id']}: no person id for {unknown}")
    utterances: List[Dict[str, Any]] = []
    turns = parsed.turns
    for number, turn in enumerate(turns):
        start = float(turn.start or 0.0)
        following = (
            turns[number + 1].start if number + 1 < len(turns) else None
        )
        end = (following - 0.5) if following is not None else start + 6.0
        before = utterances[-1] if utterances else None
        utterances.append({
            "id": f"t{number}.s0", "turn": number, "index": 0,
            "speaker": people[turn.speaker], "start": start,
            "end": round(max(end, start), 2), "text": turn.text,
            "previous_speaker": before["speaker"] if before else None,
            "previous_utterance": before["id"] if before else None,
        })
    document = {
        "schema": "transcript.v1",
        "session": {
            "id": session["id"], "title": session["title"],
            "language": language, "started_at": session["started_at"],
            "duration": utterances[-1]["end"], "type": session["type"],
            "goal": session["goal"],
        },
        "speakers": [
            {"id": people[name], "name": name, "enrolled": True,
             "identification": "threshold"}
            for name in names
        ],
        "utterances": utterances,
        "pipeline": {
            "recogniser": "none (hand-written example)",
            "voice_model": "none (hand-written example)",
            "transcriber": "none (hand-written example)",
        },
    }
    read_contract(document)
    return document


def cell(
    transcript: Transcript, catalogue: RubricCatalogue, name: str,
    competency_id: str, spec: Optional[List[Any]],
) -> CompetencyAssessment:
    """Return one competency of one person from its hand-written spec."""
    if spec is None:
        return CompetencyAssessment(
            competency_id=competency_id, status="insufficient_evidence",
        )
    wanted, items = spec
    competency = catalogue.competency(competency_id)
    evidence = []
    for turn, indicator_id in items:
        spoken = transcript.turns[turn]
        if spoken.speaker != name:
            raise ValueError(
                f"{transcript.session_id} turn {turn} is {spoken.speaker}'s, "
                f"not {name}'s"
            )
        indicator = competency.indicator(indicator_id)
        if indicator is None:
            raise ValueError(f"unknown indicator {indicator_id}")
        evidence.append(Evidence(
            turn=turn, quote=spoken.text, indicator_id=indicator_id,
            polarity=indicator.polarity,
        ))
    rating, flags = decide_rating(
        wanted, evidence, catalogue.scale, int(RULES["min_incidents"]),
        int(RULES["min_episodes"]), int(RULES["episode_gap"]),
    )
    if rating != wanted:
        raise ValueError(
            f"{transcript.session_id} {name} {competency_id}: the rules "
            f"turn {wanted} into {rating} ({flags})"
        )
    return CompetencyAssessment(
        competency_id=competency_id,
        status="rated" if rating is not None else "insufficient_evidence",
        rating=rating, evidence=evidence, flags=flags,
        proposed=len(evidence),
    )


def assessment(
    session: Dict[str, Any], transcript: Transcript,
    catalogue: RubricCatalogue, judge: str,
) -> SessionAssessment:
    """Return the assessment of one session from its hand-written levels."""
    team = describe(transcript)
    ratings: Dict[str, Dict[str, List[Any]]] = session["ratings"]
    names = {person: name for name, person in transcript.people.items()}
    cards = []
    for name in transcript.speakers():
        person = transcript.people[name]
        given = ratings.get(person, {})
        unknown = sorted(set(given) - {c.id for c in catalogue.competencies})
        if unknown:
            raise ValueError(f"unknown competencies {unknown}")
        cards.append(ParticipantScorecard(
            participant=name, person_id=person, interaction=team.of(name),
            assessments=[
                cell(transcript, catalogue, name, c.id, given.get(c.id))
                for c in catalogue.competencies
            ],
        ))
    missing = sorted(set(ratings) - set(names))
    if missing:
        raise ValueError(f"{session['id']}: {missing} did not speak")
    return SessionAssessment(
        session_id=transcript.session_id, language=catalogue.language,
        catalogue=catalogue.name, catalogue_version=catalogue.version,
        judge_model=judge, embedding_model="", min_incidents=3,
        rules=RULES, team=team, scorecards=cards,
        started_at=transcript.started_at,
        conversation_type=transcript.conversation_type,
    )


def build(target: Path) -> List[Path]:
    """Write the transcripts and assessments under ``target``."""
    spec = json.loads((EXAMPLE / "example.json").read_text("utf-8"))
    catalogue = JsonCatalogueRepository(str(ROOT / "knowledge" / "rubrics"))
    rubric = catalogue.load(spec["catalogue"], spec["language"])
    written: List[Path] = []
    for session in spec["sessions"]:
        text = (EXAMPLE / "sessions" / f"{session['id']}.txt").read_text(
            "utf-8"
        )
        document = contract(session, text, spec["people"], spec["language"])
        transcript = from_contract(read_contract(document))
        result = assessment(session, transcript, rubric, spec["judge_model"])
        for folder, name, data in (
            ("transcripts", f"{session['id']}.transcript.v1.json", document),
            ("output", f"{session['id']}.assessment.json",
             result.model_dump()),
        ):
            path = target / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            written.append(path)
    return written


def check() -> int:
    """Return 1 if a committed output differs from a fresh build."""
    with tempfile.TemporaryDirectory() as scratch:
        stale = [
            path.relative_to(scratch) for path in build(Path(scratch))
            if not (EXAMPLE / path.relative_to(scratch)).is_file()
            or (EXAMPLE / path.relative_to(scratch)).read_text("utf-8")
            != path.read_text("utf-8")
        ]
    for path in stale:
        print(f"Out of date: examples/skills/profile/{path}")
    return 1 if stale else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run ``build`` or ``check``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "check"))
    args = parser.parse_args(argv)
    if args.command == "check":
        return check()
    for path in build(EXAMPLE):
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
