"""Real-judge check of the opinions of #231 on the simulated discussion.

Runs step i's writer through ``claude -p`` (subscription, never the paid
API) on the committed synthetic ``sim-reunion-verano-audio`` assessment:
the 4 people as ``reunion_decision``, one person as ``coaching`` and as
``entrevista``, and one person with her evidence cut to two
observations: 7 calls. Writes a short summary, no transcript text of
real people (all data is invented).

    PYTHONPATH=. uv run --no-sync python experiments/skills/opinion_check.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge
from src.adapters.outbound.json_catalogue_repository import (
    JsonCatalogueRepository,
)
from src.application.skills.synthesis import ProfileSynthesizer, person_facts
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.synthesis import PersonFacts

SOURCE = Path(
    "experiments/skills/results/model_comparison/analysis/"
    "sim-reunion-verano-audio"
)
OUT = Path("experiments/skills/results/opinion_check.json")


def summary(kind: str, name: str, result: Any) -> Dict[str, Any]:
    """Return the figures kept of one profile."""
    overall = result.overall
    return {
        "type": kind, "person": name, "status": result.status,
        "dropped": result.dropped, "score": overall.score,
        "percent": overall.percent, "verdict": overall.verdict,
        "line": overall.line, "source": overall.source,
        "caveats": overall.caveats, "confidence": result.confidence,
        "top_strengths": result.top_strengths,
        "top_improvements": result.top_improvements,
        "skills_by_judge": sum(s.source == "judge" for s in result.skills),
        "skills_by_rules": sum(s.source == "rules" for s in result.skills),
    }


def main() -> None:
    """Run the 7 calls and write the summary."""
    base = SessionAssessment.model_validate_json(
        (SOURCE / "assessment.json").read_text()
    )
    catalogue = JsonCatalogueRepository().load(base.catalogue, base.language)
    acts = json.loads((SOURCE / "acts.json").read_text())
    events = json.loads((SOURCE / "events.json").read_text())
    writer = ProfileSynthesizer(ClaudeCodeJudge(model="haiku"))
    rows: List[Dict[str, Any]] = []

    def run(kind: str, name: str, cut: bool = False) -> None:
        assessment = base.model_copy(update={"conversation_type": kind})
        facts: PersonFacts = person_facts(
            assessment, catalogue, name, acts, events
        )
        if cut:
            quotes = [c for c in facts.citables if c.kind != "event"][:2]
            facts = facts.model_copy(update={
                "words": 60,
                "citables": [
                    c.model_copy(update={"kind": "observation"})
                    for c in quotes
                ],
                "competencies": [
                    c.model_copy(update={
                        "status": "insufficient_evidence", "rating": None,
                        "confidence": "",
                    }) for c in facts.competencies
                ],
            })
        row = summary(kind + ("+cut" if cut else ""), name,
                      writer.write(facts))
        rows.append(row)
        print(row["type"], name, row["score"], row["verdict"], row["line"])

    for card in base.scorecards:
        run("reunion_decision", card.participant)
    first = base.scorecards[0].participant
    run("coaching", first)
    run("entrevista", first)
    run("coaching", first, cut=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
