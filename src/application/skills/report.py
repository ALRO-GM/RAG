"""Render a session assessment as a Markdown scorecard report."""

from typing import List

from src.domain.skills.assessment import (
    CompetencyAssessment,
    SessionAssessment,
)
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.transcript import Transcript


def _rating(item: CompetencyAssessment, maximum: int) -> str:
    """Return the rating cell of a competency."""
    if item.rating is None:
        return "insufficient evidence"
    return f"{item.rating} / {maximum}"


def render_scorecards(
    assessment: SessionAssessment,
    catalogue: RubricCatalogue,
    transcript: Transcript,
) -> str:
    """Return the report: team participation, then one card per person.

    Every rating is followed by the quotes it rests on, with their
    turn number and timestamp, so a reader can check each of them.
    """
    team = assessment.team
    names = {c.id: c.name for c in catalogue.competencies}
    indicators = {
        i.id: i.text for c in catalogue.competencies for i in c.indicators
    }
    maximum = catalogue.scale.maximum
    lines: List[str] = [
        f"# Session {assessment.session_id}",
        "",
        f"Rubric: {assessment.catalogue} {assessment.catalogue_version} · "
        f"judge: {assessment.judge_model} · language: "
        f"{assessment.language} · a rating needs "
        f"{assessment.min_incidents} verified incidents.",
        "",
        "Ratings describe behaviour observed in this session only. They "
        "are a draft for a human assessor, not a decision.",
        "",
        "## Participation",
        "",
        f"{team.speakers} speakers, {team.turns} turns, {team.words} words. "
        f"Participation equality {team.participation_equality:.2f} "
        f"(1 = everyone spoke the same amount).",
        "",
        "| Speaker | Turns | Words | Share | Questions |",
        "|---|---|---|---|---|",
    ]
    for speaker in team.per_speaker:
        lines.append(
            f"| {speaker.speaker} | {speaker.turns} | {speaker.words} | "
            f"{speaker.word_share:.0%} | {speaker.questions} |"
        )
    for card in assessment.scorecards:
        lines += ["", f"## {card.participant}", "",
                  "| Competency | Rating | Evidence | Flags |",
                  "|---|---|---|---|"]
        for item in card.assessments:
            lines.append(
                f"| {names.get(item.competency_id, item.competency_id)} | "
                f"{_rating(item, maximum)} | {len(item.evidence)} | "
                f"{', '.join(item.flags) or '—'} |"
            )
        for item in card.assessments:
            if not item.evidence:
                continue
            lines += ["", f"### {names.get(item.competency_id)} — "
                      f"{_rating(item, maximum)}", ""]
            if item.rationale:
                lines += [item.rationale, ""]
            for evidence in item.evidence:
                turn = transcript.turns[evidence.turn]
                clock = (
                    "" if turn.start is None
                    else f", {int(turn.start // 60):02d}:"
                         f"{int(turn.start % 60):02d}"
                )
                sign = "+" if evidence.polarity == "positive" else "−"
                match = (
                    "" if evidence.similarity is None
                    else f" (match {evidence.similarity:.2f})"
                )
                lines.append(
                    f"- ({sign}) \"{evidence.quote}\" — turn "
                    f"{evidence.turn}{clock}. *{evidence.indicator_id}: "
                    f"{indicators.get(evidence.indicator_id, '')}*{match}"
                )
    return "\n".join(lines) + "\n"
