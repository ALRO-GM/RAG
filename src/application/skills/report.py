"""Render a session assessment as a Markdown scorecard report."""

from typing import Dict, List, Optional, Sequence, Union

from src.application.skills.person_report_texts import (
    TEXTS,
    conversation_type_label,
)
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Estimate,
    Evidence,
    Observation,
    SessionAssessment,
)
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.transcript import Transcript


def _rating(item: CompetencyAssessment, maximum: int) -> str:
    """Return the rating cell of a competency."""
    if item.rating is not None:
        return f"{item.rating} / {maximum}"
    if item.estimate is not None:
        return (
            f"provisional ~{item.estimate.rating} / {maximum} "
            f"({item.estimate.confidence} confidence) — review"
        )
    return "insufficient evidence"


def _not_firm(estimate: Estimate) -> str:
    """Return why a provisional estimate is not a firm level."""
    words = TEXTS["en"]
    values = {
        "quotes": estimate.quotes, "needed": estimate.needed,
        "words": estimate.words,
    }
    reasons = "; ".join(
        words[f"reason.{code}"].format(**values) for code in estimate.reasons
    )
    return f"Not firm: {reasons or 'below the minimum evidence'}."


def _quote_line(
    found: Union[Evidence, Observation],
    transcript: Transcript,
    indicators: Dict[str, str],
) -> str:
    """Return one quote with its turn, time, indicator and origin."""
    start = found.start
    if start is None and 0 <= found.turn < len(transcript.turns):
        start = transcript.turns[found.turn].start
    clock = (
        "" if start is None
        else f", {int(start // 60):02d}:{int(start % 60):02d}"
    )
    sign = "+" if found.polarity == "positive" else "−"
    match = (
        "" if found.similarity is None
        else f" (match {found.similarity:.2f})"
    )
    origin = (
        f" [{TEXTS['en']['source.' + found.source]}]"
        if isinstance(found, Observation) else ""
    )
    return (
        f"- ({sign}) \"{found.quote}\" — turn {found.turn}{clock}. "
        f"*{found.indicator_id}: {indicators.get(found.indicator_id, '')}*"
        f"{match}{origin}"
    )


def conversation_context(transcript: Optional[Transcript]) -> str:
    """Return the declared kind and goal of the conversation, or "".

    E.g. ``Conversation: negotiation · goal: Price (declared by whoever
    recorded it; context, not used in any rating).``
    """
    if transcript is None:
        return ""
    parts = []
    if transcript.conversation_type:
        parts.append(
            conversation_type_label("en", transcript.conversation_type)
        )
    if transcript.conversation_goal:
        parts.append(f"goal: {transcript.conversation_goal}")
    if not parts:
        return ""
    return (
        f"Conversation: {' · '.join(parts)} (declared by whoever recorded "
        "it; context, not used in any rating)."
    )


def render_scorecards(
    assessment: SessionAssessment,
    catalogue: RubricCatalogue,
    transcript: Transcript,
) -> str:
    """Return the report: team participation, then one card per person.

    Every rating is followed by the quotes it rests on, with their
    turn number and timestamp, so a reader can check each of them. A
    competency without a firm level lists its provisional estimate, why
    it is not firm and its observations, each with where it came from.
    """
    team = assessment.team
    names = {c.id: c.name for c in catalogue.competencies}
    indicators = {
        i.id: i.text for c in catalogue.competencies for i in c.indicators
    }
    maximum = catalogue.scale.maximum
    context = conversation_context(transcript)
    lines: List[str] = [
        f"# Session {assessment.session_id}",
        "",
        f"Rubric: {assessment.catalogue} {assessment.catalogue_version} · "
        f"judge: {assessment.judge_model} · language: "
        f"{assessment.language} · a rating needs "
        f"{assessment.min_incidents} verified incidents.",
        "",
        *([context, ""] if context else []),
        "Ratings describe behaviour observed in this session only. They "
        "are a draft for a human assessor, not a decision.",
        "",
        "Provisional estimates (some evidence, below the minimum) are "
        "hypotheses for the interviewer, not assessments; their "
        "observations are real sentences to check in the transcript.",
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
        firm, provisional, noted = card.tiers()
        lines += ["", f"## {card.participant}", "",
                  f"{firm} firm, {provisional} provisional, {noted} "
                  "observations.", "",
                  "| Competency | Rating | Evidence | Flags |",
                  "|---|---|---|---|"]
        for item in card.assessments:
            lines.append(
                f"| {names.get(item.competency_id, item.competency_id)} | "
                f"{_rating(item, maximum)} | {len(item.evidence)} | "
                f"{', '.join(item.flags) or '—'} |"
            )
        for item in card.assessments:
            observed = item.observed()
            if not item.evidence and not observed:
                continue
            lines += ["", f"### {names.get(item.competency_id)} — "
                      f"{_rating(item, maximum)}", ""]
            if item.estimate is not None:
                lines += [_not_firm(item.estimate), ""]
            if item.rationale:
                lines += [item.rationale, ""]
            shown: Sequence[Union[Evidence, Observation]] = (
                item.evidence if item.status == "rated" else observed
            )
            for found in shown:
                lines.append(_quote_line(found, transcript, indicators))
    return "\n".join(lines) + "\n"
