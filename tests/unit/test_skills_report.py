"""The Markdown scorecard report."""

from typing import List, Optional

from src.application.skills.report import render_scorecards
from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    ParticipantScorecard,
    SessionAssessment,
)
from src.domain.skills.interaction import describe
from src.domain.skills.transcript import Transcript
from tests.fakes import MEETING_EN, make_catalogue, make_transcript

CATALOGUE = make_catalogue()
TIMED = make_transcript([
    (speaker, text, 61.0 + 10 * i, 70.0 + 10 * i)
    for i, (speaker, text) in enumerate(MEETING_EN)
])


def evidence(
    turn: int, quote: str, indicator: str, similarity: Optional[float] = None
) -> Evidence:
    """Return a verified evidence item."""
    return Evidence(
        turn=turn, quote=quote, indicator_id=indicator,
        polarity="negative" if indicator in ("L3", "P2") else "positive",
        similarity=similarity,
    )


def report(transcript: Transcript = TIMED) -> str:
    """Render a session with one rated and several unrated cells."""
    team = describe(transcript)
    cards: List[ParticipantScorecard] = [
        ParticipantScorecard(
            participant="Maya", interaction=team.of("Maya"),
            assessments=[
                CompetencyAssessment(
                    competency_id="listening", status="rated", rating=4,
                    rationale="Asks and summarises.",
                    evidence=[
                        evidence(0, "What do you mean exactly", "L1", 0.4239),
                        evidence(3, "So if I understood correctly", "L2"),
                    ],
                ),
                CompetencyAssessment(
                    competency_id="planning",
                    status="insufficient_evidence",
                ),
            ],
        ),
        ParticipantScorecard(
            participant="Tom", interaction=team.of("Tom"),
            assessments=[
                CompetencyAssessment(
                    competency_id="listening",
                    status="insufficient_evidence",
                    evidence=[evidence(1, "That is nonsense", "L3")],
                    flags=["below_minimum_evidence", "other_flag"],
                ),
            ],
        ),
    ]
    assessment = SessionAssessment(
        session_id="s1", language="en", catalogue="core",
        catalogue_version="1.0", judge_model="fake-judge",
        embedding_model="", min_incidents=2, team=team, scorecards=cards,
    )
    return render_scorecards(assessment, CATALOGUE, transcript)


def test_report_header_names_how_it_was_produced() -> None:
    """Session, rubric version, judge and incident rule open the report."""
    text = report()
    assert text.startswith("# Session s1\n")
    assert "Rubric: core 1.0 · judge: fake-judge · language: en" in text
    assert "a rating needs 2 verified incidents" in text
    assert "not a decision" in text
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_report_participation_table() -> None:
    """Every speaker has a row with turns, words, share and questions."""
    text = report()
    assert "3 speakers, 8 turns" in text
    assert "| Speaker | Turns | Words | Share | Questions |" in text
    rows = [line for line in text.splitlines() if line.startswith("| Maya |")]
    assert rows and rows[0].split("|")[2].strip() == "3"
    assert any(line.startswith("| Priya | 2 |") for line in text.splitlines())


def test_report_ratings_and_insufficient_evidence() -> None:
    """A rating shows as n / max; no rating as 'insufficient evidence'."""
    text = report()
    assert "| Active listening | 4 / 5 | 2 | — |" in text
    assert "| Planning | insufficient evidence | 0 | — |" in text
    assert ("| Active listening | insufficient evidence | 1 | "
            "below_minimum_evidence, other_flag |") in text
    assert "### Active listening — 4 / 5" in text
    assert "### Active listening — insufficient evidence" in text
    assert "### Planning" not in text


def test_report_quotes_with_turn_number_and_timestamp() -> None:
    """Each quote is followed by its turn, clock and indicator text."""
    text = report()
    assert ('- (+) "What do you mean exactly" — turn 0, 01:01. '
            "*L1: Asks a clarifying question* (match 0.42)") in text
    assert ('- (+) "So if I understood correctly" — turn 3, 01:31. '
            "*L2: Summarises what others said*") in text
    assert ('- (−) "That is nonsense" — turn 1, 01:11. '
            "*L3: Dismisses others*") in text
    assert "Asks and summarises." in text


def test_report_match_is_shown_only_when_similarity_is_known() -> None:
    """Evidence without a cosine has no '(match ...)' suffix."""
    lines = [line for line in report().splitlines() if line.startswith("- (")]
    assert [("(match" in line) for line in lines] == [True, False, False]


def test_report_without_timestamps_shows_turn_numbers_only() -> None:
    """An untimed transcript gives 'turn n.' without a clock."""
    text = report(make_transcript())
    assert '"What do you mean exactly" — turn 0. *L1' in text
    assert "01:01" not in text


def test_report_clock_past_one_hour_counts_minutes() -> None:
    """Minutes keep counting after an hour (mm:ss)."""
    late = make_transcript([
        (speaker, text, 3725.0 + i, 3726.0 + i)
        for i, (speaker, text) in enumerate(MEETING_EN)
    ])
    assert "— turn 0, 62:05." in report(late)


def test_report_has_one_section_per_participant() -> None:
    """Scorecards appear in order, one heading each."""
    text = report()
    assert text.index("## Participation") < text.index("## Maya")
    assert text.index("## Maya") < text.index("## Tom")
    assert "## Priya" not in text
