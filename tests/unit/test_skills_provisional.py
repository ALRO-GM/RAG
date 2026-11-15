"""Provisional estimates and why few levels are firm (#103)."""

from typing import List, Optional

from src.domain.skills.assessment import (
    CompetencyAssessment,
    Evidence,
    Observation,
    ParticipantScorecard,
)
from src.domain.skills.interaction import SpeakerInteraction
from src.domain.skills.provisional import (
    REASONS,
    SHORT_PARTICIPATION_WORDS,
    WHY_FEW_FIRM,
    estimate,
    why_few_firm,
)
from src.domain.skills.rubric import Scale

FIVE = Scale(minimum=1, maximum=5)
ENOUGH = SHORT_PARTICIPATION_WORDS


def quote(turn: int, positive: bool = True) -> Evidence:
    """Return a verified quote."""
    return Evidence(turn=turn, quote="q", indicator_id="X.1",
                    polarity="positive" if positive else "negative")


def seen(turn: int, positive: bool = True,
         source: str = "confirmation") -> Observation:
    """Return an observation of step f or a weak quote."""
    return Observation(
        turn=turn, quote="s", indicator_id="X.2",
        polarity="positive" if positive else "negative",
        source="weak_match" if source == "weak_match" else "confirmation",
    )


def test_nothing_at_all_gives_no_estimate() -> None:
    assert estimate(4, [], [], [], FIVE, 3, ENOUGH) == (None, [])


def test_the_judge_rating_is_kept_inside_the_bounds() -> None:
    """3 of 5 from the judge stays 3; the reasons say why it is not firm."""
    guess, flags = estimate(3, [quote(0), quote(10, False)], [], [], FIVE,
                            3, ENOUGH)
    assert guess is not None and flags == []
    assert (guess.rating, guess.confidence) == (3, "medium")
    assert guess.reasons == ["few_quotes"]
    assert (guess.quotes, guess.confirmations, guess.needed) == (2, 0, 3)


def test_never_an_extreme_on_little_evidence() -> None:
    """A 5 or a 1 proposed on one quote moves one point to the centre."""
    top, flags = estimate(5, [quote(0)], [], [], FIVE, 3, ENOUGH)
    assert top is not None and top.rating == 4
    assert flags == ["estimate_bounded"]
    bottom, _ = estimate(1, [quote(0, False)], [], [], FIVE, 3, ENOUGH)
    assert bottom is not None and bottom.rating == 2


def test_never_against_evidence_that_all_points_one_way() -> None:
    """A low rating on only positive evidence goes to the centre."""
    up, flags = estimate(2, [quote(0)], [], [], FIVE, 3, ENOUGH)
    assert up is not None and up.rating == 3
    assert flags == ["estimate_bounded"]
    down, _ = estimate(4, [quote(0, False)], [], [], FIVE, 3, ENOUGH)
    assert down is not None and down.rating == 3


def test_without_a_judge_rating_the_sign_of_the_sentences_decides() -> None:
    """More + than −: above the centre; more −: below; even: centre."""
    cases = [
        ([seen(0), seen(9)], 4), ([seen(0, False)], 2),
        ([seen(0), seen(9, False)], 3),
    ]
    for confirmed, expected in cases:
        guess, _ = estimate(None, [], [], confirmed, FIVE, 3, ENOUGH)
        assert guess is not None and guess.rating == expected
        assert guess.confidence == "low"
        assert guess.reasons[0] == "only_confirmations"
        assert "no_judge_rating" in guess.reasons
    off, _ = estimate(9, [], [], [seen(0)], FIVE, 3, ENOUGH)
    assert off is not None and off.judge_rating == 9
    assert "no_judge_rating" in off.reasons


def test_even_and_tiny_scales() -> None:
    """On 1–4 the centre is 2.5; on 1–2 the only estimate is 1."""
    four = Scale(minimum=1, maximum=4)
    up, _ = estimate(None, [quote(0)], [], [], four, 3, ENOUGH)
    down, _ = estimate(None, [quote(0, False)], [], [], four, 3, ENOUGH)
    even, _ = estimate(None, [quote(0), quote(9, False)], [], [], four, 3,
                       ENOUGH)
    assert [g.rating for g in (up, down, even) if g] == [3, 2, 2]
    two, _ = estimate(2, [quote(0), quote(9, False)], [], [],
                      Scale(minimum=1, maximum=2), 3, ENOUGH)
    assert two is not None and two.rating == 1


def test_confidence_and_every_reason() -> None:
    """Weak quotes, one moment and a short participation lower it."""
    guess, _ = estimate(
        4, [quote(0)], [seen(1, source="weak_match")], [seen(2)], FIVE, 3,
        ENOUGH - 1,
    )
    assert guess is not None and guess.confidence == "low"
    assert guess.reasons == [
        "few_quotes", "single_episode", "weak_match", "short_participation",
    ]
    assert guess.quotes == 2 and guess.confirmations == 1
    assert set(guess.reasons) <= set(REASONS)
    spread, _ = estimate(4, [quote(0)], [], [seen(20)], FIVE, 3, ENOUGH)
    assert spread is not None and spread.confidence == "medium"


def card(
    statuses: List[str], words: int = 1000, proposed: int = 3,
    dropped: int = 0, flags: Optional[List[str]] = None,
) -> ParticipantScorecard:
    """Return a scorecard with one competency per status."""
    return ParticipantScorecard(
        participant="Ana",
        interaction=SpeakerInteraction(
            speaker="Ana", turns=5, words=words, word_share=0.5,
            turn_share=0.5, mean_words_per_turn=words / 5, questions=0,
        ),
        assessments=[
            CompetencyAssessment(
                competency_id=f"C{n}", status=status,  # type: ignore[arg-type]
                rating=3 if status == "rated" else None,
                proposed=proposed, rejected=dropped, flags=flags or [],
            )
            for n, status in enumerate(statuses)
        ],
    )


def test_why_is_empty_with_enough_firm_levels() -> None:
    assert why_few_firm(card(["rated", "insufficient_evidence"]), 3, 2) == []
    assert why_few_firm(card([]), 3, None) == []


def test_why_names_the_short_recording_and_the_quiet_person() -> None:
    codes = why_few_firm(
        card(["provisional", "insufficient_evidence"], words=120,
             proposed=0), 3, 2.5,
    )
    assert codes == [
        "short_recording", "little_participation", "few_quotes_proposed",
    ]
    assert set(codes) <= set(WHY_FEW_FIRM)


def test_why_names_the_floor_and_the_filtered_quotes() -> None:
    quiet = card(["insufficient_evidence"], words=20, proposed=0,
                 flags=["below_participation_floor"])
    assert why_few_firm(quiet, 3, None) == [
        "little_participation", "below_floor",
    ]
    filtered = card(["provisional", "provisional"], proposed=4, dropped=2)
    assert why_few_firm(filtered, 3, 30.0) == ["quotes_filtered"]
    assert why_few_firm(card(["provisional"]), 3, 30.0) == ["not_shown"]
