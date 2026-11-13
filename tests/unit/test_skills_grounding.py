"""Grounding: quote verification, evidence checks and rating rules."""

from typing import List, Optional, Sequence, Tuple

import pytest

from src.domain.skills.assessment import Evidence
from src.domain.skills.grounding import (
    ProposedEvidence,
    contains_quote,
    decide_rating,
    episodes,
    find_quote,
    ground,
    incidents,
    quote_tokens,
)
from src.domain.skills.rubric import Scale
from src.domain.skills.transcript import Turn
from tests.fakes import make_catalogue, make_transcript

TEXT = (
    "Espera, Luis, déjala terminar. Marta, ¿en qué parte del enunciado "
    "has visto eso? Así lo comprobamos entre todos."
)
LISTENING = make_catalogue().competency("listening")
TURNS = make_transcript([
    ("Maya", "What do you mean exactly by streaming the file, Priya?"),
    ("Tom", "That is nonsense, nobody will test with the big file."),
    ("Maya", "So if I understood correctly, the big file must work."),
    ("Tom", "Hold on, let her finish what she was saying please."),
    ("Maya", "Good point, let us write that down in the notes."),
]).turns
SCALE = Scale()


def item(turn: Optional[int], quote: str, indicator: str) -> ProposedEvidence:
    """Return a proposed evidence item."""
    return ProposedEvidence(turn=turn, quote=quote, indicator=indicator)


def grounded(
    proposed: Sequence[ProposedEvidence],
    participant: str = "Maya",
    turns: Sequence[Turn] = TURNS,
) -> Tuple[List[Tuple[int, str]], int]:
    """Return the kept (turn, indicator) pairs and the rejected count."""
    kept, rejected = ground(proposed, turns, participant, LISTENING)
    return [(e.turn, e.indicator_id) for e in kept], rejected


def evidence(*items: Tuple[int, str]) -> List[Evidence]:
    """Return verified evidence from (turn, sign + indicator) pairs.

    ``"+"`` is a positive item of indicator ``X``; ``"+A"`` names the
    indicator; ``"-"`` items are negative.
    """
    return [
        Evidence(turn=turn, quote="q", indicator_id=code[1:] or "X",
                 polarity="positive" if code[0] == "+" else "negative")
        for turn, code in items
    ]


# ------------------------------------------------------- contains_quote


def test_quote_tokens_fold_case_accents_and_punctuation() -> None:
    """Tokens are lowercase words without accents; ``ñ`` survives."""
    assert quote_tokens("¡DÉJALA, señor: 3 veces!") == [
        "dejala", "señor", "3", "veces",
    ]


@pytest.mark.parametrize("quote", [
    "Luis, déjala terminar",
    "Espera, Luis, déjala terminar.",
    "ESPERA LUIS DEJALA TERMINAR",
    "en que parte del enunciado has visto eso",
    "¿En qué parte del enunciado has visto eso?",
    "«Así lo comprobamos entre todos»",
    "dejala   terminar\nMarta",
    TEXT,
])
def test_contains_quote_accepts_verbatim_quotes(quote: str) -> None:
    """Case, accents, punctuation and spacing do not matter."""
    assert contains_quote(quote, TEXT)


@pytest.mark.parametrize("quote", [
    "Luis, deja que Marta acabe de hablar",
    "le pidió a Luis que la dejara terminar",
    "wait Luis let her finish",
    "comprobamos todos el enunciado",
    "Marta has visto eso en el enunciado",
    "terminar déjala Luis espera",
])
def test_contains_quote_rejects_paraphrases(quote: str) -> None:
    """Rewordings, translations and reorderings are not quotes."""
    assert not contains_quote(quote, TEXT)


@pytest.mark.parametrize("quote", ["", "   ", "Luis", "déjala", "¡¿...?!",
                                   "espera Luis"])
def test_contains_quote_rejects_quotes_under_three_words(quote: str) -> None:
    """One or two words (or none) are too little to be evidence."""
    assert not contains_quote(quote, TEXT)


def test_contains_quote_in_empty_text() -> None:
    """Nothing is quoted from an empty turn."""
    assert not contains_quote("déjala terminar ya", "")


def test_contains_quote_rejects_two_stitched_fragments() -> None:
    """Two real fragments glued together are not one verbatim quote."""
    assert not contains_quote("Espera Luis así lo comprobamos", TEXT)


def test_contains_quote_rejects_an_added_word() -> None:
    """A word the speaker did not say makes the quote a paraphrase."""
    assert not contains_quote("oye espera Luis déjala terminar", TEXT)
    assert not contains_quote("Espera Luis déjala terminar ya", TEXT)


def test_find_quote_returns_the_transcript_wording() -> None:
    """The passage comes back with its own case, accents and commas."""
    assert find_quote("ESPERA LUIS DEJALA TERMINAR", TEXT) == (
        "Espera, Luis, déjala terminar"
    )
    assert find_quote("en que parte del enunciado", TEXT) == (
        "en qué parte del enunciado"
    )
    assert find_quote("espera Luis", TEXT) is None
    assert find_quote("nada de esto se dijo", TEXT) is None


def test_find_quote_span_limit_bounds_the_skipped_words() -> None:
    """Skipped words may add at most a quarter to the quote length."""
    said = "one two three four five six seven eight nine ten"
    assert find_quote("one two three five", said) == "one two three four five"
    assert find_quote("one two three six", said) is None
    assert find_quote("one two four", said) is None
    assert find_quote("one two three four six seven eight ten", said) == said


def test_find_quote_prefers_the_tightest_passage() -> None:
    """Among several matches the shortest passage is returned."""
    said = "we can, well, do it now. Yes, we can do it tomorrow"
    assert find_quote("we can do it", said) == "we can do it"


def test_contains_quote_does_not_match_inside_words() -> None:
    """Matching is by whole words, not by substrings."""
    assert not contains_quote("par del enun", TEXT)


def test_contains_quote_tolerates_a_dropped_filler_word() -> None:
    """A quote that only skips a filler word is still accepted."""
    said = "I think, um, we should ship it today"
    assert contains_quote("I think we should ship it today", said)


def test_contains_quote_rejects_a_replaced_word() -> None:
    """A quote with a word the speaker did not say is not verbatim."""
    said = "we should ship it today because the client asked"
    assert not contains_quote("we should ship it tomorrow", said)


# --------------------------------------------------------------- ground


def test_ground_keeps_verified_evidence_with_rubric_polarity() -> None:
    """A verbatim quote under a real indicator is kept, sorted by turn."""
    kept, rejected = ground(
        [item(2, "if I understood correctly", "L2"),
         item(0, "what do you mean exactly", "L1")],
        TURNS, "Maya", LISTENING,
    )
    assert rejected == 0
    assert [(e.turn, e.indicator_id, e.polarity) for e in kept] == [
        (0, "L1", "positive"), (2, "L2", "positive"),
    ]
    assert kept[0].quote == "What do you mean exactly"
    assert kept[1].quote == "if I understood correctly"


def test_ground_polarity_comes_from_the_rubric() -> None:
    """A negative indicator yields negative evidence."""
    kept, _ = ground(
        [item(1, "That is nonsense, nobody will test", "L3")],
        TURNS, "Tom", LISTENING,
    )
    assert [e.polarity for e in kept] == ["negative"]


def test_ground_rejects_hallucinated_quotes() -> None:
    """A quote nobody said is counted as rejected."""
    assert grounded([item(0, "I really value your opinion Priya", "L1")]) == (
        [], 1,
    )


def test_ground_rejects_paraphrases_of_a_real_turn() -> None:
    """A summary of the turn in other words is rejected."""
    assert grounded([item(0, "she asked Priya to clarify streaming", "L1")
                     ]) == ([], 1)


def test_ground_rejects_quotes_of_another_speaker() -> None:
    """Words said by somebody else are not the participant's evidence."""
    quote = "That is nonsense, nobody will test with the big file"
    assert grounded([item(1, quote, "L3")], "Maya") == ([], 1)
    assert grounded([item(0, quote, "L3")], "Maya") == ([], 1)
    assert grounded([item(None, quote, "L3")], "Maya") == ([], 1)


def test_ground_corrects_a_wrong_turn_number() -> None:
    """A real quote cited under the wrong turn is moved to its turn."""
    assert grounded([item(0, "the big file must work", "L2")]) == (
        [(2, "L2")], 0,
    )
    assert grounded([item(1, "the big file must work", "L2")]) == (
        [(2, "L2")], 0,
    )


@pytest.mark.parametrize("turn", [None, 99, -1])
def test_ground_finds_the_turn_when_the_citation_is_unusable(
    turn: Optional[int],
) -> None:
    """A missing or impossible turn number is recovered from the quote."""
    assert grounded([item(turn, "what do you mean exactly", "L1")]) == (
        [(0, "L1")], 0,
    )


def test_ground_only_searches_the_candidate_turns() -> None:
    """A real quote from a turn the judge was not shown is rejected."""
    shown = [TURNS[0], TURNS[1]]
    assert grounded(
        [item(2, "the big file must work", "L2")], turns=shown
    ) == ([], 1)


@pytest.mark.parametrize("indicator", ["P1", "", "L9", "l1", "listening"])
def test_ground_rejects_unknown_indicators(indicator: str) -> None:
    """An indicator that is not of this competency is rejected."""
    assert grounded([item(0, "what do you mean exactly", indicator)]) == (
        [], 1,
    )


def test_ground_strips_spaces_around_the_indicator_id() -> None:
    """Padding around the indicator id is forgiven."""
    assert grounded([item(0, "what do you mean exactly", " L1 ")]) == (
        [(0, "L1")], 0,
    )


def test_ground_collapses_duplicates_without_rejecting_them() -> None:
    """Two quotes of one turn under one indicator count once."""
    assert grounded([
        item(0, "what do you mean exactly", "L1"),
        item(0, "by streaming the file, Priya", "L1"),
        item(0, "What do you mean exactly", "L1"),
    ]) == ([(0, "L1")], 0)


def test_ground_keeps_one_turn_under_two_indicators() -> None:
    """The same turn can be evidence of two different indicators."""
    assert grounded([
        item(0, "what do you mean exactly", "L2"),
        item(0, "what do you mean exactly", "L1"),
    ]) == ([(0, "L1"), (0, "L2")], 0)


def test_ground_stores_the_transcript_wording_not_the_models() -> None:
    """The kept quote is the passage of the turn, filler included."""
    kept, _ = ground(
        [item(0, "  WHAT do\n you   mean by streaming the file priya ", "L1")],
        TURNS, "Maya", LISTENING,
    )
    assert kept[0].quote == (
        "What do you mean exactly by streaming the file, Priya"
    )


def test_ground_counts_every_rejection() -> None:
    """Rejected items are counted one by one, kept ones are not."""
    assert grounded([
        item(0, "what do you mean exactly", "L1"),
        item(0, "invented words never said", "L1"),
        item(0, "what do you mean exactly", "nope"),
        item(2, "", "L2"),
    ]) == ([(0, "L1")], 3)


def test_ground_with_nothing_proposed_or_unknown_participant() -> None:
    """No proposal keeps nothing; a stranger owns no turn."""
    assert grounded([]) == ([], 0)
    assert grounded(
        [item(0, "what do you mean exactly", "L1")], "Nobody"
    ) == ([], 1)


def test_incidents_are_distinct_turns() -> None:
    """Two indicators on one turn are one incident."""
    assert incidents(evidence((0, "+"), (0, "-"), (3, "+"))) == 2
    assert incidents([]) == 0


# -------------------------------------------------------- decide_rating


def test_episodes_group_turns_at_most_gap_apart() -> None:
    """Evidence turns close together are one exchange."""
    assert episodes([]) == 0
    assert episodes(evidence((4, "+"))) == 1
    assert episodes(evidence((4, "+"), (4, "-"), (7, "+"))) == 1
    assert episodes(evidence((4, "+"), (8, "+"))) == 2
    assert episodes(evidence((9, "+"), (0, "+"), (3, "+"), (6, "+"))) == 1
    assert episodes(evidence((0, "+"), (10, "+"), (20, "+"))) == 3


def test_episodes_gap_is_configurable() -> None:
    """Gap 0 makes every turn its own episode; a wide gap merges all."""
    items = evidence((1, "+"), (2, "+"), (9, "+"))
    assert episodes(items, gap=0) == 3
    assert episodes(items, gap=1) == 2
    assert episodes(items, gap=7) == 1


def test_decide_rating_keeps_a_supported_rating() -> None:
    """Enough incidents of the right polarity keep the rating unflagged."""
    assert decide_rating(4, evidence((0, "+"), (2, "+")), SCALE, 2) == (4, [])
    assert decide_rating(2, evidence((0, "-"), (2, "-")), SCALE, 2) == (2, [])


def test_decide_rating_without_a_rating() -> None:
    """A judge that gave no rating yields none, whatever the evidence."""
    assert decide_rating(None, evidence((0, "+"), (2, "+")), SCALE, 2) == (
        None, [],
    )
    assert decide_rating(None, [], SCALE, 2, min_episodes=2) == (None, [])


@pytest.mark.parametrize("rating", [0, 6, -1, 100])
def test_decide_rating_off_the_scale(rating: int) -> None:
    """A rating outside the scale is withheld and flagged."""
    assert decide_rating(rating, evidence((0, "+"), (2, "+")), SCALE, 2) == (
        None, ["rating_off_scale"],
    )


def test_decide_rating_below_the_minimum_incidents() -> None:
    """Fewer distinct turns than required withhold the rating."""
    below = (None, ["below_minimum_evidence"])
    assert decide_rating(4, [], SCALE, 2) == below
    assert decide_rating(4, evidence((0, "+")), SCALE, 2) == below
    assert decide_rating(4, evidence((0, "+"), (0, "-")), SCALE, 2) == below
    assert decide_rating(4, evidence((0, "+"), (1, "+")), SCALE, 3) == below
    assert decide_rating(4, evidence((0, "+")), SCALE, 1) == (4, [])


def test_decide_rating_single_episode_is_withheld() -> None:
    """Enough turns from one exchange only do not make a rating."""
    close = evidence((0, "+"), (2, "+"), (3, "+"))
    spread = evidence((0, "+"), (2, "+"), (9, "+"))
    assert decide_rating(4, close, SCALE, 2, min_episodes=2) == (
        None, ["single_episode"],
    )
    assert decide_rating(4, spread, SCALE, 2, min_episodes=2) == (4, [])
    assert decide_rating(4, close, SCALE, 2) == (4, [])
    assert decide_rating(4, close, SCALE, 2, 2, episode_gap=1) == (4, [])


def test_decide_rating_incident_rule_is_checked_before_episodes() -> None:
    """Too few turns is reported as such, not as a single episode."""
    assert decide_rating(4, evidence((0, "+")), SCALE, 2, 2) == (
        None, ["below_minimum_evidence"],
    )


def test_decide_rating_top_needs_two_positive_indicators() -> None:
    """A 5 resting on one positive indicator is lowered to 4."""
    narrow = evidence((0, "+A"), (5, "+A"))
    broad = evidence((0, "+A"), (5, "+B"))
    assert decide_rating(5, narrow, SCALE, 2) == (
        4, ["capped_narrow_evidence"],
    )
    assert decide_rating(5, broad, SCALE, 2) == (5, [])
    assert decide_rating(5, evidence((0, "+A"), (5, "-B")), SCALE, 2) == (
        4, ["capped_narrow_evidence"],
    )
    assert decide_rating(4, narrow, SCALE, 2) == (4, [])


def test_decide_rating_bottom_needs_two_negative_incidents() -> None:
    """A 1 resting on one negative turn is raised to 2."""
    one = evidence((0, "-A"), (0, "-B"), (5, "+C"))
    two = evidence((0, "-A"), (5, "-A"))
    assert decide_rating(1, one, SCALE, 2) == (
        2, ["raised_narrow_evidence"],
    )
    assert decide_rating(1, two, SCALE, 2) == (1, [])
    assert decide_rating(2, one, SCALE, 2) == (2, [])


def test_decide_rating_flags_a_rating_contradicting_its_evidence() -> None:
    """High with only negatives, or low with only positives, is flagged."""
    flag = ["rating_contradicts_evidence"]
    assert decide_rating(4, evidence((0, "-"), (1, "-")), SCALE, 2) == (
        4, flag,
    )
    assert decide_rating(2, evidence((0, "+"), (1, "+")), SCALE, 2) == (
        2, flag,
    )


def test_decide_rating_contradiction_is_judged_after_the_adjustment() -> None:
    """An adjusted extreme is still flagged when it contradicts."""
    assert decide_rating(5, evidence((0, "-"), (1, "-")), SCALE, 2) == (
        4, ["capped_narrow_evidence", "rating_contradicts_evidence"],
    )
    assert decide_rating(1, evidence((0, "+A"), (1, "+B")), SCALE, 2) == (
        2, ["raised_narrow_evidence", "rating_contradicts_evidence"],
    )


def test_decide_rating_mixed_evidence_or_midpoint_is_not_flagged() -> None:
    """Mixed polarity, or the middle rating, is no contradiction."""
    mixed = evidence((0, "+"), (1, "-"))
    assert decide_rating(4, mixed, SCALE, 2) == (4, [])
    assert decide_rating(2, mixed, SCALE, 2) == (2, [])
    assert decide_rating(3, evidence((0, "-"), (1, "-")), SCALE, 2) == (3, [])
    assert decide_rating(3, evidence((0, "+"), (1, "+")), SCALE, 2) == (3, [])


def test_decide_rating_midpoint_of_an_even_scale() -> None:
    """On a 1-4 scale, 2 is low and 3 is high."""
    scale = Scale(minimum=1, maximum=4)
    positive = evidence((0, "+"), (1, "+"))
    assert decide_rating(2, positive, scale, 2)[1] == [
        "rating_contradicts_evidence"
    ]
    assert decide_rating(3, positive, scale, 2) == (3, [])


def test_decide_rating_adjustment_stays_on_a_two_point_scale() -> None:
    """Capping never leaves the scale, even with two points only."""
    scale = Scale(minimum=1, maximum=2)
    rating, _ = decide_rating(2, evidence((0, "+A"), (5, "+A")), scale, 2)
    assert rating in (1, 2)


@pytest.mark.parametrize("quote, text", [
    ("we should ship it", "we should not ship it today"),
    ("estoy de acuerdo con eso", "estoy nunca de acuerdo con eso"),
    ("lo veo claro", "lo veo nada claro"),
])
def test_quote_skipping_a_negation_is_rejected(quote: str, text: str) -> None:
    """Dropping a negation inverts the meaning, so it is no quote."""
    assert find_quote(quote, text) is None
