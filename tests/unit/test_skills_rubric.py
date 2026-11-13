"""Rubric catalogue validation and lookups; interaction descriptors."""

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.domain.errors import InvalidInputError
from src.domain.skills.interaction import (
    count_questions,
    describe,
    equality,
    gini,
)
from src.domain.skills.rubric import Indicator, RubricCatalogue, Scale
from tests.fakes import catalogue_data, make_catalogue, make_transcript


def broken(**changes: Any) -> Dict[str, Any]:
    """Return the test catalogue data with top-level ``changes``."""
    return {**catalogue_data(), **changes}


# -------------------------------------------------------------- rubric


def test_make_catalogue_is_valid_in_both_languages() -> None:
    """The test catalogue has the same structure in English and Spanish."""
    english, spanish = make_catalogue("en"), make_catalogue("es")
    assert english.structure() == spanish.structure() == {
        "listening": ["L1:positive", "L2:positive", "L3:negative"],
        "planning": ["P1:positive", "P2:negative"],
    }
    assert spanish.language == "es"


def test_indicator_query_joins_text_and_examples() -> None:
    """The retrieval query is the indicator text plus its examples."""
    indicator = Indicator(id="X", text="Asks", examples=["why?", "how?"])
    assert indicator.query() == "Asks why? how?"
    assert indicator.polarity == "positive"
    assert Indicator(id="X", text="Asks").query() == "Asks"


def test_competency_indicator_lookup() -> None:
    """An indicator is found only in its own competency."""
    listening = make_catalogue().competency("listening")
    found = listening.indicator("L3")
    assert found is not None and found.polarity == "negative"
    assert listening.indicator("P1") is None
    assert listening.indicator("") is None


def test_scale_points_and_midpoint() -> None:
    """A scale lists its points and its centre."""
    assert Scale().points() == [1, 2, 3, 4, 5]
    assert Scale().midpoint() == 3.0
    assert Scale(minimum=0, maximum=3).points() == [0, 1, 2, 3]
    assert Scale(minimum=1, maximum=4).midpoint() == 2.5


@pytest.mark.parametrize("minimum, maximum", [(5, 1), (3, 3)])
def test_scale_must_be_increasing(minimum: int, maximum: int) -> None:
    """An inverted or one-point scale is rejected."""
    with pytest.raises(ValidationError, match="above its minimum"):
        Scale(minimum=minimum, maximum=maximum)


def test_catalogue_unknown_competency_names_the_choices() -> None:
    """An unknown id is a user error listing the valid ids."""
    with pytest.raises(InvalidInputError, match="listening, planning"):
        make_catalogue().competency("leadership")


def test_catalogue_select() -> None:
    """``select`` keeps the requested order and defaults to everything."""
    catalogue = make_catalogue()
    assert [c.id for c in catalogue.select(None)] == ["listening", "planning"]
    assert [c.id for c in catalogue.select([])] == ["listening", "planning"]
    assert [c.id for c in catalogue.select(["planning"])] == ["planning"]
    with pytest.raises(InvalidInputError, match="Unknown competency"):
        catalogue.select(["planning", "nope"])


def test_catalogue_rejects_duplicate_competency_ids() -> None:
    """Two competencies cannot share an id."""
    data = catalogue_data()
    data["competencies"][1]["id"] = "listening"
    with pytest.raises(ValidationError, match="duplicate competency id"):
        RubricCatalogue.model_validate(data)


def test_catalogue_rejects_duplicate_indicator_ids() -> None:
    """Indicator ids are unique across the whole catalogue."""
    data = catalogue_data()
    data["competencies"][1]["indicators"][0]["id"] = "L1"
    with pytest.raises(ValidationError, match="duplicate indicator id L1"):
        RubricCatalogue.model_validate(data)


def test_catalogue_rejects_anchors_outside_the_scale() -> None:
    """An anchor for a point the scale does not have is an error."""
    data = catalogue_data()
    data["competencies"][0]["anchors"] = {6: "beyond"}
    with pytest.raises(ValidationError, match="anchors outside the scale"):
        RubricCatalogue.model_validate(data)


def test_catalogue_needs_a_positive_indicator_per_competency() -> None:
    """A competency described only by what not to do is rejected."""
    data = catalogue_data()
    for indicator in data["competencies"][1]["indicators"]:
        indicator["polarity"] = "negative"
    with pytest.raises(ValidationError, match="needs a positive indicator"):
        RubricCatalogue.model_validate(data)


@pytest.mark.parametrize("changes", [
    {"competencies": []}, {"name": ""}, {"version": ""}, {"language": "e"},
    {"scale": {"minimum": 4, "maximum": 2}},
])
def test_catalogue_rejects_empty_or_invalid_fields(
    changes: Dict[str, Any],
) -> None:
    """Required fields cannot be empty."""
    with pytest.raises(ValidationError):
        RubricCatalogue.model_validate(broken(**changes))


def test_catalogue_rejects_unknown_polarity_and_empty_indicators() -> None:
    """Polarity is positive or negative; a competency has indicators."""
    data = catalogue_data()
    data["competencies"][0]["indicators"][0]["polarity"] = "neutral"
    with pytest.raises(ValidationError):
        RubricCatalogue.model_validate(data)
    data = catalogue_data()
    data["competencies"][0]["indicators"] = []
    with pytest.raises(ValidationError):
        RubricCatalogue.model_validate(data)


# --------------------------------------------------------- interaction


def test_gini_and_equality_extremes() -> None:
    """Equal shares give Gini 0; one talker alone gives equality 0."""
    assert gini([5.0, 5.0, 5.0]) == pytest.approx(0.0)
    assert gini([]) == 0.0
    assert gini([0.0, 0.0]) == 0.0
    assert equality([5.0, 5.0, 5.0]) == pytest.approx(1.0)
    assert equality([9.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert equality([7.0]) == 1.0
    assert equality([]) == 1.0
    assert 0.0 < equality([6.0, 3.0, 1.0]) < 1.0


def test_count_questions_spanish_and_english() -> None:
    """Questions are counted once, by closing or by opening marks."""
    assert count_questions("¿Vale? ¿Seguro?") == 2
    assert count_questions("Really? Why?") == 2
    assert count_questions("¿Y si lo dejamos") == 1
    assert count_questions("No question.") == 0


def test_describe_counts_turns_words_and_shares() -> None:
    """Per-speaker counts add up to the team totals."""
    team = describe(make_transcript([
        ("A", "one two three four?"), ("B", "five six"),
        ("A", "seven eight"), ("C", "nine ten. Really? Yes?"),
    ]))
    assert (team.speakers, team.turns, team.words) == (3, 4, 12)
    a = team.of("A")
    assert (a.turns, a.words, a.questions) == (2, 6, 1)
    assert a.word_share == pytest.approx(0.5)
    assert a.turn_share == pytest.approx(0.5)
    assert a.mean_words_per_turn == pytest.approx(3.0)
    assert team.of("C").questions == 2
    assert (team.dominant_speaker, team.dominant_share) == ("A", 0.5)
    assert [s.speaker for s in team.per_speaker] == ["A", "B", "C"]
    assert sum(s.word_share for s in team.per_speaker) == pytest.approx(1.0)


def test_describe_without_timestamps_reports_no_durations() -> None:
    """Untimed transcripts have no seconds and no overlap counts."""
    team = describe(make_transcript([("A", "hello there"), ("B", "hi")]))
    assert not team.has_timestamps
    assert team.of("A").seconds is None
    assert team.of("A").overlapping_starts is None


def test_describe_durations_and_overlapping_starts() -> None:
    """A start well before the previous speaker stops is an overlap."""
    team = describe(make_transcript([
        ("A", "first turn", 0.0, 10.0),
        ("B", "cuts in", 9.0, 12.0),
        ("A", "barely early", 11.8, 15.0),
        ("B", "after the end", 15.0, 16.5),
    ]))
    assert team.has_timestamps
    assert team.of("A").seconds == pytest.approx(13.2)
    assert team.of("B").seconds == pytest.approx(4.5)
    assert team.of("B").overlapping_starts == 1
    assert team.of("A").overlapping_starts == 0


def test_describe_partially_timed_transcript_is_untimed() -> None:
    """One turn without times disables every duration."""
    team = describe(make_transcript([
        ("A", "timed", 0.0, 1.0), ("B", "also timed", 1.0, 2.0),
    ]).model_copy(update={"turns": make_transcript([
        ("A", "timed", 0.0, 1.0), ("B", "untimed"),
    ]).turns}))
    assert not team.has_timestamps
    assert team.of("A").seconds is None


def test_describe_participation_equality() -> None:
    """Equal talkers score 1; a monologue with echoes scores near 0."""
    even = describe(make_transcript([("A", "a b c"), ("B", "d e f")]))
    uneven = describe(make_transcript([
        ("A", "a b c d e f g h i j k l m n o p q r s t"), ("B", "ok"),
    ]))
    assert even.participation_equality == pytest.approx(1.0)
    assert uneven.participation_equality < 0.2


def test_describe_single_speaker_and_wordless_turns() -> None:
    """One speaker, or turns of punctuation only, do not divide by zero."""
    alone = describe(make_transcript([("A", "talking to myself")]))
    assert alone.participation_equality == 1.0
    assert alone.dominant_share == 1.0
    silent = describe(make_transcript([("A", "..."), ("B", "?!")]))
    assert silent.words == 0
    assert silent.of("A").word_share == 0.0
    assert silent.of("B").questions == 1


def test_team_interaction_of_unknown_speaker() -> None:
    """Asking for somebody who did not speak raises ``KeyError``."""
    team = describe(make_transcript([("A", "hello")]))
    with pytest.raises(KeyError):
        team.of("Z")
