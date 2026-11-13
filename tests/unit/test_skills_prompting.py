"""Judge prompt construction and tolerant parsing of its reply."""

import json
from typing import Any

import pytest

from src.domain.skills.prompting import (
    format_rubric,
    format_turns,
    parse_judgement,
    system_prompt,
    user_prompt,
)
from src.domain.skills.rubric import Scale
from tests.fakes import make_catalogue, make_transcript

CATALOGUE = make_catalogue()
LISTENING = CATALOGUE.competency("listening")
TRANSCRIPT = make_transcript()


def reply(**fields: Any) -> str:
    """Return a JSON reply with the given fields."""
    return json.dumps(fields)


# -------------------------------------------------------------- prompts


def test_system_prompt_states_scale_incidents_and_language() -> None:
    """The scale bounds, the incident rule and the language are filled."""
    prompt = system_prompt(Scale(minimum=0, maximum=4), 3, "es")
    assert "at least 3 different turns" in prompt
    assert "<integer 0-4 or null>" in prompt
    assert "sentences in Spanish" in prompt
    assert '{"evidence": [{"turn": <turn number>' in prompt
    assert "{min_incidents}" not in prompt and "{{" not in prompt


def test_system_prompt_unknown_language_code_is_used_as_is() -> None:
    """A language without a known name is passed through."""
    assert "sentences in pt>" in system_prompt(Scale(), 2, "pt")
    assert "sentences in English" in system_prompt(Scale(), 2, "en")


def test_system_prompt_demands_verbatim_quotes_and_restraint() -> None:
    """The rules ask for verbatim quotes and allow citing nothing."""
    prompt = system_prompt(Scale(), 2, "en")
    assert "verbatim quote" in prompt
    assert "Most turns are NOT evidence" in prompt
    assert "Absence of evidence is not evidence of a low level" in prompt


def test_format_rubric_lists_indicators_with_polarity_and_anchors() -> None:
    """Indicators carry their sign; anchors fall back to scale labels."""
    text = format_rubric(LISTENING, CATALOGUE.scale)
    lines = text.splitlines()
    assert lines[0] == "Competency: Active listening (listening)"
    assert "  L1 (+) Asks a clarifying question" in lines
    assert "  L3 (-) Dismisses others" in lines
    assert "  1: rarely shown" in lines
    assert "  2: below" in lines
    assert "  5: shown consistently" in lines
    assert lines[-1] == "Do not infer: tone of voice"


def test_format_rubric_without_limits_or_anchors() -> None:
    """Optional sections are left out when empty."""
    bare = LISTENING.model_copy(update={"limits": "", "anchors": {}})
    text = format_rubric(bare, Scale())
    assert "Do not infer" not in text
    assert text.splitlines()[-1] == "Rating anchors:"


def test_format_turns_numbers_turns_and_adds_context() -> None:
    """Each candidate is numbered and preceded by the turn before it."""
    turns = [TRANSCRIPT.turns[3], TRANSCRIPT.turns[0]]
    assert format_turns(TRANSCRIPT, turns, "Maya").splitlines() == [
        f"[0] Maya: {TRANSCRIPT.turns[0].text}",
        f"(context) Priya: {TRANSCRIPT.turns[2].text}",
        f"[3] Maya: {TRANSCRIPT.turns[3].text}",
    ]


def test_format_turns_context_is_not_numbered_as_evidence() -> None:
    """Other speakers' words are shown without a turn number."""
    text = format_turns(TRANSCRIPT, [TRANSCRIPT.turns[1]], "Tom")
    assert text.splitlines() == [
        f"(context) Maya: {TRANSCRIPT.turns[0].text}",
        f"[1] Tom: {TRANSCRIPT.turns[1].text}",
    ]


def test_format_turns_does_not_repeat_turns() -> None:
    """A duplicated candidate is printed once."""
    turn = TRANSCRIPT.turns[0]
    assert format_turns(TRANSCRIPT, [turn, turn], "Maya").count("[0]") == 1
    assert format_turns(TRANSCRIPT, [], "Maya") == ""


def test_user_prompt_has_rubric_participant_and_excerpts() -> None:
    """The user turn names the participant and ends asking for JSON."""
    prompt = user_prompt(
        TRANSCRIPT, [TRANSCRIPT.turns[0]], "Maya", LISTENING, CATALOGUE.scale
    )
    assert prompt.startswith("Competency: Active listening (listening)")
    assert "Participant to assess: Maya" in prompt
    assert "[0] Maya: What do you mean exactly" in prompt
    assert prompt.endswith("JSON:")


# ------------------------------------------------------ parse_judgement


def test_parse_judgement_well_formed_reply() -> None:
    """Evidence, rating and rationale are read from a JSON object."""
    judgement = parse_judgement(reply(
        evidence=[{"turn": 3, "quote": "so if I understood", "indicator":
                   "L2"}],
        rating=4, rationale="Clear  summary.\nGood.",
    ))
    assert not judgement.parse_error
    assert judgement.rating == 4
    assert judgement.rationale == "Clear summary. Good."
    assert [(e.turn, e.quote, e.indicator) for e in judgement.evidence] == [
        (3, "so if I understood", "L2"),
    ]


@pytest.mark.parametrize("wrapper", [
    "Here is my assessment:\n{}\nI hope it helps.",
    "```json\n{}\n```",
    "```\n{}\n```",
    "<think>short thought</think>\n{}",
    "\n\n   {}   \n",
])
def test_parse_judgement_json_wrapped_in_prose_or_fences(
    wrapper: str,
) -> None:
    """Text or code fences around the object are tolerated."""
    body = reply(evidence=[], rating=2, rationale="ok")
    judgement = parse_judgement(wrapper.replace("{}", body))
    assert not judgement.parse_error
    assert judgement.rating == 2


def test_parse_judgement_braces_inside_strings() -> None:
    """Braces and quotes inside a quote do not break the parser."""
    quote = 'use {"key": {"a": 1}} and } then {'
    judgement = parse_judgement("Answer: " + reply(
        evidence=[{"turn": 1, "quote": quote, "indicator": "L1"}],
        rating=None, rationale="{nested}",
    ))
    assert judgement.evidence[0].quote == quote
    assert judgement.rationale == "{nested}"
    assert not judgement.parse_error


@pytest.mark.parametrize("value, expected", [
    (4, 4), ("4", 4), (" 4 ", 4), (4.0, 4), ("-1", -1), (0, 0),
    (None, None), (4.5, None), ("4.5", None), ("four", None), ("", None),
    (True, None), (False, None), ([4], None), ({"value": 4}, None),
    ("4/5", None),
])
def test_parse_judgement_rating_types(value: Any, expected: Any) -> None:
    """Integers (also as text or whole floats) pass; the rest is null."""
    judgement = parse_judgement(reply(rating=value))
    assert judgement.rating == expected
    assert not judgement.parse_error


def test_parse_judgement_non_finite_numbers() -> None:
    """``NaN`` and overflowing numbers are not ratings."""
    assert parse_judgement('{"rating": NaN}').rating is None
    assert parse_judgement('{"rating": 1e999}').rating is None
    assert parse_judgement('{"rating": -Infinity}').rating is None


@pytest.mark.parametrize("value", ["--4", "²", "---1"])
def test_parse_judgement_odd_numeric_strings_do_not_crash(
    value: str,
) -> None:
    """A string that only looks numeric gives a null, not an exception."""
    assert parse_judgement(reply(rating=value)).rating is None
    judgement = parse_judgement(reply(evidence=[{"turn": value}]))
    assert judgement.evidence[0].turn is None


def test_parse_judgement_missing_keys() -> None:
    """An empty object is an empty judgement, not a parse error."""
    judgement = parse_judgement("{}")
    assert (judgement.evidence, judgement.rating, judgement.rationale) == (
        [], None, "",
    )
    assert not judgement.parse_error


def test_parse_judgement_evidence_item_types() -> None:
    """Non-object items are skipped; missing fields get neutral values."""
    judgement = parse_judgement(reply(evidence=[
        "a quote alone", 7, None, ["x"],
        {"turn": "2", "quote": "said this"},
        {"turn": None, "quote": None, "indicator": None},
        {"turn": 1.0, "quote": 5, "indicator": 7},
        {},
    ]))
    assert [(e.turn, e.quote, e.indicator) for e in judgement.evidence] == [
        (2, "said this", ""), (None, "", ""), (1, "5", "7"), (None, "", ""),
    ]


@pytest.mark.parametrize("evidence", [None, "none", 3, {"turn": 1}, True])
def test_parse_judgement_evidence_that_is_not_a_list(evidence: Any) -> None:
    """A wrongly typed ``evidence`` is read as no evidence."""
    judgement = parse_judgement(reply(evidence=evidence, rating=3))
    assert judgement.evidence == []
    assert judgement.rating == 3


def test_parse_judgement_rationale_types() -> None:
    """A null rationale is empty; other types are rendered as text."""
    assert parse_judgement(reply(rationale=None)).rationale == ""
    assert parse_judgement(reply(rationale=12)).rationale == "12"


@pytest.mark.parametrize("text", [
    "", "   ", "I cannot assess this.", "rating: 4", "{", "}", "}{",
    "{rating: 4}", "{'rating': 4}", '{"rating": 4', '"just a string"',
    "42", "null", "[1, 2, 3]", "[]", '{"rating": 4,}',
])
def test_parse_judgement_garbage_is_a_parse_error(text: str) -> None:
    """Anything without a JSON object yields an empty, flagged judgement."""
    judgement = parse_judgement(text)
    assert judgement.parse_error
    assert (judgement.evidence, judgement.rating) == ([], None)


def test_parse_judgement_two_objects_is_a_parse_error() -> None:
    """Two objects in one reply are ambiguous: nothing is invented."""
    judgement = parse_judgement('{"rating": 5} or maybe {"rating": 1}')
    assert judgement.parse_error and judgement.rating is None


def test_parse_judgement_prose_with_braces_is_a_parse_error() -> None:
    """Braces in the surrounding prose defeat the extraction, safely."""
    judgement = parse_judgement('Using {rubric}: {"rating": 5}')
    assert judgement.parse_error and judgement.rating is None


def test_parse_judgement_object_inside_an_array_is_a_parse_error() -> None:
    """A reply that is a JSON array is not the object asked for."""
    for text in ('[{"rating": 4}]', '  \n[{"rating": 4}, {"rating": 1}]'):
        judgement = parse_judgement(text)
        assert judgement.parse_error and judgement.rating is None
