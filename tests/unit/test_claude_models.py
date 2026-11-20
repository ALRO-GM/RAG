"""Choosing the Claude model of each judged step: allow-list and defaults."""

import pytest
from pydantic import ValidationError

from src.domain.errors import InvalidInputError
from src.domain.skills.claude_models import (
    JUDGE_STEPS,
    ClaudeModels,
    check_claude_model,
)


@pytest.mark.parametrize("name", [
    "haiku", "sonnet", "opus", "opus[1m]", "claude-haiku-4-5",
    "claude-haiku-4-5-20251001", "claude-opus-4.1", " sonnet ",
])
def test_aliases_and_full_ids_pass(name: str) -> None:
    """Aliases, full ids and the long-context suffix are accepted."""
    assert check_claude_model(name) == name.strip()


@pytest.mark.parametrize("name", [
    "", "--model", "-p", "opus; rm -rf ~", "opus --tools Bash", "OPUS",
    "gpt-4o", "claude-", "claude--x", "hai ku", "o'pus", "$(id)",
    "claude-" + "x" * 64, "opus[2m]",
])
def test_anything_else_is_refused(name: str) -> None:
    """Options, spaces, quotes, shell syntax and other vendors fail."""
    with pytest.raises(InvalidInputError):
        check_claude_model(name)


def test_the_empty_choice_means_the_default() -> None:
    """``""`` is accepted only where it means "the configured model"."""
    assert check_claude_model("", empty_ok=True) == ""
    assert check_claude_model("  ", empty_ok=True) == ""
    models = ClaudeModels(confirmation="haiku")
    assert models.model_for("confirmation", "opus") == "haiku"
    assert models.model_for("assessment", "opus") == "opus"
    assert models.chosen() == {"confirmation": "haiku"}
    assert ClaudeModels().chosen() == {}
    assert JUDGE_STEPS == ("confirmation", "assessment", "synthesis")


def test_a_bad_choice_or_an_unknown_step_is_a_validation_error() -> None:
    """The model refuses names off the pattern and unknown steps."""
    with pytest.raises(ValidationError):
        ClaudeModels(synthesis="opus && echo")
    with pytest.raises(ValidationError):
        ClaudeModels.model_validate({"acts": "haiku"})
