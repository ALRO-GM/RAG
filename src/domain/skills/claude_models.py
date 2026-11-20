"""Which Claude model judges each step of an analysis.

Three steps of ``skills analyze`` ask a judge: ``confirmation`` (f),
``assessment`` (g) and ``synthesis`` (i). When the judge is Claude
Code, each may use its own model: an alias (``haiku``, ``sonnet``,
``opus``) or a full model id (``claude-haiku-4-5-20251001``). An empty
choice means the run's default model (``claude_model``), so nothing
changes unless someone chooses.

A model name ends up as one argument of the ``claude`` command, so it
is checked against an allow-list pattern here: lower-case letters,
digits, dots and dashes, starting with an alias or ``claude-``, with
Claude Code's optional ``[1m]`` (long context) suffix. Nothing that
could read as an option (``--…``) or carry a space or a quote passes.

The measured trade-off between the models is in
``docs/skills/MODEL_COMPARISON.md``. Plain data: no I/O.
"""

import re
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, field_validator

from src.domain.errors import InvalidInputError

JudgeStep = Literal["confirmation", "assessment", "synthesis"]
JUDGE_STEPS: Tuple[JudgeStep, ...] = (
    "confirmation", "assessment", "synthesis",
)
"""The steps of an analysis that ask a judge, in the order they run."""

CLAUDE_MODEL_ALIASES: Tuple[str, ...] = ("haiku", "sonnet", "opus")
"""Claude Code's aliases, cheapest first."""

MAX_MODEL_LENGTH = 64
"""Longest model name accepted."""

CLAUDE_MODEL_PATTERN = re.compile(
    r"(?:haiku|sonnet|opus|claude-[a-z0-9]+(?:[.-][a-z0-9]+)*)(?:\[1m\])?"
)
"""What a Claude model name may look like (matched in full)."""


def check_claude_model(name: str, empty_ok: bool = False) -> str:
    """Return ``name`` stripped if it is an acceptable Claude model.

    ``empty_ok`` accepts ``""`` (the default model). Raises
    :class:`~src.domain.errors.InvalidInputError` otherwise; the
    message never repeats more than the first characters of the input.
    """
    value = name.strip() if isinstance(name, str) else ""
    if not value and empty_ok and isinstance(name, str):
        return ""
    if len(value) > MAX_MODEL_LENGTH or not CLAUDE_MODEL_PATTERN.fullmatch(
        value
    ):
        raise InvalidInputError(
            f"Invalid Claude model {str(name)[:20]!r}: use "
            f"{', '.join(CLAUDE_MODEL_ALIASES)} or a full id such as "
            "claude-haiku-4-5."
        )
    return value


class ClaudeModels(BaseModel):
    """The Claude model chosen for each judged step (``""``: default)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation: str = ""
    assessment: str = ""
    synthesis: str = ""

    @field_validator("confirmation", "assessment", "synthesis")
    @classmethod
    def _valid(cls, value: str) -> str:
        """Accept only names the allow-list pattern matches, or ``""``."""
        try:
            return check_claude_model(value, empty_ok=True)
        except InvalidInputError as exc:
            raise ValueError(str(exc)) from exc

    def model_for(self, step: JudgeStep, default: str) -> str:
        """Return the model of ``step``: the chosen one or ``default``."""
        chosen: str = getattr(self, step)
        return chosen or default

    def chosen(self) -> "dict[str, str]":
        """Return only the steps with a model chosen."""
        return {step: getattr(self, step) for step in JUDGE_STEPS
                if getattr(self, step)}
