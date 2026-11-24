"""The prepared report of a scripted support call (``live.demo.v1``, #250).

A demo call may carry a report for the supervisor: what each line of the
agent did, how the call followed the reference script, how other agents
word the same moments, and whether anything was left unsaid. In the
demo all of it is written by hand for the fictitious call and says so;
nothing here measures a real agent. Every reference names a sentence of
the call (checked by the demo), so the page can point at the exact line.

Texts are Spanish with an ``en`` mapping of the same keys in English.
Plain data and rules: no I/O.
"""

from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator

Effect = Literal["helped", "neutral", "hurt"]
"""What a line did, judged by the customer's next reaction."""
StepStatus = Literal["followed", "skipped", "reordered"]
"""What became of a step of the reference script."""
ItemOutcome = Literal["said", "said_late", "never"]
"""What became of a mandatory item."""
Gap = Literal["question", "data"]
"""A customer's question, or a data point the customer gave."""


class _Text(BaseModel):
    """A block of texts with their English version."""

    model_config = ConfigDict(extra="forbid")

    en: Optional[Dict[str, str]] = None
    """The same text keys in English."""

    def lines(self) -> List[str]:
        """Return the sentence ids this block points at."""
        return []

    def texts(self) -> List[str]:
        """Return the names of the fields that must be translated."""
        return []


class LineReview(_Text):
    """What one line of the agent did."""

    utterance: str
    effect: Effect
    reaction: str
    """The customer sentence that shows it: the next reaction."""
    why: str = Field(min_length=1)
    better: Optional[str] = None
    """A better wording, always given when the line hurt."""

    def lines(self) -> List[str]:
        """Return the line and its reaction."""
        return [self.utterance, self.reaction]

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["why", "better"] if self.better else ["why"]

    @model_validator(mode="after")
    def _hurt_has_better(self) -> "LineReview":
        """A line that hurt always comes with a better wording."""
        if self.effect == "hurt" and not self.better:
            raise ValueError("a line that hurt needs a better wording")
        return self


class Point(_Text):
    """A strength or something to improve, with the lines it comes from."""

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    about: List[str] = Field(min_length=1)

    def lines(self) -> List[str]:
        """Return the lines it points at."""
        return list(self.about)

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["title", "text"]


class Summary(_Text):
    """The short summary on top of the report."""

    result: str = Field(min_length=1)
    strengths: List[Point] = Field(min_length=2, max_length=2)
    improve: List[Point] = Field(min_length=2, max_length=2)

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["result"]


class ScriptStep(_Text):
    """One step of the reference script and what became of it."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: StepStatus
    about: List[str] = Field(default_factory=list)
    """The lines where the step was done (none when it was skipped)."""
    note: Optional[str] = None

    def lines(self) -> List[str]:
        """Return the lines it points at."""
        return list(self.about)

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["name", "note"] if self.note else ["name"]

    @model_validator(mode="after")
    def _skipped_has_no_lines(self) -> "ScriptStep":
        """A skipped step has no line; any other has at least one."""
        if (self.status == "skipped") == bool(self.about):
            raise ValueError("skipped steps have no lines, others do")
        return self


class ScriptVerdict(_Text):
    """Whether the script itself looks improvable, and why."""

    improvable: bool
    step: Optional[str] = None
    """The script step customers reacted badly to."""
    text: str = Field(min_length=1)
    about: List[str] = Field(default_factory=list)
    caution: str = Field(min_length=1)
    """What one call can and cannot tell."""

    def lines(self) -> List[str]:
        """Return the lines it points at."""
        return list(self.about)

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["text", "caution"]


class Script(_Text):
    """The reference script of this type of call and how the call went."""

    name: str = Field(min_length=1)
    steps: List[ScriptStep] = Field(min_length=1)
    verdict: ScriptVerdict

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["name"]


class TopMoment(_Text):
    """How other agents word a moment of the call."""

    id: str = Field(min_length=1)
    moment: str = Field(min_length=1)
    utterance: str
    """The agent's line at that moment."""
    top: str = Field(min_length=1)
    """The illustrative wording of a top-rated agent."""
    why: str = Field(min_length=1)

    def lines(self) -> List[str]:
        """Return the line it points at."""
        return [self.utterance]

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["moment", "top", "why"]


class TopScorers(_Text):
    """The top-scorers block: always illustrative sample data."""

    illustrative: Literal[True]
    note: str = Field(min_length=1)
    moments: List[TopMoment] = Field(min_length=1)

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["note"]


class MandatoryResult(_Text):
    """What became of one mandatory item in the call."""

    item: str
    outcome: ItemOutcome
    utterance: Optional[str] = None
    """The line that said it (none when never said)."""
    note: str = Field(min_length=1)

    def lines(self) -> List[str]:
        """Return the line it points at."""
        return [self.utterance] if self.utterance else []

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["note"]

    @model_validator(mode="after")
    def _said_has_a_line(self) -> "MandatoryResult":
        """An item never said has no line; any other has one."""
        if (self.outcome == "never") == bool(self.utterance):
            raise ValueError("never-said items have no line, others do")
        return self


class Unanswered(_Text):
    """A customer question or data point the agent left without an answer."""

    utterance: str
    kind: Gap
    text: str = Field(min_length=1)

    def lines(self) -> List[str]:
        """Return the customer line."""
        return [self.utterance]

    def texts(self) -> List[str]:
        """Return the translatable fields."""
        return ["text"]


class CallReport(_Text):
    """The report of a demo call: five blocks, all prepared by hand."""

    summary: Summary
    lines_review: List[LineReview] = Field(min_length=1)
    script: Script
    top_scorers: TopScorers
    mandatory: List[MandatoryResult] = Field(min_length=1)
    unanswered: List[Unanswered] = Field(default_factory=list)

    def blocks(self) -> List[_Text]:
        """Return every block that carries texts and lines."""
        found: List[_Text] = [self.summary, *self.summary.strengths,
                              *self.summary.improve, *self.lines_review,
                              self.script, *self.script.steps,
                              self.script.verdict, self.top_scorers,
                              *self.top_scorers.moments, *self.mandatory,
                              *self.unanswered]
        return found

    def sentences(self) -> Set[str]:
        """Return every sentence id the report points at."""
        return {u for block in self.blocks() for u in block.lines()}

    def missing_english(self) -> List[str]:
        """Return the texts of the report that have no English version."""
        missing: List[str] = []
        for block in self.blocks():
            known = block.en or {}
            missing += [f"{type(block).__name__}.{key}"
                        for key in block.texts() if not known.get(key)]
        return missing
