"""The pipeline settings of the web's «Ajustes» page (``settings.v1``).

Every step of the pipeline that has measured alternatives gets one
choice among a closed list of options (an allow-list: no free text ever
reaches a command line). Three presets fill every choice at once:
``quality`` (what the services do today: best measured models),
``balanced`` (Haiku on the step that makes most Claude calls) and
``cost`` (no Claude quota, smaller models); any other combination is
``custom``. The figures behind each option are in
``docs/SETTINGS.md``.

The **first option of every step is the service's own configuration**
(the default before this page existed), so a missing or unreadable
settings file changes nothing. ``applies`` tells when a change takes
effect: ``next_job`` (the next transcription, analysis, upload or live
session reads it) or ``restart`` (the model is loaded when the service
starts). Steps whose ``consumer`` is ``web`` are read by the live page
itself (what it shows), not by a service.

The RAG owns the document (``data/settings.json``); the transcriber
forwards the web's ``/api/settings`` to it and the orchestrator reads
it over HTTP. Plain data and validation: no I/O.
"""

import math
import re
from datetime import datetime
from typing import Dict, List, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.errors import InvalidInputError

SCHEMA = "settings.v1"

Applies = Literal["next_job", "restart"]
Consumer = Literal["transcriber", "rag", "orchestrator", "web"]
PresetName = Literal["quality", "balanced", "cost"]
PRESET_NAMES: Tuple[PresetName, ...] = ("quality", "balanced", "cost")
CUSTOM = "custom"


SPEAKER_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
"""An enrolled person's id (the transcriber's speaker id pattern)."""


class StepSpec(BaseModel):
    """One configurable step: its options, default and who reads it.

    Besides its closed ``options``, a step may accept a number within
    ``range`` (``low``, ``high``, the page's slider ``step``), kept as
    its shortest decimal text with at most three decimals (``"1.1"``),
    or a value matching ``pattern`` (a person's id, for instance).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    group: Literal[
        "transcription", "live", "analysis", "knowledge", "voicebot",
        "interview",
    ]
    options: Tuple[str, ...]
    applies: Applies = "next_job"
    consumer: Consumer
    range: Optional[Tuple[float, float, float]] = None
    pattern: Optional[str] = None

    @property
    def default(self) -> str:
        """Return the first option: the service's own configuration."""
        return self.options[0]

    def accept(self, option: object) -> Optional[str]:
        """Return ``option`` as stored, or ``None`` if it is not allowed."""
        if not isinstance(option, str):
            return None
        if option in self.options:
            return option
        if self.pattern is not None and re.fullmatch(self.pattern, option):
            return option
        if self.range is not None:
            return number_text(option, self.range[0], self.range[1])
        return None


NUMBER = re.compile(r"[0-9]{1,2}(?:\.[0-9]{1,3})?")


def number_text(text: str, low: float, high: float) -> Optional[str]:
    """Return ``text`` as a canonical number in ``[low, high]``, or None.

    At most three decimals; ``"1.10"`` becomes ``"1.1"``, ``"1.0"``
    becomes ``"1"``.
    """
    if not NUMBER.fullmatch(text):
        return None
    value = float(text)
    if not math.isfinite(value) or not low <= value <= high:
        return None
    return f"{round(value, 3):g}"


CLAUDE_ALIASES: Tuple[str, ...] = ("opus", "sonnet", "haiku")

STEPS: Tuple[StepSpec, ...] = (
    StepSpec(id="transcription.recogniser", group="transcription",
             options=("whisper-large-v3-turbo", "whisper-large-v3"),
             applies="restart", consumer="transcriber"),
    StepSpec(id="transcription.diarizer", group="transcription",
             options=("auto", "diarizen", "nemotron", "pyannote",
                      "builtin"),
             consumer="transcriber"),
    StepSpec(id="transcription.names", group="transcription",
             options=("qwen3-8b", "qwen3-4b", "rules"),
             consumer="transcriber"),
    StepSpec(id="live.asr", group="live", options=("gpu", "cpu"),
             consumer="transcriber"),
    StepSpec(id="live.diarization", group="live", options=("on", "off"),
             consumer="orchestrator"),
    StepSpec(id="live.suggester_model", group="live",
             options=("haiku", "sonnet", "opus"), consumer="orchestrator"),
    StepSpec(id="live.suggester_effort", group="live",
             options=("low", "medium", "high"), consumer="rag"),
    StepSpec(id="live.deadline", group="live",
             options=("product", "1800", "2800"), consumer="orchestrator"),
    StepSpec(id="live.assist_mode", group="live",
             options=("pincelada", "guiado"), consumer="orchestrator"),
    StepSpec(id="live.max_suggestions", group="live",
             options=("3", "1", "2", "5"), consumer="web"),
    StepSpec(id="live.show_employee", group="live", options=("on", "off"),
             consumer="web"),
    StepSpec(id="live.show_other", group="live", options=("on", "off"),
             consumer="web"),
    StepSpec(id="analysis.judge", group="analysis",
             options=("claude_code", "local", "both", "opencode"),
             consumer="rag"),
    StepSpec(id="analysis.confirmation_model", group="analysis",
             options=CLAUDE_ALIASES, consumer="rag"),
    StepSpec(id="analysis.assessment_model", group="analysis",
             options=CLAUDE_ALIASES, consumer="rag"),
    StepSpec(id="analysis.synthesis_model", group="analysis",
             options=CLAUDE_ALIASES, consumer="rag"),
    StepSpec(id="analysis.local_judge", group="analysis",
             options=("qwen3-8b", "qwen3-4b"), consumer="rag"),
    StepSpec(id="analysis.act_tagger", group="analysis",
             options=("encoder", "combined", "rules"), consumer="rag"),
    StepSpec(id="knowledge.ocr", group="knowledge", options=("on", "off"),
             consumer="rag"),
    StepSpec(id="knowledge.captions", group="knowledge",
             options=("florence", "claude", "off"), consumer="rag"),
    StepSpec(id="voicebot.reasoner_model", group="voicebot",
             options=("haiku", "sonnet", "opus"), consumer="orchestrator"),
    StepSpec(id="voicebot.voice", group="voicebot",
             options=("davefx", "sharvard"), consumer="orchestrator",
             pattern=rf"clone:{SPEAKER_ID}"),
    # Piper's own controls; the defaults are the voices' (davefx and
    # sharvard both say noise_scale 0.5, noise_w 0.333 in their
    # .onnx.json), so nothing saved changes nothing.
    StepSpec(id="voicebot.length_scale", group="voicebot", options=("1",),
             consumer="orchestrator", range=(0.7, 1.5, 0.05)),
    StepSpec(id="voicebot.noise_scale", group="voicebot", options=("0.5",),
             consumer="orchestrator", range=(0.0, 1.0, 0.05)),
    StepSpec(id="voicebot.noise_w", group="voicebot", options=("0.333",),
             consumer="orchestrator", range=(0.0, 1.0, 0.05)),
    StepSpec(id="voicebot.sentence_silence", group="voicebot",
             options=("0",), consumer="orchestrator",
             range=(0.0, 1.0, 0.05)),
    # "off", or the id of an enrolled person whose measured speaking
    # rate sets the length scale (it then wins over the one above).
    StepSpec(id="voicebot.match_rate", group="voicebot", options=("off",),
             consumer="orchestrator", pattern=SPEAKER_ID),
    StepSpec(id="interview.slot_filler", group="interview",
             options=("haiku", "heuristic"), consumer="orchestrator"),
    StepSpec(id="interview.max_probes", group="interview",
             options=("1", "0", "2"), consumer="orchestrator"),
    StepSpec(id="interview.minutes", group="interview",
             options=("20", "15", "30", "45"), consumer="orchestrator"),
    StepSpec(id="interview.post_call", group="interview",
             options=("off", "on"), consumer="orchestrator"),
)
"""Every configurable step, in the order the page shows them."""

STEP_BY_ID: Dict[str, StepSpec] = {s.id: s for s in STEPS}

Choices = Dict[str, str]


def defaults() -> Choices:
    """Return every step at its default (the services as configured)."""
    return {s.id: s.default for s in STEPS}


PRESETS: Dict[PresetName, Choices] = {
    # Today's defaults: the best measured option of every step.
    "quality": defaults(),
    # Haiku only where Claude is called most (confirmations: ~100 of
    # ~150 calls of a session); assessment and synthesis keep Opus.
    "balanced": {**defaults(), "analysis.confirmation_model": "haiku"},
    # No Claude quota for the analysis and the smallest local models.
    "cost": {
        **defaults(),
        "analysis.judge": "local",
        "analysis.local_judge": "qwen3-4b",
        "analysis.confirmation_model": "haiku",
        "analysis.assessment_model": "haiku",
        "analysis.synthesis_model": "haiku",
        "transcription.names": "qwen3-4b",
        "transcription.diarizer": "nemotron",
    },
}
"""What each preset chooses (the reasons are in ``docs/SETTINGS.md``)."""


def check_choices(given: Mapping[str, str], partial: bool = True) -> Choices:
    """Return ``given`` checked against the allow-lists.

    Raises :class:`~src.domain.errors.InvalidInputError` for an unknown
    step or option (``partial=False`` also wants every step). The
    message repeats at most the first characters of the input.
    """
    checked: Choices = {}
    for step, option in given.items():
        spec = STEP_BY_ID.get(step) if isinstance(step, str) else None
        if spec is None:
            raise InvalidInputError(f"Unknown setting {str(step)[:40]!r}.")
        accepted = spec.accept(option)
        if accepted is None:
            raise InvalidInputError(
                f"Invalid option {str(option)[:20]!r} for {step}; choose "
                f"one of {', '.join(spec.options)}{_also(spec)}."
            )
        checked[step] = accepted
    if not partial and set(checked) != set(STEP_BY_ID):
        missing = sorted(set(STEP_BY_ID) - set(checked))
        raise InvalidInputError(f"Missing settings: {', '.join(missing)}.")
    return checked


def _also(spec: StepSpec) -> str:
    """Say what else a step accepts, for an error message."""
    if spec.range is not None:
        return f", or a number from {spec.range[0]:g} to {spec.range[1]:g}"
    if spec.pattern is not None:
        return ", or an enrolled person's id"
    return ""


def resolve(stored: Mapping[str, str]) -> Choices:
    """Return every step: the valid stored choices, defaults elsewhere.

    Lenient on purpose (a reader of a hand-edited or older file): an
    unknown step or option is ignored, never an error.
    """
    chosen = defaults()
    for step, option in stored.items():
        spec = STEP_BY_ID.get(step)
        accepted = spec.accept(option) if spec is not None else None
        if accepted is not None:
            chosen[step] = accepted
    return chosen


def preset_of(choices: Mapping[str, str]) -> str:
    """Return the preset whose choices are exactly ``choices``, or custom."""
    full = resolve(choices)
    for name in PRESET_NAMES:
        if PRESETS[name] == full:
            return name
    return CUSTOM


class SettingsDocument(BaseModel):
    """The stored ``settings.v1`` document.

    ``revision`` grows by one at every change (a writer that read an
    older one is refused); ``choices`` holds every step.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["settings.v1"] = Field(
        default="settings.v1", alias="schema"
    )
    revision: int = Field(default=0, ge=0)
    updated_at: Optional[datetime] = None
    preset: str = "quality"
    choices: Choices = Field(default_factory=defaults)

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, value: str) -> str:
        """Accept a preset name or ``custom``."""
        if value != CUSTOM and value not in PRESET_NAMES:
            raise ValueError(f"unknown preset {value[:20]!r}")
        return value

    @field_validator("choices")
    @classmethod
    def _allowed(cls, value: Choices) -> Choices:
        """Accept only the allow-listed steps and options."""
        try:
            return check_choices(value)
        except InvalidInputError as exc:
            raise ValueError(str(exc)) from exc

    def to_json(self) -> Dict[str, object]:
        """Return the document as ``settings.v1`` JSON."""
        return self.model_dump(mode="json", by_alias=True)


def catalogue() -> Dict[str, object]:
    """Return the steps and presets, for the page to draw."""
    steps: List[Dict[str, object]] = []
    for s in STEPS:
        step: Dict[str, object] = {
            "id": s.id, "group": s.group, "options": list(s.options),
            "default": s.default, "applies": s.applies,
            "consumer": s.consumer,
        }
        if s.range is not None:
            step["range"] = {"low": s.range[0], "high": s.range[1],
                             "step": s.range[2]}
        if s.pattern is not None:
            step["free"] = "speaker"
        steps.append(step)
    return {"steps": steps,
            "presets": {n: dict(PRESETS[n]) for n in PRESET_NAMES}}


class RunChoices(BaseModel):
    """What the analysis launcher passes to one ``skills analyze`` run.

    Empty fields keep the run's configuration (nothing is passed).
    """

    model_config = ConfigDict(frozen=True)

    act_tagger: str = ""
    local_preference: Tuple[str, ...] = ()


LOCAL_JUDGES: Dict[str, Tuple[str, ...]] = {
    "qwen3-8b": (),
    "qwen3-4b": ("Qwen/Qwen3-4B",),
}
"""Local judge option → ``local_preference`` (empty: the configured)."""

ACT_TAGGERS: Dict[str, str] = {
    "encoder": "", "combined": "combined", "rules": "rules",
}
"""Act tagger option → ``--act_tagger`` (empty: the act card's choice)."""


def run_choices(choices: Mapping[str, str]) -> RunChoices:
    """Return the analysis run's options for ``choices``."""
    full = resolve(choices)
    return RunChoices(
        act_tagger=ACT_TAGGERS[full["analysis.act_tagger"]],
        local_preference=LOCAL_JUDGES[full["analysis.local_judge"]],
    )


def step_models(choices: Mapping[str, str]) -> Dict[str, str]:
    """Return the Claude model of each judged step for ``choices``."""
    full = resolve(choices)
    return {
        step: full[f"analysis.{step}_model"]
        for step in ("confirmation", "assessment", "synthesis")
    }
