"""Requests and responses of the reasoner (``reasoner.*.v1``).

The reasoner answers in one of three roles: ``judge`` (the analyzer's
existing confirmation, unchanged), ``suggester`` (what to ask or say
next in a live conversation) and ``conversational`` (a reply for a
voice bot). The models mirror ``contracts/reasoner.request.v1`` and
``contracts/reasoner.response.v1`` field by field. Plain data: no I/O.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["judge", "suggester", "conversational"]
ROLES: Tuple[Role, ...] = ("judge", "suggester", "conversational")

Status = Literal[
    "ok", "no_suggestion", "timeout", "cancelled", "quota", "auth", "error"
]
SuggestionKind = Literal[
    "question", "follow_up", "answer", "reminder", "caution"
]
Signal = Literal[
    "complaint", "stated_frustration", "urgency", "churn_risk",
    "confusion", "satisfaction",
]
SIGNALS: Tuple[Signal, ...] = (
    "complaint", "stated_frustration", "urgency", "churn_risk",
    "confusion", "satisfaction",
)
"""What a customer may say in the text (``ARCHITECTURE.md`` §8.1)."""

Backend = Literal["claude-code", "anthropic-api", "local", "opencode"]

AssistMode = Literal["pincelada", "guiado"]
ASSIST_MODES: Tuple[AssistMode, ...] = ("pincelada", "guiado")
"""How a ``@2`` suggester phrases its help: ``pincelada`` a hint of a
few words saying where to steer, ``guiado`` the exact sentence or
question to say (``docs/live/ASSIST_MODES.md``)."""

MAX_SUGGESTION_CHARS = 300
"""Longest suggestion text or rationale (the contract's limit)."""


class _Strict(BaseModel):
    """Base of the contract models: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


class Line(_Strict):
    """One utterance of the window."""

    id: str
    speaker: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


class SessionInfo(_Strict):
    """The conversation the request is about."""

    id: Optional[str] = None
    type: Optional[str] = None
    goal: Optional[str] = None


class Context(_Strict):
    """What was said lately and where the conversation stands."""

    session: Optional[SessionInfo] = None
    speaker_roles: Dict[str, str] = Field(default_factory=dict)
    window: List[Line] = Field(max_length=40)
    focus: Optional[str] = None
    script_phase: Optional[str] = None
    covered_phases: List[str] = Field(default_factory=list)


class Retrieved(_Strict):
    """One knowledge chunk the answer may cite."""

    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Constraints(_Strict):
    """Limits of one call."""

    max_output_tokens: Optional[int] = Field(default=None, ge=1, le=4000)
    max_suggestions: Optional[int] = Field(default=None, ge=0, le=5)
    deadline_ms: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = None
    mode: Optional[AssistMode] = None


class ReasonRequest(_Strict):
    """``reasoner.request.v1``."""

    schema_: Literal["reasoner.request.v1"] = Field(
        default="reasoner.request.v1", alias="schema"
    )
    request_id: str = Field(min_length=1, max_length=64)
    role: Role
    model: str = "haiku"
    template: str = Field(pattern=r"^[a-z_.]+@[0-9]+$")
    language: Optional[str] = None
    context: Context
    retrieved: List[Retrieved] = Field(default_factory=list, max_length=10)
    payload: Dict[str, Any] = Field(default_factory=dict)
    constraints: Constraints = Field(default_factory=Constraints)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Suggestion(_Strict):
    """One thing to ask or say, with the evidence it rests on."""

    kind: SuggestionKind
    text: str = Field(max_length=MAX_SUGGESTION_CHARS)
    signals: List[Signal] = Field(default_factory=list)
    rationale: Optional[str] = Field(
        default=None, max_length=MAX_SUGGESTION_CHARS
    )
    cites: List[str] = Field(default_factory=list)
    about: List[str] = Field(default_factory=list)
    script_phase: Optional[str] = None


class Reply(_Strict):
    """A voice bot's reply."""

    text: str
    cites: List[str] = Field(default_factory=list)


class Usage(_Strict):
    """Tokens of the call (mirrors ``skills.usage.Usage``)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd_equivalent: float = 0.0


class Latency(_Strict):
    """Where the time of a call went, in milliseconds.

    ``first_token_ms`` is ``None`` for a backend that does not stream.
    """

    queued_ms: float = 0.0
    process_start_ms: float = 0.0
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0


class ErrorInfo(_Strict):
    """Why a call failed."""

    kind: Literal["quota", "auth", "transient", "timeout", "invalid_output"]
    message: str
    resets_at: Optional[str] = None


class ReasonResponse(_Strict):
    """``reasoner.response.v1``."""

    schema_: Literal["reasoner.response.v1"] = Field(
        default="reasoner.response.v1", alias="schema"
    )
    request_id: str
    role: Role
    status: Status
    suggestions: List[Suggestion] = Field(default_factory=list)
    reply: Optional[Reply] = None
    verdict: Optional[Dict[str, Any]] = None
    model: str
    backend: Backend
    usage: Usage = Field(default_factory=Usage)
    latency: Latency
    error: Optional[ErrorInfo] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_contract(self) -> Dict[str, Any]:
        """Return the JSON document of the contract."""
        return self.model_dump(mode="json", by_alias=True)
