"""The suggestion log record: one line per reasoner call, with latency.

Plain data: no I/O. ``stages`` hold the per-stage latency of the call in
milliseconds of wall clock, measured by the orchestrator; ``reasoner``
repeats the backend's own breakdown (``reasoner.response.v1.latency``).
``retrieved`` keeps the chunks the call was given (id, text, score,
metadata) so a page can show what a suggestion cites.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

CallStatus = Literal[
    "ok", "no_suggestion", "timeout", "cancelled", "quota", "auth",
    "error", "stale",
]


class Stages(BaseModel):
    """Per-stage latency of one call (milliseconds)."""

    knowledge_ms: Optional[float] = None
    reasoner_ms: Optional[float] = None
    total_ms: Optional[float] = None
    reasoner: Dict[str, Any] = Field(default_factory=dict)


class TriggerInfo(BaseModel):
    """The utterance that asked for the call."""

    utterance: str
    speaker: str
    role: str
    stream_time: float
    text: str


class SuggestionRecord(BaseModel):
    """One reasoner call and what came of it."""

    session_id: str
    call: int
    request_id: str
    trigger: TriggerInfo
    status: CallStatus
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    reply: Optional[Dict[str, Any]] = None
    knowledge_status: str = "none"
    chunks: int = 0
    retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    stages: Stages = Field(default_factory=Stages)


class TurnStages(BaseModel):
    """Where the time of one voice-bot turn went (milliseconds).

    Every value counts from the moment the turn started (the user's
    final reached the orchestrator), except ``event_lag_ms``, the
    transcriber's own end of speech → final (``timings.lag_ms``), and
    ``first_tts_ms``, the synthesis time of the first sentence. The end
    of the user's turn to the first audio is ``event_lag_ms +
    first_audio_ms`` plus the transport to the browser.
    """

    event_lag_ms: Optional[float] = None
    knowledge_ms: Optional[float] = None
    first_text_ms: Optional[float] = None
    first_sentence_ms: Optional[float] = None
    first_tts_ms: Optional[float] = None
    first_audio_ms: Optional[float] = None
    reasoner_ms: Optional[float] = None
    total_ms: Optional[float] = None
    reasoner: Dict[str, Any] = Field(default_factory=dict)


class TurnRecord(BaseModel):
    """One turn of the voice bot: what it heard, said and when.

    ``reply`` is the whole reply the reasoner wrote; ``spoken`` the
    sentences that were synthesised and sent (fewer when the turn was
    interrupted, ``interrupted_by`` says by what). No audio.
    """

    session_id: str
    turn: int
    request_id: str
    trigger: TriggerInfo
    status: CallStatus
    reply: Optional[str] = None
    cites: List[str] = Field(default_factory=list)
    spoken: List[str] = Field(default_factory=list)
    audio_ms: float = 0.0
    interrupted_by: Optional[str] = None
    knowledge_status: str = "none"
    chunks: int = 0
    retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    stages: TurnStages = Field(default_factory=TurnStages)
