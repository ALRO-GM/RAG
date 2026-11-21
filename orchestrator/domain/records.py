"""The suggestion log record: one line per reasoner call, with latency.

Plain data: no I/O. ``stages`` hold the per-stage latency of the call in
milliseconds of wall clock, measured by the orchestrator; ``reasoner``
repeats the backend's own breakdown (``reasoner.response.v1.latency``).
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
    model: Optional[str] = None
    backend: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    stages: Stages = Field(default_factory=Stages)
