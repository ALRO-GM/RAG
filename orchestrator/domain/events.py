"""``live.event.v1`` and the transcript it can be replayed from.

The event models mirror ``contracts/live.event.v1.schema.json`` for the
fields the orchestrator reads and writes; the transcript model reads
the few fields of ``transcript.v1`` a replay needs (other fields are
ignored). Plain data: no I/O.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "session_started", "partial", "final", "speaker_update", "turn_end",
    "session_ended", "warning",
]


class _Strict(BaseModel):
    """Base of the event models: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


class Source(_Strict):
    """The audio source of the session."""

    kind: Literal["stereo_channels", "mono_diarized", "single_speaker"]
    sample_rate: int
    channels: int = Field(ge=1, le=2)


class Speaker(_Strict):
    """One speaker and the role the product gives them."""

    id: str = Field(min_length=1)
    name: Optional[str] = None
    channel: Optional[int] = Field(default=None, ge=0)
    enrolled: Optional[bool] = None
    role: str


class Utterance(_Strict):
    """One sentence of a speaker."""

    id: str
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    channel: Optional[int] = Field(default=None, ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    revision: int = Field(ge=0)
    language: Optional[str] = None


class Timings(_Strict):
    """Where the time between speech and event went (milliseconds)."""

    endpoint_ms: Optional[float] = Field(default=None, ge=0)
    asr_ms: Optional[float] = Field(default=None, ge=0)
    attribution_ms: Optional[float] = Field(default=None, ge=0)
    lag_ms: Optional[float] = Field(default=None, ge=0)


class LiveEvent(_Strict):
    """``live.event.v1``."""

    schema_: Literal["live.event.v1"] = Field(
        default="live.event.v1", alias="schema"
    )
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    seq: int = Field(ge=0)
    type: EventType
    at: str
    stream_time: float = Field(ge=0)
    source: Optional[Source] = None
    speakers: Optional[List[Speaker]] = None
    utterance: Optional[Utterance] = None
    timings: Optional[Timings] = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_contract(self) -> Dict[str, Any]:
        """Return the JSON document of the contract."""
        return self.model_dump(mode="json", by_alias=True,
                               exclude_none=True)


class TranscriptSession(BaseModel):
    """The session block of a ``transcript.v1``."""

    id: str
    language: Optional[str] = None
    type: Optional[str] = None
    goal: Optional[str] = None
    started_at: Optional[str] = None
    title: Optional[str] = None


class TranscriptSpeaker(BaseModel):
    """A speaker of a ``transcript.v1``."""

    id: str
    name: Optional[str] = None
    enrolled: Optional[bool] = None


class TranscriptUtterance(BaseModel):
    """An utterance of a ``transcript.v1``."""

    id: str
    turn: int
    index: int
    speaker: str
    start: float
    end: float
    text: str
    language: Optional[str] = None


class Transcript(BaseModel):
    """The part of ``transcript.v1`` a replay reads."""

    session: TranscriptSession
    speakers: List[TranscriptSpeaker]
    utterances: List[TranscriptUtterance]
