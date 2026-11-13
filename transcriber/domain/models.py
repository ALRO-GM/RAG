"""Domain model of the transcriber and the ``transcript.v1`` contract.

A recorded :class:`Session` becomes a :class:`Transcript`: the people
heard (:class:`Speaker`), and the :class:`Utterance` list, one sentence
of one turn each, with its time in the audio and the speaker who spoke
before it. :meth:`Transcript.to_contract` and
:meth:`Transcript.from_contract` translate to and from the JSON
contract shared with the RAG (``contracts/transcript.v1.schema.json``).
"""

import enum
from typing import Any, Dict, List, Mapping, Optional

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from transcriber.domain.errors import ContractError

SCHEMA = "transcript.v1"

SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
UTTERANCE_ID_PATTERN = r"^t[0-9]+\.s[0-9]+$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Word(BaseModel):
    """A recognised word with its time span in seconds.

    ``speaker`` is set once diarization has attributed the word;
    ``probability`` is the recogniser's confidence, when it reports one.
    Words are kept by the transcriber but are not part of the contract.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: Optional[str] = None
    probability: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Word":
        """Reject a word that ends before it starts."""
        if self.end < self.start:
            raise ValueError("a word cannot end before it starts")
        return self


class Turn(BaseModel):
    """One uninterrupted contribution of a speaker, before splitting.

    ``start`` and ``end`` are seconds from the start of the session
    audio. ``speaker_confidence`` is the similarity that supports the
    attribution of the turn, when the speaker was identified by voice.
    """

    model_config = ConfigDict(frozen=True)

    speaker: str = Field(min_length=1)
    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    words: List[Word] = Field(default_factory=list)
    speaker_confidence: Optional[float] = Field(default=None, ge=-1, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Turn":
        """Reject a turn that ends before it starts."""
        if self.end < self.start:
            raise ValueError("a turn cannot end before it starts")
        return self


class Utterance(BaseModel):
    """One sentence of one turn, linked to its place in the audio.

    ``id`` is ``t<turn>.s<index>``. ``previous_speaker`` is the speaker
    of the turn before this one and ``previous_utterance`` the last
    sentence of that turn (both ``None`` in the first turn; the same for
    every sentence of a turn). ``words`` are not part of the contract.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=UTTERANCE_ID_PATTERN)
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)
    previous_speaker: Optional[str]
    previous_utterance: Optional[str] = Field(pattern=UTTERANCE_ID_PATTERN)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    speaker_confidence: Optional[float] = Field(default=None, ge=-1, le=1)
    words: List[Word] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "Utterance":
        """Check the id against turn and index, and the time order."""
        if self.id != f"t{self.turn}.s{self.index}":
            raise ValueError(
                f"id {self.id} does not match turn {self.turn}"
                f" and index {self.index}"
            )
        if self.end < self.start:
            raise ValueError("an utterance cannot end before it starts")
        return self


class Speaker(BaseModel):
    """A person heard in a session.

    ``id`` is an enrolled user id when the voice was recognised, or an
    anonymous label (``S1``, ``S2``, ...) stable inside the session.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: Optional[str] = None
    enrolled: bool


class Voiceprint(BaseModel):
    """The enrolled voice of one person: one embedding and its consent.

    Nothing else of the enrolment is kept: the sample audio is deleted.
    ``model`` names the voice encoder that produced ``vector``; vectors
    of different models are not comparable.
    """

    model_config = ConfigDict(frozen=True)

    speaker_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    vector: List[float] = Field(min_length=1)
    consent_version: str = Field(min_length=1)
    enrolled_at: AwareDatetime


class SessionState(str, enum.Enum):
    """Where a session is in its life, from recording to transcript."""

    RECORDING = "recording"
    RECORDED = "recorded"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    FAILED = "failed"


class Session(BaseModel):
    """One recorded meeting.

    ``duration`` is in seconds; ``audio_sha256`` identifies the audio
    the transcript was made from, without carrying it.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=SESSION_ID_PATTERN)
    title: Optional[str] = None
    language: str = Field(min_length=2)
    started_at: Optional[AwareDatetime] = None
    duration: float = Field(default=0.0, ge=0)
    state: SessionState = SessionState.RECORDING
    audio_sha256: Optional[str] = Field(default=None, pattern=SHA256_PATTERN)


class Pipeline(BaseModel):
    """The models and the transcriber version that made a transcript."""

    model_config = ConfigDict(frozen=True)

    recogniser: str
    voice_model: str
    transcriber: str


_SESSION_FIELDS = frozenset({
    "id", "title", "language", "started_at", "duration", "audio_sha256",
})
_SPEAKER_FIELDS = frozenset({"id", "name", "enrolled"})
_UTTERANCE_FIELDS = frozenset({
    "id", "turn", "index", "speaker", "start", "end", "text",
    "previous_speaker", "previous_utterance", "confidence",
    "speaker_confidence",
})
_PIPELINE_FIELDS = frozenset({"recogniser", "voice_model", "transcriber"})
_TOP_FIELDS = frozenset({
    "schema", "session", "speakers", "utterances", "pipeline",
})
_SESSION_REQUIRED = frozenset({"id", "language", "duration"})
_SPEAKER_REQUIRED = frozenset({"id", "enrolled"})
_UTTERANCE_REQUIRED = _UTTERANCE_FIELDS - {
    "confidence", "speaker_confidence",
}


class Transcript(BaseModel):
    """The attributed sentences of one session, ready to hand over."""

    model_config = ConfigDict(frozen=True)

    session: Session
    speakers: List[Speaker]
    utterances: List[Utterance]
    pipeline: Pipeline

    @model_validator(mode="after")
    def _consistent(self) -> "Transcript":
        """Check ids are unique and every speaker referenced is listed."""
        known = [s.id for s in self.speakers]
        if len(set(known)) != len(known):
            raise ValueError("speaker ids must be unique")
        ids = [u.id for u in self.utterances]
        if len(set(ids)) != len(ids):
            raise ValueError("utterance ids must be unique")
        listed = set(known)
        for utterance in self.utterances:
            for who in (utterance.speaker, utterance.previous_speaker):
                if who is not None and who not in listed:
                    raise ValueError(
                        f"utterance {utterance.id} names an unlisted"
                        f" speaker {who}"
                    )
            if (
                utterance.previous_utterance is not None
                and utterance.previous_utterance not in ids
            ):
                raise ValueError(
                    f"utterance {utterance.id} follows an unknown"
                    f" utterance {utterance.previous_utterance}"
                )
        return self

    def to_contract(self) -> Dict[str, Any]:
        """Return the transcript as a ``transcript.v1`` document."""
        session = self.session
        return {
            "schema": SCHEMA,
            "session": {
                "id": session.id,
                "title": session.title,
                "language": session.language,
                "started_at": (
                    session.started_at.isoformat()
                    if session.started_at is not None else None
                ),
                "duration": session.duration,
                "audio_sha256": session.audio_sha256,
            },
            "speakers": [s.model_dump() for s in self.speakers],
            "utterances": [
                u.model_dump(include=set(_UTTERANCE_FIELDS))
                for u in self.utterances
            ],
            "pipeline": self.pipeline.model_dump(),
        }

    @classmethod
    def from_contract(cls, document: Mapping[str, Any]) -> "Transcript":
        """Build a transcript from a ``transcript.v1`` document.

        The session of a published transcript is ``transcribed``.
        Raises :class:`ContractError` when the document does not follow
        the contract or breaks a rule of the model.
        """
        if not isinstance(document, Mapping):
            raise ContractError("A transcript must be a JSON object.")
        if document.get("schema") != SCHEMA:
            raise ContractError(
                f"Expected schema {SCHEMA}, got {document.get('schema')!r}."
            )
        _only(document, _TOP_FIELDS, "transcript")
        session = _section(
            document, "session", _SESSION_FIELDS, _SESSION_REQUIRED
        )
        speakers = [
            _only(s, _SPEAKER_FIELDS, "speaker", _SPEAKER_REQUIRED)
            for s in _items(document, "speakers")
        ]
        utterances = [
            _only(u, _UTTERANCE_FIELDS, "utterance", _UTTERANCE_REQUIRED)
            for u in _items(document, "utterances")
        ]
        pipeline = _section(
            document, "pipeline", _PIPELINE_FIELDS, _PIPELINE_FIELDS
        )
        try:
            return cls.model_validate({
                "session": {**session, "state": SessionState.TRANSCRIBED},
                "speakers": speakers,
                "utterances": utterances,
                "pipeline": pipeline,
            })
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ()))
            raise ContractError(
                f"Invalid transcript: {where}: {first.get('msg', '')}"
            ) from exc


def _only(
    item: Any,
    allowed: frozenset[str],
    what: str,
    required: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    """Return ``item`` if it is an object with the contract's keys.

    Every key must be in ``allowed`` and every key of ``required`` must
    be present (it may be ``null`` where the contract allows it).
    """
    if not isinstance(item, Mapping):
        raise ContractError(f"A {what} must be a JSON object.")
    extra = sorted(set(item) - allowed)
    if extra:
        raise ContractError(f"Unknown {what} field(s): {', '.join(extra)}.")
    missing = sorted(required - set(item))
    if missing:
        raise ContractError(
            f"Missing {what} field(s): {', '.join(missing)}."
        )
    return item


def _section(
    document: Mapping[str, Any],
    key: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, Any]:
    """Return the required object ``key`` of ``document``."""
    if key not in document:
        raise ContractError(f"Missing {key}.")
    return _only(document[key], allowed, key, required)


def _items(document: Mapping[str, Any], key: str) -> List[Any]:
    """Return the required list ``key`` of ``document``."""
    value = document.get(key)
    if not isinstance(value, list):
        raise ContractError(f"{key} must be a list.")
    return value
