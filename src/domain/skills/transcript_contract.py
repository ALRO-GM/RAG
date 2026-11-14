"""The ``transcript.v1`` contract, as the analyzer reads it.

The transcriber (a separate package) publishes finished transcripts as
``transcript.v1`` documents (``contracts/transcript.v1.schema.json``,
``docs/transcriber/ARCHITECTURE.md`` §4). The RAG and the analyzer
share nothing else with it, so this module is their own reading of the
contract: strict pydantic models that refuse what the JSON schema
refuses (unknown fields, a string where a number belongs, ``0.0`` as a
turn number, ``NaN``), plus the rules of §4 the schema cannot express
(ids that match turn and index, the order of turns and sentences, the
previous speaker and utterance of every turn).
"""

import json
import re
from typing import Any, List, Optional

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from src.domain.errors import InvalidInputError, short_reason

SCHEMA = "transcript.v1"
SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
UTTERANCE_ID_PATTERN = r"^t[0-9]+\.s[0-9]+$"
_SESSION_ID = re.compile(SESSION_ID_PATTERN)


def is_session_id(value: str) -> bool:
    """Tell whether ``value`` is a valid ``transcript.v1`` session id."""
    return _SESSION_ID.fullmatch(value) is not None


class _Strict(BaseModel):
    """Base of the contract models: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ContractSession(_Strict):
    """The recorded session a transcript belongs to."""

    id: str = Field(pattern=SESSION_ID_PATTERN)
    title: Optional[str] = None
    language: str = Field(min_length=2)
    started_at: Optional[AwareDatetime] = None
    duration: float = Field(ge=0, allow_inf_nan=False)
    audio_sha256: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ContractSpeaker(_Strict):
    """A voice of the session: an enrolled person or an anonymous label."""

    id: str = Field(min_length=1)
    name: Optional[str] = None
    enrolled: bool


class ContractUtterance(_Strict):
    """One sentence of one turn, with its place in the audio."""

    id: str = Field(pattern=UTTERANCE_ID_PATTERN)
    turn: int = Field(ge=0)
    index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)
    text: str = Field(min_length=1)
    previous_speaker: Optional[str]
    previous_utterance: Optional[str]
    confidence: Optional[float] = Field(
        default=None, ge=0, le=1, allow_inf_nan=False
    )
    speaker_confidence: Optional[float] = Field(
        default=None, ge=-1, le=1, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _consistent(self) -> "ContractUtterance":
        """The id names the turn and index; the end is not before the start."""
        if self.id != f"t{self.turn}.s{self.index}":
            raise ValueError(
                f"id {self.id} does not match turn {self.turn} "
                f"and index {self.index}"
            )
        if self.end < self.start:
            raise ValueError(f"utterance {self.id} ends before it starts")
        return self


class ContractPipeline(_Strict):
    """The models and the transcriber version that made the transcript."""

    recogniser: str
    voice_model: str
    transcriber: str


class ContractTranscript(_Strict):
    """A whole ``transcript.v1`` document."""

    schema_: str = Field(alias="schema", pattern=r"^transcript\.v1$")
    session: ContractSession
    speakers: List[ContractSpeaker]
    utterances: List[ContractUtterance]
    pipeline: ContractPipeline

    @model_validator(mode="after")
    def _consistent(self) -> "ContractTranscript":
        """Check the ids and the turn structure of §4."""
        ids = [s.id for s in self.speakers]
        if len(set(ids)) != len(ids):
            raise ValueError("speaker ids must be unique")
        listed = set(ids)
        for utterance in self.utterances:
            for who in (utterance.speaker, utterance.previous_speaker):
                if who is not None and who not in listed:
                    raise ValueError(
                        f"utterance {utterance.id} names an unlisted "
                        f"speaker {who}"
                    )
        check_turns(self.utterances)
        return self

    def turns(self) -> List[List[ContractUtterance]]:
        """Return the utterances grouped by turn, in order."""
        grouped: List[List[ContractUtterance]] = []
        for utterance in self.utterances:
            if grouped and grouped[-1][0].turn == utterance.turn:
                grouped[-1].append(utterance)
            else:
                grouped.append([utterance])
        return grouped


def check_turns(utterances: List[ContractUtterance]) -> None:
    """Check the order and the previous pair of every turn (§4).

    Turns come in order ``0, 1, 2, ...`` and, inside a turn, sentences
    ``0, 1, 2, ...``. Every sentence of a turn has its speaker and its
    previous pair. The first turn has no previous speaker or utterance;
    any other turn points at the last sentence of the turn before and
    at that turn's speaker. Raises :class:`ValueError` otherwise.
    """
    turns: List[List[ContractUtterance]] = []
    for utterance in utterances:
        if turns and utterance.turn == turns[-1][0].turn:
            turns[-1].append(utterance)
            continue
        if utterance.turn != len(turns):
            raise ValueError(
                f"utterance {utterance.id} is out of order: expected "
                f"turn {len(turns)}"
            )
        turns.append([utterance])
    for number, sentences in enumerate(turns):
        first = sentences[0]
        for index, utterance in enumerate(sentences):
            if utterance.index != index:
                raise ValueError(
                    f"utterance {utterance.id} is out of order: expected "
                    f"index {index} in turn {number}"
                )
            if utterance.speaker != first.speaker:
                raise ValueError(f"turn {number} has two speakers")
            if (utterance.previous_speaker, utterance.previous_utterance) != (
                first.previous_speaker, first.previous_utterance
            ):
                raise ValueError(
                    f"utterance {utterance.id} does not share the previous "
                    f"speaker and utterance of turn {number}"
                )
        if number == 0:
            if first.previous_speaker or first.previous_utterance:
                raise ValueError("the first turn has no previous speaker")
            continue
        before = turns[number - 1]
        if first.previous_utterance != before[-1].id:
            raise ValueError(f"turn {number} must follow {before[-1].id}")
        if first.previous_speaker != before[0].speaker:
            raise ValueError(
                f"turn {number} must have previous speaker "
                f"{before[0].speaker}"
            )


def read_contract(document: Any) -> ContractTranscript:
    """Validate a ``transcript.v1`` document (already parsed from JSON).

    Nothing is coerced: the document goes through JSON text in strict
    mode, so ``"14.2"`` is not a number, ``1`` not a boolean, ``0.0``
    not a turn number and ``NaN`` not JSON at all. Raises
    :class:`~src.domain.errors.InvalidInputError` with the first
    problem found.
    """
    if not isinstance(document, dict):
        raise InvalidInputError("A transcript.v1 document is a JSON object.")
    if document.get("schema") != SCHEMA:
        raise InvalidInputError(
            f"Expected schema {SCHEMA}, got {document.get('schema')!r}."
        )
    try:
        text = json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"Invalid transcript: not JSON data: {exc}")
    try:
        return ContractTranscript.model_validate_json(text, strict=True)
    except ValidationError as exc:
        raise InvalidInputError(f"Invalid transcript: {short_reason(exc)}")
