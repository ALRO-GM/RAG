"""Transcripts: diarized conversations as ordered speaker turns.

Three formats are read: ``[hh:mm:ss] Speaker: text`` lines, the
analyzer's own JSON (``{"turns": [...]}``, or diarizer ``segments``)
and the transcriber's ``transcript.v1`` contract
(:func:`from_contract`), whose sentences and their ids are kept as they
were published.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.domain.errors import CorruptedDataError, InvalidInputError
from src.domain.skills.transcript_contract import (
    ContractTranscript,
    Identification,
    read_contract,
)

_LINE = re.compile(
    r"^\s*(?:\[?(\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\]?\s*[-–]?\s*)?"
    r"([^\s:\[\]{}\"][^:\[\]{}\n]{0,39}):\s+(.*)$"
)
_WORDS = re.compile(r"\w+", re.UNICODE)


class Word(BaseModel):
    """A recognised word with its time span in seconds.

    ``probability`` is the recogniser's confidence in the word, when it
    reports one.
    """

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: Optional[str] = None
    probability: Optional[float] = None

    def middle(self) -> float:
        """Return the midpoint of the word."""
        return (self.start + self.end) / 2


class TurnSentence(BaseModel):
    """A sentence of a turn as its source split it, with a fixed id.

    Transcripts from the transcriber arrive already split into
    sentences (``t<turn>.s<n>`` utterances with their own times); they
    are kept as they are instead of being split again.
    """

    id: str
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


class Turn(BaseModel):
    """One uninterrupted contribution of a speaker.

    ``start`` and ``end`` are seconds from the start of the recording,
    when the source has timestamps. Transcripts made from audio also
    keep the timed ``words`` of the turn and the recogniser's mean
    ``confidence`` in them; hand-written transcripts have neither.
    ``sentences`` are the turn's sentences when the source gave them
    (``transcript.v1``), and ``previous_speaker`` who spoke in the turn
    before, when the source says so; both are empty otherwise.
    ``speaker_identification`` is how the transcriber identified the
    speaker (``threshold`` or ``expected_floor``), when it says so.
    """

    index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    words: List[Word] = Field(default_factory=list)
    confidence: Optional[float] = None
    sentences: List[TurnSentence] = Field(default_factory=list)
    previous_speaker: Optional[str] = None
    speaker_identification: Optional[Identification] = None

    def named_through_floor(self) -> bool:
        """Tell whether the speaker's name is only a guess from a list.

        ``True`` when the transcriber named the speaker through the
        expected-speakers floor: someone who spoke without being listed
        is given a listed name that way, so the attribution of the
        turn's sentences should be down-weighted or left out.
        """
        return self.speaker_identification == "expected_floor"

    def n_words(self) -> int:
        """Return the number of words of the turn."""
        return len(_WORDS.findall(self.text))


class Transcript(BaseModel):
    """A recorded session as a list of turns in speaking order."""

    session_id: str
    language: Optional[str] = None
    turns: List[Turn]

    def speakers(self) -> List[str]:
        """Return the speakers in order of first appearance."""
        seen: Dict[str, None] = {}
        for turn in self.turns:
            seen.setdefault(turn.speaker, None)
        return list(seen)

    def text(self) -> str:
        """Return every turn as ``speaker: text`` lines."""
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

    def has_timestamps(self) -> bool:
        """Tell whether every turn carries a start and an end time."""
        return bool(self.turns) and all(
            t.start is not None and t.end is not None for t in self.turns
        )


def parse_clock(value: str) -> float:
    """Convert ``mm:ss`` or ``hh:mm:ss(.fff)`` to seconds."""
    seconds = 0.0
    for part in value.replace(",", ".").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _seconds(value: Any) -> Optional[float]:
    """Return a timestamp as seconds, or ``None`` when it is unusable."""
    if value is None or isinstance(value, bool):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) else None


def _words(value: Any) -> List[Word]:
    """Return the timed words of a segment, or none if they are unusable.

    Words are optional detail: a segment whose ``words`` are missing or
    malformed (some diarizers drop the times of a few words) is kept
    without them rather than rejected.
    """
    if not isinstance(value, list) or not value:
        return []
    try:
        return [
            w if isinstance(w, Word) else Word.model_validate({
                "text": w.get("text", w.get("word", "")),
                "start": w["start"], "end": w["end"],
                "speaker": w.get("speaker"),
                "probability": w.get("probability", w.get("score")),
            })
            for w in value
        ]
    except (AttributeError, KeyError, TypeError, ValueError):
        return []


def _mean_confidence(words: Sequence[Word]) -> Optional[float]:
    """Mean recogniser confidence of ``words``, when they report it."""
    known = [w.probability for w in words if w.probability is not None]
    return round(sum(known) / len(known), 4) if known else None


def build_transcript(
    session_id: str,
    segments: Sequence[Dict[str, Any]],
    language: Optional[str] = None,
    merge_gap: float = 1.0,
) -> Transcript:
    """Build a transcript from ``speaker``/``text`` segments.

    Consecutive segments of the same speaker are merged into one turn
    when they have no timestamps or are at most ``merge_gap`` seconds
    apart: diarizers cut one contribution into several segments.
    """
    turns: List[Turn] = []
    for segment in segments:
        text = " ".join(str(segment.get("text") or "").split())
        name = segment.get("speaker")
        speaker = "" if name is None else str(name).strip()
        if not text or not speaker:
            continue
        start = _seconds(segment.get("start"))
        end = _seconds(segment.get("end"))
        last = turns[-1] if turns else None
        close = (
            last is not None
            and (start is None or last.end is None
                 or start - last.end <= merge_gap)
        )
        words = _words(segment.get("words"))
        if last is not None and last.speaker == speaker and close:
            last.text = f"{last.text} {text}"
            last.end = end if end is not None else last.end
            # Word times only stay usable if every merged part has them.
            last.words = last.words + words if last.words and words else []
            last.confidence = _mean_confidence(last.words)
            continue
        turns.append(Turn(
            index=len(turns), speaker=speaker, text=text,
            start=start, end=end, words=words,
            confidence=_mean_confidence(words),
        ))
    if not turns:
        raise InvalidInputError(
            f"Transcript {session_id!r} has no speaker turn."
        )
    return Transcript(session_id=session_id, language=language, turns=turns)


def speaker_names(contract: ContractTranscript) -> Dict[str, str]:
    """Return how each speaker id of ``contract`` is shown.

    A speaker is shown by name when it has one, else by id. A name
    that two speakers share, or that is another speaker's id, is
    followed by the id (``Ana (ana-2)``), so two people never merge.
    """
    names = {s.id: (s.name or "").strip() or s.id for s in contract.speakers}
    shown = list(names.values())
    clashing = {
        n for n in shown
        if shown.count(n) > 1 or (n in names and names[n] != n)
    }
    return {
        sid: f"{name} ({sid})" if name in clashing and name != sid else name
        for sid, name in names.items()
    }


def from_contract(contract: ContractTranscript) -> Transcript:
    """Return the analyzer's transcript of a ``transcript.v1`` document.

    One turn per contract turn, keeping its number; its sentences are
    the contract's utterances with their ids and times; the speaker and
    the previous speaker are shown as :func:`speaker_names` says, and
    the speaker's identification is kept (``named_through_floor``). The
    turn's confidence is the mean of its utterances' when they have one.
    """
    names = speaker_names(contract)
    turns: List[Turn] = []
    for utterances in contract.turns():
        first, last = utterances[0], utterances[-1]
        known = [u.confidence for u in utterances if u.confidence is not None]
        previous = first.previous_speaker
        turns.append(Turn(
            index=first.turn,
            speaker=names[first.speaker],
            text=" ".join(u.text for u in utterances),
            start=first.start,
            end=last.end,
            confidence=(
                round(sum(known) / len(known), 4) if known else None
            ),
            sentences=[
                TurnSentence(id=u.id, text=u.text, start=u.start, end=u.end)
                for u in utterances
            ],
            previous_speaker=None if previous is None else names[previous],
            speaker_identification=contract.identification_of(first),
        ))
    if not turns:
        raise InvalidInputError(
            f"Transcript {contract.session.id!r} has no speaker turn."
        )
    return Transcript(
        session_id=contract.session.id,
        language=contract.session.language,
        turns=turns,
    )


def parse_json(session_id: str, text: str) -> Transcript:
    """Parse a JSON transcript.

    Accepted shapes: a ``transcript.v1`` document (recognised by its
    ``schema`` field and validated strictly), our own (``{"turns":
    [...]}``), a diarizer output (``{"segments": [...]}``, as WhisperX
    writes) or a bare list of segments. Each item needs ``speaker`` and
    ``text``; ``start`` and ``end`` are optional.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise CorruptedDataError(
            f"Transcript {session_id!r} is not valid JSON: {exc}"
        )
    language: Optional[str] = None
    if isinstance(data, dict) and "schema" in data:
        try:
            contract = read_contract(data)
        except InvalidInputError as exc:
            raise CorruptedDataError(f"Transcript {session_id!r}: {exc}")
        return from_contract(contract)
    if isinstance(data, dict):
        language = data.get("language")
        session_id = str(data.get("session_id") or session_id)
        data = data.get("turns", data.get("segments"))
    if not isinstance(data, list) or not all(
        isinstance(item, dict) for item in data
    ):
        raise CorruptedDataError(
            f"Transcript {session_id!r} must contain a list of turns "
            "or segments."
        )
    try:
        return build_transcript(session_id, data, language)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, InvalidInputError):
            raise
        raise CorruptedDataError(
            f"Transcript {session_id!r} has an invalid segment: {exc}"
        )


def parse_text(session_id: str, text: str) -> Transcript:
    """Parse ``[hh:mm:ss] Speaker: words`` lines (timestamp optional).

    A line without a ``Speaker:`` prefix continues the previous turn.
    """
    segments: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _LINE.match(line)
        if match is None:
            if segments:
                segments[-1]["text"] += " " + line.strip()
            continue
        clock, speaker, words = match.groups()
        if segments and clock and segments[-1]["end"] is None:
            segments[-1]["end"] = parse_clock(clock)
        segments.append({
            "speaker": speaker.strip(), "text": words,
            "start": parse_clock(clock) if clock else None, "end": None,
        })
    for segment in segments:
        if segment["start"] is not None and segment["end"] is None:
            segment["end"] = segment["start"]
    return build_transcript(session_id, segments, merge_gap=0.0)


def parse_transcript(session_id: str, text: str) -> Transcript:
    """Parse a transcript, choosing the format from its content.

    A document starting with ``{`` is JSON. One starting with ``[`` is
    JSON when it parses as such, and otherwise text whose first line
    opens with a bracketed timestamp.
    """
    stripped = text.strip()
    if not stripped:
        raise InvalidInputError(f"Transcript {session_id!r} is empty.")
    if stripped[0] == "{":
        return parse_json(session_id, stripped)
    if stripped[0] == "[":
        try:
            json.loads(stripped)
        except ValueError:
            return parse_text(session_id, text)
        return parse_json(session_id, stripped)
    return parse_text(session_id, text)
