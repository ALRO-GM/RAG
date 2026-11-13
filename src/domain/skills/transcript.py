"""Transcripts: diarized conversations as ordered speaker turns."""

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.domain.errors import CorruptedDataError, InvalidInputError

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


class Turn(BaseModel):
    """One uninterrupted contribution of a speaker.

    ``start`` and ``end`` are seconds from the start of the recording,
    when the source has timestamps. Transcripts made from audio also
    keep the timed ``words`` of the turn and the recogniser's mean
    ``confidence`` in them; hand-written transcripts have neither.
    """

    index: int = Field(ge=0)
    speaker: str = Field(min_length=1)
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    words: List[Word] = Field(default_factory=list)
    confidence: Optional[float] = None

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


def parse_json(session_id: str, text: str) -> Transcript:
    """Parse a JSON transcript.

    Accepted shapes: our own (``{"turns": [...]}``), a diarizer output
    (``{"segments": [...]}``, as WhisperX writes) or a bare list of
    segments. Each item needs ``speaker`` and ``text``; ``start`` and
    ``end`` are optional.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise CorruptedDataError(
            f"Transcript {session_id!r} is not valid JSON: {exc}"
        )
    language: Optional[str] = None
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
