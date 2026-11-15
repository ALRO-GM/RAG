"""Simulated discussions: a written script becomes a ``transcript.v1``.

A script (``examples/skills/simulated/*.txt``) has ``key: value``
header lines (``id``, ``title``, ``language``, ``type``, ``goal``,
``started_at``), then one turn per line, ``Name: text``. A turn line
that starts with ``+`` begins before the previous turn ends (a short
overlap, as people do when they cut in). Lines starting with ``#``
are comments.

Times are not measured: each sentence lasts its words divided by a
speaking rate, turns are separated by a short pause, and an
overlapping turn starts ``overlap`` seconds before the previous one
ends. The same timing is used when the script is rendered to audio,
so the reference times of the text and the audio agree. Plain text
and arithmetic: no I/O, no model.
"""

import re
import unicodedata
from typing import Any, Dict, List, NamedTuple, Optional

from src.domain.errors import InvalidInputError
from src.domain.skills.speech import split_sentences
from src.domain.skills.transcript_contract import SCHEMA, is_session_id

HEADER_KEYS = ("id", "title", "language", "type", "goal", "started_at")
"""Header fields a script may set."""

_HEADER = re.compile(r"^([a-z_]+):\s*(.*)$")
_TURN = re.compile(r"^(\+?)([^:+#][^:]{0,60}):\s*(\S.*)$")
_WORD = re.compile(r"\w+", re.UNICODE)


class ScriptTurn(NamedTuple):
    """One turn of a script: who, what, and whether it cuts in."""

    speaker: str
    text: str
    overlaps: bool


class Script(NamedTuple):
    """A parsed script: its header fields and its turns in order."""

    header: Dict[str, str]
    turns: List[ScriptTurn]

    def speakers(self) -> List[str]:
        """Return the speakers in order of first appearance."""
        seen: Dict[str, None] = {}
        for turn in self.turns:
            seen.setdefault(turn.speaker, None)
        return list(seen)


class TimedSentence(NamedTuple):
    """A sentence of the script with its simulated times."""

    turn: int
    position: int
    speaker: str
    text: str
    start: float
    end: float
    overlap: bool


def parse_script(text: str) -> Script:
    """Read a script; refuse a line that is neither header nor turn."""
    header: Dict[str, str] = {}
    turns: List[ScriptTurn] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        found = _HEADER.match(line)
        if not turns and found and found.group(1) in HEADER_KEYS:
            header[found.group(1)] = found.group(2).strip()
            continue
        matched = _TURN.match(line)
        if matched is None:
            raise InvalidInputError(
                f"Line {number}: expected 'Name: text', got {line[:40]!r}."
            )
        overlaps = bool(matched.group(1))
        if overlaps and not turns:
            raise InvalidInputError(
                f"Line {number}: the first turn cannot overlap."
            )
        turns.append(ScriptTurn(
            matched.group(2).strip(), matched.group(3).strip(), overlaps,
        ))
    if not turns:
        raise InvalidInputError("The script has no turns.")
    session_id = header.get("id", "")
    if not is_session_id(session_id):
        raise InvalidInputError(
            f"The script needs a valid 'id:' line, not {session_id!r}."
        )
    for previous, turn in zip(turns, turns[1:]):
        if previous.speaker == turn.speaker:
            raise InvalidInputError(
                f"Two turns of {turn.speaker} in a row: join them."
            )
    return Script(header, turns)


def speaker_id(name: str) -> str:
    """Return a plain id for a speaker name (``Lucía`` → ``lucia``)."""
    plain = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in plain if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    return slug or "speaker"


def timed_sentences(
    script: Script,
    words_per_second: float = 2.4,
    pause: float = 0.45,
    sentence_pause: float = 0.25,
    overlap: float = 0.6,
) -> List[TimedSentence]:
    """Give every sentence of the script a start and an end time.

    A sentence lasts ``words / words_per_second`` (at least 0.6 s);
    sentences of one turn are ``sentence_pause`` apart, turns
    ``pause`` apart, and an overlapping turn starts ``overlap``
    seconds before the previous turn's end (never before its start).
    """
    if words_per_second <= 0:
        raise InvalidInputError("words_per_second must be positive.")
    out: List[TimedSentence] = []
    clock = 0.3
    turn_start = 0.0
    for number, turn in enumerate(script.turns):
        if turn.overlaps:
            start = max(clock - overlap, turn_start + 0.3)
        else:
            start = clock + (pause if out else 0.0)
        turn_start = start
        for index, text in enumerate(split_sentences(turn.text)):
            if index:
                start += sentence_pause
            length = max(len(_WORD.findall(text)) / words_per_second, 0.6)
            end = start + length
            out.append(TimedSentence(
                number, index, turn.speaker, text, round(start, 2),
                round(end, 2), turn.overlaps and index == 0,
            ))
            start = end
        clock = start
    return out


def script_transcript(
    script: Script, sentences: Optional[List[TimedSentence]] = None
) -> Dict[str, Any]:
    """Return the script as a ``transcript.v1`` document.

    Every speaker is named and marked enrolled (the script says who
    speaks); ``pipeline`` says the text was simulated, not recognised.
    """
    timed = sentences if sentences is not None else timed_sentences(script)
    ids = {name: speaker_id(name) for name in script.speakers()}
    if len(set(ids.values())) != len(ids):
        raise InvalidInputError("Two speakers would get the same id.")
    h = script.header
    session: Dict[str, Any] = {
        "id": h["id"], "title": h.get("title") or None,
        "language": h.get("language", "es"),
        "started_at": h.get("started_at") or None,
        "duration": round(max(s.end for s in timed) + 0.5, 2),
        "audio_sha256": None,
    }
    if h.get("type"):
        session["type"] = h["type"]
    if h.get("goal"):
        session["goal"] = h["goal"]
    utterances: List[Dict[str, Any]] = []
    for s in timed:
        before = next(
            (u for u in reversed(utterances) if u["turn"] == s.turn - 1),
            None,
        )
        utterances.append({
            "id": f"t{s.turn}.s{s.position}", "turn": s.turn,
            "index": s.position,
            "speaker": ids[s.speaker], "start": s.start, "end": s.end,
            "text": s.text,
            "previous_speaker": before["speaker"] if before else None,
            "previous_utterance": before["id"] if before else None,
            "overlap": s.overlap,
        })
    return {
        "schema": SCHEMA,
        "session": session,
        "speakers": [
            {"id": ids[name], "name": name, "enrolled": True}
            for name in script.speakers()
        ],
        "utterances": utterances,
        "pipeline": {
            "recogniser": "simulated-script", "voice_model": "none",
            "transcriber": "simulation",
        },
    }
