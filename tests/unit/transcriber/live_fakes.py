"""In-memory fakes of the live ports: VAD, streaming recogniser, timer.

:class:`LoudnessVad` calls a frame speech when it is loud;
:class:`ScriptedAsr` answers each decode with the words a function of
the audio's length gives, and spends ``cost`` seconds of the
:class:`VirtualTimer`, so the use case's timings are exact numbers.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple,
)

import numpy as np

from transcriber.application.live import FloatArray
from transcriber.domain.live import SAMPLE_RATE, Hypothesis
from transcriber.domain.models import Word

REPO = Path(__file__).resolve().parents[3]
LIVE_SCHEMA: Dict[str, Any] = json.loads(
    (REPO / "contracts" / "live.event.v1.schema.json").read_text("utf-8")
)


class VirtualTimer:
    """A monotonic clock that moves only when slept on or spent."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += seconds
            self.slept += seconds


class TimerClock:
    """Wall time that follows a :class:`VirtualTimer`."""

    def __init__(self, timer: VirtualTimer) -> None:
        self._timer = timer
        self._origin = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._origin + timedelta(seconds=self._timer.t)


class LoudnessVad:
    """Speech when the frame's peak is above ``level``."""

    def __init__(self, level: float = 0.1) -> None:
        self.level = level
        self.frames = 0
        self.resets = 0

    def probability(self, frame: FloatArray) -> float:
        self.frames += 1
        return 0.9 if float(np.abs(frame).max()) > self.level else 0.05

    def reset(self) -> None:
        self.resets += 1


Script = Callable[[float, bool], str]


def words_of(text: str, seconds: float) -> List[Word]:
    """Spread the words of ``text`` evenly over ``seconds``."""
    tokens = text.split()
    step = seconds / max(len(tokens), 1)
    return [
        Word(text=t, start=n * step, end=(n + 1) * step, probability=0.8)
        for n, t in enumerate(tokens)
    ]


class ScriptedAsr:
    """Hears ``script(seconds, final)``; each decode costs ``cost`` s."""

    def __init__(
        self, script: Script, timer: Optional[VirtualTimer] = None,
        cost: float = 0.0,
    ) -> None:
        self.script = script
        self.timer = timer
        self.cost = cost
        self.calls: List[Tuple[float, bool]] = []
        self.warm = 0

    def warm_up(self) -> None:
        self.warm += 1

    def decode(self, samples: FloatArray, final: bool) -> Hypothesis:
        seconds = len(samples) / SAMPLE_RATE
        self.calls.append((seconds, final))
        if self.timer is not None:
            self.timer.t += self.cost
        text = self.script(seconds, final)
        return Hypothesis(
            text=text, words=words_of(text, seconds) if final else [],
            language="es", confidence=0.8 if final else None,
        )


def bursts(
    spans: List[Tuple[float, float]], total: float, level: float = 0.5
) -> FloatArray:
    """Silence of ``total`` seconds with a tone inside each span."""
    out = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    for start, end in spans:
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        t = np.arange(b - a, dtype=np.float32) / SAMPLE_RATE
        out[a:b] = level * np.sin(2 * np.pi * 220 * t)
    return out


def pcm(*channels: FloatArray) -> bytes:
    """Interleave ``channels`` as 16-bit little-endian PCM bytes."""
    stacked = np.stack(channels, axis=1).reshape(-1)
    return (np.clip(stacked, -1, 1) * 32767).astype("<i2").tobytes()


def chunks_of(data: bytes, size: int) -> List[bytes]:
    """Cut ``data`` into messages of ``size`` bytes (the last shorter)."""
    return [data[n:n + size] for n in range(0, len(data), size)]


class FakeLiveLease:
    """The GPU lease: acquired or refused; the ack comes when told."""

    def __init__(self, held_by_other: bool = False, ack: bool = True) -> None:
        self.held_by_other = held_by_other
        self.ack_given = ack
        self.acquired: List[int] = []
        self.released = 0

    def acquire(self, need_mib: int, starts_at: object = None) -> Any:
        from transcriber.domain.errors import StateError

        if self.held_by_other:
            raise StateError("The GPU lease is held by someone else.")
        self.acquired.append(need_mib)

    def release(self) -> None:
        self.released += 1

    def current(self) -> None:
        return None

    def ack(self) -> None:
        return None

    def freed(self) -> bool:
        return self.ack_given and bool(self.acquired)

    def clear(self) -> bool:
        return False


class FakeFreeMemory:
    """Free GPU memory that follows a list of readings (last one stays)."""

    def __init__(self, readings: List[Optional[int]]) -> None:
        self.readings = list(readings)
        self.asked = 0

    def free_mib(self) -> Optional[int]:
        self.asked += 1
        if len(self.readings) > 1:
            return self.readings.pop(0)
        return self.readings[0]


class MemoryLiveLog:
    """A live log in memory."""

    def __init__(self, fail: bool = False) -> None:
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.transcripts: Dict[str, Dict[str, Any]] = {}
        self.attribution: Dict[str, List[Dict[str, Any]]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.handed: Set[str] = set()
        self.closed: List[str] = []
        self.fail = fail

    def append(self, session_id: str, event: Mapping[str, Any]) -> None:
        if self.fail:
            raise OSError("disk full")
        self.events.setdefault(session_id, []).append(dict(event))

    def save_transcript(self, session_id: str, document: Any) -> None:
        if self.fail:
            raise OSError("disk full")
        self.transcripts[session_id] = dict(document)

    def save_attribution(
        self, session_id: str, records: Sequence[Mapping[str, Any]]
    ) -> None:
        if self.fail:
            raise OSError("disk full")
        self.attribution[session_id] = [dict(r) for r in records]

    def close(self, session_id: str) -> None:
        self.closed.append(session_id)

    def save_session(self, session_id: str, document: Any) -> None:
        if self.fail:
            raise OSError("disk full")
        self.sessions[session_id] = dict(document)

    def pending(self) -> List[str]:
        return [k for k in self.sessions if k not in self.handed]

    def mark_handed(self, session_id: str) -> None:
        self.handed.add(session_id)

    def has_transcript(self, session_id: str) -> bool:
        return session_id in self.transcripts

    def read_transcript(self, session_id: str) -> Optional[Dict[str, Any]]:
        found = self.transcripts.get(session_id)
        return dict(found) if found is not None else None

    def read_events(self, session_id: str) -> List[Dict[str, Any]]:
        return [dict(e) for e in self.events.get(session_id, [])]

    def read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        found = self.sessions.get(session_id)
        return dict(found) if found is not None else None


class SchemaCheck:
    """The contract check, with jsonschema (as the real adapter)."""

    def problem(self, event: Mapping[str, Any]) -> Optional[str]:
        import jsonschema

        try:
            jsonschema.validate(event, LIVE_SCHEMA)
        except jsonschema.ValidationError as exc:
            return exc.message
        return None


class RecordingPublisher:
    """Keeps what is published; may refuse."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.error = error

    def publish(self, document: Any) -> None:
        if self.error is not None:
            raise self.error
        self.documents.append(dict(document))


class RecordingAnalysis:
    """Keeps the analyses started."""

    def __init__(self) -> None:
        self.started: List[str] = []

    def start(self, session_id: str, restart: bool, synthetic: bool,
              *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.started.append(session_id)
        return {}

    def status(self, session_id: str) -> Dict[str, Any]:
        return {}

    def result(self, session_id: str, step: str) -> Any:
        return None

    def cancel(self, session_id: str) -> None:
        pass

    def report(self, session_id: str, position: int) -> str:
        return ""
