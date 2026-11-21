"""In-memory fakes of the live ports: VAD, streaming recogniser, timer.

:class:`LoudnessVad` calls a frame speech when it is loud;
:class:`ScriptedAsr` answers each decode with the words a function of
the audio's length gives, and spends ``cost`` seconds of the
:class:`VirtualTimer`, so the use case's timings are exact numbers.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
