"""Live adapters without their models: faster-whisper stream, timer.

The Whisper model is replaced by a fake that records the decode
options; the real model runs in
``tests/slow/test_transcriber_live_models.py``.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from transcriber.adapters.outbound.faster_whisper_stream import (
    FasterWhisperStream,
)
from transcriber.adapters.outbound.system_clock import SystemTimer


class FakeWhisper:
    """Returns scripted segments and records the decode options."""

    def __init__(self, segments: List[Any], language: str = "es") -> None:
        self.segments = segments
        self.language = language
        self.calls: List[Dict[str, Any]] = []

    def transcribe(self, audio: Any, **options: Any) -> Tuple[Any, Any]:
        self.calls.append(dict(options, samples=len(audio)))
        return iter(self.segments), SimpleNamespace(language=self.language)


def word(text: str, start: float, end: float, p: float) -> SimpleNamespace:
    return SimpleNamespace(word=text, start=start, end=end, probability=p)


def test_faster_whisper_returns_words_and_confidence() -> None:
    segments = [
        SimpleNamespace(text=" Hola,  qué", words=[
            word(" Hola,", 0.0, 0.4, 0.9), word(" ", 0.4, 0.4, 0.1),
            word(" qué", 0.5, 0.7, 0.7),
        ]),
        SimpleNamespace(text=" tal.", words=[word(" tal.", 0.7, 0.6, 1.2)]),
    ]
    model = FakeWhisper(segments)
    loads: List[Tuple[str, str, str]] = []

    def factory(name: str, device: str, compute: str) -> FakeWhisper:
        loads.append((name, device, compute))
        return model

    asr = FasterWhisperStream(device="cpu", factory=factory)
    heard = asr.decode(np.zeros(16000, dtype=np.float32), True)
    assert heard.text == "Hola, qué tal."
    assert [w.text for w in heard.words] == ["Hola,", "qué", "tal."]
    assert heard.words[2].end == heard.words[2].start == 0.7
    assert heard.words[2].probability == 1.0
    assert heard.confidence == pytest.approx((0.9 + 0.7 + 1.0) / 3)
    assert heard.language == "es"
    options = model.calls[0]
    assert options["word_timestamps"] is True
    assert options["beam_size"] == 1 and options["vad_filter"] is False
    assert options["language"] == "es"
    assert loads == [("large-v3-turbo", "cpu", "int8")]


def test_faster_whisper_partials_skip_word_times_and_stay_loaded() -> None:
    model = FakeWhisper([SimpleNamespace(text=" hola", words=None)], "")
    asr = FasterWhisperStream(language=None, factory=lambda *a: model)
    asr.warm_up()
    heard = asr.decode(np.zeros(8000, dtype=np.float32), False)
    assert heard.text == "hola" and heard.words == []
    assert heard.confidence is None and heard.language is None
    assert [c["word_timestamps"] for c in model.calls] == [True, False]
    assert model.calls[0]["samples"] == 16000


def test_system_timer_moves_forward() -> None:
    timer = SystemTimer()
    before = timer.monotonic()
    timer.sleep(0)
    timer.sleep(-1)
    timer.sleep(0.001)
    assert timer.monotonic() > before
