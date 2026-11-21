"""The Silero VAD adapter without onnxruntime: a fake ONNX session.

The fake records what the adapter gives the model (frame, context,
recurrent state); the real model runs in
``tests/slow/test_transcriber_live_models.py``.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from transcriber.adapters.outbound.silero_vad import (
    CONTEXT_SAMPLES,
    SileroVad,
)
from transcriber.domain.live import FRAME_SAMPLES


class FakeSession:
    """Answers the frame's mean and counts the state it was given."""

    def __init__(self) -> None:
        self.inputs: List[Dict[str, Any]] = []

    def run(
        self, outputs: Any, feeds: Dict[str, Any]
    ) -> Tuple[Any, Any, Any]:
        self.inputs.append(feeds)
        return (
            np.array([[float(np.abs(feeds["input"]).max())]]),
            feeds["h"] + 1, feeds["c"] + 2,
        )


def test_silero_feeds_context_and_carries_state() -> None:
    session = FakeSession()
    opened: List[Path] = []

    def factory(path: Path) -> FakeSession:
        opened.append(path)
        return session

    vad = SileroVad(Path("vad.onnx"), session_factory=factory)
    first = np.full(FRAME_SAMPLES, 0.25, dtype=np.float32)
    second = np.linspace(0, 0.5, FRAME_SAMPLES, dtype=np.float32)
    assert vad.probability(first) == pytest.approx(0.25)
    assert vad.probability(second) == pytest.approx(0.5)
    assert opened == [Path("vad.onnx")]
    a, b = session.inputs
    assert a["input"].shape == (1, FRAME_SAMPLES + CONTEXT_SAMPLES)
    assert a["input"].dtype == np.float32
    assert not a["input"][0, :CONTEXT_SAMPLES].any()
    assert np.allclose(b["input"][0, :CONTEXT_SAMPLES], 0.25)
    assert b["h"].max() == 1 and b["c"].max() == 2
    vad.reset()
    vad.probability(first)
    c = session.inputs[2]
    assert c["h"].max() == 0 and not c["input"][0, :CONTEXT_SAMPLES].any()


def test_silero_rejects_a_frame_of_the_wrong_size() -> None:
    vad = SileroVad(Path("vad.onnx"), session_factory=lambda p: FakeSession())
    with pytest.raises(ValueError):
        vad.probability(np.zeros(100, dtype=np.float32))
