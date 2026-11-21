"""The real Piper voice through the adapter (workstation only)."""

from pathlib import Path

import pytest

from orchestrator.adapters.outbound.piper_tts import (
    DEFAULT_PYTHON,
    DEFAULT_VOICES,
    PiperTts,
)
from orchestrator.application.ports import CancelToken
from orchestrator.domain.speech import tts_request

VOICE = "es_ES-davefx-medium"

pytestmark = pytest.mark.slow


def test_the_spanish_voice_speaks_sentence_by_sentence() -> None:
    python = DEFAULT_PYTHON.expanduser()
    if not python.exists() or not (
        DEFAULT_VOICES.expanduser() / f"{VOICE}.onnx"
    ).exists():
        pytest.skip("Piper's venv or voice is not installed")
    tts = PiperTts(python=Path(python))
    try:
        chunks = list(tts.synthesize(tts_request(
            "s1", "Hola, ¿en qué puedo ayudarte? Dime tu pedido.", VOICE,
            "es"), CancelToken()))
    finally:
        tts.close()
    assert [c.seq for c in chunks] == [0, 1]
    assert all(c.sample_rate == 22050 and c.duration_ms > 500
               for c in chunks)
