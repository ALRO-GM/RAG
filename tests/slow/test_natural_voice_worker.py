"""The real Pocket TTS worker on the CPU through the adapter (#214;
workstation only: its venv under ``~/.local/share/ragputin/
natural-tts``). Qwen3-TTS needs the GPU lease and is measured by hand
(``docs/live/NATURAL_TTS.md`` §7)."""

import pytest

from orchestrator.adapters.outbound.subprocess_speech import WorkerEngines
from orchestrator.application.natural_voice import NaturalVoice
from orchestrator.application.ports import CancelToken
from orchestrator.adapters.outbound.fakes import FakeTts
from orchestrator.domain.speech import tts_request
from orchestrator.domain.voices import EngineVoice

pytestmark = pytest.mark.slow
POCKET = EngineVoice("pocket", "cpu", voice="lola")


def test_pocket_on_the_cpu_streams_each_sentence() -> None:
    engines = WorkerEngines()
    if "pocket" not in engines.installed():
        pytest.skip("Pocket TTS is not installed")
    voice = NaturalVoice([POCKET], engines, FakeTts(), budget_ms=60000.0)
    try:
        voice.warm()
        chunks = list(voice.synthesize(tts_request(
            "s1", "Hola, ¿en qué puedo ayudarte? Dime tu pedido.",
            "es_ES-davefx-medium", "es", audio_format="pcm_s16le"),
            CancelToken()))
    finally:
        voice.close()
        engines.close()
    assert voice.spoken == {"pocket-cpu": 2}
    assert {c.seq for c in chunks} == {0, 1}
    assert len(chunks) > 2, "streamed in several pieces"
    assert all(c.sample_rate == 24000 for c in chunks)
    assert sum(c.duration_ms for c in chunks) > 1500
