"""The live models (Silero VAD, faster-whisper turbo int8) for real.

Needs the ``live`` dependency group and the model in the local cache
(``HF_HUB_OFFLINE=1``). Silence and noise are checked always; the
Piper-rendered simulation (``SIM_WAV``, made by
``experiments/skills/render_simulation.py``) is used when the machine
has it, to check that speech is found and its first words are heard.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile

from transcriber.adapters.outbound.silero_vad import SileroVad
from transcriber.domain.live import FRAME_SAMPLES, Endpointer, SpeechStart
from transcriber.domain.voices import FloatArray

pytestmark = pytest.mark.slow

SIM_WAV = Path(os.environ.get(
    "SIM_WAV",
    Path.home() / ".local/share/ragputin/sim96/audio/sim-reunion-verano.wav",
))


def frames_of(audio: FloatArray) -> List[FloatArray]:
    n = len(audio) // FRAME_SAMPLES
    return [audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            for i in range(n)]


def test_silero_hears_no_speech_in_silence_or_noise() -> None:
    pytest.importorskip("faster_whisper")
    vad = SileroVad()
    silence = np.zeros(16000, dtype=np.float32)
    noise = np.random.default_rng(3).normal(0, 0.05, 16000).astype(
        np.float32
    )
    for audio in (silence, noise):
        vad.reset()
        assert max(vad.probability(f) for f in frames_of(audio)) < 0.5


@pytest.mark.skipif(not SIM_WAV.exists(), reason="no simulation audio")
def test_silero_and_whisper_on_the_simulation() -> None:
    pytest.importorskip("faster_whisper")
    from transcriber.adapters.outbound.faster_whisper_stream import (
        FasterWhisperStream,
    )

    audio, rate = soundfile.read(str(SIM_WAV), dtype="float32")
    assert rate == 16000
    audio = audio[:rate * 15]
    vad, endpointer = SileroVad(), Endpointer()
    starts = [
        e for f in frames_of(audio) for e in endpointer.push(
            vad.probability(f)
        ) if isinstance(e, SpeechStart)
    ]
    assert starts and starts[0].start < 1.0
    device = os.environ.get("LIVE_DEVICE", "cpu")
    asr = FasterWhisperStream(device=device)
    heard = asr.decode(audio[:rate * 5], True)
    assert "buenos días" in heard.text.lower()
    assert heard.words and heard.confidence is not None
