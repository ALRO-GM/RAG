"""The transcriber's real models (Whisper, WavLM) on generated audio.

There is no consented speech recording in the repository and nothing
is downloaded. Where the machine has them, the spoken channel names
that ALSA ships (``/usr/share/sounds/alsa``, alsa-utils, GPL-2: a voice
saying "Front Center"...) give a few seconds of real speech to check
that the word scores follow the audio; the rest checks what generated
audio can show: the models load from the local
cache (``HF_HUB_OFFLINE=1``), the output has the right shape and
order, what silence gives, the voice embeddings are unit vectors
that tell different sounds apart, and a whole job through the real
container runs the real models and unloads them afterwards. They say
nothing about recognition accuracy or about telling real voices apart.
"""

import io
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile

from transcriber.adapters.outbound.av_audio_decoder import resample, to_mono
from transcriber.adapters.outbound.wavlm_voice_encoder import (
    WavLMVoiceEncoder,
)
from transcriber.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
)
from transcriber.application.ports import Audio, RecordedAudio
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.models import SessionState, Word
from transcriber.domain.voices import FloatArray

pytestmark = pytest.mark.slow

RATE = 16000


def tone(seconds: float, hz: float) -> FloatArray:
    """A sine wave at half amplitude."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_whisper_on_silence_invents_at_most_one_word() -> None:
    """A known Whisper failure, pinned so a change is noticed.

    On 4 s and 5 s of digital silence large-v3-turbo returned one word,
    " Gracias.", spanning the whole clip (run on the CPU, 2026-11-14).
    Silence does not give an empty transcript; a voice-activity gate
    would be needed for that (not part of this adapter).
    """
    recogniser = WhisperRecogniser()
    audio = Audio(np.zeros(5 * RATE, dtype=np.float32), RATE)
    speech = recogniser.recognise(audio, "es")
    assert speech.language == "es"
    assert len(speech.words) <= 1
    for word in speech.words:
        assert 0.0 <= word.start <= word.end <= audio.duration
    print(f"\nwhisper on 5 s of silence: {speech.words}")
    assert recogniser.loaded
    recogniser.release()
    assert not recogniser.loaded


def test_whisper_word_times_stay_ordered_and_inside_the_audio() -> None:
    """Whatever it makes of noise and tones, the times are sound."""
    rng = np.random.default_rng(3)
    samples = np.concatenate([
        rng.normal(0, 0.05, 20 * RATE), tone(20.0, 300.0),
        rng.normal(0, 0.05, 25 * RATE),
    ]).astype(np.float32)  # 65 s: three 30-second windows, batch of 2
    audio = Audio(samples, RATE)
    recogniser = WhisperRecogniser()
    words = recogniser.recognise(audio, "es").words
    recogniser.release()
    for word in words:
        assert 0.0 <= word.start <= word.end <= audio.duration
    starts = [w.start for w in words]
    assert starts == sorted(starts)


def test_whisper_scores_words_it_is_given() -> None:
    """Every word gets a probability, also out of time order."""
    rng = np.random.default_rng(4)
    audio = Audio(rng.normal(0, 0.05, 40 * RATE).astype(np.float32), RATE)
    given = [Word(text=" hola", start=31.0, end=31.5),
             Word(text=" buenos", start=1.0, end=1.4),
             Word(text=" días.", start=1.4, end=2.0)]
    recogniser = WhisperRecogniser()
    scored = recogniser.score_words(audio, given, "es").words
    recogniser.release()
    assert [w.text for w in scored] == [w.text for w in given]
    for word in scored:
        assert word.probability is not None
        assert 0.0 <= word.probability <= 1.0
    print(f"\nwhisper word probabilities on noise:"
          f" {[round(w.probability or 0, 3) for w in scored]}")


ALSA_SOUNDS = Path("/usr/share/sounds/alsa")
CHANNELS = ("Front_Center", "Front_Left", "Rear_Right", "Side_Left")


def channel_names() -> FloatArray:
    """ALSA's spoken channel names, 1 s apart, then 40 s of silence."""
    pieces: List[FloatArray] = [np.zeros(RATE, dtype=np.float32)]
    for name in CHANNELS:
        data, rate = soundfile.read(
            str(ALSA_SOUNDS / f"{name}.wav"), dtype="float32",
            always_2d=True,
        )
        pieces.append(resample(to_mono(data), rate, RATE))
        pieces.append(np.zeros(RATE, dtype=np.float32))
    pieces.append(np.zeros(40 * RATE, dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


@pytest.mark.skipif(
    not all((ALSA_SOUNDS / f"{n}.wav").exists() for n in CHANNELS),
    reason="the ALSA test sounds are not installed",
)
def test_whisper_scores_follow_the_audio() -> None:
    """Real words score high; the same times with wrong words, or the
    real words moved onto silence, score low.

    If the scores were taken against the wrong stretch of audio (as
    before seams were resolved first), the real words would not stand
    out from the wrong ones.
    """
    audio = Audio(channel_names(), RATE)
    recogniser = WhisperRecogniser()
    heard = recogniser.recognise(audio, "en").words
    text = " ".join(w.text for w in heard).lower()
    assert "front" in text and "rear" in text
    wrong = [w.model_copy(update={"text": f" {o}"}) for w, o in zip(
        heard, ["banana", "tractor", "umbrella", "violin", "pepper",
                "garden", "winter", "candle"] * 4)]
    later = audio.duration - 30.0
    moved = [w.model_copy(update={"start": w.start + later,
                                  "end": w.end + later}) for w in heard]
    real = recogniser.score_words(audio, heard, "en").words
    other = recogniser.score_words(audio, wrong, "en").words
    silent = recogniser.score_words(audio, moved, "en").words
    recogniser.release()

    def mean(words: List[Word]) -> float:
        """The mean probability of ``words`` (all must have one)."""
        values = [w.probability for w in words if w.probability is not None]
        assert len(values) == len(words)
        return float(np.mean(values))

    print(f"\nwhisper on {len(heard)} ALSA words: real {mean(real):.3f},"
          f" wrong words {mean(other):.3f}, on silence {mean(silent):.3f}")
    assert mean(real) > 0.5
    assert mean(real) > mean(other) + 0.3
    assert mean(real) > mean(silent) + 0.3


def test_wavlm_embeds_windows_as_unit_vectors() -> None:
    encoder = WavLMVoiceEncoder()
    rng = np.random.default_rng(5)
    samples = np.concatenate([
        tone(3.0, 220.0), rng.normal(0, 0.1, 3 * RATE).astype(np.float32),
    ])
    audio = Audio(samples, RATE)
    windows = [(0.0, 1.5), (1.5, 3.0), (0.0, 1.5), (3.2, 5.8)]
    vectors = encoder.embed(audio, windows)
    encoder.release()
    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 4 and vectors.shape[1] > 0
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    same_clip = float(vectors[0] @ vectors[2])
    tone_vs_noise = float(vectors[0] @ vectors[3])
    assert same_clip > 0.999
    assert tone_vs_noise < same_clip
    print(
        f"\nwavlm dim={vectors.shape[1]} same clip={same_clip:.4f}"
        f" tone/tone={float(vectors[0] @ vectors[1]):.4f}"
        f" tone/noise={tone_vs_noise:.4f}"
    )


def test_a_real_job_runs_the_models_and_unloads_them(tmp_path: Path) -> None:
    import torch

    container = build_container(TranscriberSettings(data_dir=tmp_path))
    buffer = io.BytesIO()
    soundfile.write(buffer, np.zeros(4 * RATE, dtype=np.float32), RATE,
                    format="WAV")
    container.import_recording(
        "quiet", "es", RecordedAudio(buffer.getvalue(), "wav")
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    container.queue_transcription("quiet")
    assert container.runner.wait(600)
    session = container.sessions.get("quiet")
    assert session is not None
    # Whisper may invent a word in the silence (see above); the speech
    # activity check drops it, so the job always fails at "recognise".
    assert (session.state, session.failure) == (
        SessionState.FAILED, "recognise: No speech recognised.",
    )
    assert container.sessions.transcript("quiet") is None
    print(f"\nreal job on 4 s of silence ended {session.state.value}")
    assert isinstance(container.recogniser, WhisperRecogniser)
    assert isinstance(container.encoder, WavLMVoiceEncoder)
    assert not container.recogniser.loaded
    assert not container.encoder.loaded
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        left = torch.cuda.memory_allocated() / 2**30
        print(f"\nGPU peak {peak:.2f} GiB, after the job {left:.2f} GiB")
    container.runner.close(10)
