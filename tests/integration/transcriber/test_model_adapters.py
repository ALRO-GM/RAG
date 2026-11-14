"""Model adapters without loading a model, and the fake models."""

import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tests.integration.transcriber.audio import (
    RATE,
    noise,
    scene,
    silence,
    tone,
)
from transcriber.adapters.outbound import torch_runtime
from transcriber.adapters.outbound.fake_models import (
    FAKE_RECOGNISER,
    FAKE_VOICE_MODEL,
    SENTENCE_WORDS,
    FakeOverlapDetector,
    FakeSpeechRecogniser,
    FakeVoiceEncoder,
)
from transcriber.adapters.outbound.wavlm_voice_encoder import (
    WavLMVoiceEncoder,
)
from transcriber.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
    mean_token_probabilities,
    scoring_windows,
    whisper_language,
    without_encoder_attentions,
    words_from_chunks,
)
from transcriber.adapters.outbound.whisper_languages import (
    WHISPER_LANGUAGES,
)
from transcriber.application.ports import Audio
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Word


class TestWordProbabilities:
    def test_windows_start_at_a_word_and_hold_what_ends_within(self) -> None:
        words = [Word(text=" a", start=s, end=s + 1.0)
                 for s in (0.0, 10.0, 28.0, 29.5, 45.0)]
        assert scoring_windows(words, 30.0) == [
            (0.0, [0, 1, 2]), (29.5, [3, 4]),
        ]

    def test_a_word_longer_than_a_window_is_alone(self) -> None:
        words = [Word(text=" a", start=0.0, end=40.0),
                 Word(text=" b", start=40.0, end=41.0)]
        assert scoring_windows(words, 30.0) == [(0.0, [0]), (40.0, [1])]

    def test_a_word_is_the_mean_of_its_tokens(self) -> None:
        assert mean_token_probabilities([1, 2, 0], [0.9, 0.5, 0.7]) == [
            0.9, pytest.approx(0.6), None,
        ]

    def test_tokens_cut_off_leave_the_word_unknown(self) -> None:
        assert mean_token_probabilities([2, 2], [0.4, 0.6, 0.8]) == [
            pytest.approx(0.5), None,
        ]

    def test_values_stay_probabilities(self) -> None:
        assert mean_token_probabilities([1], [1.0000001]) == [1.0]


class TestWhisperChunks:
    def test_words_and_times(self) -> None:
        words = words_from_chunks([
            {"text": " Hola", "timestamp": (0.0, 0.4)},
            {"text": " mundo.", "timestamp": (0.5, 1.0)},
        ], duration=2.0)
        assert [(w.text, w.start, w.end) for w in words] == [
            (" Hola", 0.0, 0.4), (" mundo.", 0.5, 1.0),
        ]

    def test_open_end_closes_at_next_word_or_end_of_audio(self) -> None:
        words = words_from_chunks([
            {"text": " a", "timestamp": (0.0, None)},
            {"text": " b", "timestamp": (0.7, None)},
        ], duration=1.5)
        assert [(w.start, w.end) for w in words] == [(0.0, 0.7), (0.7, 1.5)]

    def test_unusable_chunks_are_dropped_and_times_clamped(self) -> None:
        words = words_from_chunks([
            {"text": " lost", "timestamp": (None, 0.3)},
            {"text": "  ", "timestamp": (0.1, 0.2)},
            {"text": " missing"},
            {"text": " late", "timestamp": (1.9, 2.6)},
            {"text": " reversed", "timestamp": (1.0, 0.5)},
        ], duration=2.0)
        assert [(w.text, w.start, w.end) for w in words] == [
            (" late", 1.9, 2.0), (" reversed", 1.0, 1.0),
        ]


def test_encoder_is_never_asked_for_attention_maps() -> None:
    """Whatever the caller requests, the encoder gets ``False``."""

    class Encoder:
        def __init__(self) -> None:
            self.seen: List[Dict[str, Any]] = []

        def forward(self, *args: Any, **kwargs: Any) -> str:
            self.seen.append(kwargs)
            return "encoded"

    class Model:
        def __init__(self) -> None:
            self.encoder = Encoder()

        def get_encoder(self) -> Encoder:
            return self.encoder

    model = Model()
    without_encoder_attentions(model)
    assert model.encoder.forward(1, output_attentions=True, x=2) == "encoded"
    assert model.encoder.forward(1) == "encoded"
    assert model.encoder.seen == [
        {"output_attentions": False, "x": 2}, {"output_attentions": False},
    ]


class TestLazyModels:
    def test_recogniser_loads_nothing_until_used(self) -> None:
        recogniser = WhisperRecogniser("some/model", device="cpu")
        assert recogniser.model_name == "some/model"
        assert recogniser.batch_size == 2
        assert not recogniser.loaded
        recogniser.release()  # nothing loaded: nothing to do
        recogniser._pipeline = object()
        assert recogniser.loaded
        recogniser.release()
        assert not recogniser.loaded

    def test_encoder_loads_nothing_until_used(self) -> None:
        encoder = WavLMVoiceEncoder("some/voice", device="cpu")
        assert encoder.model_name == "some/voice"
        assert not encoder.loaded
        audio = Audio(samples=silence(1.0), sample_rate=RATE)
        assert encoder.embed(audio, []).shape == (0, 0)
        assert not encoder.loaded
        encoder.release()
        encoder._loaded = (object(), object())
        encoder.release()
        assert not encoder.loaded

    def test_encoder_checks_its_input_before_loading(self) -> None:
        encoder = WavLMVoiceEncoder("some/voice", device="cpu")
        slow = Audio(samples=silence(1.0, 8000), sample_rate=8000)
        with pytest.raises(InvalidInputError, match="16000 Hz"):
            encoder.embed(slow, [(0.0, 1.0)])
        audio = Audio(samples=silence(1.0), sample_rate=RATE)
        with pytest.raises(InvalidInputError, match="outside the audio"):
            encoder.embed(audio, [(0.0, 0.5), (2.0, 3.0)])
        assert not encoder.loaded


class TestTorchRuntime:
    def test_devices_and_precision(self) -> None:
        import torch

        assert torch_runtime.pick_device("cpu") == "cpu"
        assert torch_runtime.pick_device("cuda") == "cuda"
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert torch_runtime.pick_device("auto") == expected
        assert torch_runtime.model_kwargs("cpu") == {}
        assert torch_runtime.model_kwargs("cuda") == {"dtype": torch.float16}

    def test_release_is_safe_anywhere(self) -> None:
        torch_runtime.release("cpu")
        torch_runtime.release("cuda")


class TestWhisperLanguages:
    def test_the_copied_codes_are_whisper_s(self) -> None:
        from transformers.models.whisper.tokenization_whisper import (
            LANGUAGES,
        )

        assert WHISPER_LANGUAGES == frozenset(LANGUAGES)

    def test_a_code_is_mapped_to_its_token_or_refused(self) -> None:
        assert whisper_language("es-ES") == "es"
        assert whisper_language(None) is None
        with pytest.raises(InvalidInputError):
            whisper_language("xx")


class TestFakeRecogniser:
    def test_words_follow_the_loud_slots(self) -> None:
        samples = scene([("220", 2.0), ("-", 1.2), ("440", 0.8)])
        speech = FakeSpeechRecogniser().recognise(
            Audio(samples=samples, sample_rate=RATE), "es"
        )
        assert speech.language == "es"
        assert FakeSpeechRecogniser.model_name == FAKE_RECOGNISER
        times = [(w.start, w.end) for w in speech.words]
        assert times == [
            (0.0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 2.0),
            (3.2, 3.6), (3.6, 4.0),
        ]
        texts = [w.text for w in speech.words]
        assert texts[0] == " Hola" and texts[4].endswith(".")
        assert texts[5][1].isupper() and texts[-1].endswith(".")

    def test_long_speech_is_cut_into_sentences(self) -> None:
        samples = scene([("220", 0.4 * (SENTENCE_WORDS + 2))])
        words = FakeSpeechRecogniser().recognise(
            Audio(samples=samples, sample_rate=RATE)
        ).words
        assert words[SENTENCE_WORDS - 1].text.endswith(".")
        assert words[SENTENCE_WORDS].text[1].isupper()

    def test_silence_and_quiet_noise_have_no_words(self) -> None:
        recogniser = FakeSpeechRecogniser()
        for samples in (silence(3.0), noise(3.0, level=0.001)):
            audio = Audio(samples=samples, sample_rate=RATE)
            assert recogniser.recognise(audio).words == []
        recogniser.release()


class TestFakeEncoder:
    def test_pitch_is_the_voice(self) -> None:
        samples = scene([("220", 2.0), ("880", 2.0), ("220", 2.0)])
        audio = Audio(samples=samples, sample_rate=RATE)
        encoder = FakeVoiceEncoder()
        assert encoder.model_name == FAKE_VOICE_MODEL
        vectors = encoder.embed(
            audio, [(0.0, 1.5), (2.2, 3.8), (4.2, 5.8), (6.5, 7.0)]
        )
        assert vectors.shape == (4, 24) and vectors.dtype == np.float32
        same = float(vectors[0] @ vectors[2])
        other = float(vectors[0] @ vectors[1])
        assert same > 0.99 and other < 0.1
        assert np.allclose(np.linalg.norm(vectors[:3], axis=1), 1.0)
        assert not vectors[3].any()  # outside the audio: zero row
        assert encoder.embed(audio, []).shape == (0, 0)
        encoder.release()


class _SlowLoader:
    """Stands in for transformers: counts loads, each takes a moment."""

    def __init__(self) -> None:
        self.loads = 0

    def load(self) -> Any:
        self.loads += 1
        threading.Event().wait(0.1)
        loaded = types.SimpleNamespace()
        loaded.model = types.SimpleNamespace(
            get_encoder=lambda: types.SimpleNamespace(forward=None)
        )
        loaded.to = lambda device: loaded
        loaded.eval = lambda: loaded
        return loaded


def _fake_transformers(
    monkeypatch: pytest.MonkeyPatch, loader: _SlowLoader
) -> None:
    module = types.ModuleType("transformers")
    setattr(module, "pipeline", lambda *a, **k: loader.load())
    for name in ("AutoFeatureExtractor", "WavLMForXVector"):
        setattr(module, name, types.SimpleNamespace(
            from_pretrained=lambda *a, **k: loader.load()
        ))
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.mark.parametrize("adapter, loads_per_model", [
    (lambda: WhisperRecogniser("m", device="cpu"), 1),
    (lambda: WavLMVoiceEncoder("m", device="cpu"), 2),
])
def test_two_first_uses_at_once_load_one_model(
    monkeypatch: pytest.MonkeyPatch, adapter: Any, loads_per_model: int
) -> None:
    loader = _SlowLoader()
    _fake_transformers(monkeypatch, loader)
    model = adapter()
    start = threading.Barrier(2)

    def first_use() -> None:
        start.wait()
        model._load()

    threads = [threading.Thread(target=first_use) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert loader.loads == loads_per_model
    assert model.loaded
    model.release()
    assert not model.loaded


def test_the_fake_recogniser_scores_nothing() -> None:
    words = [Word(text="hola", start=0.0, end=0.4)]
    audio = Audio(samples=silence(1.0), sample_rate=RATE)
    scored = FakeSpeechRecogniser().score_words(audio, words, "es").words
    assert scored == words
    assert "es" in FakeSpeechRecogniser().supported_languages


class TestFakeOverlapDetector:
    def test_two_tones_at_once_are_an_overlap(self) -> None:
        both = (tone(1.0, hz=220.0, level=0.3)
                + tone(1.0, hz=660.0, level=0.3)).astype(np.float32)
        samples = np.concatenate([tone(1.0), silence(1.0), both])
        found = FakeOverlapDetector().overlaps(
            Audio(samples=samples, sample_rate=RATE))
        assert found == [(2.0, 2.5), (2.5, 3.0)]
        FakeOverlapDetector().release()


def test_the_settings_pick_the_overlap_detector(tmp_path: Path) -> None:
    off = build_container(TranscriberSettings(data_dir=tmp_path,
                                              runtime="fake"))
    assert off.overlaps is None
    on = build_container(TranscriberSettings(
        data_dir=tmp_path, runtime="fake", overlap_detector="fake"))
    assert isinstance(on.overlaps, FakeOverlapDetector)
    on.release_models()
