"""Audio adapters without models: decoding, Whisper chunks, diarization."""

import shutil
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest
import soundfile

from src.adapters.outbound.soundfile_audio_reader import (
    FloatArray,
    SoundFileAudioReader,
    resample,
    to_mono,
)
from src.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
    words_from_chunks,
)
from src.adapters.outbound.xvector_diarizer import XVectorDiarizer
from src.application.skills.ports import Audio
from src.domain.errors import InvalidInputError, NotFoundError


def tone(seconds: float, rate: int, hz: float = 220.0) -> FloatArray:
    """A sine wave."""
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    wave: FloatArray = (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return wave


class TestAudioReader:
    def test_stereo_44k_becomes_mono_16k(self, tmp_path: Path) -> None:
        path = tmp_path / "a.wav"
        left = tone(1.0, 44100)
        soundfile.write(path, np.stack([left, -left * 0.5], axis=1), 44100)
        audio = SoundFileAudioReader().read(str(path))
        assert audio.sample_rate == 16000
        assert audio.samples.dtype == np.float32
        assert audio.samples.ndim == 1
        assert audio.duration == pytest.approx(1.0, abs=0.01)
        # (left - left / 2) / 2 = left / 4: amplitude 0.125
        assert float(np.abs(audio.samples).max()) == pytest.approx(
            0.125, abs=0.02
        )

    def test_flac_at_the_target_rate_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "a.flac"
        wave = tone(0.5, 16000)
        soundfile.write(path, wave, 16000)
        audio = SoundFileAudioReader().read(str(path))
        assert audio.samples.shape == wave.shape
        assert np.allclose(audio.samples, wave, atol=1e-3)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(NotFoundError, match="Audio file not found"):
            SoundFileAudioReader().read(str(tmp_path / "nope.wav"))

    def test_empty_recording(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        soundfile.write(path, np.zeros(0, dtype=np.float32), 16000)
        with pytest.raises(InvalidInputError, match="no audio"):
            SoundFileAudioReader().read(str(path))

    def test_unknown_format_without_ffmpeg_explains(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "notes.m4a"
        path.write_bytes(b"this is not audio")
        monkeypatch.setattr(
            shutil, "which", lambda name: None
        )
        with pytest.raises(InvalidInputError, match="install ffmpeg"):
            SoundFileAudioReader().read(str(path))

    def test_ffmpeg_fallback_decodes_or_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stand-in ffmpeg: prints float samples, or fails."""
        path = tmp_path / "notes.m4a"
        path.write_bytes(b"this is not audio")
        good = tmp_path / "ffmpeg"
        good.write_text(
            "#!/usr/bin/env python3\nimport sys, struct\n"
            "sys.stdout.buffer.write(struct.pack('<4f', 0, .5, -.5, 0))\n"
        )
        good.chmod(0o755)
        monkeypatch.setattr(
            shutil, "which", lambda name: str(good)
        )
        audio = SoundFileAudioReader().read(str(path))
        assert audio.samples.tolist() == [0.0, 0.5, -0.5, 0.0]
        bad = tmp_path / "ffmpeg-bad"
        bad.write_text(
            "#!/usr/bin/env python3\nimport sys\n"
            "sys.stderr.write('moov atom not found')\nsys.exit(1)\n"
        )
        bad.chmod(0o755)
        monkeypatch.setattr(
            shutil, "which", lambda name: str(bad)
        )
        with pytest.raises(InvalidInputError, match="moov atom not found"):
            SoundFileAudioReader().read(str(path))

    def test_helpers(self) -> None:
        mono = tone(0.1, 8000)
        assert to_mono(mono).shape == mono.shape
        assert resample(mono, 8000, 8000).shape == mono.shape
        assert resample(mono, 8000, 16000).shape[0] == 2 * mono.shape[0]
        empty = np.zeros(0, dtype=np.float32)
        assert resample(empty, 8000, 16000).size == 0


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

    def test_recogniser_is_lazy(self) -> None:
        recogniser = WhisperRecogniser("some/model", batch_size=2)
        assert recogniser.model_name == "some/model"
        assert recogniser._pipeline is None


class FakeVoices(XVectorDiarizer):
    """A diarizer whose embedding is the mean level of each clip."""

    def embed(self, clips: Sequence[FloatArray], rate: int) -> FloatArray:
        rows: List[List[float]] = []
        for clip in clips:
            level = float(np.mean(clip)) if clip.size else 0.0
            rows.append([1.0, 0.0] if level > 0 else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


class TestDiarizer:
    def audio(self) -> Audio:
        """Speaker A (+) for 3 s, B (-) for 3 s, A again for 2 s."""
        rate = 100
        samples = np.concatenate([
            np.full(3 * rate, 0.5), np.full(3 * rate, -0.5),
            np.full(2 * rate, 0.5),
        ]).astype(np.float32)
        return Audio(samples, rate)

    def test_spans_follow_the_voices(self) -> None:
        diarizer = FakeVoices(window=1.0, step=1.0)
        spans = diarizer.diarize(
            self.audio(), [(0.0, 3.0), (3.0, 6.0), (6.0, 8.0)], 2
        )
        assert [(s.speaker, s.start, s.end) for s in spans] == [
            ("SPEAKER_01", 0.0, 3.0),
            ("SPEAKER_02", 3.0, 6.0),
            ("SPEAKER_01", 6.0, 8.0),
        ]

    def test_number_of_speakers_found_by_threshold(self) -> None:
        diarizer = FakeVoices(window=1.0, step=1.0, threshold=0.5)
        spans = diarizer.diarize(self.audio(), [(0.0, 8.0)])
        assert {s.speaker for s in spans} == {"SPEAKER_01", "SPEAKER_02"}

    def test_no_speech_no_spans(self) -> None:
        assert FakeVoices().diarize(self.audio(), []) == []
        assert FakeVoices().diarize(self.audio(), [(0.0, 0.1)]) == []

    def test_diarizer_is_lazy(self) -> None:
        diarizer = XVectorDiarizer("some/voice-model")
        assert diarizer.model_name == "some/voice-model"
        assert diarizer._loaded is None


def test_encoder_is_never_asked_for_attention_maps() -> None:
    """Whatever the caller requests, the encoder gets ``False``."""
    from typing import Any, Dict

    from src.adapters.outbound.whisper_recogniser import (
        without_encoder_attentions,
    )

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
