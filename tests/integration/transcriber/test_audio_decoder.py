"""The PyAV/libsndfile decoder on generated recordings of every kind."""

import numpy as np
import pytest

from tests.integration.transcriber.audio import (
    RATE,
    mp4_aac,
    noise,
    pcm_bytes,
    sound_file,
    tone,
    video_only_mp4,
    webm_opus,
)
from transcriber.adapters.outbound.av_audio_decoder import (
    AvAudioDecoder,
    pcm_samples,
    resample,
    to_mono,
)
from transcriber.application.ports import PCM_FORMAT, RecordedAudio
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.voices import FloatArray


def decode(data: bytes, kind: str) -> FloatArray:
    """Decode ``data`` and check the shape every result must have."""
    audio = AvAudioDecoder().decode(RecordedAudio(data, kind))
    assert audio.sample_rate == RATE
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    return audio.samples


def peak(samples: FloatArray) -> float:
    """The largest absolute sample."""
    return float(np.abs(samples).max())


class TestFormats:
    def test_pcm_from_the_phone(self) -> None:
        wave = tone(1.0)
        samples = decode(pcm_bytes(wave), PCM_FORMAT)
        assert samples.shape == wave.shape
        assert np.allclose(samples, wave, atol=1e-4)

    def test_stereo_44k_wav_becomes_mono_16k(self) -> None:
        left = tone(1.0, 44100)
        data = sound_file(np.stack([left, -left * 0.5], axis=1), 44100, "wav")
        samples = decode(data, "wav")
        assert samples.shape[0] == pytest.approx(RATE, abs=2)
        # (left - left / 2) / 2 = left / 4: amplitude 0.125
        assert peak(samples) == pytest.approx(0.125, abs=0.02)

    def test_flac_at_the_target_rate_is_unchanged(self) -> None:
        wave = noise(0.5)
        samples = decode(sound_file(wave, RATE, "flac"), "FLAC")
        assert samples.shape == wave.shape
        assert np.allclose(samples, wave, atol=1e-3)

    def test_mp4_aac_made_with_pyav(self) -> None:
        samples = decode(mp4_aac(tone(1.0, 44100, 440.0)), "mp4")
        # AAC adds a priming delay of a frame or two.
        assert samples.shape[0] / RATE == pytest.approx(1.0, abs=0.1)
        assert peak(samples) == pytest.approx(0.5, abs=0.05)

    def test_webm_opus_as_a_browser_records(self) -> None:
        samples = decode(webm_opus(tone(1.0, 48000, 440.0)), "webm")
        assert samples.shape[0] / RATE == pytest.approx(1.0, abs=0.1)
        assert peak(samples) == pytest.approx(0.5, abs=0.05)

    def test_a_wav_with_the_wrong_extension_still_decodes(self) -> None:
        """libsndfile refuses nothing it knows; PyAV reads the rest."""
        wave = tone(0.5)
        samples = decode(sound_file(wave, RATE, "wav"), "m4a")
        assert samples.shape == wave.shape

    def test_an_mp4_named_wav_falls_back_to_pyav(self) -> None:
        samples = decode(mp4_aac(tone(0.5, 44100)), "wav")
        assert samples.shape[0] / RATE == pytest.approx(0.5, abs=0.1)


class TestErrors:
    def test_empty_recording(self) -> None:
        with pytest.raises(InvalidInputError, match="empty"):
            AvAudioDecoder().decode(RecordedAudio(b"", "wav"))

    def test_garbage(self) -> None:
        with pytest.raises(InvalidInputError, match="Cannot decode the m4a"):
            AvAudioDecoder().decode(
                RecordedAudio(b"this is not audio" * 10, "m4a")
            )

    def test_garbage_named_wav(self) -> None:
        with pytest.raises(InvalidInputError, match="Cannot decode"):
            AvAudioDecoder().decode(
                RecordedAudio(b"RIFF but not really" * 10, "wav")
            )

    def test_no_audio_stream(self) -> None:
        with pytest.raises(InvalidInputError, match="no audio stream"):
            AvAudioDecoder().decode(RecordedAudio(video_only_mp4(), "mp4"))

    def test_a_wav_without_frames(self) -> None:
        data = sound_file(np.zeros(0, dtype=np.float32), RATE, "wav")
        with pytest.raises(InvalidInputError, match="no audio"):
            AvAudioDecoder().decode(RecordedAudio(data, "wav"))

    def test_odd_pcm(self) -> None:
        with pytest.raises(InvalidInputError, match="odd number of bytes"):
            AvAudioDecoder().decode(RecordedAudio(b"\x00\x01\x02", PCM_FORMAT))


def test_helpers() -> None:
    mono = tone(0.1, 8000)
    assert to_mono(mono).shape == mono.shape
    assert resample(mono, 8000, 8000).shape == mono.shape
    assert resample(mono, 8000, 16000).shape[0] == 2 * mono.shape[0]
    assert resample(np.zeros(0, dtype=np.float32), 8000, 16000).size == 0
    assert pcm_samples(b"\x00\x80\xff\x7f").tolist() == [
        -1.0, pytest.approx(32767 / 32768),
    ]


def test_other_sample_rates() -> None:
    audio = AvAudioDecoder(sample_rate=8000).decode(
        RecordedAudio(pcm_bytes(tone(1.0)), PCM_FORMAT)
    )
    assert audio.sample_rate == 8000
    assert audio.samples.shape[0] == 8000
