"""The speech check on synthetic signals: a voice passes, noise fails."""

import numpy as np
import numpy.typing as npt

from transcriber.domain.speech_check import (
    MAX_FLATNESS,
    MIN_AUTOCORRELATION,
    NO_SPEECH,
    check_speech,
)

RATE = 16000


def voiced(seconds: float = 10.0, rate: int = RATE,
           seed: int = 0) -> npt.NDArray[np.float32]:
    """Return a synthetic voice: harmonics of 120 Hz in syllables.

    Harmonics up to 3.6 kHz falling as 1/k, switched on and off four
    times a second (pauses between syllables), with a little noise.
    """
    t = np.arange(int(seconds * rate)) / rate
    tone = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 31))
    syllables = np.clip(np.sin(2 * np.pi * 4 * t), 0, None)
    noise = np.random.default_rng(seed).normal(0, 0.005, t.size)
    out = 0.1 * tone * syllables + noise
    return np.asarray(out, dtype=np.float32)


def band_noise(seconds: float = 10.0, rate: int = RATE,
               top: float = 4000.0, seed: int = 1) -> npt.NDArray[np.float32]:
    """Return steady white noise cut above ``top`` Hz (as in CALLHOME)."""
    size = int(seconds * rate)
    white = np.random.default_rng(seed).normal(0, 0.1, size)
    spectrum = np.fft.rfft(white)
    spectrum[np.fft.rfftfreq(size, 1 / rate) > top] = 0
    return np.asarray(np.fft.irfft(spectrum, size), dtype=np.float32)


class TestSpeechCheck:
    def test_a_voice_passes(self) -> None:
        check = check_speech(voiced(), RATE)
        assert check.speech and check.reason() is None
        assert check.autocorrelation > MIN_AUTOCORRELATION
        assert check.flatness < MAX_FLATNESS
        # 10 s in frames of 512 samples: 312 frames, the loudest 30 %
        assert check.frames == 94

    def test_band_limited_noise_fails(self) -> None:
        check = check_speech(band_noise(), RATE)
        assert not check.speech
        assert check.reason() == NO_SPEECH
        assert check.autocorrelation < MIN_AUTOCORRELATION
        assert check.flatness > MAX_FLATNESS
        assert check.describe() == (
            f"lag-1 autocorrelation {check.autocorrelation:.3f},"
            f" spectral flatness {check.flatness:.3f}"
        )

    def test_white_noise_fails(self) -> None:
        check = check_speech(band_noise(top=RATE / 2), RATE)
        assert not check.speech

    def test_a_voice_in_noise_still_passes(self) -> None:
        # the voice's loud frames stand some 10 dB over the noise
        mixed = voiced() + 0.2 * band_noise(seed=3)
        assert check_speech(mixed, RATE).speech

    def test_silence_holds_no_speech(self) -> None:
        check = check_speech(np.zeros(RATE * 2, dtype=np.float32), RATE)
        assert not check.speech

    def test_shorter_than_a_frame_cannot_tell(self) -> None:
        check = check_speech(band_noise(seconds=0.01), RATE)
        assert check.speech and check.frames == 0

    def test_other_sample_rates(self) -> None:
        assert check_speech(voiced(rate=8000), 8000).speech
        assert not check_speech(band_noise(rate=8000), 8000).speech
