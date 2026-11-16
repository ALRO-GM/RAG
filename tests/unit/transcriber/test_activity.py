"""Transcriber speech activity: frames with energy, words kept on them."""

import numpy as np
import pytest

from transcriber.domain.activity import (
    SpeechActivity,
    heard_words,
    speech_activity,
    unheard_tail,
)
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Word
from transcriber.domain.voices import FloatArray

RATE = 100


def audio(*parts: tuple[float, float]) -> FloatArray:
    """Concatenate ``(seconds, level)`` parts of a constant signal."""
    return np.concatenate([
        np.full(int(seconds * RATE), level, dtype=np.float32)
        for seconds, level in parts
    ]).astype(np.float32)


class TestSpeechActivity:
    def test_frames_at_or_above_the_level_are_active(self) -> None:
        samples = audio((0.1, 0.0), (0.1, 0.5), (0.1, 0.002))
        activity = speech_activity(samples, RATE, 0.05, 0.003)
        assert activity.frame == pytest.approx(0.05)
        assert activity.active.tolist() == [
            False, False, True, True, False, False,
        ]
        exact = speech_activity(audio((0.1, 0.003)), RATE, 0.05, 0.003)
        assert exact.active.all()

    def test_the_level_is_root_mean_square(self) -> None:
        """A loud click in a quiet frame lifts its RMS above the level."""
        samples = np.zeros(10, dtype=np.float32)
        samples[3] = 0.1  # RMS of the frame: 0.1 / sqrt(5), about 0.045
        activity = speech_activity(samples, RATE, 0.05, 0.04)
        assert activity.active.tolist() == [True, False]
        quieter = speech_activity(samples, RATE, 0.05, 0.05)
        assert quieter.active.tolist() == [False, False]

    def test_a_short_last_frame_is_averaged_over_its_own_samples(
        self,
    ) -> None:
        samples = audio((0.05, 0.0), (0.02, 0.01))
        activity = speech_activity(samples, RATE, 0.05, 0.009)
        assert activity.active.tolist() == [False, True]

    def test_a_frame_is_at_least_one_sample(self) -> None:
        activity = speech_activity(audio((0.03, 0.5)), RATE, 0.001, 0.1)
        assert activity.frame == pytest.approx(0.01)
        assert activity.active.tolist() == [True, True, True]

    def test_empty_audio_has_no_active_frame(self) -> None:
        activity = speech_activity(np.zeros(0, np.float32), RATE, 0.05, 0.0)
        assert activity.active.shape == (0,)
        assert not activity.heard(0.0, 1.0)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(InvalidInputError, match="1-D"):
            speech_activity(np.zeros((2, 2), np.float32), RATE, 0.05, 0.1)
        with pytest.raises(InvalidInputError, match="positive"):
            speech_activity(np.zeros(4, np.float32), 0, 0.05, 0.1)
        with pytest.raises(InvalidInputError, match="positive"):
            speech_activity(np.zeros(4, np.float32), RATE, 0.0, 0.1)


class TestHeardWords:
    def activity(self) -> SpeechActivity:
        """Frames of 0.1 s; only 0.2-0.4 s is active."""
        return SpeechActivity(
            0.1, np.array([False, False, True, True, False], dtype=bool)
        )

    def test_spans_touching_an_active_frame(self) -> None:
        activity = self.activity()
        assert activity.heard(0.25, 0.3)
        assert activity.heard(0.0, 0.25)  # overlaps the first active one
        assert activity.heard(0.35, 0.6)
        assert not activity.heard(0.0, 0.2)  # ends where speech starts
        assert not activity.heard(0.4, 0.5)
        assert activity.heard(0.3, 0.3)  # zero length: its own frame
        assert not activity.heard(0.45, 0.45)
        assert not activity.heard(0.6, 0.9)  # past the end of the audio

    def test_words_over_silence_are_dropped(self) -> None:
        words = [
            Word(text=" Gracias.", start=0.0, end=0.15),
            Word(text=" Hola", start=0.2, end=0.3),
            Word(text=" ya.", start=0.42, end=0.48),
        ]
        assert heard_words(words, self.activity()) == [words[1]]
        assert heard_words([], self.activity()) == []


class TestUnheardTail:
    """Speech that goes on after the last recognised word (#125)."""

    def activity(self) -> SpeechActivity:
        """Frames of 0.1 s; 0.0-0.3 s and 0.5-0.8 s are active."""
        return SpeechActivity(0.1, np.array(
            [True, True, True, False, False, True, True, True], dtype=bool
        ))

    def test_seconds_of_active_frames_from_a_time_on(self) -> None:
        activity = self.activity()
        assert activity.seconds_after(0.0) == pytest.approx(0.6)
        assert activity.seconds_after(0.3) == pytest.approx(0.3)
        # The frame holding 0.55 s is not counted.
        assert activity.seconds_after(0.55) == pytest.approx(0.2)
        assert activity.seconds_after(0.8) == 0.0
        assert activity.seconds_after(5.0) == 0.0

    def test_speech_after_the_last_word_starts_at_its_end(self) -> None:
        words = [
            Word(text=" Hola", start=0.25, end=0.3),
            Word(text=" ya.", start=0.0, end=0.2),
        ]
        activity = self.activity()
        assert unheard_tail(words, activity, 0.3) == pytest.approx(0.3)
        assert unheard_tail(words, activity, 0.31) is None

    def test_no_tail_when_the_words_reach_the_end(self) -> None:
        words = [Word(text=" adiós.", start=0.5, end=0.8)]
        assert unheard_tail(words, self.activity(), 0.1) is None

    def test_no_words_or_no_minimum_turn_the_check_off(self) -> None:
        words = [Word(text=" Hola", start=0.0, end=0.1)]
        assert unheard_tail([], self.activity(), 0.1) is None
        assert unheard_tail(words, self.activity(), 0.0) is None
