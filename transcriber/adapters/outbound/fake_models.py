"""Deterministic stand-ins for the speech and voice models.

Selected with ``runtime="fake"`` in the settings, they let the CLI, the
HTTP API and the whole pipeline run in a subprocess or on CI without
torch or a download. They do not understand speech; they react to the
signal in a fixed, predictable way:

- :class:`FakeSpeechRecogniser` cuts the audio into 0.4-second slots
  and "hears" one word in every slot loud enough to be speech. A
  sentence ends at a pause or after eight words.
- :class:`FakeVoiceEncoder` describes each window by how its energy is
  spread over frequency bands, so tones of different pitch sound like
  different people and the same tone like the same person.
- :class:`FakeOverlapDetector` hears two people at once where a slot
  holds two tones of clearly different pitch (two strong peaks).
"""

from typing import AbstractSet, List, Optional, Sequence, Tuple

import numpy as np

from transcriber.adapters.outbound.whisper_languages import (
    WHISPER_LANGUAGES,
)
from transcriber.application.ports import Audio, RecognisedSpeech, WordScores
from transcriber.domain.models import Word
from transcriber.domain.voices import FloatArray

FAKE_RECOGNISER = "fake/recogniser"
FAKE_VOICE_MODEL = "fake/voice-encoder"

WORD_SECONDS = 0.4
"""Length of the slot one fake word occupies."""

SPEECH_RMS = 0.01
"""Root-mean-square level above which a slot counts as speech."""

SENTENCE_WORDS = 8
"""Longest fake sentence, in words."""

VOCABULARY = (
    "hola", "vamos", "a", "repartir", "las", "tareas", "del",
    "proyecto", "esta", "semana", "con", "calma",
)

BANDS = 24
"""Number of log-spaced frequency bands of a fake voice embedding."""

LOWEST_HZ = 50.0


class FakeSpeechRecogniser:
    """One word per loud 0.4-second slot; punctuation at pauses."""

    model_name = FAKE_RECOGNISER

    @property
    def supported_languages(self) -> AbstractSet[str]:
        """Return the codes the real recogniser knows (Whisper's)."""
        return WHISPER_LANGUAGES

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        """Return a word for every slot of ``audio`` above the level."""
        size = max(int(WORD_SECONDS * audio.sample_rate), 1)
        count = int(audio.samples.shape[0]) // size
        loud = [
            float(np.sqrt(np.mean(
                np.square(audio.samples[n * size:(n + 1) * size],
                          dtype=np.float64)
            ))) >= SPEECH_RMS
            for n in range(count)
        ]
        words: List[Word] = []
        in_sentence = 0
        for slot, speech in enumerate(loud):
            if not speech:
                continue
            text = VOCABULARY[len(words) % len(VOCABULARY)]
            if in_sentence == 0:
                text = text.capitalize()
            in_sentence += 1
            last = slot + 1 >= count or not loud[slot + 1]
            if last or in_sentence >= SENTENCE_WORDS:
                text += "."
                in_sentence = 0
            words.append(Word(
                text=f" {text}",
                start=round(slot * WORD_SECONDS, 3),
                end=round((slot + 1) * WORD_SECONDS, 3),
            ))
        return RecognisedSpeech(words=words, language=language)

    def score_words(
        self, audio: Audio, words: Sequence[Word],
        language: Optional[str] = None,
    ) -> WordScores:
        """Score nothing: every word keeps ``probability`` ``None``."""
        return WordScores(words=list(words))

    def release(self) -> None:
        """Nothing to unload."""


OVERLAP_SECONDS = 0.5
"""Length of the slots the fake overlap detector judges."""


class FakeOverlapDetector:
    """Two voices where a slot's spectrum has two strong, distant peaks."""

    model_name = "fake/overlap-detector"

    def overlaps(self, audio: Audio) -> List[Tuple[float, float]]:
        """Return the slots whose two strongest bins are far apart.

        A slot is loud enough to be speech (:data:`SPEECH_RMS`), and its
        second strongest frequency, at least 50 Hz from the strongest,
        has at least a third of its power: two tones at once.
        """
        size = max(int(OVERLAP_SECONDS * audio.sample_rate), 1)
        spans: List[Tuple[float, float]] = []
        for first in range(0, int(audio.samples.shape[0]) - size + 1, size):
            clip = audio.samples[first:first + size].astype(np.float64)
            if float(np.sqrt(np.mean(np.square(clip)))) < SPEECH_RMS:
                continue
            power = np.square(np.abs(np.fft.rfft(clip)))
            hz = np.fft.rfftfreq(clip.size, 1.0 / audio.sample_rate)
            top = int(np.argmax(power))
            far = np.abs(hz - hz[top]) >= 50.0
            if far.any() and float(power[far].max()) >= power[top] / 3:
                start = first / audio.sample_rate
                spans.append((start, start + OVERLAP_SECONDS))
        return spans

    def release(self) -> None:
        """Nothing to unload."""


class FakeVoiceEncoder:
    """Embed each window as its unit spectral-band energy profile."""

    model_name = FAKE_VOICE_MODEL

    def embed(
        self, audio: Audio, windows: Sequence[Tuple[float, float]]
    ) -> FloatArray:
        """Return one unit row per window; silence gives a zero row."""
        if not windows:
            return np.zeros((0, 0), dtype=np.float32)
        edges = np.geomspace(
            LOWEST_HZ, audio.sample_rate / 2, BANDS + 1
        )
        rows = np.zeros((len(windows), BANDS), dtype=np.float32)
        for row, (start, end) in enumerate(windows):
            clip = audio.cut(start, end)
            if clip.size == 0:
                continue
            power = np.square(np.abs(np.fft.rfft(clip)))
            hz = np.fft.rfftfreq(clip.size, 1.0 / audio.sample_rate)
            band = np.digitize(hz, edges) - 1
            inside = (band >= 0) & (band < BANDS)
            energy = np.bincount(
                band[inside], weights=power[inside], minlength=BANDS
            )
            norm = float(np.linalg.norm(energy))
            if norm > 1e-9:
                rows[row] = energy / norm
        return rows

    def release(self) -> None:
        """Nothing to unload."""
