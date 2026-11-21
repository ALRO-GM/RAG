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
- :class:`FakeVad` and :class:`FakeAsrStream` stand in for the live
  mode's VAD and streaming recogniser, by loudness too.
- :class:`FakeStreamDiarizer` stands in for the streaming diarizer of
  mono live sessions: a loud 80 ms frame is the first speaker below
  :data:`PITCH_SPLIT_HZ` and the second above it, ``lag_frames``
  frames after the audio.
"""

from typing import AbstractSet, List, Optional, Sequence, Tuple

import numpy as np

from transcriber.adapters.outbound.whisper_languages import (
    WHISPER_LANGUAGES,
)
from transcriber.application.ports import Audio, RecognisedSpeech, WordScores
from transcriber.domain.live import SAMPLE_RATE, Hypothesis
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


class FakeVad:
    """Live stand-in for the VAD: a frame is speech when it is loud.

    The probability is 0.9 above :data:`SPEECH_RMS` and 0.05 below.
    """

    def probability(self, frame: FloatArray) -> float:
        """Return 0.9 for a loud frame, 0.05 for a quiet one."""
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0
        return 0.9 if rms > SPEECH_RMS else 0.05

    def reset(self) -> None:
        """Nothing to forget."""


class FakeAsrStream:
    """Live stand-in for the streaming recogniser.

    Hears one word (``palabra<n>``) per 0.4-second slot loud enough to
    be speech; a final ends with a period and carries word times.
    """

    def __init__(self, language: Optional[str] = "es") -> None:
        """Keep the language reported."""
        self.language = language or "es"
        self.warmed = 0

    def warm_up(self) -> None:
        """Count the warm-ups."""
        self.warmed += 1

    def decode(self, samples: FloatArray, final: bool) -> Hypothesis:
        """Return one word per loud slot of ``samples``."""
        slot = int(WORD_SECONDS * SAMPLE_RATE)
        words: List[Word] = []
        for n in range(len(samples) // slot):
            piece = samples[n * slot:(n + 1) * slot]
            if float(np.sqrt(np.mean(np.square(piece)))) > SPEECH_RMS:
                words.append(Word(
                    text=f"palabra{len(words)}",
                    start=n * WORD_SECONDS, end=(n + 1) * WORD_SECONDS,
                    probability=0.9,
                ))
        text = " ".join(w.text for w in words)
        if final and text:
            text += "."
            last = words[-1]
            words[-1] = last.model_copy(update={"text": last.text + "."})
        return Hypothesis(
            text=text, words=words if final else [],
            language=self.language, confidence=0.9 if words else None,
        )


PITCH_SPLIT_HZ = 300.0
"""Pitch that separates the fake diarizer's two speakers."""

FAKE_DIARIZER = "fake/stream-diarizer"


class FakeStreamDiarizer:
    """Streaming diarizer stand-in: speaker by pitch, frame by frame."""

    model_name = FAKE_DIARIZER
    frame_seconds = 0.08
    slots = 4

    def __init__(self, lag_frames: int = 0) -> None:
        """Answer each frame ``lag_frames`` frames after its audio."""
        self.lag_frames = lag_frames
        self.started = 0
        self.closed = 0
        self._audio: FloatArray = np.zeros(0, dtype=np.float32)
        self._rows: List[List[float]] = []
        self._given = 0
        self._ended = False

    def start(self) -> None:
        """Forget the stream."""
        self.started += 1
        self._audio = np.zeros(0, dtype=np.float32)
        self._rows, self._given, self._ended = [], 0, False

    def feed(self, samples: FloatArray) -> None:
        """Add audio; compute the frames it completes."""
        self._audio = np.concatenate([self._audio, samples])
        self._compute()

    def _compute(self) -> None:
        """Turn every whole frame of audio into a row of probabilities."""
        size = int(self.frame_seconds * SAMPLE_RATE)
        while (len(self._rows) + 1) * size <= len(self._audio):
            n = len(self._rows)
            frame = self._audio[n * size:(n + 1) * size]
            row = [0.0] * self.slots
            if float(np.sqrt(np.mean(np.square(frame)))) > SPEECH_RMS:
                power = np.abs(np.fft.rfft(frame))
                pitch = float(np.fft.rfftfreq(size, 1 / SAMPLE_RATE)[
                    int(np.argmax(power))
                ])
                row[0 if pitch < PITCH_SPLIT_HZ else 1] = 0.9
            self._rows.append(row)

    def frames(
        self, until: float = 0.0, timeout: float = 0.0
    ) -> Tuple[int, List[List[float]]]:
        """Return the frames ready (all of them once the stream ended)."""
        ready = len(self._rows) if self._ended else max(
            len(self._rows) - self.lag_frames, 0
        )
        first = self._given
        self._given = max(ready, first)
        return first, self._rows[first:self._given]

    def finish(self) -> None:
        """Release the frames held back."""
        self._ended = True

    def close(self) -> None:
        """Count the closes."""
        self.closed += 1
