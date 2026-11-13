"""TranscriptionService with fakes of the audio, speech and voice ports."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

from src.application.skills.ports import Audio, RecognisedSpeech
from src.application.skills.transcription import TranscriptionService
from src.domain.errors import InvalidInputError
from src.domain.skills.speech import (
    SpeakerSpan,
    smooth_speakers,
    transcript_document,
    transcript_sentences,
)
from src.domain.skills.transcript import Word, parse_json


class FakeReader:
    """Returns one second of silence whatever the path."""

    def __init__(self) -> None:
        self.paths: List[str] = []

    def read(self, path: str) -> Audio:
        self.paths.append(path)
        return Audio(np.zeros(16000, dtype=np.float32), 16000)


class FakeRecogniser:
    """Returns the words it was built with."""

    model_name = "fake-asr"

    def __init__(self, words: List[Word], language: Optional[str] = "es"):
        self.words = words
        self.language = language
        self.asked: List[Optional[str]] = []

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        self.asked.append(language)
        return RecognisedSpeech(words=self.words, language=self.language)


class FakeDiarizer:
    """Returns fixed spans and records what it was asked."""

    def __init__(self, spans: List[SpeakerSpan]):
        self.spans = spans
        self.calls: List[Tuple[List[Tuple[float, float]], Optional[int]]] = []

    def diarize(
        self,
        audio: Audio,
        regions: Sequence[Tuple[float, float]],
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSpan]:
        self.calls.append((list(regions), num_speakers))
        return self.spans


def spoken() -> List[Word]:
    """Two people: a greeting, an answer, then the first one again."""
    items = [
        (" Hola", 0.0, 0.4), (" a", 0.5, 0.6), (" todos.", 0.6, 1.0),
        (" ¿Empezamos?", 1.1, 1.8),
        (" Sí,", 2.5, 2.8), (" claro.", 2.8, 3.2),
        (" Vale.", 6.0, 6.4), ("  ", 6.5, 6.6),
    ]
    return [Word(text=t, start=s, end=e, probability=0.9)
            for t, s, e in items]


SPANS = [
    SpeakerSpan(speaker="SPEAKER_01", start=0.0, end=2.0),
    SpeakerSpan(speaker="SPEAKER_02", start=2.0, end=4.0),
    SpeakerSpan(speaker="SPEAKER_01", start=5.5, end=7.0),
]


def service(
    diarizer: Optional[FakeDiarizer] = None, **kwargs: float
) -> Tuple[TranscriptionService, FakeRecogniser]:
    recogniser = FakeRecogniser(spoken())
    return TranscriptionService(
        FakeReader(), recogniser, diarizer, **kwargs  # type: ignore[arg-type]
    ), recogniser


def test_words_become_turns_with_speakers_times_and_words() -> None:
    diarizer = FakeDiarizer(SPANS)
    svc, recogniser = service(diarizer)
    transcript = svc.transcribe("meeting/standup.wav", num_speakers=2)
    assert transcript.session_id == "standup"
    assert transcript.language == "es"
    assert [(t.speaker, t.text, t.start, t.end) for t in transcript.turns] == [
        ("SPEAKER_01", "Hola a todos. ¿Empezamos?", 0.0, 1.8),
        ("SPEAKER_02", "Sí, claro.", 2.5, 3.2),
        ("SPEAKER_01", "Vale.", 6.0, 6.4),
    ]
    assert transcript.has_timestamps()
    assert transcript.turns[0].confidence == pytest.approx(0.9)
    assert len(transcript.turns[0].words) == 4
    assert recogniser.asked == [None]
    regions, asked_speakers = diarizer.calls[0]
    assert asked_speakers == 2
    assert regions == [(0.0, 1.8), (2.5, 3.2), (6.0, 6.4)]


def test_sentences_get_exact_times_from_words() -> None:
    svc, _ = service(FakeDiarizer(SPANS))
    sentences = transcript_sentences(svc.transcribe("a.wav"))
    assert [(s.id, s.speaker, s.text, s.start, s.end) for s in sentences] == [
        ("t0.s0", "SPEAKER_01", "Hola a todos.", 0.0, 1.0),
        ("t0.s1", "SPEAKER_01", "¿Empezamos?", 1.1, 1.8),
        ("t1.s0", "SPEAKER_02", "Sí, claro.", 2.5, 3.2),
        ("t2.s0", "SPEAKER_01", "Vale.", 6.0, 6.4),
    ]


def test_names_session_and_language_are_applied() -> None:
    svc, recogniser = service(FakeDiarizer(SPANS))
    transcript = svc.transcribe(
        "a.wav", session_id="s1", language="es",
        names={"SPEAKER_01": "Ana", "SPEAKER_02": "Luis"},
    )
    assert transcript.session_id == "s1"
    assert transcript.speakers() == ["Ana", "Luis"]
    assert recogniser.asked == ["es"]


def test_without_diarizer_or_with_one_speaker_everything_is_one_voice(
) -> None:
    svc, _ = service(None)
    assert svc.transcribe("a.wav").speakers() == ["SPEAKER_00"]
    diarizer = FakeDiarizer(SPANS)
    svc, _ = service(diarizer)
    assert svc.transcribe("a.wav", num_speakers=1).speakers() == [
        "SPEAKER_00",
    ]
    assert diarizer.calls == []


def test_requested_language_is_kept_when_none_is_detected() -> None:
    recogniser = FakeRecogniser(spoken(), language=None)
    svc = TranscriptionService(FakeReader(), recogniser)
    assert svc.transcribe("a.wav", language="en").language == "en"


def test_silence_is_an_error() -> None:
    svc = TranscriptionService(
        FakeReader(), FakeRecogniser([Word(text=" ", start=0, end=1)])
    )
    with pytest.raises(InvalidInputError, match="No speech"):
        svc.transcribe("quiet.wav")


@pytest.mark.parametrize("kwargs", [{"turn_gap": -1.0}, {"smooth": -1}])
def test_invalid_settings(kwargs: dict[str, float]) -> None:
    with pytest.raises(InvalidInputError):
        service(None, **kwargs)


def test_invalid_number_of_speakers() -> None:
    svc, _ = service(None)
    with pytest.raises(InvalidInputError, match="num_speakers"):
        svc.transcribe("a.wav", num_speakers=0)


def test_turn_gap_controls_the_split() -> None:
    svc, _ = service(FakeDiarizer(SPANS), turn_gap=10.0)
    turns = svc.transcribe("a.wav").turns
    assert [t.speaker for t in turns] == ["SPEAKER_01", "SPEAKER_02",
                                          "SPEAKER_01"]
    one_voice, _ = service(None, turn_gap=10.0)
    assert len(one_voice.transcribe("a.wav").turns) == 1


class TestSmoothSpeakers:
    @staticmethod
    def labelled(pattern: str) -> List[Word]:
        return [Word(text=c, start=i, end=i + 1, speaker=c)
                for i, c in enumerate(pattern)]

    @pytest.mark.parametrize("pattern, max_run, expected", [
        ("AABAA", 1, "AAAAA"),
        ("AABBAA", 1, "AABBAA"),
        ("AABBAA", 2, "AAAAAA"),
        ("AABCC", 1, "AABCC"),   # different neighbours: a real change
        ("BAAA", 1, "BAAA"),     # edges are kept
        ("AAAB", 1, "AAAB"),
        ("", 1, ""),
    ])
    def test_patterns(self, pattern: str, max_run: int,
                      expected: str) -> None:
        out = smooth_speakers(self.labelled(pattern), max_run)
        assert "".join(w.speaker or "?" for w in out) == expected

    def test_service_can_disable_smoothing(self) -> None:
        spans = [SpeakerSpan(speaker="SPEAKER_01", start=0.0, end=0.45),
                 SpeakerSpan(speaker="SPEAKER_02", start=0.45, end=0.62),
                 SpeakerSpan(speaker="SPEAKER_01", start=0.62, end=7.0)]
        smoothed, _ = service(FakeDiarizer(spans))
        raw, _ = service(FakeDiarizer(spans), smooth=0)
        assert smoothed.transcribe("a.wav").speakers() == ["SPEAKER_01"]
        assert raw.transcribe("a.wav").speakers() == [
            "SPEAKER_01", "SPEAKER_02",
        ]


def test_document_round_trip_keeps_turns_words_and_sentences() -> None:
    import json

    svc, _ = service(FakeDiarizer(SPANS))
    transcript = svc.transcribe("a.wav", session_id="s1")
    document = transcript_document(transcript)
    assert document["speakers"] == ["SPEAKER_01", "SPEAKER_02"]
    assert [s["id"] for s in document["sentences"]] == [
        "t0.s0", "t0.s1", "t1.s0", "t2.s0",
    ]
    again = parse_json("x", json.dumps(document))
    assert again.session_id == "s1"
    assert [t.text for t in again.turns] == [t.text for t in transcript.turns]
    assert transcript_sentences(again) == transcript_sentences(transcript)


def test_audio_cut_and_duration() -> None:
    audio = Audio(np.arange(32000, dtype=np.float32), 16000)
    assert audio.duration == 2.0
    assert audio.cut(0.5, 1.0).shape[0] == 8000
    assert audio.cut(1.5, 9.0).shape[0] == 8000
    assert audio.cut(-1.0, 0.25).shape[0] == 4000
    assert audio.cut(3.0, 4.0).shape[0] == 0
