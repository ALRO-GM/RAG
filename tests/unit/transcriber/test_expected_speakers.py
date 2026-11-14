"""TR-17: the expected-speakers rule, languages and the overlap detector."""

from typing import Optional

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    DIM,
    FakeOverlapDetector,
    Line,
    Scene,
    meeting,
    voice,
)
from tests.unit.transcriber.test_transcribe import World, world
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
    TranscriptionOptions,
)
from transcriber.domain.models import Transcript
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.voices import FloatArray


def weak(index: int, cosine: float) -> FloatArray:
    """``voice(index)`` turned off its axis until its cosine is ``cosine``.

    Adds a component on an axis no voice uses (DIM - 2), so the cosine
    with the true voice is ``cosine``.
    """
    base = voice(index)
    norm = float(np.linalg.norm(base))
    extra = norm * np.sqrt(1.0 / cosine ** 2 - 1.0)
    out = base.copy()
    out[DIM - 2] = extra
    return out


def options(*listed: str, count: Optional[int] = None) -> TranscriptionOptions:
    """Options listing ``listed``, with ``count`` speakers (default: all)."""
    return TranscriptionOptions(
        expected_speakers=list(listed),
        num_speakers=len(listed) if count is None else count,
    )


def speakers(transcript: Transcript) -> dict[str, Optional[str]]:
    """``id → identification`` of every listed speaker."""
    return {s.id: s.identification for s in transcript.speakers}


class TestExpectedSpeakers:
    def test_a_listed_weak_voice_is_named_through_the_floor(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", weak(1, 0.85), name="Luis")
        transcript = w.run(options("ana", "luis"))
        assert speakers(transcript) == {
            "ana": "threshold", "luis": "expected_floor",
        }
        document = transcript.to_contract()
        identified = {s["id"]: s.get("identification")
                      for s in document["speakers"]}
        assert identified == {"ana": "threshold", "luis": "expected_floor"}
        assert Transcript.from_contract(document).to_contract() == document

    def test_an_open_session_keeps_the_match_threshold(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", weak(1, 0.85), name="Luis")
        transcript = w.run(TranscriptionOptions(num_speakers=2))
        assert speakers(transcript) == {"ana": "threshold", "S1": None}

    def test_a_listed_person_absent_stays_anonymous(self) -> None:
        """Eva is listed but Luis speaks: his voice is far from hers."""
        w = world()
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("eva", voice(2), name="Eva")
        transcript = w.run(options("ana", "eva"))
        assert speakers(transcript) == {"ana": "threshold", "S1": None}

    def test_the_floor_is_a_parameter(self) -> None:
        w = world(parameters=TranscriptionParameters(expected_floor=0.9))
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", weak(1, 0.85), name="Luis")
        transcript = w.run(options("ana", "luis"))
        assert speakers(transcript) == {"ana": "threshold", "S1": None}

    def test_a_count_other_than_the_list_is_an_open_session(self) -> None:
        """An unlisted third voice: the count says 3, the list 2."""
        scene = Scene(
            lines=[
                Line("ana", 0.0, 10.0, "Hola. " * 10),
                Line("luis", 11.0, 21.0, "Vale. " * 10),
                Line("eva", 22.0, 32.0, "Bien. " * 10),
            ],
            voices={"ana": voice(0), "luis": voice(1), "eva": voice(2)},
        )
        w = world(scene)
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", weak(1, 0.85), name="Luis")
        transcript = w.run(options("ana", "luis", count=3))
        assert speakers(transcript) == {
            "ana": "threshold", "S1": None, "S2": None,
        }

    def test_an_unlisted_voice_in_a_listed_count_names_nobody_twice(
        self,
    ) -> None:
        """The list says 2 and a third voice speaks: the list guards
        the names, so the third voice joins a group, but no person is
        given to two groups."""
        scene = Scene(
            lines=[
                Line("ana", 0.0, 10.0, "Hola. " * 10),
                Line("luis", 11.0, 21.0, "Vale. " * 10),
                Line("eva", 22.0, 32.0, "Bien. " * 10),
            ],
            voices={"ana": voice(0), "luis": voice(1), "eva": voice(2)},
        )
        w = world(scene)
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = w.run(options("ana", "luis"))
        ids = [s.id for s in transcript.speakers]
        assert len(ids) == len(set(ids)) == 2

    def test_a_voiceprint_close_to_both_voices_names_one(self) -> None:
        """Ana's voiceprint is near both voices, Luis's near his only.

        Both clusters are most similar to Ana's voiceprint; the one to
        one assignment gives each person one cluster.
        """
        w = world(parameters=TranscriptionParameters(expected_floor=0.3))
        w.voiceprints.enrol("ana", voice(0) + 0.9 * voice(1), name="Ana")
        w.voiceprints.enrol("luis", weak(1, 0.6), name="Luis")
        transcript = w.run(options("ana", "luis"))
        assert sorted(speakers(transcript)) == ["ana", "luis"]
        first = transcript.utterances[0]
        assert first.speaker == "ana"


def with_detector(w: World, detector: FakeOverlapDetector) -> World:
    """``w`` with an overlap detector in its use cases."""
    w.transcribe = TranscribeSession(
        w.sessions, w.voiceprints, w.decoder, w.recogniser, w.encoder,
        w.locks, TranscriptionParameters(), detector,
    )
    w.queue = QueueTranscription(w.sessions, w.voiceprints, w.runner,
                                 w.transcribe, w.locks)
    return w


class TestOverlapDetector:
    def test_a_detected_overlap_flags_the_sentence(self) -> None:
        detector = FakeOverlapDetector([(14.0, 15.0)])
        w = with_detector(world(meeting()), detector)
        transcript = w.run()
        flagged = [u for u in transcript.utterances if u.overlap]
        assert len(flagged) == 1
        assert flagged[0].start <= 14.0 and flagged[0].end >= 15.0
        assert flagged[0].overlap_source == "detector"
        assert detector.calls == 1
        others = [u for u in transcript.utterances if not u.overlap]
        assert all(u.overlap is False and u.overlap_source is None
                   for u in others)

    def test_too_short_an_overlap_does_not(self) -> None:
        w = with_detector(world(), FakeOverlapDetector([(14.0, 14.3)]))
        assert not any(u.overlap for u in w.run().utterances)

    def test_a_failed_detector_keeps_the_job(self) -> None:
        detector = FakeOverlapDetector([(14.0, 15.0)])
        detector.failures.append(RuntimeError("no GPU"))
        w = with_detector(world(), detector)
        transcript = w.run()
        assert not any(u.overlap for u in transcript.utterances)
        status = w.runner.status("s1")
        assert status is not None
        assert "overlap detector failed" in (status.message or "")


class TestLanguages:
    def test_auto_detects_and_records_the_language(self) -> None:
        w = world()
        w.recogniser.detected = [(0.0, 12.5, "es"), (12.5, 40.0, "en")]
        transcript = w.run(TranscriptionOptions(language="auto"))
        assert w.recogniser.languages == [None]
        assert transcript.session.language == "auto"
        found = {(u.speaker, u.language, u.language_differs)
                 for u in transcript.utterances}
        assert found == {("S1", "es", None), ("S2", "en", None),
                         ("S1", "en", None)}

    def test_a_fixed_language_flags_another_one(self) -> None:
        w = world()
        w.recogniser.detected = [(0.0, 12.5, "es"), (12.5, 25.5, "en"),
                                 (25.5, 40.0, "es")]
        transcript = w.run()
        assert w.recogniser.languages == ["es"]
        differs = {u.speaker: u.language_differs
                   for u in transcript.utterances}
        assert differs == {"S1": False, "S2": True}
        document = transcript.to_contract()
        assert document["utterances"][0]["language"] == "es"
        assert Transcript.from_contract(document).to_contract() == document

    def test_without_detection_nothing_is_said(self) -> None:
        transcript = world().run()
        assert {u.language for u in transcript.utterances} == {None}
        assert "language" not in transcript.to_contract()["utterances"][0]

    @pytest.mark.parametrize("language", ["auto", "AUTO", "es-ES"])
    def test_accepted_session_languages(self, language: str) -> None:
        w = world()
        w.run(TranscriptionOptions(language=language))
        session = w.sessions.get("s1")
        assert session is not None and session.language == language
