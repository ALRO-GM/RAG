"""TR-17: the expected-speakers rule, languages and the overlap detector."""

from typing import Optional, Sequence

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
from transcriber.application.ports import Audio, LanguageSpan, WordScores
from transcriber.application.transcribe import (
    QueueTranscription,
    TranscribeSession,
    TranscriptionOptions,
)
from transcriber.domain.models import Transcript, Word
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

    def test_an_unlisted_person_is_given_a_listed_name(self) -> None:
        """Known failure, pinned: it cannot be solved from the voices.

        Ana and Luis speak; the list says Ana and Eva. Luis is not
        enrolled and Eva's voiceprint is weak, 0.85 from Luis's voice:
        above the 0.8 floor. The list makes two groups and gives them
        to Ana and Eva, so every sentence of Luis is attributed to Eva.
        Only the speaker-level ``expected_floor`` says the name is a
        guess; readers derive it per sentence from it.
        """
        w = world()  # meeting(): ana and luis speak
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("eva", weak(1, 0.85), name="Eva")
        transcript = w.run(options("ana", "eva"))
        assert speakers(transcript) == {
            "ana": "threshold", "eva": "expected_floor",
        }
        said = {(u.start < 12.5, u.speaker) for u in transcript.utterances
                if 13.0 <= u.start < 25.0 or u.start < 12.5}
        assert said == {(True, "ana"), (False, "eva")}
        document = transcript.to_contract()
        floor = {s["id"] for s in document["speakers"]
                 if s.get("identification") == "expected_floor"}
        luis = [u for u in document["utterances"]
                if 13.0 <= u["start"] < 25.0]
        assert luis and all(u["speaker"] in floor for u in luis)

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


class Interrupted(Scene):
    """The meeting with Luis's voice heard for a moment inside Ana's turn.

    Only the window whose middle falls in 8.0-8.3 s hears Luis: too
    little to take any of Ana's words, enough to flag her sentence as
    an overlap through the voice windows.
    """

    def speaker_at(self, time: float) -> Optional[str]:
        """Luis at 8.0-8.3 s, otherwise as the scene says."""
        if 8.0 <= time <= 8.3:
            return "luis"
        return super().speaker_at(time)


def interrupted() -> Interrupted:
    """:func:`meeting` with Luis's moment inside Ana's first turn."""
    plain = meeting()
    return Interrupted(lines=plain.lines, voices=plain.voices)


class TestDetectorFailure:
    def test_a_failed_detector_keeps_the_voice_flags(self) -> None:
        plain = world(interrupted()).run()
        flags = [(u.overlap, u.overlap_source) for u in plain.utterances]
        assert (True, "voices") in flags
        detector = FakeOverlapDetector([(30.0, 32.0)])
        detector.failures.append(RuntimeError("no GPU"))
        failed = with_detector(world(interrupted()), detector).run()
        assert [(u.overlap, u.overlap_source)
                for u in failed.utterances] == flags

    def test_a_working_detector_adds_to_the_voice_flags(self) -> None:
        detector = FakeOverlapDetector([(30.0, 32.0)])
        found = with_detector(world(interrupted()), detector).run()
        sources = [u.overlap_source for u in found.utterances]
        assert "voices" in sources and "detector" in sources


class TestLanguages:
    def test_a_failed_scoring_sets_no_language(self) -> None:
        w = world()
        w.recogniser.detected = [(0.0, 12.5, "es"), (12.5, 40.0, "en")]
        w.recogniser.score_failures.append(RuntimeError("out of memory"))
        transcript = w.run()
        assert {(u.language, u.language_differs)
                for u in transcript.utterances} == {(None, None)}

    def test_scores_for_other_words_set_no_language(self) -> None:
        """The recogniser scored fewer words than it was given."""
        w = world()
        w.recogniser.detected = [(0.0, 40.0, "en")]
        score = w.recogniser.score_words

        def short(audio: Audio, words: Sequence[Word],
                  language: Optional[str] = None) -> WordScores:
            """Score all but the last word, with the languages."""
            full = score(audio, words, language)
            assert full.languages == [
                LanguageSpan(start=0.0, end=40.0, language="en")]
            return WordScores(words=full.words[:-1],
                              languages=full.languages)

        w.recogniser.score_words = short  # type: ignore[method-assign]
        transcript = w.run()
        assert {(u.language, u.language_differs)
                for u in transcript.utterances} == {(None, None)}
        assert {u.confidence for u in transcript.utterances} == {None}

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
