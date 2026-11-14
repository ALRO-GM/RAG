"""Transcriber enrolment use cases: enrol with consent, forget."""

import inspect
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    NOW,
    FakeDecoder,
    FakeJobRunner,
    FakeRecogniser,
    FakeSessionRepository,
    FakeVoiceEncoder,
    FakeVoiceprintRepository,
    FixedClock,
    Line,
    Scene,
    reading,
    voice,
)
from transcriber.application.enrol import EnrolSpeaker, ForgetSpeaker
from transcriber.application.ports import (
    Audio,
    RecognisedSpeech,
    RecordedAudio,
)
from transcriber.domain.enrolment import Rejection, SampleRules
from transcriber.domain.errors import (
    ConsentRequiredError,
    InvalidInputError,
    NotFoundError,
    SampleRejectedError,
    StateError,
)
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
)
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.sentences import build_utterances
from transcriber.domain.speakers import turns_from_words
from transcriber.domain.voices import FloatArray

SAMPLE = RecordedAudio(b"sample-bytes", "wav")


def enrol_use_case(
    scene: Scene,
    voiceprints: FakeVoiceprintRepository,
    clock: FixedClock = FixedClock(),
    rules: SampleRules = SampleRules(),
) -> EnrolSpeaker:
    """The use case around fakes that hear ``scene``."""
    return EnrolSpeaker(
        FakeDecoder(scene), FakeRecogniser(scene), FakeVoiceEncoder(scene),
        voiceprints, clock, FakeJobRunner(), TranscriptionParameters(),
        rules,
    )


class TestEnrolSpeaker:
    def test_enrolment_keeps_voiceprint_and_consent(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        enrol = enrol_use_case(reading(), voiceprints)
        voiceprint = enrol("ana", " Ana  María ", "es", SAMPLE, True,
                           "consent-1")
        unit = voice(0) / np.linalg.norm(voice(0))
        assert float(np.dot(voiceprint.vector, unit)) > 0.99
        assert voiceprint.model == "fake-voice"
        assert voiceprint.consent_version == "consent-1"
        assert voiceprint.enrolled_at == NOW
        assert voiceprints.get("ana") == voiceprint
        consent = voiceprints.consent("ana")
        assert consent is not None
        assert (consent.name, consent.consent_version,
                consent.text_version, consent.given_at) == (
            "Ana María", "consent-1", "es-1", NOW,
        )
        assert consent.given_at.tzinfo is not None

    def test_the_models_run_as_a_job_of_the_runner(self) -> None:
        """Enrolment never uses the models beside another job."""
        scene = reading()
        runner = FakeJobRunner()
        seen: List[bool] = []

        class Watched(FakeRecogniser):
            def recognise(
                self, audio: Audio, language: Optional[str] = None
            ) -> RecognisedSpeech:
                seen.append(runner.inside)
                return super().recognise(audio, language)

        class WatchedEncoder(FakeVoiceEncoder):
            def embed(
                self, audio: Audio, windows: Sequence[Tuple[float, float]]
            ) -> FloatArray:
                seen.append(runner.inside)
                return super().embed(audio, windows)

        EnrolSpeaker(FakeDecoder(scene), Watched(scene),
                     WatchedEncoder(scene), FakeVoiceprintRepository(),
                     FixedClock(), runner)(
            "ana", "Ana", "es", SAMPLE, True, "consent-1")
        assert seen == [True, True]
        assert runner.calls == 1

    def test_a_refused_job_stores_nothing(self) -> None:
        scene = reading()
        runner = FakeJobRunner()
        runner.refuse = StateError("The job runner is closed.")
        voiceprints = FakeVoiceprintRepository()
        with pytest.raises(StateError, match="closed"):
            EnrolSpeaker(FakeDecoder(scene), FakeRecogniser(scene),
                         FakeVoiceEncoder(scene), voiceprints,
                         FixedClock(), runner)(
                "ana", "Ana", "es", SAMPLE, True, "consent-1")
        assert voiceprints.voiceprints == {}

    def test_the_sample_is_cut_into_long_windows(self) -> None:
        scene = reading()
        encoder = FakeVoiceEncoder(scene)
        EnrolSpeaker(FakeDecoder(scene), FakeRecogniser(scene), encoder,
                     FakeVoiceprintRepository(), FixedClock(),
                     FakeJobRunner())(
            "ana", "Ana", "es", SAMPLE, True, "consent-1")
        lengths = {round(end - start, 3) for start, end in encoder.calls[0]}
        assert lengths == {5.0}

    def test_the_audio_is_never_stored(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        scene = reading()
        decoder = FakeDecoder(scene)
        EnrolSpeaker(decoder, FakeRecogniser(scene), FakeVoiceEncoder(scene),
                     voiceprints, FixedClock(), FakeJobRunner())(
            "ana", "Ana", "es", SAMPLE, True, "consent-1"
        )
        assert decoder.decoded == [SAMPLE]  # the audio was used...
        stored = repr(voiceprints.voiceprints) + repr(voiceprints.consents)
        assert "sample-bytes" not in stored  # ...and not kept
        # The only store the use case can write to is the voiceprints':
        # it is given no port that could keep audio.
        ports = inspect.signature(EnrolSpeaker.__init__).parameters
        assert not any(
            "Session" in str(param.annotation) for param in ports.values()
        )

    @pytest.mark.parametrize("consent", [False, None, "yes", 1])
    def test_enrolment_without_consent_is_refused(
        self, consent: object
    ) -> None:
        voiceprints = FakeVoiceprintRepository()
        scene = reading()
        decoder = FakeDecoder(scene)
        enrol = EnrolSpeaker(decoder, FakeRecogniser(scene),
                             FakeVoiceEncoder(scene), voiceprints,
                             FixedClock(), FakeJobRunner())
        with pytest.raises(ConsentRequiredError):
            enrol("ana", "Ana", "es", SAMPLE, consent,  # type: ignore
                  "consent-1")
        assert voiceprints.voiceprints == {}
        assert decoder.decoded == []

    def test_re_enrolling_replaces_the_voiceprint(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        enrol_use_case(reading(), voiceprints)(
            "ana", "Ana", "es", SAMPLE, True, "consent-1"
        )
        other = reading()
        other.voices["ana"] = voice(3)
        later = FixedClock(NOW + timedelta(days=1))
        enrol_use_case(other, voiceprints, later)(
            "ana", "Ana", "en", SAMPLE, True, "consent-2"
        )
        assert list(voiceprints.voiceprints) == ["ana"]
        stored = voiceprints.voiceprints["ana"]
        assert stored.vector.index(max(stored.vector)) == 3
        assert stored.consent_version == "consent-2"
        consent = voiceprints.consents["ana"]
        assert consent.text_version == "en-1"
        assert consent.given_at == NOW + timedelta(days=1)

    def test_a_rejected_sample_stores_nothing(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        with pytest.raises(SampleRejectedError) as error:
            enrol_use_case(reading(seconds=10.0), voiceprints)(
                "ana", "Ana", "es", SAMPLE, True, "consent-1"
            )
        assert error.value.reason == Rejection.TOO_SHORT.value
        assert voiceprints.voiceprints == {}

    def test_a_rejected_sample_leaves_an_existing_voiceprint(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        enrol_use_case(reading(), voiceprints)(
            "ana", "Ana", "es", SAMPLE, True, "consent-1"
        )
        voiceprint = voiceprints.voiceprints["ana"]
        consent = voiceprints.consents["ana"]
        later = FixedClock(NOW + timedelta(days=1))
        with pytest.raises(SampleRejectedError):
            enrol_use_case(reading(seconds=10.0), voiceprints, later)(
                "ana", "Ana Nueva", "es", SAMPLE, True, "consent-2"
            )
        assert voiceprints.voiceprints == {"ana": voiceprint}
        assert voiceprints.consents == {"ana": consent}

    def test_two_voices_are_rejected(self) -> None:
        scene = Scene(
            lines=[Line("ana", 0.5, 15.0, " ".join(["hola"] * 30)),
                   Line("luis", 15.5, 29.5, " ".join(["vale"] * 30))],
            voices={"ana": voice(0), "luis": voice(1)}, duration=30.0,
        )
        with pytest.raises(SampleRejectedError) as error:
            enrol_use_case(scene, FakeVoiceprintRepository())(
                "ana", "Ana", "es", SAMPLE, True, "consent-1"
            )
        assert error.value.reason == Rejection.SEVERAL_VOICES.value

    @pytest.mark.parametrize("speaker_id", ["S", "Sara", "s1", "S1a"])
    def test_ids_near_the_reserved_shape_are_allowed(
        self, speaker_id: str
    ) -> None:
        voiceprint = enrol_use_case(reading(), FakeVoiceprintRepository())(
            speaker_id, "Ana", "es", SAMPLE, True, "consent-1"
        )
        assert voiceprint.speaker_id == speaker_id

    def test_the_rules_come_from_the_settings(self) -> None:
        rules = SampleRules(min_duration=5.0, min_windows=3,
                            min_window_seconds=5.0)
        voiceprint = enrol_use_case(
            reading(seconds=10.0), FakeVoiceprintRepository(), rules=rules
        )("ana", "Ana", "es", SAMPLE, True, "consent-1")
        assert voiceprint.speaker_id == "ana"

    def test_invalid_requests(self) -> None:
        enrol = enrol_use_case(reading(), FakeVoiceprintRepository())
        with pytest.raises(InvalidInputError, match="speaker id"):
            enrol("../ana", "Ana", "es", SAMPLE, True, "consent-1")
        for reserved in ("S1", "S12", "S0"):
            with pytest.raises(InvalidInputError, match="reserved"):
                enrol(reserved, "Ana", "es", SAMPLE, True, "consent-1")
        with pytest.raises(InvalidInputError, match="consent version"):
            enrol("ana", "Ana", "es", SAMPLE, True, " ")
        with pytest.raises(InvalidInputError, match="name"):
            enrol("ana", "  ", "es", SAMPLE, True, "consent-1")
        with pytest.raises(InvalidInputError, match="fr"):
            enrol("ana", "Ana", "fr", SAMPLE, True, "consent-1")
        naive = enrol_use_case(reading(), FakeVoiceprintRepository(),
                               FixedClock(datetime(2026, 11, 14)))
        with pytest.raises(InvalidInputError, match="aware"):
            naive("ana", "Ana", "es", SAMPLE, True, "consent-1")


class TestForgetSpeaker:
    def test_forget_removes_voiceprint_and_consent(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        voiceprints.enrol("ana", voice(0), name="Ana")
        ForgetSpeaker(voiceprints)("ana")
        assert voiceprints.get("ana") is None
        assert voiceprints.consent("ana") is None
        with pytest.raises(NotFoundError):
            ForgetSpeaker(voiceprints)("ana")

    def test_past_transcripts_are_unchanged(self) -> None:
        voiceprints = FakeVoiceprintRepository()
        voiceprints.enrol("ana", voice(0), name="Ana")
        sessions = FakeSessionRepository()
        words = reading().words()
        utterances = build_utterances(turns_from_words(
            [w.model_copy(update={"speaker": "ana"}) for w in words]
        ))
        transcript = Transcript(
            session=Session(id="s1", language="es", duration=30.0,
                            state=SessionState.DONE,
                            started_at=datetime(2026, 11, 14,
                                                tzinfo=timezone.utc)),
            speakers=[Speaker(id="ana", name="Ana", enrolled=True)],
            utterances=utterances,
            pipeline=Pipeline(recogniser="r", voice_model="v",
                              transcriber="0"),
        )
        sessions.save_transcript(transcript)
        before = transcript.to_contract()
        ForgetSpeaker(voiceprints)("ana")
        after = sessions.transcript("s1")
        assert after is not None
        assert after.to_contract() == before
