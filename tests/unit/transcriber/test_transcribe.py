"""Transcriber transcription use cases, with a scripted meeting."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    FakeDecoder,
    FakeJobRunner,
    FakeRecogniser,
    FakeSessionLocks,
    FakeSessionRepository,
    FakeVoiceEncoder,
    FakeVoiceprintRepository,
    Line,
    RecordingProgress,
    Scene,
    meeting,
    voice,
)
from transcriber import __version__
from transcriber.application.ports import Audio, RecordedAudio
from transcriber.application.record import ImportRecording
from transcriber.application.transcribe import (
    STAGES,
    QueueTranscription,
    TranscribeSession,
    TranscriptionOptions,
    TranscriptionStatus,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StateError,
    VoiceModelMismatchError,
)
from transcriber.domain.models import SessionState, Transcript
from transcriber.domain.parameters import TranscriptionParameters


@dataclass
class World:
    """The fakes and the use cases wired together."""

    scene: Scene
    sessions: FakeSessionRepository
    voiceprints: FakeVoiceprintRepository
    decoder: FakeDecoder
    recogniser: FakeRecogniser
    encoder: FakeVoiceEncoder
    runner: FakeJobRunner
    transcribe: TranscribeSession
    queue: QueueTranscription
    status: TranscriptionStatus
    locks: FakeSessionLocks

    def upload(self, session_id: str = "s1") -> str:
        """Import a recording as an uploaded session."""
        ImportRecording(self.sessions, self.locks)(
            session_id, "es", RecordedAudio(b"audio", "wav")
        )
        return session_id

    def run(
        self, options: TranscriptionOptions = TranscriptionOptions()
    ) -> Transcript:
        """Upload, queue and run session ``s1``; return its transcript."""
        self.upload()
        self.queue("s1", options)
        self.runner.run_all()
        if "s1" in self.runner.errors:
            raise self.runner.errors["s1"]
        transcript = self.sessions.transcript("s1")
        assert transcript is not None
        return transcript


def world(
    scene: Optional[Scene] = None,
    parameters: TranscriptionParameters = TranscriptionParameters(),
    sessions: Optional[FakeSessionRepository] = None,
    voiceprints: Optional[FakeVoiceprintRepository] = None,
) -> World:
    """Fresh fakes around ``scene`` (the default meeting).

    ``sessions`` and ``voiceprints`` replace the empty stores (their
    writes are then checked against the world's locks).
    """
    scene = scene or meeting()
    locks = FakeSessionLocks()
    sessions = sessions or FakeSessionRepository()
    sessions.locks = locks
    voiceprints = voiceprints or FakeVoiceprintRepository()
    decoder = FakeDecoder(scene)
    recogniser = FakeRecogniser(scene)
    encoder = FakeVoiceEncoder(scene)
    runner = FakeJobRunner()
    transcribe = TranscribeSession(sessions, voiceprints, decoder,
                                   recogniser, encoder, locks, parameters)
    return World(
        scene, sessions, voiceprints, decoder, recogniser, encoder, runner,
        transcribe, QueueTranscription(sessions, voiceprints, runner,
                                       transcribe, locks),
        TranscriptionStatus(sessions, runner), locks,
    )


def monologue(seconds: float = 60.0) -> Scene:
    """Ana alone for ``seconds``: four-second sentences, short pauses."""
    lines = []
    start = 0.0
    while start + 4.0 <= seconds:
        lines.append(Line("ana", start, start + 4.0,
                          "Hoy os cuento como va el proyecto nuevo."))
        start += 4.6
    return Scene(lines=lines, voices={"ana": voice(0)})


def turns(transcript: Transcript) -> List[Tuple[str, Optional[str]]]:
    """``(speaker, previous speaker)`` of each turn's first sentence."""
    return [(u.speaker, u.previous_speaker)
            for u in transcript.utterances if u.index == 0]


class TestTranscribeSession:
    def test_no_one_enrolled_gives_anonymous_labels(self) -> None:
        w = world()
        transcript = w.run()
        assert turns(transcript) == [
            ("S1", None), ("S2", "S1"), ("S1", "S2"),
        ]
        assert [(s.id, s.name, s.enrolled) for s in transcript.speakers] == [
            ("S1", None, False), ("S2", None, False),
        ]
        assert {u.speaker_confidence for u in transcript.utterances} == {
            None
        }

    def test_some_enrolled_gives_ids_and_anonymous_labels(self) -> None:
        w = world()
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        w.voiceprints.enrol("eva", voice(4), name="Eva")
        transcript = w.run()
        assert turns(transcript) == [
            ("S1", None), ("luis", "S1"), ("S1", "luis"),
        ]
        assert [(s.id, s.name, s.enrolled) for s in transcript.speakers] == [
            ("S1", None, False), ("luis", "Luis", True),
        ]
        luis = [u for u in transcript.utterances if u.speaker == "luis"]
        assert all(u.speaker_confidence is not None
                   and u.speaker_confidence > 0.9 for u in luis)

    def test_everyone_enrolled(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = w.run()
        assert turns(transcript) == [
            ("ana", None), ("luis", "ana"), ("ana", "luis"),
        ]

    def test_the_transcript_is_complete(self) -> None:
        w = world()
        transcript = w.run()
        session = transcript.session
        assert session.state is SessionState.DONE
        assert session.duration == pytest.approx(37.0)
        assert session.audio_sha256 is not None
        assert transcript.pipeline.recogniser == "fake-whisper"
        assert transcript.pipeline.voice_model == "fake-voice"
        assert transcript.pipeline.transcriber == __version__
        texts = [u.text for u in transcript.utterances]
        assert texts[:2] == [
            "Hola a todos.",
            "Hoy repartimos las tareas del proyecto nuevo.",
        ]
        assert transcript.utterances[0].start == 0.0
        assert w.sessions.sessions["s1"].state is SessionState.DONE
        assert w.recogniser.languages == ["es"]
        assert transcript.to_contract()["schema"] == "transcript.v1"

    def test_words_are_saved_with_their_speakers(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0))
        w.run()
        words = w.sessions.words("s1")
        assert len(words) == len(w.scene.words())
        assert words[0].speaker == "ana"
        assert {wd.speaker for wd in words} == {"ana", "S1"}

    def test_stages_are_reported_in_order(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        progress = RecordingProgress()
        w.runner.pending.clear()
        w.transcribe("s1", progress=progress)
        assert progress.reports == list(STAGES) + [("done", 1.0)]

    def test_progress_is_optional(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        w.runner.pending.clear()
        transcript = w.transcribe("s1")
        assert transcript is not None
        assert transcript.session.state is SessionState.DONE

    def test_the_runner_status_follows_the_job(self) -> None:
        w = world()
        w.upload()
        assert w.status("s1").state is SessionState.UPLOADED
        w.queue("s1")
        status = w.status("s1")
        assert (status.state, status.fraction) == (SessionState.QUEUED, 0.0)
        w.runner.run_all()
        done = w.status("s1")
        assert (done.state, done.stage, done.fraction) == (
            SessionState.DONE, "done", 1.0,
        )

    def test_status_while_running(self) -> None:
        w = world()
        w.upload()
        w.queue("s1")
        seen = []
        original = w.recogniser.recognise

        def spy(audio, language=None):  # type: ignore[no-untyped-def]
            seen.append(w.status("s1"))
            return original(audio, language)

        w.recogniser.recognise = spy  # type: ignore[method-assign]
        w.runner.run_all()
        assert seen[0].state is SessionState.RUNNING
        assert seen[0].stage == "recognise"

    def test_num_speakers_is_passed_to_the_grouping(self) -> None:
        w = world()
        transcript = w.run(TranscriptionOptions(num_speakers=1))
        assert turns(transcript) == [("S1", None)]

    def test_expected_speakers_restrict_the_voiceprints(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0))
        w.voiceprints.enrol("luis", voice(1))
        transcript = w.run(TranscriptionOptions(expected_speakers=["luis"]))
        assert turns(transcript) == [
            ("S1", None), ("luis", "S1"), ("S1", "luis"),
        ]

    def test_an_unknown_expected_speaker_is_refused_at_queue_time(
        self,
    ) -> None:
        w = world()
        w.upload()
        with pytest.raises(NotFoundError, match="nadie"):
            w.queue("s1", TranscriptionOptions(expected_speakers=["nadie"]))
        assert w.sessions.sessions["s1"].state is SessionState.UPLOADED
        assert w.runner.pending == []

    def test_voiceprints_of_another_model_are_not_compared(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), model="other-model")
        transcript = w.run()
        assert [s.id for s in transcript.speakers] == ["S1", "S2"]

    def test_thresholds_come_from_the_parameters(self) -> None:
        near_ana = voice(0).copy()
        near_ana[2] = 0.3  # cosine with Ana's voice about 0.96
        lenient = world()
        lenient.voiceprints.enrol("ana", near_ana)
        assert [s.id for s in lenient.run().speakers] == ["ana", "S1"]
        strict = world(parameters=TranscriptionParameters(
            match_threshold=0.99
        ))
        strict.voiceprints.enrol("ana", near_ana)
        assert [s.id for s in strict.run().speakers] == ["S1", "S2"]
        merged = world(parameters=TranscriptionParameters(
            group_threshold=-1.0
        ))
        assert [s.id for s in merged.run().speakers] == ["S1"]

    def test_without_voice_windows_one_anonymous_speaker(self) -> None:
        """Words too short to embed still give a transcript."""
        scene = Scene(lines=[Line("ana", 0.0, 0.3, "Sí.")],
                      voices={"ana": voice(0)}, duration=2.0)
        w = world(scene)
        w.voiceprints.enrol("ana", voice(0))
        transcript = w.run()
        assert turns(transcript) == [("S1", None)]
        assert w.encoder.calls == []

    def test_only_long_windows_still_attribute(self) -> None:
        params = TranscriptionParameters(fine_window=0.5, fine_step=0.25,
                                         window_minimum=0.5)
        scene = Scene(lines=[Line("ana", 0.0, 0.45, "Sí.")],
                      voices={"ana": voice(0)}, duration=1.0)
        w = world(scene, params)
        assert turns(w.run()) == [("S1", None)]

    def test_groups_are_numbered_by_first_appearance_in_words(self) -> None:
        """Luis speaks first: he is S1 even if Ana's windows come first."""
        scene = meeting()
        scene.lines = [
            Line("luis", 0.0, 6.0, "Empiezo yo hoy con la base."),
            Line("ana", 7.0, 13.0, "Perfecto, yo sigo con la interfaz."),
        ]
        w = world(scene)
        transcript = w.run()
        assert turns(transcript) == [("S1", None), ("S2", "S1")]
        words = w.sessions.words("s1")
        assert words[0].speaker == "S1"


class TestSingleSpeaker:
    def test_a_monologue_is_one_anonymous_speaker(self) -> None:
        w = world(monologue())
        transcript = w.run()
        assert [s.id for s in transcript.speakers] == ["S1"]
        assert turns(transcript) == [("S1", None)]
        assert {wd.speaker for wd in w.sessions.words("s1")} == {"S1"}

    def test_a_monologue_of_an_enrolled_person(self) -> None:
        w = world(monologue())
        w.voiceprints.enrol("ana", voice(0), name="Ana")
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = w.run()
        assert [(s.id, s.enrolled) for s in transcript.speakers] == [
            ("ana", True),
        ]
        assert turns(transcript) == [("ana", None)]

    def test_the_threshold_comes_from_the_parameters(self) -> None:
        """At a threshold no window reaches, the old split comes back."""
        w = world(monologue(), TranscriptionParameters(
            single_voice_threshold=1.0
        ))
        assert len(w.run().speakers) > 1


class TestOrdering:
    def test_first_heard_is_s1_even_if_clustered_second(self) -> None:
        """Luis opens, but the first long window is Ana's voice.

        Luis's opening second only fits in the first long window, whose
        middle is Ana's, so clustering numbers Ana's group first; the
        short windows still give the opening words to Luis's group.
        """
        scene = Scene(
            lines=[
                Line("luis", 0.0, 1.0, "Buenos días."),
                Line("ana", 1.0, 12.0,
                     "Hola a todos. Hoy repartimos las tareas del"
                     " proyecto nuevo."),
                Line("luis", 13.0, 25.0,
                     "Vale. Yo me encargo de la interfaz y de las"
                     " pruebas."),
            ],
            voices={"ana": voice(0), "luis": voice(1)},
        )
        w = world(scene)
        transcript = w.run()
        words = w.sessions.words("s1")
        assert (words[0].text, words[0].speaker) == ("Buenos", "S1")
        assert turns(transcript) == [
            ("S1", None), ("S2", "S1"), ("S1", "S2"),
        ]


class TestSilence:
    def test_words_invented_over_silence_are_dropped(self) -> None:
        w = world()
        w.decoder.silent.append((13.0, 25.0))  # Luis's line is silent
        transcript = w.run()
        assert "Necesito" not in " ".join(
            u.text for u in transcript.utterances
        )
        assert [s.id for s in transcript.speakers] == ["S1"]
        heard = w.sessions.words("s1")
        assert heard and all(not 13.0 <= wd.start < 25.0 for wd in heard)

    def test_a_silent_recording_fails_with_no_speech(self) -> None:
        w = world()
        w.decoder.silent.append((0.0, w.scene.duration))
        with pytest.raises(InvalidInputError, match="No speech"):
            w.run()
        session = w.sessions.sessions["s1"]
        assert (session.state, session.failure) == (
            SessionState.FAILED, "recognise: No speech recognised.",
        )
        assert w.sessions.transcript("s1") is None

    def test_the_level_comes_from_the_parameters(self) -> None:
        w = world(parameters=TranscriptionParameters(activity_threshold=0.5))
        with pytest.raises(InvalidInputError, match="No speech"):
            w.run()


class TestVoiceModels:
    def test_an_expected_speaker_of_another_model_is_refused(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), model="other-model")
        w.upload()
        with pytest.raises(VoiceModelMismatchError, match="ana"):
            w.queue("s1", TranscriptionOptions(expected_speakers=["ana"]))
        assert w.sessions.sessions["s1"].state is SessionState.UPLOADED
        assert w.runner.pending == []

    def test_skipped_voiceprints_are_reported(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), model="other-model")
        w.voiceprints.enrol("eva", voice(4), model="old-model")
        w.voiceprints.enrol("luis", voice(1))
        w.upload()
        w.queue("s1")
        w.runner.pending.clear()
        progress = RecordingProgress()
        transcript = w.transcribe("s1", progress=progress)
        assert transcript is not None
        assert [s.id for s in transcript.speakers] == ["S1", "luis"]
        assert len(progress.notes) == 1
        assert progress.notes[0].startswith("2 enrolled voiceprint(s)")

    def test_the_skipped_count_reaches_the_job_status(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0), model="other-model")
        w.run()
        message = w.status("s1").message
        assert message is not None and message.startswith("1 enrolled")

    def test_nothing_is_reported_when_nothing_is_skipped(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0))
        w.run()
        assert w.status("s1").message is None


class TestReservedLabels:
    def test_an_enrolled_s1_never_labels_an_anonymous_voice(self) -> None:
        w = world()
        w.voiceprints.enrol("S1", voice(5))  # stored before the rule
        w.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = w.run(TranscriptionOptions(expected_speakers=["luis"]))
        assert turns(transcript) == [
            ("S2", None), ("luis", "S2"), ("S2", "luis"),
        ]

    def test_without_voice_windows_the_label_skips_enrolled_ids(
        self,
    ) -> None:
        scene = Scene(lines=[Line("ana", 0.0, 0.3, "Sí.")],
                      voices={"ana": voice(0)}, duration=2.0)
        w = world(scene)
        w.voiceprints.enrol("S1", voice(0), model="other-model")
        assert turns(w.run()) == [("S2", None)]


class TestFailures:
    def test_a_failing_stage_leaves_the_session_failed(self) -> None:
        w = world()
        w.recogniser.failures.append(RuntimeError("CUDA out of memory"))
        w.upload()
        w.queue("s1")
        w.runner.run_all()
        session = w.sessions.sessions["s1"]
        assert session.state is SessionState.FAILED
        assert session.failure == "recognise: CUDA out of memory"
        assert isinstance(w.runner.errors["s1"], RuntimeError)
        assert w.sessions.transcript("s1") is None
        assert w.status("s1").state is SessionState.FAILED

    def test_a_retry_succeeds(self) -> None:
        w = world()
        w.decoder.failures.append(ValueError())
        w.upload()
        w.queue("s1")
        w.runner.run_all()
        failed = w.sessions.sessions["s1"]
        assert failed.failure == "decode: ValueError"
        w.queue("s1")
        assert w.sessions.sessions["s1"].failure is None
        w.runner.run_all()
        session = w.sessions.sessions["s1"]
        assert session.state is SessionState.DONE
        assert session.failure is None
        assert w.sessions.transcript("s1") is not None

    def test_no_speech_fails_the_session(self) -> None:
        w = world(Scene(lines=[Line("ana", 0.0, 1.0, " ")],
                        voices={"ana": voice(0)}))
        with pytest.raises(InvalidInputError, match="No speech"):
            w.run()
        assert w.sessions.sessions["s1"].failure == (
            "recognise: No speech recognised."
        )

    def test_a_forgotten_expected_speaker_fails_the_job(self) -> None:
        w = world()
        w.voiceprints.enrol("ana", voice(0))
        w.upload()
        w.queue("s1", TranscriptionOptions(expected_speakers=["ana"]))
        w.voiceprints.delete("ana")
        w.runner.run_all()
        assert w.sessions.sessions["s1"].failure == (
            "identify: No enrolled speaker ana."
        )

    def test_a_refused_job_fails_the_session(self) -> None:
        w = world()
        w.runner.refuse = RuntimeError("queue full")
        w.upload()
        with pytest.raises(RuntimeError):
            w.queue("s1")
        session = w.sessions.sessions["s1"]
        assert (session.state, session.failure) == (
            SessionState.FAILED, "queue: queue full",
        )

    def test_only_queued_sessions_are_transcribed(self) -> None:
        w = world()
        w.upload()
        with pytest.raises(StateError, match="queued"):
            w.transcribe("s1")
        assert w.sessions.sessions["s1"].state is SessionState.UPLOADED
        with pytest.raises(NotFoundError):
            w.transcribe("nope")

    @pytest.mark.parametrize("state", [
        SessionState.RECORDING, SessionState.QUEUED, SessionState.RUNNING,
    ])
    def test_only_uploaded_failed_or_done_sessions_are_queued(
        self, state: SessionState
    ) -> None:
        w = world()
        w.upload()
        w.sessions.save(w.sessions.sessions["s1"].model_copy(
            update={"state": state}
        ))
        with pytest.raises(StateError):
            w.queue("s1")

    def test_options_are_validated(self) -> None:
        with pytest.raises(ValueError):
            TranscriptionOptions(num_speakers=0)


def test_group_vectors_are_compared_raw() -> None:
    """The voiceprint of a person matches their group with cosine ~1."""
    w = world()
    w.voiceprints.enrol("ana", voice(0))
    transcript = w.run()
    ana = next(u for u in transcript.utterances if u.speaker == "ana")
    assert ana.speaker_confidence is not None
    assert ana.speaker_confidence > 0.99


def test_audio_cut_is_clamped_to_the_recording() -> None:
    audio = Audio(samples=np.arange(10, dtype=np.float32), sample_rate=2)
    assert audio.duration == 5.0
    assert audio.cut(1.0, 2.0).tolist() == [2.0, 3.0]
    assert audio.cut(-1.0, 0.5).tolist() == [0.0]
    assert audio.cut(4.0, 9.0).tolist() == [8.0, 9.0]
    assert audio.cut(3.0, 1.0).tolist() == []
