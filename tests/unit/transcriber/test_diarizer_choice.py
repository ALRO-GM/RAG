"""Transcription with a lab diarizer (TR-18): rule, voices, fallback."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    FakeDecoder,
    FakeJobRunner,
    FakeOverlapDetector,
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
from transcriber.application.ports import Audio, RecordedAudio
from transcriber.application.record import ImportRecording
from transcriber.application.transcribe import (
    DIARIZER_FALLBACK_NOTE,
    QueueTranscription,
    TranscribeSession,
    TranscriptionOptions,
)
from transcriber.domain.diarization import (
    BUILTIN_RULE,
    DEFAULT_DIARIZER,
    TWO_SPEAKER_DIARIZER,
    Diarization,
    DiarizerRule,
    exclusive_regions,
    label_order,
    label_windows,
    name_labels,
    segment_spans,
    segment_voices,
    spread,
)
from transcriber.domain.errors import LabStepError
from transcriber.domain.lab import Segment
from transcriber.domain.models import SessionState, Transcript, Voiceprint
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.voices import (
    FloatArray,
    Normalisation,
    centre_on_session,
    match_voices,
    unit_vectors,
)


@dataclass
class FakeDiarizer:
    """A diarizer that hears the scene's lines as labelled segments.

    ``problems`` say which models cannot run; ``fail`` makes every call
    raise; ``overlaps`` are reported as the diarizer's own overlaps.
    """

    scene: Scene
    problems: Dict[str, str] = field(default_factory=dict)
    fail: bool = False
    silent: bool = False
    overlaps: List[Tuple[float, float]] = field(default_factory=list)
    calls: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    extra: List[Segment] = field(default_factory=list)

    def problem(self, model: str) -> Optional[str]:
        """Return the configured problem of ``model``."""
        return self.problems.get(model)

    def diarize(
        self, model: str, audio: Audio, num_speakers: Optional[int],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Diarization:
        """Return one segment per line, labelled by person (``spkN``)."""
        self.calls.append((model, num_speakers))
        if self.fail:
            raise RuntimeError("out of memory")
        people = list(dict.fromkeys(line.speaker for line in
                                    self.scene.lines))
        segments = [] if self.silent else [
            Segment(start=line.start, end=line.end,
                    speaker=f"spk{people.index(line.speaker)}")
            for line in self.scene.lines
        ] + list(self.extra)
        return Diarization(
            model=model, licence="CC-BY-NC-4.0",
            commercial=model != DEFAULT_DIARIZER and None,
            segments=segments, overlaps=self.overlaps,
        )


@dataclass
class Setup:
    """The fakes around one transcription."""

    sessions: FakeSessionRepository
    voiceprints: FakeVoiceprintRepository
    diarizer: FakeDiarizer
    transcribe: TranscribeSession
    locks: FakeSessionLocks


def setup(
    scene: Optional[Scene] = None,
    rule: DiarizerRule = DiarizerRule(),
    diarizer: Optional[FakeDiarizer] = None,
    overlaps: Optional[FakeOverlapDetector] = None,
    parameters: TranscriptionParameters = TranscriptionParameters(),
) -> Setup:
    """Wire a transcription that uses ``diarizer`` by ``rule``."""
    scene = scene or meeting()
    locks = FakeSessionLocks()
    sessions = FakeSessionRepository()
    sessions.locks = locks
    voiceprints = FakeVoiceprintRepository()
    diarizer = diarizer or FakeDiarizer(scene)
    transcribe = TranscribeSession(
        sessions, voiceprints, FakeDecoder(scene), FakeRecogniser(scene),
        FakeVoiceEncoder(scene), locks, parameters,
        overlaps, diarizer, rule,
    )
    return Setup(sessions, voiceprints, diarizer, transcribe, locks)


def run(
    s: Setup, options: TranscriptionOptions = TranscriptionOptions(),
) -> Transcript:
    """Import and transcribe session ``s1``."""
    ImportRecording(s.sessions, s.locks)(
        "s1", "es", RecordedAudio(b"audio", "wav")
    )
    runner = FakeJobRunner()
    QueueTranscription(
        s.sessions, s.voiceprints, runner, s.transcribe, s.locks
    )("s1", options)
    runner.run_all()
    if "s1" in runner.errors:
        raise runner.errors["s1"]
    transcript = s.sessions.transcript("s1")
    assert transcript is not None
    return transcript


def turns(transcript: Transcript) -> List[str]:
    """The speaker of each turn."""
    return [u.speaker for u in transcript.utterances if u.index == 0]


class TestDiarizerRule:
    def test_default_is_diarizen_and_nemotron_for_two(self) -> None:
        rule = DiarizerRule()
        assert rule.model_for(None) == DEFAULT_DIARIZER
        assert rule.model_for(4) == DEFAULT_DIARIZER
        assert rule.model_for(2) == TWO_SPEAKER_DIARIZER
        assert rule.models() == [DEFAULT_DIARIZER, TWO_SPEAKER_DIARIZER]

    def test_builtin_rule(self) -> None:
        assert BUILTIN_RULE.model_for(2) == "builtin"
        assert BUILTIN_RULE.models() == ["builtin"]

    def test_rule_from_settings_json(self) -> None:
        rule = DiarizerRule.model_validate(
            {"model": "pyannote-3.1", "by_speakers": {"3": "x"}}
        )
        assert rule.model_for(3) == "x"
        assert rule.model_for(2) == "pyannote-3.1"


class TestSegmentHelpers:
    segments = [
        Segment(start=5.0, end=9.0, speaker="b"),
        Segment(start=0.0, end=6.0, speaker="a"),
    ]

    def test_label_order_by_first_speech(self) -> None:
        assert label_order(self.segments) == {"a": 0, "b": 1}

    def test_spans_are_renamed_and_sorted(self) -> None:
        spans = segment_spans(self.segments, label_order(self.segments))
        assert [(s.speaker, s.start, s.end) for s in spans] == [
            ("SPEAKER_01", 0.0, 6.0), ("SPEAKER_02", 5.0, 9.0),
        ]

    def test_exclusive_regions_leave_the_overlap_out(self) -> None:
        regions = exclusive_regions(self.segments, 10.0)
        assert regions["a"] == [(0.0, 5.0)]
        assert regions["b"] == [(6.0, 9.0)]

    def test_windows_of_each_label(self) -> None:
        order = label_order(self.segments)
        windows, owners = label_windows(
            self.segments, order, 10.0, 1.5, 0.75, 0.4
        )
        assert set(owners) == {0, 1}
        for (start, end), owner in zip(windows, owners):
            if owner == 0:
                assert end <= 5.0
            else:
                assert start >= 6.0

    def test_a_label_never_alone_is_cut_over_its_segments(self) -> None:
        segments = [
            Segment(start=0.0, end=10.0, speaker="a"),
            Segment(start=2.0, end=4.0, speaker="b"),
        ]
        windows, owners = label_windows(
            segments, label_order(segments), 10.0, 1.5, 0.75, 0.4
        )
        assert 1 in owners
        assert all(
            2.0 <= w[0] and w[1] <= 4.0
            for w, o in zip(windows, owners) if o == 1
        )

    def test_spread_keeps_at_most_the_limit(self) -> None:
        windows = [(float(n), float(n) + 1) for n in range(10)]
        picked = spread(windows, 3)
        assert len(picked) == 3
        assert picked[0] == windows[0] and picked[-1] == windows[-1]
        assert spread(windows, 20) == windows


class TestSegmentVoices:
    def test_vectors_and_scores(self) -> None:
        a, b = voice(0), voice(1)
        label_vectors = np.stack([a, a, b]).astype(np.float32)
        fine = np.stack([a, b]).astype(np.float32)
        voices = segment_voices(label_vectors, [0, 0, 1], 2, fine)
        assert voices.vectors.shape == (2, a.shape[0])
        assert np.allclose(np.linalg.norm(voices.vectors, axis=1), 1.0)
        assert voices.fine_scores is not None
        assert list(np.argmax(voices.fine_scores, axis=1)) == [0, 1]

    def test_uncentred_scores(self) -> None:
        a, b = voice(0), voice(1)
        voices = segment_voices(
            np.stack([a, b]).astype(np.float32), [0, 1], 2,
            np.stack([b]).astype(np.float32), centre_windows=False,
        )
        assert voices.fine_scores is not None
        assert int(np.argmax(voices.fine_scores[0])) == 1

    def test_a_label_without_window_is_a_zero_row_never_preferred(
        self,
    ) -> None:
        a, b = voice(0), voice(1)
        voices = segment_voices(
            np.stack([a, b]).astype(np.float32), [0, 2], 3,
            np.stack([a, b]).astype(np.float32),
        )
        assert not voices.vectors[1].any()
        assert voices.fine_scores is not None
        assert (voices.fine_scores[:, 1] == -1.0).all()

    def test_one_voice_gives_no_scores(self) -> None:
        a = voice(0)
        voices = segment_voices(
            np.stack([a]).astype(np.float32), [0], 1,
            np.stack([a]).astype(np.float32),
        )
        assert voices.fine_scores is None

    def test_owner_count_must_match(self) -> None:
        with pytest.raises(ValueError):
            segment_voices(
                np.zeros((2, 3), dtype=np.float32), [0], 1,
                np.zeros((0, 3), dtype=np.float32),
            )

    def test_short_windows_of_another_size(self) -> None:
        with pytest.raises(ValueError):
            segment_voices(
                np.eye(2, 3, dtype=np.float32), [0, 1], 2,
                np.zeros((1, 4), dtype=np.float32),
            )


class TestTranscribeWithDiarizer:
    def test_default_rule_uses_diarizen_and_records_its_licence(
        self,
    ) -> None:
        s = setup()
        transcript = run(s)
        assert s.diarizer.calls == [(DEFAULT_DIARIZER, None)]
        assert turns(transcript) == ["S1", "S2", "S1"]
        pipeline = transcript.pipeline
        assert pipeline.diarizer == DEFAULT_DIARIZER
        assert pipeline.diarizer_licence == "CC-BY-NC-4.0"
        contract = transcript.to_contract()["pipeline"]
        assert contract["diarizer"] == DEFAULT_DIARIZER
        assert contract["diarizer_licence"] == "CC-BY-NC-4.0"
        assert Transcript.from_contract(
            transcript.to_contract()
        ).pipeline == pipeline

    def test_two_speakers_use_nemotron(self) -> None:
        s = setup()
        run(s, TranscriptionOptions(num_speakers=2))
        assert s.diarizer.calls == [(TWO_SPEAKER_DIARIZER, 2)]

    def test_enrolled_people_are_named_from_the_labels(self) -> None:
        s = setup()
        s.voiceprints.enrol("ana", voice(0), name="Ana")
        s.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = run(s)
        assert turns(transcript) == ["ana", "luis", "ana"]
        assert all(sp.enrolled for sp in transcript.speakers)

    def test_unmatched_label_stays_anonymous(self) -> None:
        s = setup()
        s.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = run(s)
        assert turns(transcript) == ["S1", "luis", "S1"]

    def test_sentences_keep_their_certainty(self) -> None:
        transcript = run(setup())
        assert any(
            u.speaker_certainty is not None for u in transcript.utterances
        )

    def test_diarizer_overlaps_flag_sentences(self) -> None:
        scene = meeting()
        s = setup(scene, diarizer=FakeDiarizer(scene, overlaps=[(1.0, 4.0)]))
        transcript = run(s)
        first = transcript.utterances[0]
        assert first.overlap is True
        assert first.overlap_source in ("detector", "both")

    def test_builtin_rule_does_not_call_the_diarizer(self) -> None:
        s = setup(rule=BUILTIN_RULE)
        transcript = run(s)
        assert s.diarizer.calls == []
        assert transcript.pipeline.diarizer == "builtin"
        assert transcript.pipeline.diarizer_licence is None
        assert "diarizer_licence" not in transcript.to_contract()["pipeline"]


class TestFallback:
    def _fallback(self, diarizer: FakeDiarizer) -> Transcript:
        return run(setup(diarizer.scene, diarizer=diarizer))

    def test_missing_environment_falls_back_with_a_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        scene = meeting()
        diarizer = FakeDiarizer(
            scene, problems={DEFAULT_DIARIZER: "not installed"}
        )
        with caplog.at_level("WARNING"):
            transcript = self._fallback(diarizer)
        assert diarizer.calls == []
        assert transcript.pipeline.diarizer == "builtin"
        assert turns(transcript) == ["S1", "S2", "S1"]
        assert "not installed" in caplog.text

    def test_a_failing_diarizer_falls_back(self) -> None:
        diarizer = FakeDiarizer(meeting(), fail=True)
        transcript = self._fallback(diarizer)
        assert diarizer.calls == [(DEFAULT_DIARIZER, None)]
        assert transcript.pipeline.diarizer == "builtin"

    def test_a_diarizer_hearing_no_one_falls_back(self) -> None:
        diarizer = FakeDiarizer(meeting(), silent=True)
        transcript = self._fallback(diarizer)
        assert transcript.pipeline.diarizer == "builtin"

    def test_the_fallback_is_noted_on_the_job(self) -> None:
        scene = meeting()
        s = setup(scene, diarizer=FakeDiarizer(scene, fail=True))
        ImportRecording(s.sessions, s.locks)(
            "s1", "es", RecordedAudio(b"audio", "wav")
        )
        runner = FakeJobRunner()
        QueueTranscription(
            s.sessions, s.voiceprints, runner, s.transcribe, s.locks
        )("s1")
        session = s.sessions.get("s1")
        assert session is not None
        progress = RecordingProgress()
        s.transcribe("s1", progress=progress, run_id=session.run_id)
        expected = DIARIZER_FALLBACK_NOTE.format(
            model=DEFAULT_DIARIZER, reason="out of memory"
        )
        assert expected in progress.notes


def print_of(speaker_id: str, vector: FloatArray) -> Voiceprint:
    """A voiceprint of ``vector`` (unit length) for ``speaker_id``."""
    unit = vector / np.linalg.norm(vector)
    return Voiceprint(
        speaker_id=speaker_id, model="fake-voice",
        vector=[float(x) for x in unit], consent_version="consent-1",
        enrolled_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestNameLabels:
    """A label with no voice window is never matched (review of #68)."""

    a, b = voice(0), voice(1)
    rows = np.stack([a, b, np.zeros_like(a)]).astype(np.float32)
    # Centred on the session, the zero row becomes -mean: a voiceprint
    # along -(a + b) then looks exactly like it.
    carla = print_of("carla", -(a + b))

    def test_the_hazard_is_real_with_centring(self) -> None:
        matched = match_voices(self.rows, [self.carla], 0.95,
                               centre_on_session)
        assert matched[2].speaker == "carla"

    @pytest.mark.parametrize("normalisation", [
        unit_vectors, centre_on_session,
    ])
    def test_a_label_without_window_stays_anonymous(
        self, normalisation: Normalisation,
    ) -> None:
        names = name_labels(
            self.rows, [True, True, False],
            [self.carla, print_of("ana", self.a)], 0.95, normalisation,
        )
        assert [n.group for n in names] == [0, 1, 2]
        assert not names[2].enrolled and names[2].speaker.startswith("S")
        assert all(n.speaker != "carla" for n in names)

    def test_heard_labels_are_matched_and_anonymous_ones_numbered(
        self,
    ) -> None:
        names = name_labels(
            self.rows, [True, True, False], [print_of("ana", self.a)],
            0.95, taken={"S1"},
        )
        assert [n.speaker for n in names] == ["ana", "S2", "S3"]
        assert names[0].similarity is not None
        assert names[1].nearest == "ana"

    def test_nobody_to_match(self) -> None:
        names = name_labels(self.rows, [False, False, False], [], 0.95)
        assert [n.speaker for n in names] == ["S1", "S2", "S3"]

    def test_one_flag_per_row(self) -> None:
        with pytest.raises(ValueError):
            name_labels(self.rows, [True], [], 0.95)

    def test_segment_voices_lists_the_heard_labels(self) -> None:
        voices = segment_voices(
            np.stack([self.a, self.b]).astype(np.float32), [0, 2], 3,
            np.stack([self.a]).astype(np.float32),
        )
        assert voices.heard == (0, 2)

    def test_the_transcription_leaves_a_windowless_label_anonymous(
        self,
    ) -> None:
        # Eva says one word of 0.3 s: too short for any voice window, so
        # her label has a zero vector. Centred on the session (three
        # labels), that zero row would look like a voiceprint along
        # -(Ana + Luis); before the fix it took that person's name.
        scene = Scene(
            lines=[
                Line("ana", 0.0, 12.0, "Hola a todos. Hoy repartimos las"
                     " tareas del proyecto nuevo y lo revisamos."),
                Line("eva", 12.2, 12.5, "Sí."),
                Line("luis", 13.0, 25.0, "Vale. Yo me encargo de la"
                     " interfaz y de las pruebas del sistema."),
            ],
            voices={"ana": voice(0), "eva": voice(2), "luis": voice(1)},
        )
        s = setup(scene, parameters=TranscriptionParameters(
            normalisation="centre_on_session"))
        s.voiceprints.enrol("carla", -(voice(0) + voice(1)), name="Carla")
        transcript = run(s)
        said = {u.text: u.speaker for u in transcript.utterances}
        assert said["Sí."].startswith("S")
        assert "carla" not in {sp.id for sp in transcript.speakers}


class TestFloorAndJoin:
    def test_the_floor_names_only_listed_people(self) -> None:
        # Ana's voiceprint is ~0.88 from her voice: below the match
        # threshold (0.95), above the expected floor (0.8).
        off = voice(0).copy()
        off[5] = 0.56
        cosine = float(off @ voice(0) / np.linalg.norm(off)
                       / np.linalg.norm(voice(0)))
        p = TranscriptionParameters()
        assert p.expected_floor < cosine < p.match_threshold
        listed = setup()
        listed.voiceprints.enrol("ana", off, name="Ana")
        listed.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = run(listed, TranscriptionOptions(
            num_speakers=2, expected_speakers=["ana", "luis"]
        ))
        assert turns(transcript) == ["ana", "luis", "ana"]
        ana = next(s for s in transcript.speakers if s.id == "ana")
        assert ana.identification == "expected_floor"
        open_session = setup()
        open_session.voiceprints.enrol("ana", off, name="Ana")
        open_session.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = run(open_session, TranscriptionOptions(num_speakers=2))
        assert turns(transcript) == ["S1", "luis", "S1"]

    def test_diarizer_and_detector_overlaps_are_joined(self) -> None:
        scene = meeting()
        s = setup(
            scene, diarizer=FakeDiarizer(scene, overlaps=[(1.0, 4.0)]),
            overlaps=FakeOverlapDetector([(27.0, 30.0)]),
        )
        transcript = run(s)

        def source(at: float) -> Optional[str]:
            sentence = next(u for u in transcript.utterances
                            if u.start <= at <= u.end)
            assert sentence.overlap is True
            return sentence.overlap_source

        assert source(2.0) in ("detector", "both")  # the diarizer's
        assert source(28.0) in ("detector", "both")  # the detector's


@dataclass
class CancellingDiarizer(FakeDiarizer):
    """A diarizer that lets the test act mid-run, then polls cancelled."""

    during: Callable[[], None] = lambda: None
    saw: List[bool] = field(default_factory=list)

    def diarize(
        self, model: str, audio: Audio, num_speakers: Optional[int],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> Diarization:
        """Run ``during``, record ``cancelled()``, stop if it is true."""
        self.saw.append(cancelled())
        self.during()
        self.saw.append(cancelled())
        if self.saw[-1]:
            raise LabStepError("cancelled", "The run was cancelled.")
        return super().diarize(model, audio, num_speakers, cancelled)


class TestCancellation:
    def _queued(self, s: Setup) -> str:
        ImportRecording(s.sessions, s.locks)(
            "s1", "es", RecordedAudio(b"audio", "wav")
        )
        QueueTranscription(
            s.sessions, s.voiceprints, FakeJobRunner(), s.transcribe,
            s.locks,
        )("s1")
        session = s.sessions.get("s1")
        assert session is not None and session.run_id is not None
        return session.run_id

    def test_a_session_queued_again_cancels_the_diarizer(self) -> None:
        scene = meeting()
        diarizer = CancellingDiarizer(scene)
        s = setup(scene, diarizer=diarizer)
        run_id = self._queued(s)

        def queue_again() -> None:
            session = s.sessions.get("s1")
            assert session is not None
            with s.locks.hold("s1"):
                s.sessions.save(session.model_copy(update={
                    "run_id": "another", "state": SessionState.QUEUED,
                }))

        diarizer.during = queue_again
        progress = RecordingProgress()
        assert s.transcribe("s1", progress=progress, run_id=run_id) is None
        assert diarizer.saw == [False, True]
        # stopped, not fallen back to the built-in grouping
        assert not any("built-in" in n for n in progress.notes)

    def test_a_deleted_session_cancels_the_diarizer(self) -> None:
        scene = meeting()
        diarizer = CancellingDiarizer(scene)
        s = setup(scene, diarizer=diarizer)
        run_id = self._queued(s)

        def delete() -> None:
            with s.locks.hold("s1"):
                s.sessions.delete("s1")

        diarizer.during = delete
        assert s.transcribe("s1", run_id=run_id) is None
        assert diarizer.saw == [False, True]
        assert s.sessions.get("s1") is None

    def test_a_stopping_service_fails_the_job_without_fallback(
        self,
    ) -> None:
        scene = meeting()
        diarizer = CancellingDiarizer(scene)
        s = setup(scene, diarizer=diarizer)
        run_id = self._queued(s)

        def stop() -> None:
            raise LabStepError("cancelled", "The service is stopping.")

        diarizer.during = stop
        with pytest.raises(LabStepError):
            s.transcribe("s1", run_id=run_id)
        session = s.sessions.get("s1")
        assert session is not None and session.state is SessionState.FAILED
