"""Transcription with a lab diarizer (TR-18): rule, voices, fallback."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    segment_spans,
    segment_voices,
    spread,
)
from transcriber.domain.lab import Segment
from transcriber.domain.models import Transcript
from transcriber.domain.parameters import TranscriptionParameters


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

    def problem(self, model: str) -> Optional[str]:
        """Return the configured problem of ``model``."""
        return self.problems.get(model)

    def diarize(
        self, model: str, audio: Audio, num_speakers: Optional[int]
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
        ]
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
        FakeVoiceEncoder(scene), locks, TranscriptionParameters(),
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

    def test_expected_speakers_take_the_floor(self) -> None:
        s = setup()
        s.voiceprints.enrol("ana", voice(0), name="Ana")
        s.voiceprints.enrol("luis", voice(1), name="Luis")
        transcript = run(s, TranscriptionOptions(
            num_speakers=2, expected_speakers=["ana", "luis"]
        ))
        assert turns(transcript) == ["ana", "luis", "ana"]

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

    def test_diarizer_and_detector_overlaps_are_joined(self) -> None:
        scene = meeting()
        s = setup(
            scene, diarizer=FakeDiarizer(scene, overlaps=[(1.0, 4.0)]),
            overlaps=FakeOverlapDetector(),
        )
        transcript = run(s)
        assert transcript.utterances[0].overlap is True

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
