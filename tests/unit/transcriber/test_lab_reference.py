"""Reference corpora: parsing, scoring by hand, import and summary."""

import dataclasses
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pytest

from tests.unit.transcriber.fakes import (
    FakeSessionLocks,
    FakeSessionRepository,
    FixedClock,
)
from tests.unit.transcriber.lab_fakes import FakeLabRepository
from tests.unit.transcriber.test_lab_use_cases import SESSION, TURNS, Lab
from transcriber.application.lab_ports import CorpusRecording
from tests.unit.transcriber.test_speech_check import band_noise, voiced
from transcriber.application.lab_reference import (
    CheckReferenceSpeech,
    ImportedRecording,
    INVALID_REFERENCE,
    ImportReference,
    ReferenceSummary,
)
from transcriber.application.ports import Audio, RecordedAudio
from transcriber.application.record import ImportRecording
from transcriber.domain.comparison import (
    ComparisonParameters,
    Track,
    compare_transcripts,
    jaccard_error,
)
from transcriber.domain.errors import InvalidInputError, StorageError
from transcriber.domain.lab import LabRun, Segment
from transcriber.domain.lab_reference import (
    CORPORA,
    REFERENCE_TRACK,
    SessionReference,
    check_split,
    corpus_spec,
    make_reference,
    recording_name,
    recording_session_id,
    reference_segments,
    score_speakers,
    score_track,
    spread,
)
from transcriber.domain.models import Session, SessionState, Word
from transcriber.domain.speech_check import NO_SPEECH, SpeechCheck

NO_COLLAR = ComparisonParameters(collar=0.0)
AMI = CORPORA["ami"]
CALLHOME = CORPORA["callhome"]


def seg(start: float, end: float, speaker: str) -> Segment:
    """A segment."""
    return Segment(start=start, end=end, speaker=speaker)


def ten_seconds(words: Optional[List[Word]] = None) -> SessionReference:
    """The 10-second example: A 0-4, B 3-8 (1 s of overlap), then silence.

    Speaker time 9 s (A 4, B 5), speech 8 s, overlap 1 s.
    """
    return make_reference(
        AMI, "test", "EX0001a", 10.0,
        [seg(0.0, 4.0, "A"), seg(3.0, 8.0, "B")], words,
    )


HYPOTHESIS = Track("none+z", [
    seg(0.0, 3.8, "~x"), seg(3.5, 6.0, "~y"), seg(6.0, 7.5, "~z"),
    seg(7.5, 9.0, "~y"),
], "z")
"""Three anonymous voices: ~x is A, ~y mostly B, ~z takes 1.5 s of B.

Frame by frame (no collar), with ~x→A and ~y→B: 0-3 right; 3-3.5 B
missed (0.5 s); 3.5-3.8 both right; 3.8-4 A missed (0.2 s); 4-6 right;
6-7.5 ~z for B: confusion 1.5 s; 7.5-8 right; 8-9 ~y in silence: false
alarm 1.0 s. DER = (0.7 + 1.0 + 1.5) / 9.
"""


class TestCorpora:
    def test_known_corpora_and_splits(self) -> None:
        assert corpus_spec("ami").licence_commercial is True
        assert CALLHOME.gated and not CALLHOME.licence_commercial
        assert check_split(AMI, None) == "test"
        assert check_split(CALLHOME, None) == "data"
        with pytest.raises(InvalidInputError, match="Unknown corpus"):
            corpus_spec("rtve")
        with pytest.raises(InvalidInputError, match="no split"):
            check_split(AMI, "data")

    def test_names(self) -> None:
        assert recording_name("ES2004a.Array1-01.wav", "x") == "ES2004a"
        assert recording_name("dir/sub/EN2002b.wav", "x") == "EN2002b"
        assert recording_name(None, "test-0001") == "test-0001"
        assert recording_session_id("ami", "ES2004a") == "ami-ES2004a"
        assert recording_session_id("ami", "a b/c") == "ami-a-b-c"
        assert len(recording_session_id("ami", "x" * 90)) == 64
        with pytest.raises(InvalidInputError):
            recording_session_id("", "///")


class TestReferenceParsing:
    def test_turns_kept_with_overlap_cut_and_sorted(self) -> None:
        segments = reference_segments(
            [3.0, 0.0, 9.5, 11.0, 2.0], [8.0, 4.0, 10.5, 12.0, 2.0],
            ["B", "A", "C", "D", "E"], 10.0,
        )
        assert [(s.start, s.end, s.speaker) for s in segments] == [
            (0.0, 4.0, "A"), (3.0, 8.0, "B"), (9.5, 10.0, "C"),
        ]

    @pytest.mark.parametrize("starts,ends,speakers", [
        ([0.0], [1.0, 2.0], ["A"]),
        ([2.0], [1.0], ["A"]),
        ([float("nan")], [1.0], ["A"]),
        ([-1.0], [1.0], ["A"]),
        ([0.0], [1.0], [" "]),
    ])
    def test_bad_columns(
        self, starts: List[float], ends: List[float], speakers: List[str]
    ) -> None:
        with pytest.raises(InvalidInputError):
            reference_segments(starts, ends, speakers, 10.0)

    def test_reference_record(self) -> None:
        words = [Word(text="hi", start=1.0, end=1.2),
                 Word(text="late", start=10.5, end=11.0)]
        reference = ten_seconds(words)
        assert reference.speakers() == ["A", "B"]
        assert reference.licence == "CC BY 4.0"
        assert reference.source_url.startswith("https://huggingface.co/")
        assert [w.text for w in reference.words or []] == ["hi"]
        assert reference.track().id == REFERENCE_TRACK
        with pytest.raises(InvalidInputError, match="no speaker turn"):
            make_reference(AMI, "test", "x", 10.0, [], None)


class TestScoringByHand:
    def test_der_jer_overlap_and_count_without_collar(self) -> None:
        score = score_track(ten_seconds(), HYPOTHESIS, NO_COLLAR)
        assert score.reference_speech == pytest.approx(9.0)
        assert score.missed == pytest.approx(0.7)
        assert score.false_alarm == pytest.approx(1.0)
        assert score.confusion == pytest.approx(1.5)
        assert score.der == pytest.approx(3.2 / 9, abs=1e-4)
        # A~x: 3.8 / 4 shared; B~y: 3.0 shared over 6.0 united
        assert score.jer == pytest.approx((0.05 + 0.5) / 2, abs=1e-4)
        assert score.overlap_reference_seconds == pytest.approx(1.0)
        assert score.overlap_hypothesis_seconds == pytest.approx(0.3)
        assert score.overlap_precision == pytest.approx(1.0)
        assert score.overlap_recall == pytest.approx(0.3)
        assert (score.reference_speakers, score.hypothesis_speakers,
                score.speaker_count_error) == (2, 3, 1)

    def test_the_lab_collar_leaves_out_the_reference_boundaries(
        self,
    ) -> None:
        # boundaries 0, 3, 4, 8 ± 0.25 s: scored speaker time A 3.0 +
        # B 4.0; missed 3.25-3.5 only, false alarm 8.25-9, confusion
        # untouched (6-7.5).
        score = score_track(ten_seconds(), HYPOTHESIS)
        assert score.reference_speech == pytest.approx(7.0)
        assert score.missed == pytest.approx(0.25)
        assert score.false_alarm == pytest.approx(0.75)
        assert score.confusion == pytest.approx(1.5)
        assert score.der == pytest.approx(2.5 / 7, abs=1e-4)
        # A~x: 3.0 of 3.0 scored frames shared; B~y: 2.25 over 4.75
        assert score.jer == pytest.approx((0 + 1 - 2.25 / 4.75) / 2,
                                          abs=1e-4)

    def test_perfect_and_empty_tracks(self) -> None:
        reference = ten_seconds()
        same = Track("none+same", [seg(0.0, 4.0, "~p"),
                                   seg(3.0, 8.0, "~q")], "same")
        perfect = score_track(reference, same, NO_COLLAR)
        assert perfect.der == 0.0 and perfect.jer == 0.0
        assert perfect.overlap_precision == perfect.overlap_recall == 1.0
        silent = score_track(reference, Track("none+mute", [], "mute"))
        assert silent.der == 1.0 and silent.jer == 1.0
        assert silent.overlap_precision is None
        assert silent.speaker_count_error == -2

    def test_errors_split_by_reference_speaker(self) -> None:
        """No collar: A loses 3.8-4 to ~y (0.2 s of 4); B loses 3-3.5 to
        ~x and 6-7.5 to ~z (2.0 s of 5). Nobody is silent while A or B
        talks, so nothing is missed outright; the unattributed time adds
        up to DER's missed + confusion (0.7 + 1.5)."""
        a, b = score_speakers(ten_seconds(), HYPOTHESIS, NO_COLLAR)
        assert (a.speaker, b.speaker) == ("A", "B")
        assert (a.speech, b.speech) == (pytest.approx(4.0),
                                        pytest.approx(5.0))
        assert a.missed == b.missed == 0.0
        assert a.unattributed == pytest.approx(0.2)
        assert b.unattributed == pytest.approx(2.0)
        assert (a.error_rate, b.error_rate) == (0.05, 0.4)
        assert a.unattributed + b.unattributed == pytest.approx(
            score_track(ten_seconds(), HYPOTHESIS, NO_COLLAR).missed
            + score_track(ten_seconds(), HYPOTHESIS, NO_COLLAR).confusion
        )

    def test_speaker_scores_of_a_silent_track_and_the_collar(self) -> None:
        silent = score_speakers(ten_seconds(), Track("none+mute", [], "m"))
        assert [s.missed_rate for s in silent] == [1.0, 1.0]
        assert [s.error_rate for s in silent] == [1.0, 1.0]
        a, b = score_speakers(ten_seconds(), HYPOTHESIS)
        assert (a.speech, b.speech) == (pytest.approx(3.0),
                                        pytest.approx(4.0))
        short = make_reference(AMI, "test", "x", 10.0,
                               [seg(1.0, 1.4, "A"), seg(3.0, 8.0, "B")],
                               None)
        tiny = score_speakers(short, HYPOTHESIS)[0]
        assert tiny.speech == 0.0 and tiny.error_rate is None

    def test_jaccard_edges(self) -> None:
        nobody = np.zeros((10, 2), dtype=bool)
        assert jaccard_error(nobody, nobody, np.ones(10, dtype=bool)) is None

    def test_wer_against_the_reference_words(self) -> None:
        reference = [Word(text="Hello,", start=0.0, end=0.5),
                     Word(text="world.", start=0.5, end=1.0),
                     Word(text="Good", start=2.0, end=2.4)]
        heard = [Word(text="hello", start=0.0, end=0.5),
                 Word(text="word", start=0.5, end=1.0),
                 Word(text="good", start=2.0, end=2.4),
                 Word(text="morning", start=2.4, end=2.9)]
        compared = compare_transcripts(REFERENCE_TRACK, reference,
                                       "none+r", heard)
        # one substitution and one insertion over three reference words
        assert compared.wer == pytest.approx(2 / 3)
        assert compared.reference_tokens == 3

    def test_spread(self) -> None:
        assert spread([None]).n == 0 and spread([]).mean is None
        one = spread([0.5, None])
        assert (one.n, one.mean, one.sd) == (1, 0.5, None)
        two = spread([0.2, 0.4])
        assert (two.mean, two.min, two.max) == (0.3, 0.2, 0.4)
        assert two.sd == pytest.approx(0.1414, abs=1e-4)


# ------------------------------------------------------------ the import


NOISE = b"NOISE"
"""The bytes of a fake recording that holds steady noise, not speech."""


class SyntheticDecoder:
    """Decode a recording as ``seconds`` of a synthetic voice at 16 kHz.

    A recording whose bytes are :data:`NOISE` holds band-limited noise.
    """

    def __init__(self, seconds: float = 10.0) -> None:
        """Keep the duration."""
        self.seconds = seconds
        self.decoded = 0

    def decode(self, recording: RecordedAudio) -> Audio:
        """Return the voice (or the noise) of the duration."""
        self.decoded += 1
        make = band_noise if recording.data == NOISE else voiced
        return Audio(make(self.seconds), 16000)


class FakeCorpus:
    """A tiny corpus of recordings with the 10-second example's turns.

    The recordings numbered in ``noisy`` hold noise instead of speech;
    those in ``broken`` have a second turn that ends before it starts
    (as a CALLHOME row does).
    """

    def __init__(self, count: int = 3,
                 words: Optional[List[Word]] = None,
                 noisy: Sequence[int] = (),
                 broken: Sequence[int] = ()) -> None:
        """Make ``count`` recordings."""
        self.items = [
            CorpusRecording(
                recording=f"REC{n}",
                audio=RecordedAudio(NOISE if n in noisy else b"RIFF", "wav"),
                starts=[0.0, 3.0],
                ends=[4.0, -5.25 if n in broken else 8.0],
                speakers=["A", "B"], words=words,
            )
            for n in range(count)
        ]
        self.asked: List[tuple[str, str, Optional[int]]] = []

    def recordings(
        self, spec: object, split: str, limit: Optional[int]
    ) -> Iterator[CorpusRecording]:
        """Yield the recordings, at most ``limit``."""
        self.asked.append((getattr(spec, "id"), split, limit))
        yield from self.items[:limit]


class Importer:
    """The import use case over fakes."""

    def __init__(self, corpus: FakeCorpus, seconds: float = 10.0) -> None:
        """Wire the fakes."""
        self.sessions = FakeSessionRepository()
        self.repository = FakeLabRepository()
        locks = FakeSessionLocks()
        self.decoder = SyntheticDecoder(seconds)
        self.use = ImportReference(
            self.sessions, self.repository, corpus, self.decoder,
            ImportRecording(self.sessions, locks), locks, FixedClock(),
        )
        self.check = CheckReferenceSpeech(
            self.sessions, self.repository, self.decoder, locks,
        )


class TestImport:
    def test_one_session_per_recording(self) -> None:
        corpus = FakeCorpus(3)
        importer = Importer(corpus)
        heard: List[ImportedRecording] = []
        done = importer.use("ami", limit=2, report=heard.append)
        # the corpus is read on until two usable recordings are in
        assert corpus.asked == [("ami", "test", None)]
        assert [d.session_id for d in done] == ["ami-REC0", "ami-REC1"]
        assert heard == done
        first = done[0]
        assert (first.state, first.duration, first.speakers, first.turns,
                first.speaker_seconds, first.words) == (
            "imported", 10.0, 2, 2, 9.0, None)
        session = importer.sessions.get("ami-REC0")
        assert session is not None
        assert session.state is SessionState.UPLOADED
        assert session.language == "en" and session.duration == 10.0
        assert session.title == "AMI (single distant microphone) REC0"
        assert session.expected_speakers is None
        reference = importer.repository.reference("ami-REC0")
        assert reference is not None
        assert (reference.corpus, reference.recording, reference.split,
                reference.licence) == ("ami", "REC0", "test", "CC BY 4.0")
        assert reference.imported_at == FixedClock().now()
        assert reference.speech is not None and reference.speech.speech
        assert reference.excluded() is None
        assert first.speech == reference.speech and first.usable

    def test_noise_is_skipped_and_the_limit_counts_usable_ones(
        self,
    ) -> None:
        importer = Importer(FakeCorpus(5, noisy=(1, 2)))
        heard: List[ImportedRecording] = []
        done = importer.use("ami", limit=2, report=heard.append)
        assert [(d.session_id, d.state) for d in done] == [
            ("ami-REC0", "imported"), ("ami-REC1", "skipped"),
            ("ami-REC2", "skipped"), ("ami-REC3", "imported"),
        ]
        assert heard == done
        skipped = done[1]
        assert skipped.reason == NO_SPEECH and not skipped.usable
        assert skipped.speech is not None and not skipped.speech.speech
        assert skipped.duration == 10.0 and skipped.turns == 0
        assert importer.sessions.get("ami-REC1") is None
        assert importer.repository.reference("ami-REC1") is None
        assert importer.sessions.get("ami-REC4") is None
        # again: the noise is decoded and skipped again, nothing made
        again = importer.use("ami", limit=3)
        assert [(d.session_id, d.state) for d in again] == [
            ("ami-REC0", "exists"), ("ami-REC1", "skipped"),
            ("ami-REC2", "skipped"), ("ami-REC3", "exists"),
            ("ami-REC4", "imported"),
        ]

    def test_an_invalid_reference_is_skipped(self) -> None:
        importer = Importer(FakeCorpus(4, noisy=(2,), broken=(1, 2)))
        heard: List[ImportedRecording] = []
        done = importer.use("ami", limit=2, report=heard.append)
        assert [(d.session_id, d.state) for d in done] == [
            ("ami-REC0", "imported"), ("ami-REC1", "skipped"),
            ("ami-REC2", "skipped"), ("ami-REC3", "imported"),
        ]
        assert heard == done
        skipped = done[1]
        assert skipped.reason == (
            "invalid reference (Invalid reference turn from 3 s to -5.25 s)"
        )
        assert not skipped.usable and skipped.speech is None
        assert skipped.duration == 10.0 and skipped.turns == 0
        # the reference is refused before the speech check: noise and a
        # broken reference say the reference
        assert done[2].reason is not None
        assert done[2].reason.startswith(INVALID_REFERENCE)
        for name in ("ami-REC1", "ami-REC2"):
            assert importer.sessions.get(name) is None
            assert importer.repository.reference(name) is None

    def test_a_row_without_turns_is_skipped(self) -> None:
        corpus = FakeCorpus(2)
        corpus.items[0] = dataclasses.replace(
            corpus.items[0], starts=[], ends=[], speakers=[],
        )
        done = Importer(corpus).use("ami", limit=1)
        assert [(d.state, d.reason) for d in done] == [
            ("skipped", "invalid reference (Recording 'REC0' has no"
             " speaker turn to score)"),
            ("imported", None),
        ]

    def test_a_repaired_session_keeps_refusing_a_broken_reference(
        self,
    ) -> None:
        corpus = FakeCorpus(1)
        importer = Importer(corpus)
        importer.use("ami")
        del importer.repository.references["ami-REC0"]
        corpus.items = FakeCorpus(1, broken=(0,)).items
        with pytest.raises(InvalidInputError, match="Invalid reference"):
            importer.use("ami")

    def test_a_repaired_session_without_speech_is_marked(self) -> None:
        corpus = FakeCorpus(1)
        importer = Importer(corpus)
        importer.use("ami")
        del importer.repository.references["ami-REC0"]
        corpus.items = FakeCorpus(1, noisy=(0,)).items
        (repaired,) = importer.use("ami")
        assert (repaired.state, repaired.reason) == ("repaired", NO_SPEECH)
        reference = importer.repository.reference("ami-REC0")
        assert reference is not None and reference.excluded() == NO_SPEECH
        # it no longer counts for the limit
        (exists,) = importer.use("ami", limit=1)
        assert (exists.state, exists.usable) == ("exists", False)

    def test_again_skips_what_exists(self) -> None:
        importer = Importer(FakeCorpus(2))
        importer.use("ami", limit=1)
        again = importer.use("ami")
        assert [(d.session_id, d.state) for d in again] == [
            ("ami-REC0", "exists"), ("ami-REC1", "imported"),
        ]

    def test_a_session_left_without_reference_is_repaired(self) -> None:
        importer = Importer(FakeCorpus(1))
        importer.use("ami")
        # the process stopped between the session and its reference
        del importer.repository.references["ami-REC0"]
        again = importer.use("ami")
        assert [(d.session_id, d.state, d.turns) for d in again] == [
            ("ami-REC0", "repaired", 2),
        ]
        assert importer.repository.reference("ami-REC0") is not None
        assert importer.use("ami")[0].state == "exists"

    def test_a_foreign_session_of_that_id_is_refused(self) -> None:
        importer = Importer(FakeCorpus(1))
        importer.sessions.create(Session(id="ami-REC0", language="es",
                                         title="my meeting"))
        with pytest.raises(InvalidInputError, match="not imported"):
            importer.use("ami")
        assert importer.repository.reference("ami-REC0") is None

    def test_words_and_a_non_commercial_corpus(self) -> None:
        words = [Word(text="hola", start=1.0, end=1.4)]
        importer = Importer(FakeCorpus(1, words))
        done = importer.use("callhome")
        assert done[0].session_id == "callhome-REC0" and done[0].words == 1
        reference = importer.repository.reference("callhome-REC0")
        assert reference is not None and not reference.licence_commercial
        assert reference.split == "data" and reference.language == "es"

    def test_refusals_and_rollback(self) -> None:
        importer = Importer(FakeCorpus(1))
        with pytest.raises(InvalidInputError):
            importer.use("ami", limit=0)
        with pytest.raises(InvalidInputError):
            importer.use("ami", split="nope")
        empty = Importer(FakeCorpus(1), seconds=0.0)
        with pytest.raises(InvalidInputError, match="no audio"):
            empty.use("ami")

        def broken(session_id: str, reference: SessionReference) -> None:
            """Fail to store the reference."""
            raise StorageError("disk full")

        importer.repository.save_reference = broken  # type: ignore
        with pytest.raises(StorageError):
            importer.use("ami")
        assert importer.sessions.get("ami-REC0") is None


class TestCheckAgain:
    def test_sessions_imported_before_are_checked(self) -> None:
        importer = Importer(FakeCorpus(3))
        importer.use("ami")
        importer.use("callhome", limit=1)
        # as imported before the check existed: no figures, and the
        # second recording's audio turns out to be noise
        for ident in ("ami-REC0", "ami-REC1", "ami-REC2"):
            old = importer.repository.reference(ident)
            assert old is not None
            importer.repository.save_reference(
                ident, old.model_copy(update={"speech": None}))
        importer.sessions.recordings["ami-REC1"] = RecordedAudio(NOISE, "wav")
        heard: List[ImportedRecording] = []
        done = importer.check("ami", heard.append)
        assert heard == done
        assert [(d.session_id, d.state, d.reason) for d in done] == [
            ("ami-REC0", "checked", None),
            ("ami-REC1", "checked", NO_SPEECH),
            ("ami-REC2", "checked", None),
        ]
        bad = importer.repository.reference("ami-REC1")
        assert bad is not None and bad.excluded() == NO_SPEECH
        good = importer.repository.reference("ami-REC0")
        assert good is not None and good.speech is not None
        assert importer.sessions.get("ami-REC1") is not None  # kept
        with pytest.raises(InvalidInputError):
            importer.check("nope")


# ------------------------------------------------- scoring inside a run


def with_reference(lab: Lab, corpus: str = "ami") -> SessionReference:
    """Give the lab's session a reference equal to diarizer z1's turns."""
    reference = make_reference(
        CORPORA[corpus], "test", "R", lab.scene.duration,
        [seg(a, b, f"spk-{who}") for a, b, who in TURNS], None,
    )
    lab.repository.save_reference(SESSION, reference)
    return reference


class TestRunScores:
    def test_a_run_is_measured_against_the_reference(self) -> None:
        lab = Lab()
        with_reference(lab)
        run_id = lab.run(diarizers=["z1", "z2"], recognisers=["r1"])
        results = lab.results(SESSION, run_id)
        block = results["reference"]
        assert block["corpus"] == "ami" and block["licence_commercial"]
        assert block["reference_tokens"] is None
        assert block["transcripts"] == []
        tracks = {t["track"]: t for t in block["tracks"]}
        assert tracks["none+z1"]["der"] == 0.0
        assert tracks["none+z1"]["speaker_count_error"] == 0
        assert tracks["none+z2"]["der"] > 0.0
        rows = {p["id"]: p for p in block["pipelines"]}
        assert rows["none+z1+r1"]["der"] == 0.0
        assert rows["none+z1+r1"]["wer"] is None
        assert rows["none+z2+r1"]["licence_commercial"] is False

    def test_a_reference_transcript_gives_a_wer(self) -> None:
        lab = Lab()
        reference = with_reference(lab, "callhome").model_copy(update={
            "words": [Word(text="hola", start=1.0, end=1.5),
                      Word(text="vale", start=14.0, end=14.5)],
        })
        lab.repository.save_reference(SESSION, reference)
        run_id = lab.run(diarizers=["z1"], recognisers=["r1"])
        block = lab.results(SESSION, run_id)["reference"]
        assert block["reference_tokens"] == 2
        assert block["transcripts"][0]["wer"] == 0.0
        assert not block["licence_commercial"]
        assert all(not p["licence_commercial"] for p in block["pipelines"])

    def test_corpus_speakers_are_never_given_enrolled_ids(self) -> None:
        lab = Lab()
        named = lab.run(diarizers=["builtin", "z1"])
        speakers = {s for t in lab.results(SESSION, named)["tracks"]
                    for s in t["speakers"]}
        assert {"ana", "luis"} <= speakers  # enrolled people are found
        with_reference(lab)
        run_id = lab.run(diarizers=["builtin", "z1"])
        tracks = lab.results(SESSION, run_id)["tracks"]
        assert tracks and all(
            m["speaker"] is None and m["method"] == "none"
            for t in tracks for m in t["mapping"]
        )
        assert not {"ana", "luis"} & {s for t in tracks
                                      for s in t["speakers"]}

    def test_sessions_without_reference_have_none(self) -> None:
        lab = Lab()
        run_id = lab.run(diarizers=["z1"])
        assert lab.results(SESSION, run_id)["reference"] is None


class TestSummary:
    def test_mean_and_spread_over_recordings(self) -> None:
        lab = Lab()
        with_reference(lab)
        run_id = lab.run(diarizers=["z1", "z2"], recognisers=["r1"])
        results = lab.repository.stored_results[run_id]
        # a second recording with the same run, z2's DER changed
        session = lab.sessions.get(SESSION)
        reference = lab.repository.reference(SESSION)
        assert session is not None and reference is not None
        lab.sessions.create(session.model_copy(update={"id": "ami-two"}))
        lab.repository.save_reference("ami-two", reference)
        other = lab.repository.run(SESSION, run_id)
        assert other is not None
        copy = other.model_copy(update={"id": "f" * 32,
                                        "session_id": "ami-two"})
        lab.repository.save_run(copy)
        block = dict(results["reference"])
        block["tracks"] = [
            dict(t, der=0.5) if t["track"] == "none+z2" else t
            for t in block["tracks"]
        ]
        lab.repository.save_results("ami-two", copy.id,
                                    dict(results, reference=block))
        lab.sessions.create(session.model_copy(update={"id": "plain"}))
        summary = ReferenceSummary(lab.sessions, lab.repository)()
        assert summary.corpora == ["ami"] and not summary.non_commercial
        assert [r.session_id for r in summary.recordings] == [
            "ami-two", SESSION,
        ]
        models = {(m.kind, m.id): m for m in summary.models}
        z1 = models[("diarization", "none+z1")]
        assert z1.recordings == 2
        assert (z1.metrics["der"].mean, z1.metrics["der"].sd) == (0.0, 0.0)
        z2 = models[("diarization", "none+z2")].metrics["der"]
        first = next(t for t in results["reference"]["tracks"]
                     if t["track"] == "none+z2")["der"]
        assert z2.mean == pytest.approx((first + 0.5) / 2, abs=1e-4)
        assert z2.min == pytest.approx(min(first, 0.5), abs=1e-4)
        assert models[("pipeline", "none+z1+r1")].metrics["wer"].n == 0
        assert ReferenceSummary(lab.sessions, lab.repository)(
            "callhome"
        ).recordings == []

    def test_recordings_without_speech_are_excluded(self) -> None:
        lab = Lab()
        noise = SpeechCheck(autocorrelation=0.68, flatness=0.39, frames=9,
                            speech=False)
        reference = with_reference(lab).model_copy(update={"speech": noise})
        lab.repository.save_reference(SESSION, reference)
        lab.run(diarizers=["z1"])
        summary = ReferenceSummary(lab.sessions, lab.repository)()
        assert summary.recordings == [] and summary.models == []
        assert summary.unscored == []
        (excluded,) = summary.excluded
        assert (excluded.session_id, excluded.corpus, excluded.reason) == (
            SESSION, "ami", NO_SPEECH,
        )
        assert excluded.speech == noise

    def test_unscored_and_unreadable(self) -> None:
        lab = Lab()
        with_reference(lab, "callhome")
        failed = LabRun.model_validate({
            "id": "e" * 32, "session_id": SESSION,
            "created_at": "2026-01-01T00:00:00Z", "request": {},
            "state": "failed",
        })
        lab.repository.save_run(failed)
        summary = ReferenceSummary(lab.sessions, lab.repository)()
        assert summary.unscored == [SESSION] and summary.non_commercial
        assert summary.recordings[0].run_id is None

        def unreadable(session_id: str) -> SessionReference:
            """A damaged reference file."""
            raise StorageError("bad")

        lab.repository.reference = unreadable  # type: ignore
        assert ReferenceSummary(lab.sessions, lab.repository)()\
            .recordings == []
        with pytest.raises(InvalidInputError):
            ReferenceSummary(lab.sessions, lab.repository)("nope")
