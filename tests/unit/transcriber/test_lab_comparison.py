"""The lab's comparison on synthetic cases whose answer is known."""

from typing import List, Sequence, Tuple

import pytest

from transcriber.domain.comparison import (
    NO_CONSENSUS,
    ComparisonParameters,
    Track,
    agreement_with,
    align,
    attribute,
    change_times,
    collar_mask,
    compare_tracks,
    compare_transcripts,
    consensus,
    consensus_coverage,
    consensus_spans,
    cut_points,
    distinct_texts,
    error_components,
    fragments,
    frame_count,
    local_names,
    match_times,
    normalise_text,
    summarise_track,
    text_of,
    to_frames,
    vote_weights,
    word_agreement,
    word_error,
    words_between,
    words_on_consensus,
)
from transcriber.domain.lab import Segment
from transcriber.domain.models import Word
from transcriber.domain.speakers import SpeakerSpan, assign_speakers

NO_COLLAR = ComparisonParameters(collar=0.0)


def seg(start: float, end: float, speaker: str) -> Segment:
    """A segment."""
    return Segment(start=start, end=end, speaker=speaker)


def track(
    name: str, parts: Sequence[Tuple[float, float, str]],
    group: str = "",
) -> Track:
    """A track from ``(start, end, speaker)`` parts."""
    return Track(name, [seg(*p) for p in parts], group or name)


def words(*parts: Tuple[str, float, float]) -> List[Word]:
    """Words from ``(text, start, end)``."""
    return [Word(text=t, start=s, end=e) for t, s, e in parts]


A_THEN_B = track("x", [(0, 5, "ana"), (5, 10, "luis")])


class TestFrames:
    def test_a_frame_belongs_to_the_segment_holding_its_middle(
        self
    ) -> None:
        grid = to_frames([seg(0.004, 0.026, "a")], 4, 0.01)
        assert grid.active[:, 0].tolist() == [True, True, True, False]
        assert grid.speakers().tolist() == [1, 1, 1, 0]
        assert frame_count(0.035, 0.01) == 4
        assert frame_count(0.0, 0.01) == 0

    def test_fixed_names_keep_empty_columns(self) -> None:
        grid = to_frames([seg(0, 1, "a"), seg(0, 1, "z")], 100, 0.01,
                         ["a", "b"])
        assert grid.names == ["a", "b"]
        assert grid.active[:, 1].sum() == 0

    def test_collar_marks_both_sides_of_every_edge(self) -> None:
        mask = collar_mask([seg(1.0, 2.0, "a")], 300, 0.01, 0.25)
        assert mask.sum() == 100  # 0.5 s around each of the two edges
        assert not collar_mask([seg(1, 2, "a")], 300, 0.01, 0.0).any()

    def test_anonymous_names_are_local_to_their_track(self) -> None:
        named = local_names(track("t1", [(0, 1, "~S1"), (1, 2, "ana")]))
        assert [s.speaker for s in named] == ["t1~S1", "ana"]


class TestTrackComparison:
    def test_identical_tracks_agree_fully(self) -> None:
        result = compare_tracks(A_THEN_B, A_THEN_B, 10.0)
        assert result.agreement == pytest.approx(1.0)
        assert result.name_agreement == pytest.approx(1.0)
        assert result.error_rate == pytest.approx(0.0)
        assert result.change_points.f1 == pytest.approx(1.0)
        assert result.overlap.iou is None

    def test_other_labels_agree_after_mapping_but_not_by_name(
        self
    ) -> None:
        other = track("y", [(0, 5, "~S1"), (5, 10, "~S2")])
        result = compare_tracks(A_THEN_B, other, 10.0)
        assert result.agreement == pytest.approx(1.0)
        assert result.name_agreement == pytest.approx(0.0)
        assert result.error_rate == pytest.approx(0.0)

    def test_swapped_names_are_all_confusion_by_name_only(self) -> None:
        swapped = track("y", [(0, 5, "luis"), (5, 10, "ana")])
        result = compare_tracks(A_THEN_B, swapped, 10.0, NO_COLLAR)
        assert result.agreement == pytest.approx(1.0)
        assert result.name_agreement == pytest.approx(0.0)

    def test_der_components_against_the_reference(self) -> None:
        # Reference: ana 0-4, luis 6-10. Hypothesis: ana 0-2 (2 s missed),
        # ana 6-8 instead of luis (2 s confusion), luis 8-10, and a false
        # alarm from 4 to 5.
        reference = track("r", [(0, 4, "ana"), (6, 10, "luis")])
        hypothesis = track("h", [(0, 2, "ana"), (4, 5, "luis"),
                                 (6, 8, "ana"), (8, 10, "luis")])
        result = compare_tracks(reference, hypothesis, 10.0, NO_COLLAR)
        assert result.reference_speech == pytest.approx(8.0)
        assert result.missed == pytest.approx(2.0)
        assert result.false_alarm == pytest.approx(1.0)
        # one-to-one mapping keeps ana=ana, so 6-8 is a confusion
        assert result.confusion == pytest.approx(2.0)
        assert result.error_rate == pytest.approx(5.0 / 8.0)

    def test_overlap_counts_twice_in_speaker_time(self) -> None:
        reference = track("r", [(0, 4, "ana"), (2, 4, "luis")])
        hypothesis = track("h", [(0, 4, "ana")])
        result = compare_tracks(reference, hypothesis, 4.0, NO_COLLAR)
        assert result.reference_speech == pytest.approx(6.0)
        assert result.missed == pytest.approx(2.0)
        assert result.overlap.reference_seconds == pytest.approx(2.0)
        assert result.overlap.hypothesis_seconds == pytest.approx(0.0)
        assert result.overlap.iou == pytest.approx(0.0)

    def test_overlap_iou(self) -> None:
        left = track("l", [(0, 10, "a"), (2, 6, "b")])
        right = track("r", [(0, 10, "a"), (4, 8, "b")])
        result = compare_tracks(left, right, 10.0, NO_COLLAR)
        assert result.overlap.both_seconds == pytest.approx(2.0)
        assert result.overlap.iou == pytest.approx(2.0 / 6.0)

    def test_the_collar_hides_small_shifts_of_a_turn(self) -> None:
        shifted = track("y", [(0, 5.2, "ana"), (5.2, 10, "luis")])
        result = compare_tracks(A_THEN_B, shifted, 10.0)
        assert result.agreement == pytest.approx(1.0)
        assert result.error_rate == pytest.approx(0.0)
        strict = compare_tracks(A_THEN_B, shifted, 10.0, NO_COLLAR)
        assert strict.agreement is not None and strict.agreement < 1.0
        assert strict.confusion == pytest.approx(0.2)

    def test_silence_on_both_sides_is_not_agreement(self) -> None:
        sparse = track("s", [(0, 1, "ana")])
        result = compare_tracks(sparse, sparse, 100.0, NO_COLLAR)
        assert result.scored_seconds == pytest.approx(1.0)
        nothing = track("n", [])
        empty = compare_tracks(nothing, nothing, 10.0)
        assert empty.agreement is None and empty.error_rate is None
        assert empty.change_points.f1 is None

    def test_change_points_within_the_tolerance(self) -> None:
        late = track("y", [(0, 5.4, "ana"), (5.4, 10, "luis")])
        assert compare_tracks(A_THEN_B, late, 10.0).change_points.f1 == 1.0
        far = track("z", [(0, 6, "ana"), (6, 10, "luis")])
        points = compare_tracks(A_THEN_B, far, 10.0).change_points
        assert (points.matched, points.f1) == (0, 0.0)

    def test_change_times_and_matching(self) -> None:
        grid = to_frames(A_THEN_B.segments, 1000, 0.01)
        assert change_times(grid, 0.01) == [pytest.approx(5.0)]
        assert change_times(to_frames([], 10, 0.01), 0.01) == []
        assert match_times([1.0, 2.0, 3.0], [1.4, 2.1, 9.0], 0.5) == 2
        assert match_times([1.0], [1.6], 0.5) == 0
        assert match_times([1.0, 1.2], [1.1], 0.5) == 1  # one to one

    def test_error_components_by_hand(self) -> None:
        import numpy as np

        ref = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=bool)
        hyp = np.array([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=bool)
        scored = np.ones(4, dtype=bool)
        total, missed, false_alarm, confusion = error_components(
            ref, hyp, scored
        )
        assert (total, missed, false_alarm, confusion) == (4, 1, 1, 1)

    def test_summary(self) -> None:
        overlapped = track("o", [(0, 4, "ana"), (3, 6, "luis"),
                                 (8, 8.5, "~S1")])
        summary = summarise_track(overlapped, 10.0)
        assert summary.speakers == ["ana", "luis"]  # ~S1 under 1 s
        assert summary.speech_seconds == pytest.approx(6.5)
        assert summary.overlap_seconds == pytest.approx(1.0)
        assert summary.speaker_seconds["~S1"] == pytest.approx(0.5)


class TestConsensus:
    def test_majority_per_frame_and_ties(self) -> None:
        tracks = [
            track("a", [(0, 10, "ana")]),
            track("b", [(0, 5, "ana"), (5, 10, "luis")]),
            track("c", [(0, 10, "luis")]),
        ]
        agreed = consensus(tracks, 10.0)
        spans = consensus_spans(agreed, 0.01)
        assert [(s.start, s.end, s.speakers, s.agreed) for s in spans] == [
            (0.0, 5.0, ["ana"], True), (5.0, 10.0, ["luis"], True),
        ]
        two = consensus(tracks[:1] + tracks[2:], 10.0)
        assert set(two.frame_states.tolist()) == {NO_CONSENSUS}
        assert consensus_coverage(two) == pytest.approx(0.0)
        assert consensus_spans(two, 0.01)[0].agreed is False

    def test_one_vote_per_group(self) -> None:
        # Three near-duplicates say ana; two independent tracks say luis.
        tracks = [
            track("p1", [(0, 10, "ana")], "pyannote"),
            track("p2", [(0, 10, "ana")], "pyannote"),
            track("p3", [(0, 10, "ana")], "pyannote"),
            track("s", [(0, 10, "luis")]),
            track("n", [(0, 10, "luis")]),
        ]
        weights = vote_weights(tracks)
        assert weights["p1"] == pytest.approx(1 / 3)
        assert weights["s"] == pytest.approx(1.0)
        agreed = consensus(tracks, 10.0)
        assert consensus_spans(agreed, 0.01)[0].speakers == ["luis"]
        assert agreement_with(agreed, "s", 0.01).agreement == 1.0
        assert agreement_with(agreed, "p1", 0.01).agreement == 0.0

    def test_anonymous_voices_of_two_tracks_never_agree(self) -> None:
        tracks = [track("a", [(0, 10, "~S1")]),
                  track("b", [(0, 10, "~S1")])]
        assert consensus_coverage(consensus(tracks, 10.0)) == 0.0

    def test_agreement_leaves_out_silence_on_both_sides(self) -> None:
        tracks = [track("a", [(0, 2, "ana")]), track("b", [(0, 2, "ana")]),
                  track("c", [(0, 1, "ana")])]
        agreed = consensus(tracks, 100.0)
        result = agreement_with(agreed, "c", 0.01)
        assert result.scored_seconds == pytest.approx(2.0)
        assert result.agreement == pytest.approx(0.5)

    def test_no_track_no_frames(self) -> None:
        assert consensus([], 1.0).frame_states.tolist() == [-1] * 100
        assert consensus_spans(consensus([], 1.0), 0.01)[0].agreed is False
        assert consensus_spans(consensus([], 0.0), 0.01) == []
        assert consensus_coverage(consensus([], 0.0)) is None

    def test_words_on_the_consensus(self) -> None:
        agreed = consensus([A_THEN_B, A_THEN_B], 10.0)
        heard = words(("a", 1, 2), ("b", 6, 7), ("c", 20, 21))
        assert words_on_consensus(
            heard, ["ana", "ana", "ana"], agreed, 0.01
        ) == pytest.approx(0.5)
        silent = consensus([track("q", [])], 10.0)
        assert words_on_consensus(heard, ["x"] * 3, silent, 0.01) is None


class TestWords:
    def test_attribution_follows_the_service_rule(self) -> None:
        segments = [seg(0, 4, "ana"), seg(3, 8, "luis"), seg(12, 14, "ana")]
        heard = words(("a", 1, 2), ("b", 3.2, 4.5), ("c", 9, 9.5),
                      ("d", 10.9, 11.3), ("e", 3.5, 3.9))
        expected = [
            w.speaker for w in assign_speakers(heard, [
                SpeakerSpan(speaker=s.speaker, start=s.start, end=s.end)
                for s in segments
            ])
        ]
        assert attribute(heard, segments) == expected
        assert attribute(heard, []) == [None] * 5
        assert attribute([], segments) == []

    def test_word_agreement(self) -> None:
        assert word_agreement(["a", "b", None], ["a", "c", None]) == (
            pytest.approx(1 / 3)
        )
        assert word_agreement([], []) is None

    def test_words_between_by_their_middle(self) -> None:
        heard = words(("a", 0, 1), ("b", 1, 3), ("c", 3, 3.2))
        assert words_between(heard, 0.5, 2.5) == [0, 1]
        assert words_between(heard, 0.5, 2.0) == [0]  # end excluded
        assert text_of(heard, [0, 2]) == "a c"


class TestRecognition:
    def test_normalisation(self) -> None:
        assert normalise_text("¿Qué TAL? Tengo 25 años, ¡vale!") == [
            "qué", "tal", "tengo", "25", "años", "vale",
        ]
        assert normalise_text("ﬁn_de-semana") == ["fin", "de", "semana"]

    def test_word_error_rate(self) -> None:
        assert word_error(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"]) \
            == pytest.approx(0.5)
        assert word_error([], ["a"]) is None
        assert word_error(["a"], []) == pytest.approx(1.0)
        assert align(["a", "b"], ["a", "b"]).pairs == [(0, 0), (1, 1)]

    @pytest.mark.parametrize("ref, hyp, errors", [
        ("kitten", "sitting", 3), ("abc", "abc", 0), ("abc", "", 3),
        ("", "ab", 2), ("abcdef", "azced", 3), ("aaaa", "aa", 2),
    ])
    def test_alignment_distance_is_levenshtein(
        self, ref: str, hyp: str, errors: int
    ) -> None:
        assert align(list(ref), list(hyp)).errors == errors

    def test_time_agreement_of_matched_words(self) -> None:
        left = words(("Hola,", 0.0, 0.4), ("qué", 0.5, 0.7),
                     ("tal", 0.8, 1.0))
        right = words(("hola", 0.1, 0.5), ("que", 0.5, 0.7),
                      ("tal", 1.5, 1.7))
        result = compare_transcripts("l", left, "r", right)
        assert result.wer == pytest.approx(1 / 3)
        assert result.matched_tokens == 2
        assert result.time_agreement == pytest.approx(0.5)

    def test_distinct_texts(self) -> None:
        assert distinct_texts({"a": "Hola.", "b": "hola", "c": "adiós"}) == 2


class TestFragments:
    def test_cuts_merge_when_close(self) -> None:
        tracks = [A_THEN_B, track("y", [(0, 5.1, "ana"), (5.1, 10, "luis")]),
                  track("z", [(0, 7, "ana"), (7, 10, "luis")])]
        assert cut_points(tracks, 10.0) == [0.0, 5.05, 7.0, 10.0]

    def test_each_fragment_has_every_answer(self) -> None:
        tracks = [A_THEN_B, track("y", [(0, 7, "ana"), (7, 10, "~S1")])]
        pieces = fragments(tracks, 10.0)
        assert [(f.start, f.end) for f in pieces] == [
            (0.0, 5.0), (5.0, 7.0), (7.0, 10.0),
        ]
        first, second, third = pieces
        assert first.distinct_answers == 1 and first.disagreement == 0.0
        assert second.answers == {"x": ["luis"], "y": ["ana"]}
        assert second.distinct_answers == 2
        assert second.disagreement == pytest.approx(0.5)
        assert third.answers["y"] == ["~S1"]

    def test_a_fragment_takes_the_answer_given_most_of_its_time(
        self
    ) -> None:
        tracks = [track("a", [(0, 10, "ana"), (4.9, 5.0, "luis")])]
        piece = fragments(tracks, 10.0, ComparisonParameters(cut_merge=1.0))
        # the two cuts (4.9, 5.0) merge into one at 4.95; the 0.1 s of
        # luis is a minority of either fragment
        assert [f.start for f in piece] == [0.0, 4.95]
        assert [f.answers["a"] for f in piece] == [["ana"], ["ana"]]

    def test_no_track_is_one_quiet_fragment(self) -> None:
        only = fragments([], 3.0)
        assert len(only) == 1 and only[0].disagreement == 0.0
