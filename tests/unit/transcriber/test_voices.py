"""Transcriber voices: clustering, centroids and matching to voiceprints.

The clustering and centroid tests are copied from
``tests/unit/test_skills_speech.py`` with transcriber imports.
"""

from datetime import datetime, timezone
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Voiceprint
from transcriber.domain.voices import (
    Attribution,
    FloatArray,
    anonymous_label,
    centre,
    centre_on_session,
    cluster_speakers,
    match_voices,
    nearest_voice,
    unit_rows,
    unit_vectors,
    voice_centroids,
)


def voices() -> FloatArray:
    """Six embeddings: three voices, two windows each, first-seen order."""
    a, b, c = np.eye(3, dtype=np.float32)
    stacked: FloatArray = np.stack(
        [a, b, a * 0.9 + b * 0.1, c, b * 0.95 + c * 0.05, c]
    ).astype(np.float32)
    return stacked


class TestClusterSpeakers:
    def test_known_number_of_speakers(self) -> None:
        assert cluster_speakers(voices(), num_speakers=3) == [0, 1, 0, 2, 1, 2]

    def test_threshold_finds_the_number(self) -> None:
        assert cluster_speakers(voices(), threshold=0.5) == [0, 1, 0, 2, 1, 2]

    def test_one_speaker(self) -> None:
        assert cluster_speakers(voices(), num_speakers=1) == [0] * 6

    def test_low_threshold_merges_everything(self) -> None:
        assert set(cluster_speakers(voices(), threshold=-1.0)) == {0}

    def test_max_speakers_caps_an_unknown_count(self) -> None:
        distinct = np.eye(5, dtype=np.float32)
        distinct[1, 0] = 0.4  # voices 0 and 1 are the closest pair
        distinct[3, 2] = 0.3  # then voices 2 and 3
        assert len(set(cluster_speakers(distinct, threshold=0.5))) == 5
        labels = cluster_speakers(distinct, threshold=0.5, max_speakers=3)
        assert labels == [0, 0, 1, 1, 2]

    def test_empty_and_invalid(self) -> None:
        assert cluster_speakers(np.zeros((0, 4), dtype=np.float32)) == []
        with pytest.raises(InvalidInputError):
            cluster_speakers(voices(), num_speakers=0)

    def test_zero_vectors_do_not_crash(self) -> None:
        vectors = np.zeros((2, 3), dtype=np.float32)
        assert cluster_speakers(vectors, num_speakers=2) == [0, 1]

    def test_single_vector(self) -> None:
        assert cluster_speakers(np.ones((1, 3), dtype=np.float32)) == [0]


class TestCentredVoices:
    def test_centring_removes_the_common_component(self) -> None:
        """Two voices that differ slightly around a large shared part."""
        shared = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        a = shared + np.array([0.0, 1.0, 0.0], dtype=np.float32)
        b = shared + np.array([0.0, -1.0, 0.0], dtype=np.float32)
        raw = np.stack([a, b, a, b]).astype(np.float32)
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        assert float(unit[0] @ unit[1]) > 0.97      # indistinguishable
        centred = centre(raw)
        assert float(centred[0] @ centred[1]) == pytest.approx(-1.0)
        assert float(centred[0] @ centred[2]) == pytest.approx(1.0)
        assert cluster_speakers(centred, threshold=0.5) == [0, 1, 0, 1]

    def test_given_mean_and_degenerate_inputs(self) -> None:
        raw = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        moved = centre(raw, np.array([1.0, 1.0], dtype=np.float32))
        assert np.allclose(moved, [[0.7071, -0.7071], [-0.7071, 0.7071]],
                           atol=1e-3)
        assert centre(np.zeros((0, 3), dtype=np.float32)).shape == (0, 3)
        same = centre(np.ones((2, 2), dtype=np.float32))
        assert np.allclose(same, 0.0)               # no information left

    def test_centroids_and_nearest_voice(self) -> None:
        vectors = np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
        )
        centroids = voice_centroids(vectors, [0, 0, 1])
        assert centroids.shape == (2, 2)
        assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)
        fine = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]],
                        dtype=np.float32)
        assert nearest_voice(fine, centroids) == [0, 1, 0]

    def test_empty_inputs(self) -> None:
        none = np.zeros((0, 2), dtype=np.float32)
        assert voice_centroids(none, []).shape == (0, 2)
        assert nearest_voice(none, np.eye(2, dtype=np.float32)) == []
        assert nearest_voice(np.eye(2, dtype=np.float32), none) == []

    def test_unit_rows_keeps_zero_rows(self) -> None:
        rows = unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
        assert np.allclose(rows, [[0.6, 0.8], [0.0, 0.0]])


WHEN = datetime(2026, 11, 13, 10, 0, tzinfo=timezone.utc)


def print_of(speaker_id: str, *vector: float, model: str = "m") -> Voiceprint:
    """A voiceprint with the given vector."""
    return Voiceprint(speaker_id=speaker_id, model=model,
                      vector=list(vector), consent_version="c1",
                      enrolled_at=WHEN)


def groups(*rows: Sequence[float]) -> FloatArray:
    """Session group vectors, one per row."""
    out: FloatArray = np.array(rows, dtype=np.float32)
    return out


def labels(attributions: List[Attribution]) -> List[Tuple[str, bool]]:
    """The speaker and the enrolled flag of each attribution, in order."""
    return [(a.speaker, a.enrolled) for a in attributions]


class TestMatchVoices:
    def test_each_group_takes_its_voiceprint(self) -> None:
        out = match_voices(
            groups((0.1, 1.0), (1.0, 0.1)),
            [print_of("ana", 1.0, 0.0), print_of("luis", 0.0, 1.0)],
        )
        assert labels(out) == [("luis", True), ("ana", True)]
        assert [a.group for a in out] == [0, 1]
        assert out[0].similarity == pytest.approx(1.0 / np.sqrt(1.01))
        assert out[0].nearest == "luis"

    def test_two_groups_near_one_voiceprint_only_the_better_wins(
        self,
    ) -> None:
        out = match_voices(
            groups((0.8, 0.6), (1.0, 0.05)),
            [print_of("ana", 1.0, 0.0)],
        )
        assert labels(out) == [("S1", False), ("ana", True)]
        assert out[0].similarity is None
        assert out[0].nearest == "ana"
        assert out[0].nearest_similarity == pytest.approx(0.8)

    def test_below_threshold_stays_anonymous(self) -> None:
        out = match_voices(
            groups((1.0, 1.0)), [print_of("ana", 1.0, 0.0)], threshold=0.8
        )
        assert labels(out) == [("S1", False)]
        assert out[0].nearest_similarity == pytest.approx(0.7071, abs=1e-4)
        accepted = match_voices(
            groups((1.0, 1.0)), [print_of("ana", 1.0, 0.0)], threshold=0.7
        )
        assert labels(accepted) == [("ana", True)]

    def test_sub_threshold_pairs_never_displace_accepted_ones(self) -> None:
        """Maximising the raw sum would keep one person and lose another.

        Similarities: g0-ana 0.60, g0-luis 0.52, g1-ana 0.52,
        g1-luis 0.45. The raw best sum (g0-ana + g1-luis = 1.05) accepts
        only ana at threshold 0.5; counting accepted pairs only, the
        assignment g0-luis + g1-ana (1.04) names both.
        """
        g0 = (0.6, 0.52, float(np.sqrt(1 - 0.36 - 0.2704)))
        g1 = (0.52, 0.45, float(np.sqrt(1 - 0.2704 - 0.2025)))
        out = match_voices(
            groups(g0, g1),
            [print_of("ana", 1.0, 0.0, 0.0), print_of("luis", 0.0, 1.0, 0.0)],
            threshold=0.5,
        )
        assert labels(out) == [("luis", True), ("ana", True)]
        assert out[0].similarity == pytest.approx(0.52, abs=1e-6)
        assert out[0].nearest == "ana"

    def test_more_people_enrolled_than_groups(self) -> None:
        out = match_voices(
            groups((0.0, 1.0, 0.0)),
            [print_of("ana", 1.0, 0.0, 0.0), print_of("luis", 0.0, 1.0, 0.0),
             print_of("eva", 0.0, 0.0, 1.0)],
        )
        assert labels(out) == [("luis", True)]

    def test_more_groups_than_people_enrolled(self) -> None:
        out = match_voices(
            groups((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            [print_of("ana", 1.0, 0.0, 0.0)],
        )
        assert labels(out) == [("S1", False), ("ana", True), ("S2", False)]

    def test_empty_inputs(self) -> None:
        assert match_voices(groups(), [print_of("ana", 1.0)]) == []
        assert match_voices(np.zeros((0, 2), dtype=np.float32), []) == []
        out = match_voices(groups((1.0, 0.0), (0.0, 1.0)), [])
        assert labels(out) == [("S1", False), ("S2", False)]
        assert out[0].nearest is None and out[0].nearest_similarity is None

    def test_anonymous_labels_skip_enrolled_ids(self) -> None:
        out = match_voices(
            groups((1.0, 0.0), (0.0, 1.0)), [print_of("S1", 1.0, 0.0)],
            threshold=0.99,
        )
        assert labels(out) == [("S1", True), ("S2", False)]
        assert anonymous_label(3) == "S3"

    def test_labels_are_deterministic(self) -> None:
        session = groups((1.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        prints = [print_of("ana", 1.0, 0.0), print_of("luis", 1.0, 0.0)]
        first = match_voices(session, prints)
        for _ in range(5):
            assert match_voices(session, prints) == first
        assert sorted(a.speaker for a in first if a.enrolled) == [
            "ana", "luis",
        ]
        assert labels(first)[2] == ("S1", False)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(InvalidInputError, match="one voiceprint"):
            match_voices(
                groups((1.0, 0.0)),
                [print_of("ana", 1.0, 0.0), print_of("ana", 0.0, 1.0)],
            )
        with pytest.raises(InvalidInputError, match="different sizes"):
            match_voices(groups((1.0, 0.0)), [print_of("ana", 1.0)])
        with pytest.raises(InvalidInputError, match="model x"):
            match_voices(groups((1.0, 0.0)),
                         [print_of("ana", 1.0, 0.0, model="x")], model="m")
        same = match_voices(groups((1.0, 0.0)),
                            [print_of("ana", 1.0, 0.0)], model="m")
        assert labels(same) == [("ana", True)]


class TestNormalisation:
    def test_unit_vectors_is_the_default(self) -> None:
        session, enrolled = unit_vectors(
            groups((3.0, 4.0)), groups((0.0, 2.0))
        )
        assert np.allclose(session, [[0.6, 0.8]])
        assert np.allclose(enrolled, [[0.0, 1.0]])

    def test_the_caller_chooses_the_normalisation(self) -> None:
        """A shared component makes two people look alike as raw units."""
        session = groups((10.0, 1.0, 0.0), (10.0, -1.0, 0.0))
        prints = [print_of("ana", 10.0, 1.0, 0.0),
                  print_of("luis", 10.0, -1.0, 0.0)]
        enrolled = groups((10.0, 1.0, 0.0), (10.0, -1.0, 0.0))
        raw_session, raw_prints = unit_vectors(session, enrolled)
        assert float(raw_session[0] @ raw_prints[1]) > 0.97
        centred_session, centred_prints = centre_on_session(
            session, enrolled
        )
        assert float(centred_session[0] @ centred_prints[1]) == (
            pytest.approx(-1.0)
        )
        # With a threshold above the cross similarity, raw unit vectors
        # still separate them here; centring widens the margin.
        for normalisation in (unit_vectors, centre_on_session):
            out = match_voices(session, prints, threshold=0.99,
                               normalisation=normalisation)
            assert labels(out) == [("ana", True), ("luis", True)]
        swapped = match_voices(
            session, list(reversed(prints)), threshold=0.99,
            normalisation=centre_on_session,
        )
        assert labels(swapped) == [("ana", True), ("luis", True)]

    def test_centre_on_session_without_session_vectors(self) -> None:
        session, enrolled = centre_on_session(
            np.zeros((0, 2), dtype=np.float32), groups((0.0, 2.0))
        )
        assert session.shape == (0, 2)
        assert np.allclose(enrolled, [[0.0, 1.0]])
