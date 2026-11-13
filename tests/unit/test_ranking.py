"""Ranking primitives: top-k, linear fusion, reciprocal rank fusion."""

import numpy as np
import numpy.typing as npt
import pytest

from src.domain.ranking import (
    FloatArray,
    apply_priors,
    linear_fusion,
    linear_scores,
    normalised_fusion,
    ranked,
    rerank_fusion,
    rrf_fusion,
    top_k,
)


def arr(*values: float) -> FloatArray:
    """Return a float32 vector."""
    return np.asarray(values, dtype=np.float32)


def test_top_k_orders_best_first() -> None:
    """Indices of the highest scores are returned in decreasing order."""
    assert top_k(arr(0.1, 0.9, 0.5, 0.7), 3) == [1, 3, 2]


def test_top_k_stable_on_ties() -> None:
    """Ties keep a deterministic order."""
    assert top_k(arr(1.0, 1.0, 1.0), 3) == top_k(arr(1.0, 1.0, 1.0), 3)
    assert sorted(top_k(arr(1.0, 1.0, 1.0), 3)) == [0, 1, 2]


@pytest.mark.parametrize("k, expected", [(0, []), (-1, []), (10, [1, 0])])
def test_top_k_bounds(k: int, expected: list[int]) -> None:
    """k <= 0 returns nothing; k larger than the array is clamped."""
    assert top_k(arr(0.2, 0.4), k) == expected


def test_top_k_empty() -> None:
    """An empty score vector yields no result."""
    assert top_k(arr(), 5) == []


def test_linear_fusion_normalises_lexical() -> None:
    """BM25 is max-normalised before adding the weighted cosine."""
    ranked = linear_fusion(arr(10.0, 5.0, 0.0), arr(0.0, 0.0, 1.0), 0.4, 3)
    assert [i for i, _ in ranked] == [0, 1, 2]
    scores = dict(ranked)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.4)


def test_linear_fusion_weight_can_flip_order() -> None:
    """A large semantic weight lets dense similarity dominate."""
    ranked = linear_fusion(arr(1.0, 0.8), arr(0.0, 1.0), 2.0, 1)
    assert ranked[0][0] == 1


def test_linear_fusion_zero_lexical() -> None:
    """All-zero lexical scores do not divide by zero."""
    ranked = linear_fusion(arr(0.0, 0.0), arr(0.2, 0.6), 1.0, 2)
    assert [i for i, _ in ranked] == [1, 0]
    assert [s for _, s in ranked] == pytest.approx([0.6, 0.2])


def test_rrf_fusion_combines_rankings() -> None:
    """Documents ranked well in both lists come first."""
    ranked = rrf_fusion([(1.0, [0, 1, 2]), (1.0, [1, 0, 3])], 4, rrf_k=60)
    ids = [i for i, _ in ranked]
    assert set(ids[:2]) == {0, 1}
    assert ids[2:] == [2, 3]
    assert dict(ranked)[0] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_fusion_weights_and_truncation() -> None:
    """Weights scale contributions and the result is cut to k."""
    ranked = rrf_fusion([(1.0, [0]), (3.0, [1])], 1)
    assert [i for i, _ in ranked] == [1]
    assert ranked[0][1] == pytest.approx(3.0 / 61)


def test_rrf_fusion_tie_breaks_by_index() -> None:
    """Equal fused scores are ordered by document index."""
    ranked = rrf_fusion([(1.0, [5]), (1.0, [2])], 2)
    assert [i for i, _ in ranked] == [2, 5]


def test_rrf_fusion_empty() -> None:
    """No ranking yields no result."""
    assert rrf_fusion([], 5) == []
    assert rrf_fusion([(1.0, [])], 5) == []


def test_rerank_fusion_weight_zero_keeps_first_order() -> None:
    """With weight 0 only the first-stage score counts."""
    ranked = rerank_fusion([3.0, 2.0, 1.0], [0.0, 5.0, 9.0], 0.0)
    assert [i for i, _ in ranked] == [0, 1, 2]


def test_rerank_fusion_weight_one_follows_reranker() -> None:
    """With weight 1 only the reranker score counts."""
    ranked = rerank_fusion([3.0, 2.0, 1.0], [0.0, 5.0, 9.0], 1.0)
    assert [i for i, _ in ranked] == [2, 1, 0]
    assert ranked[0][1] == pytest.approx(1.0)


def test_rerank_fusion_blends_normalised_scores() -> None:
    """Scores on different scales are min-max normalised before mixing."""
    ranked = rerank_fusion([10.0, 0.0], [-4.0, 4.0], 0.3)
    assert [i for i, _ in ranked] == [0, 1]
    assert [s for _, s in ranked] == pytest.approx([0.7, 0.3])


def test_rerank_fusion_constant_scores_keep_order() -> None:
    """A constant scoring carries no information; ties keep the order."""
    assert [i for i, _ in rerank_fusion([1.0, 1.0, 1.0],
                                        [2.0, 2.0, 2.0], 0.5)] == [0, 1, 2]
    assert rerank_fusion([], [], 0.5) == []


def test_rerank_fusion_length_mismatch() -> None:
    """Both scorings must describe the same candidates."""
    with pytest.raises(ValueError):
        rerank_fusion([1.0], [1.0, 2.0], 0.5)


def test_linear_scores_scales_lexical_and_adds_weighted_dense() -> None:
    """Every chunk gets lexical / max + weight * dense, as float32."""
    fused = linear_scores(arr(2.0, 1.0, 0.0), arr(0.0, 0.5, 1.0), 0.5)
    assert fused.dtype == np.float32
    assert fused.tolist() == pytest.approx([1.0, 0.75, 0.5])


def test_linear_scores_without_lexical_match_or_chunks() -> None:
    """A zero lexical vector is not divided; empty vectors stay empty."""
    fused = linear_scores(arr(0.0, 0.0), arr(0.2, 0.8), 0.5)
    assert fused.tolist() == pytest.approx([0.1, 0.4])
    assert linear_scores(arr(), arr(), 0.4).shape == (0,)


def test_ranked_returns_best_position_score_pairs() -> None:
    """The k best positions come with their scores, best first."""
    best = ranked(arr(0.1, 0.9, 0.5), 2)
    assert [i for i, _ in best] == [1, 2]
    assert [s for _, s in best] == pytest.approx([0.9, 0.5])
    assert [i for i, _ in ranked(arr(0.1, 0.9), 5)] == [1, 0]
    assert ranked(arr(), 3) == []


def test_linear_fusion_is_the_top_of_linear_scores() -> None:
    """linear_fusion ranks exactly the vector of linear_scores."""
    lexical, dense = arr(3.0, 1.0, 2.0), arr(0.1, 0.9, 0.2)
    assert linear_fusion(lexical, dense, 0.7, 2) == ranked(
        linear_scores(lexical, dense, 0.7), 2
    )


def files(*ids: int) -> npt.NDArray[np.int64]:
    """Return a vector of file identifiers."""
    return np.asarray(ids, dtype=np.int64)


def test_apply_priors_multiplies_scores() -> None:
    """Without file boost the scores are only scaled by the priors."""
    scores = arr(1.0, 2.0, 4.0)
    adjusted = apply_priors(scores, arr(1.0, 0.5, 0.0), files(0, 0, 1), 0.0)
    assert adjusted.dtype == np.float32
    assert adjusted.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert scores.tolist() == [1.0, 2.0, 4.0], "the input is not modified"


def test_apply_priors_file_boost_adds_share_of_file_best() -> None:
    """Each chunk gains file_boost times the best score of its file."""
    adjusted = apply_priors(
        arr(1.0, 0.2, 0.9, 0.1), arr(1.0, 1.0, 1.0, 1.0),
        files(0, 0, 1, 1), 0.5,
    )
    assert adjusted.tolist() == pytest.approx([1.5, 0.7, 1.35, 0.55])


def test_apply_priors_file_boost_keeps_intra_file_order() -> None:
    """A strong file overtakes another file; its own order is kept."""
    scores = arr(1.0, 0.5, 0.6)
    ones = arr(1.0, 1.0, 1.0)
    assert top_k(apply_priors(scores, ones, files(0, 0, 1), 0.0), 3) == [
        0, 2, 1,
    ]
    boosted = apply_priors(scores, ones, files(0, 0, 1), 1.0)
    assert top_k(boosted, 3) == [0, 1, 2]
    assert boosted[0] - boosted[1] == pytest.approx(0.5)


def test_apply_priors_boost_uses_the_scaled_scores() -> None:
    """The file best is taken after the multipliers are applied."""
    adjusted = apply_priors(arr(1.0, 1.0), arr(0.5, 1.0), files(0, 1), 1.0)
    assert adjusted.tolist() == pytest.approx([1.0, 2.0])


def test_apply_priors_interleaved_files_and_empty_input() -> None:
    """File ids need not be contiguous; no chunk means no score."""
    adjusted = apply_priors(
        arr(0.4, 1.0, 0.0), arr(1.0, 1.0, 1.0), files(1, 0, 1), 0.5
    )
    assert adjusted.tolist() == pytest.approx([0.6, 1.5, 0.2])
    empty = apply_priors(arr(), arr(), files(), 0.5)
    assert empty.shape == (0,)


class TestNormalisedFusion:
    """``convex`` and ``dbsf``: both rankings normalised, then added."""

    def test_weight_is_independent_of_the_cosine_range(self) -> None:
        """Squeezing the dense scores into a narrow band (as some models
        do) changes nothing: only their order within the band matters."""
        lexical = arr(3.0, 2.0, 1.0, 0.5)
        wide = arr(0.1, 0.9, 0.5, 0.2)
        narrow = (0.8 + wide / 10).astype(np.float32)
        for method in ("convex", "dbsf"):
            a = normalised_fusion(lexical, wide, 1.5, 4, 4, method)
            b = normalised_fusion(lexical, narrow, 1.5, 4, 4, method)
            assert [i for i, _ in a] == [i for i, _ in b]

    def test_convex_values(self) -> None:
        """Min-max on each list: best lexical 1 + weight * its dense."""
        lexical = arr(4.0, 2.0, 0.0)
        dense = arr(0.2, 0.6, 1.0)
        fused = dict(normalised_fusion(lexical, dense, 1.0, 3, 3, "convex"))
        # Candidates: lexical {0, 1} (zero scores excluded) U dense {0,1,2}
        assert fused[0] == pytest.approx(1.0 + 0.0)
        assert fused[1] == pytest.approx(0.5 + 0.5)
        assert fused[2] == pytest.approx(0.0 + 1.0)

    def test_ties_keep_lower_index_and_k_truncates(self) -> None:
        """Equal fused scores keep the lower chunk index first."""
        lexical = arr(4.0, 2.0, 0.0)
        dense = arr(0.2, 0.6, 1.0)
        best = normalised_fusion(lexical, dense, 1.0, 3, 2, "convex")
        assert [i for i, _ in best] == [0, 1]

    def test_candidates_limited_to_depth(self) -> None:
        """Only the top ``depth`` of each list can be returned."""
        lexical = arr(5.0, 4.0, 0.0, 0.0, 0.0)
        dense = arr(0.0, 0.0, 0.1, 0.2, 0.9)
        best = normalised_fusion(lexical, dense, 1.0, 1, 5, "convex")
        assert sorted(i for i, _ in best) == [0, 4]

    def test_dbsf_maps_three_deviations_to_the_unit_range(self) -> None:
        """DBSF: ``mean - 3 sd`` -> 0 and ``mean + 3 sd`` -> 1."""
        lexical = arr(3.0, 2.0, 1.0)
        dense = arr(0.5, 0.5, 0.5)  # constant: contributes nothing
        fused = dict(normalised_fusion(lexical, dense, 1.0, 3, 3, "dbsf"))
        sd = float(np.std([3.0, 2.0, 1.0]))
        low, high = 2.0 - 3 * sd, 2.0 + 3 * sd
        for i, value in enumerate((3.0, 2.0, 1.0)):
            assert fused[i] == pytest.approx((value - low) / (high - low))

    def test_constant_lists_carry_no_information(self) -> None:
        """A list with one value everywhere normalises to zeros."""
        lexical = arr(1.0, 1.0, 1.0)
        dense = arr(0.3, 0.3, 0.3)
        for method in ("convex", "dbsf"):
            best = normalised_fusion(lexical, dense, 1.0, 3, 3, method)
            assert [s for _, s in best] == [0.0, 0.0, 0.0]
            assert [i for i, _ in best] == [0, 1, 2]

    def test_empty_inputs(self) -> None:
        """No chunks: nothing to rank."""
        empty = arr()
        assert normalised_fusion(empty, empty, 1.0, 5, 5, "convex") == []

    def test_unknown_method(self) -> None:
        """Only the normalised fusions are accepted."""
        with pytest.raises(ValueError, match="Unknown normalised fusion"):
            normalised_fusion(arr(1.0), arr(1.0), 1.0, 1, 1, "rrf")
