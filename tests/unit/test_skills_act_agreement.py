"""Weighted dialogue-act labels: agreement and tagger metrics.

Every expected value is either published (Krippendorff's worked
example, the textbook 2 × 2 κ) or worked out by hand in the comment
next to it.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pytest

from src.domain.skills.act_agreement import (
    Annotation,
    Probabilities,
    _kappa_draws,
    as_distribution,
    base_rates,
    category_kappas,
    distribution_distance,
    doubt_thresholds,
    expected_calibration_error,
    full_distribution,
    log_loss,
    mean_overlap,
    overlap,
    primary_kappa,
    ranked,
    tagger_labels,
    top_k_accuracy,
    weighted_alpha,
)
from src.domain.skills.agreement import (
    cohen_kappa,
    krippendorff_alpha,
    krippendorff_alpha_distance,
)
from src.domain.skills.dialogue_acts import WeightedLabel, parse_labels


def labels(text: str) -> List[WeightedLabel]:
    """Read labels written as in the Spanish set."""
    return parse_labels(text)


# ------------------------------------------------------------- overlap


def test_overlap_is_the_sum_of_the_smaller_weights() -> None:
    both = labels("compromiso:70 aceptacion:30")
    assert overlap(both, labels("compromiso")) == pytest.approx(0.7)
    # min(70, 40) + min(30, 60) = 70
    assert overlap(both, labels("aceptacion:60 compromiso:40")) == (
        pytest.approx(0.7))
    assert overlap(both, labels("peticion")) == 0.0
    assert overlap(both, both) == pytest.approx(1.0)


def test_distance_is_one_minus_overlap_and_order_free() -> None:
    a = as_distribution(labels("compromiso:70 aceptacion:30"))
    b = as_distribution(labels("aceptacion:60 compromiso:40"))
    assert a == (("aceptacion", 30), ("compromiso", 70))
    assert distribution_distance(a, b) == pytest.approx(0.3)
    assert distribution_distance(b, a) == pytest.approx(0.3)
    assert distribution_distance(a, a) == 0.0


# ------------------------------------------------------ Krippendorff α

N = None
KRIPPENDORFF: List[List[Optional[float]]] = [
    [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
    [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
]
"""Krippendorff's 4 observers × 12 units example (rows are raters)."""

AS_ACTS = {1: "pregunta", 2: "respuesta", 3: "informacion", 4: "opinion",
           5: "propuesta"}


def test_generic_alpha_reproduces_the_published_nominal_value() -> None:
    nominal = krippendorff_alpha_distance(
        KRIPPENDORFF, lambda a, b: float(a != b))
    assert nominal == pytest.approx(0.743, abs=5e-4)
    assert nominal == pytest.approx(krippendorff_alpha(
        KRIPPENDORFF, [1, 2, 3, 4, 5], level="nominal"))


def test_generic_alpha_reproduces_the_published_interval_value() -> None:
    interval = krippendorff_alpha_distance(
        KRIPPENDORFF, lambda a, b: float((a - b) ** 2))
    assert interval == pytest.approx(0.849, abs=5e-4)


def test_weighted_alpha_with_single_labels_is_the_nominal_alpha() -> None:
    rows: List[List[Annotation]] = [
        [None if v is None else labels(AS_ACTS[int(v)]) for v in row]
        for row in KRIPPENDORFF
    ]
    assert weighted_alpha(rows) == pytest.approx(0.743, abs=5e-4)


def test_weighted_alpha_hand_example() -> None:
    # Values P = pregunta, R = respuesta, M = respuesta 60 informacion 40.
    # Units (P, P), (R, M), (R, R). Coincidences: PP 2, RM 1, MR 1, RR 2;
    # totals P 2, R 3, M 1, n = 6. δ(P,R) = δ(P,M) = 1, δ(R,M) = 0.4.
    # D_o = 0.4 + 0.4 = 0.8; D_e = 2·(2·3·1 + 2·1·1 + 3·1·0.4) = 18.4.
    # α = 1 − 5 · 0.8 / 18.4 = 18/23.
    a: List[Annotation] = [labels("pregunta"), labels("respuesta"),
                           labels("respuesta")]
    b: List[Annotation] = [labels("pregunta"),
                           labels("respuesta:60 informacion:40"),
                           labels("respuesta")]
    assert weighted_alpha([a, b]) == pytest.approx(18 / 23)
    # Nominal (M a value of its own): 1 − 5 · 2 / 22 = 6/11, lower.
    assert krippendorff_alpha_distance(
        [[as_distribution(x) for x in a if x],
         [as_distribution(y) for y in b if y]],
        lambda p, q: float(p != q)) == pytest.approx(6 / 11)
    # Mean overlap: (1 + 0.6 + 1) / 3.
    assert mean_overlap(a, b) == pytest.approx(2.6 / 3)


def test_alpha_edge_cases() -> None:
    assert weighted_alpha([[labels("otro"), None], [None, labels("otro")]]
                          ) is None
    same = [labels("otro"), labels("otro")]
    assert weighted_alpha([same, same]) == 1.0
    assert krippendorff_alpha_distance([], lambda a, b: 1.0) is None


def test_mean_overlap_needs_co_rated_sentences() -> None:
    assert mean_overlap([None], [labels("otro")]) is None
    with pytest.raises(ValueError, match="same sentences"):
        mean_overlap([None], [])


# ------------------------------------------------------------------ κ


def textbook() -> List[List[Annotation]]:
    """The 2 × 2 example: yes/yes 20, yes/no 5, no/yes 10, no/no 15.

    p_o = 35/50 = .7; p_e = .5·.6 + .5·.4 = .5; κ = (.7 − .5)/.5 = .4.
    "yes" is pregunta and "no" informacion.
    """
    yes, no = labels("pregunta"), labels("informacion")
    a = [yes] * 25 + [no] * 25
    b = [yes] * 20 + [no] * 5 + [yes] * 10 + [no] * 15
    return [list(a), list(b)]


def test_primary_kappa_textbook_value_and_interval() -> None:
    a, b = textbook()
    estimate = primary_kappa(a, b, resamples=500, seed=1)
    assert estimate.kappa == pytest.approx(0.4)
    assert estimate.sentences == 50
    assert estimate.low is not None and estimate.high is not None
    assert estimate.low < 0.4 < estimate.high
    assert primary_kappa(a, b, resamples=500, seed=1) == estimate


def test_primary_kappa_reads_only_the_primary_label() -> None:
    a, b = textbook()
    b = [labels("pregunta:60 informacion:40")
         if x == labels("pregunta") else x for x in b]
    assert primary_kappa(a, b, resamples=0).kappa == pytest.approx(0.4)


def test_category_kappas_one_against_the_rest() -> None:
    a, b = textbook()
    a.append(labels("opinion"))
    b.append(labels("opinion"))
    kappas = category_kappas(a, b, resamples=200)
    # pregunta vs rest: yes/yes 20, yes/no 5, no/yes 10, no/no 16.
    # p_o = 36/51; p_e = (25·30 + 26·21)/51²; κ = (p_o − p_e)/(1 − p_e).
    p_o, p_e = 36 / 51, (25 * 30 + 26 * 21) / 51 ** 2
    assert kappas["pregunta"].kappa == pytest.approx(
        (p_o - p_e) / (1 - p_e))
    assert (kappas["pregunta"].count_a, kappas["pregunta"].count_b) == (
        25, 30)
    assert kappas["opinion"].kappa == pytest.approx(1.0)
    assert kappas["rechazo"].kappa is None
    assert kappas["rechazo"].low is None
    assert len(kappas) == 13


def test_bootstrap_draws_match_cohen_kappa_resample_by_resample() -> None:
    x = np.array([0, 0, 1, 2, 1, 0, 2, 2, 1, 0])
    y = np.array([0, 1, 1, 2, 1, 0, 2, 0, 1, 0])
    draws = _kappa_draws(x, y, 3, resamples=5, seed=7)
    picks = np.random.default_rng(7).integers(0, 10, size=(5, 10))
    expected = [
        cohen_kappa(list(map(float, x[p])), list(map(float, y[p])),
                    [0.0, 1.0, 2.0], weights="none")
        for p in picks
    ]
    assert list(draws) == pytest.approx(expected)


def test_kappa_without_sentences() -> None:
    estimate = primary_kappa([None], [labels("otro")])
    assert estimate.kappa is None and estimate.low is None
    assert estimate.sentences == 0


# ------------------------------------------------------------ tagger

GOLD = [labels("pregunta"), labels("respuesta"),
        labels("compromiso:70 aceptacion:30"), labels("informacion")]
PREDICTED: List[Probabilities] = [
    {"pregunta": 0.9, "respuesta": 0.05, "otro": 0.05},
    {"pregunta": 0.6, "respuesta": 0.3, "informacion": 0.1},
    {"aceptacion": 0.5, "compromiso": 0.4},       # 0.1 over 11 others
    {"opinion": 0.7, "pregunta": 0.2},            # 0.1 over 11 others
]


def test_full_distribution_spreads_the_rest() -> None:
    full = full_distribution(PREDICTED[2])
    assert len(full) == 13
    assert full["otro"] == pytest.approx(0.1 / 11)
    assert sum(full.values()) == pytest.approx(1.0)
    assert full_distribution(PREDICTED[0])["otro"] == 0.05


@pytest.mark.parametrize("probabilities, message", [
    ({"nada": 0.5}, "unknown category"),
    ({"otro": 1.5}, "probability of otro"),
    ({"otro": 0.7, "pregunta": 0.6}, "more than 1"),
    ({c: 0.01 for c in ("pregunta", "respuesta", "informacion", "opinion",
                        "propuesta", "peticion", "aceptacion", "rechazo",
                        "compromiso", "escucha", "gestion", "relacion",
                        "otro")}, "less than 1"),
])
def test_full_distribution_refuses(probabilities: Dict[str, float],
                                   message: str) -> None:
    with pytest.raises(ValueError, match=message):
        full_distribution(probabilities)


def test_ranked_breaks_ties_in_category_order() -> None:
    # Missing categories tie at 0.1/11; respuesta comes before
    # informacion in the category list.
    assert [c for c, _ in ranked(PREDICTED[3])[:3]] == [
        "opinion", "pregunta", "respuesta"]


def test_top_k_accuracy_hand_example() -> None:
    # Top-1 right only on the first sentence; top-3 misses only the
    # last (informacion is not in opinion, pregunta, respuesta).
    assert top_k_accuracy(GOLD, PREDICTED, 1) == 0.25
    assert top_k_accuracy(GOLD, PREDICTED, 3) == 0.75
    with pytest.raises(ValueError, match="k must"):
        top_k_accuracy(GOLD, PREDICTED, 0)
    with pytest.raises(ValueError, match="one prediction"):
        top_k_accuracy(GOLD, PREDICTED[:2], 1)


def test_log_loss_hand_example() -> None:
    # −ln .9; −ln .3; −(.7 ln .4 + .3 ln .5); −ln(.1/11) = ln 110.
    expected = (-math.log(0.9) - math.log(0.3)
                - (0.7 * math.log(0.4) + 0.3 * math.log(0.5))
                + math.log(110)) / 4
    assert log_loss(GOLD, PREDICTED) == pytest.approx(expected)
    assert expected == pytest.approx(1.71479, abs=1e-5)


def test_log_loss_clips_a_zero_probability() -> None:
    loss = log_loss([labels("otro")], [{"pregunta": 1.0}])
    assert loss == pytest.approx(-math.log(1e-15))


def test_expected_calibration_error_hand_examples() -> None:
    # Each sentence alone in its bin: (|1 − .9| + .6 + .5 + .7) / 4.
    assert expected_calibration_error(GOLD, PREDICTED) == pytest.approx(
        0.475)
    # Two in the bin (.8, .9]: accuracy .5, mean confidence .85;
    # ECE = 2/2 · .35.
    pair = [labels("pregunta"), labels("pregunta")]
    guesses: List[Probabilities] = [{"pregunta": 0.82, "otro": 0.18},
                                    {"otro": 0.88, "pregunta": 0.12}]
    assert expected_calibration_error(pair, guesses) == pytest.approx(0.35)
    # One bin: |accuracy − mean confidence| = |.25 − .675|.
    assert expected_calibration_error(GOLD, PREDICTED, bins=1) == (
        pytest.approx(0.425))
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(GOLD, PREDICTED, bins=0)


# ---------------------------------------------- tagger labels and doubt


def test_base_rates_count_weights() -> None:
    rates = base_rates(GOLD)
    assert rates["pregunta"] == 0.25
    assert rates["compromiso"] == pytest.approx(0.175)
    assert rates["aceptacion"] == pytest.approx(0.075)
    assert rates["otro"] == 0.0
    with pytest.raises(ValueError):
        base_rates([])


def test_tagger_labels_keep_an_explicit_rest_and_mark_doubt() -> None:
    thresholds = doubt_thresholds(base_rates(GOLD), 0.1)
    out = tagger_labels(PREDICTED[2], thresholds=thresholds)
    # 50, 40, 0.909 (pregunta, first of the tied rest), rest 9.09:
    # floors 50 + 40 + 0 + 9 = 99, the largest remainder (.909) gets 1.
    assert [(x.label, x.weight) for x in out.labels] == [
        ("aceptacion", 50), ("compromiso", 40), ("pregunta", 1)]
    assert out.rest == 9
    # thresholds: aceptacion .175, compromiso .275, pregunta .35.
    assert [x.doubtful for x in out.labels] == [False, False, True]


def test_tagger_labels_round_to_exactly_one_hundred() -> None:
    out = tagger_labels({"pregunta": 0.335, "respuesta": 0.335,
                         "otro": 0.33})
    assert [x.weight for x in out.labels] == [34, 33, 33]
    assert out.rest == 0
    single = tagger_labels({"otro": 1.0}, k=1)
    assert [(x.label, x.weight) for x in single.labels] == [("otro", 100)]
    assert not single.labels[0].doubtful


def test_tagger_labels_refuse_bad_arguments() -> None:
    with pytest.raises(ValueError, match="k must"):
        tagger_labels({"otro": 1.0}, k=4)
    with pytest.raises(ValueError, match="margin"):
        doubt_thresholds({}, -0.1)
