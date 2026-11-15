"""Dialogue-act tagger: inputs, routing, scoring and calibration (#82)."""

import math
from typing import Dict, List

import numpy as np
import pytest

from src.domain.skills.act_calibration import (
    as_probabilities,
    cross_fit_outcomes,
    fit_outcomes,
    fit_temperature,
    fold_of,
    margin_curve,
    propose_margin,
    reference_rates,
    set_masks,
    set_nll,
    softmax,
)
from src.domain.skills.act_scoring import (
    expected_calibration_error,
    log_loss,
    macro_f1,
    per_category,
    ranking,
    resolve,
    score,
    sentence_log_loss,
    top_k_accuracy,
)
from src.domain.skills.act_tagging import (
    OTHER_SPEAKER,
    SAME_SPEAKER,
    ActInput,
    Reference,
    Route,
    category_precision,
    check_probabilities,
    choose_routes,
    encoder_texts,
    record_input,
    record_reference,
    route,
    sentence_inputs,
    speaker_changes,
    split_of,
    top_category,
)
from src.domain.skills.dialogue_acts import (
    CATEGORIES,
    ActRecord,
    ContextLine,
    WeightedLabel,
)


def ref(*categories: str) -> Reference:
    """Return a compatible-set reference."""
    return Reference(categories=list(categories))


def dist(**values: float) -> Dict[str, float]:
    """Return a full distribution: the given values, the rest shared."""
    rest = (1.0 - sum(values.values())) / (len(CATEGORIES) - len(values))
    return {c: values.get(c, rest) for c in CATEGORIES}


def record(categories: List[str], text: str = "Hola") -> ActRecord:
    """Return an imported record."""
    return ActRecord(
        corpus="ami", licence="CC BY 4.0", non_commercial=False,
        language="en", dialogue="D1", id="u1", order=0, speaker="A",
        text=text, context=[ContextLine(speaker="B", text="Hi")],
        source_label="inf", categories=categories)


# --- inputs ---------------------------------------------------------------


def test_split_is_deterministic_and_covers_the_three_splits() -> None:
    names = [f"meeting-{n}" for n in range(400)]
    splits = [split_of(n) for n in names]
    assert splits == [split_of(n) for n in names]
    assert set(splits) == {"train", "dev", "test"}
    assert 0.08 < splits.count("test") / 400 < 0.25
    with pytest.raises(ValueError):
        split_of("x", test=0.6, dev=0.5)


def test_record_input_and_reference() -> None:
    item = record_input(record(["informacion", "opinion"]))
    assert item.id == "D1/u1" and item.context[0].text == "Hi"
    reference = record_reference(record(["informacion", "opinion"]))
    assert reference is not None
    assert reference.categories == ["informacion", "opinion"]
    assert record_reference(record([])) is None


def test_sentence_inputs_carry_their_context() -> None:
    items = sentence_inputs([("a", "X", "uno"), ("b", "Y", "dos"),
                             ("c", "Y", "tres")], context=1)
    assert [len(i.context) for i in items] == [0, 1, 1]
    assert items[2].context[0].text == "dos"
    assert sentence_inputs([("a", "X", "uno")], context=0)[0].context == []
    with pytest.raises(ValueError):
        sentence_inputs([], context=-1)


def test_encoder_texts_mark_speakers_not_names() -> None:
    item = ActInput(id="1", speaker="Ana", text="Vale.", context=[
        ContextLine(speaker="Ana", text="Lo miro."),
        ContextLine(speaker="Luis", text="¿Puedes?")])
    context, target = encoder_texts(item)
    assert context == f"{SAME_SPEAKER} Lo miro. {OTHER_SPEAKER} ¿Puedes?"
    assert target == f"{OTHER_SPEAKER} Vale."
    assert "Ana" not in context + target
    assert speaker_changes(item)
    alone = ActInput(id="2", speaker="Ana", text="Hola")
    assert encoder_texts(alone) == ("", f"{SAME_SPEAKER} Hola")
    long = ActInput(id="3", speaker="A", text="x", context=[
        ContextLine(speaker="B", text="y" * 1000)])
    assert len(encoder_texts(long)[0]) == 400


def test_check_probabilities_normalises_and_refuses_nonsense() -> None:
    full = check_probabilities({"pregunta": 2.0, "respuesta": 2.0})
    assert full["pregunta"] == 0.5 and full["otro"] == 0.0
    assert len(full) == len(CATEGORIES)
    for bad in ({"nada": 1.0}, {"pregunta": -1.0}, {"pregunta": 0.0}):
        with pytest.raises(ValueError):
            check_probabilities(bad)


def test_top_category_breaks_ties_by_category_order() -> None:
    assert top_category({"respuesta": 0.5, "pregunta": 0.5}) == "pregunta"


def test_route_takes_the_first_matching_claim() -> None:
    routes = [Route(tagger="rules", categories=["escucha"],
                    keys=["escucha:backchannel"]),
              Route(tagger="encoder", categories=["pregunta"])]
    outputs = {"rules": dist(escucha=0.9), "encoder": dist(pregunta=0.6)}
    assert route(routes, "encoder", outputs,
                 {"rules": "escucha:backchannel"})[0] == "rules"
    assert route(routes, "encoder", outputs, {"rules": "other"})[0] == (
        "encoder")
    outputs["encoder"] = dist(opinion=0.6)
    assert route(routes, "encoder", outputs, {"rules": "x"}) == (
        "encoder", outputs["encoder"])


def test_choose_routes_gives_each_category_its_most_precise_tagger() -> None:
    gold = [["escucha"]] * 6 + [["pregunta"]] * 6
    rules = [dist(escucha=0.9)] * 6 + [dist(escucha=0.9)] * 6
    encoder = [dist(informacion=0.9)] * 6 + [dist(pregunta=0.9)] * 6
    counts = category_precision(gold, rules)
    assert counts == {"escucha": (6, 12)}
    routes, fallback = choose_routes(gold, {"rules": rules,
                                            "encoder": encoder})
    assert fallback in ("rules", "encoder")
    assert routes[0] == Route(tagger="encoder", categories=["pregunta"])
    assert Route(tagger="rules", categories=["escucha"]) in routes
    with pytest.raises(ValueError):
        choose_routes(gold, {})


# --- scoring --------------------------------------------------------------


def test_compatible_sets_count_any_member_as_right() -> None:
    refs = [ref("informacion", "opinion"), ref("pregunta")]
    preds = [dist(opinion=0.7), dist(respuesta=0.5, pregunta=0.4)]
    assert top_k_accuracy(refs, preds, 1) == 0.5
    assert top_k_accuracy(refs, preds, 3) == 1.0
    assert ranking(preds[1])[:2] == ["respuesta", "pregunta"]
    with pytest.raises(ValueError):
        top_k_accuracy(refs, preds, 0)
    with pytest.raises(ValueError):
        top_k_accuracy(refs, preds[:1])


def test_log_loss_uses_the_set_or_the_weights() -> None:
    p = dist(informacion=0.5, opinion=0.3)
    assert sentence_log_loss(ref("informacion", "opinion"), p) == (
        pytest.approx(-math.log(0.8)))
    weighted = Reference(categories=["compromiso"], labels=[
        WeightedLabel(label="compromiso", weight=70),
        WeightedLabel(label="aceptacion", weight=30)])
    q = dist(compromiso=0.6, aceptacion=0.2)
    assert sentence_log_loss(weighted, q) == pytest.approx(
        -(0.7 * math.log(0.6) + 0.3 * math.log(0.2)))
    assert log_loss([ref("otro")], [{"otro": 0.0}]) == pytest.approx(
        -math.log(1e-15))


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    refs = [ref("pregunta"), ref("pregunta")]
    preds = [dist(pregunta=0.5), dist(respuesta=0.5)]
    assert expected_calibration_error(refs, preds) == pytest.approx(0.0)
    assert expected_calibration_error(refs[:1], preds[:1]) == (
        pytest.approx(0.5))
    with pytest.raises(ValueError):
        expected_calibration_error(refs, preds, bins=0)


def test_resolve_and_per_category_only_measure_named_categories() -> None:
    refs = [ref("informacion", "opinion"), ref("pregunta"),
            ref("pregunta"), ref("propuesta", "peticion")]
    preds = [dist(opinion=0.7), dist(pregunta=0.7), dist(opinion=0.7),
             dist(escucha=0.5, peticion=0.3)]
    assert resolve(refs[0], preds[0]) == "opinion"
    assert resolve(refs[3], preds[3]) == "peticion"
    scores = per_category(refs, preds)
    assert scores["pregunta"].measurable
    assert scores["pregunta"].precision == 1.0
    assert scores["pregunta"].recall == 0.5
    assert scores["opinion"].f1 is None and not scores["opinion"].measurable
    assert scores["opinion"].predicted == 2
    assert macro_f1(scores) == pytest.approx(2 / 3)
    assert macro_f1({}) is None


def test_score_collects_every_measure() -> None:
    refs = [ref("pregunta"), ref("respuesta")]
    preds = [dist(pregunta=0.9), dist(pregunta=0.6)]
    result = score(refs, preds)
    assert result.sentences == 2 and result.top1 == 0.5
    assert result.confusion == {"pregunta": {"pregunta": 1},
                                "respuesta": {"pregunta": 1}}
    assert result.per_category["respuesta"].f1 == 0.0
    assert result.macro_f1 == pytest.approx((2 / 3 + 0.0) / 2)


# --- calibration ----------------------------------------------------------


def test_softmax_and_temperature() -> None:
    logits = np.array([[2.0, 0.0] + [0.0] * 11])
    sharp = softmax(logits, 0.5)
    flat = softmax(logits, 2.0)
    assert sharp.sum() == pytest.approx(1.0)
    assert sharp[0, 0] > softmax(logits)[0, 0] > flat[0, 0]
    with pytest.raises(ValueError):
        softmax(logits, 0.0)


def test_fit_temperature_recovers_a_known_temperature() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(4000, len(CATEGORIES)))
    true = softmax(logits, 2.0)
    labels = [rng.choice(len(CATEGORIES), p=row) for row in true]
    refs = [ref(CATEGORIES[i]) for i in labels]
    fitted = fit_temperature(logits, refs)
    assert fitted == pytest.approx(2.0, rel=0.1)
    mask = set_masks(refs)
    assert set_nll(logits, mask, fitted) < set_nll(logits, mask, 1.0)
    with pytest.raises(ValueError):
        fit_temperature(logits, refs[:3])


def test_as_probabilities_and_reference_rates() -> None:
    rows = as_probabilities(np.eye(len(CATEGORIES))[:1])
    assert rows[0]["pregunta"] == 1.0
    weighted = Reference(categories=["compromiso"], labels=[
        WeightedLabel(label="compromiso", weight=70),
        WeightedLabel(label="aceptacion", weight=30)])
    rates = reference_rates([ref("informacion", "opinion"), weighted])
    assert rates["informacion"] == 0.25 and rates["compromiso"] == 0.35
    with pytest.raises(ValueError):
        reference_rates([])


def test_outcome_tables_are_smoothed_shares() -> None:
    refs = [ref("escucha"), ref("escucha"), ref("aceptacion"),
            ref("pregunta")]
    table = fit_outcomes(["bc", "bc", "bc", "q"], refs, smoothing=0.0)
    assert table.probabilities("bc")["escucha"] == pytest.approx(2 / 3)
    assert table.counts == {"bc": 3, "q": 1}
    assert table.probabilities("never")["escucha"] == 0.5
    smooth = fit_outcomes(["bc", "bc", "bc", "q"], refs, smoothing=2.0)
    assert smooth.probabilities("q")["pregunta"] == pytest.approx(
        (1 + 2 * 0.25) / 3)
    assert sum(smooth.probabilities("q").values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fit_outcomes(["a"], [])
    with pytest.raises(ValueError):
        fit_outcomes(["a"], refs[:1], smoothing=-1)


def test_folds_keep_dialogues_whole_and_cross_fit_never_peeks() -> None:
    groups = ["a", "a", "b", "c", "c", "d"]
    folds = fold_of(groups, 2)
    assert folds[0] == folds[1] and folds[3] == folds[4]
    with pytest.raises(ValueError):
        fold_of(groups, 1)
    refs = [ref("escucha"), ref("escucha"), ref("pregunta"),
            ref("pregunta"), ref("pregunta"), ref("escucha")]
    out = cross_fit_outcomes(["k"] * 6, refs, groups, folds=2,
                             smoothing=0.0)
    # Dialogues a and c (fold 0) see only b and d: one of each.
    assert out[0]["escucha"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cross_fit_outcomes(["k"], refs, groups)


def test_margin_curve_and_proposal() -> None:
    refs = [ref("pregunta")] * 4 + [ref("escucha")] * 4
    preds = ([dist(pregunta=0.9)] * 4
             + [dist(escucha=0.3, pregunta=0.6)] * 4)
    rates = reference_rates(refs)
    curve = margin_curve(refs, preds, rates, [0.0, 0.3, 0.6])
    assert [p.margin for p in curve] == [0.0, 0.3, 0.6]
    # Margin 0: never-referenced categories (base rate 0) are clear too.
    assert curve[0].clear_top1 == 8 and curve[0].clear_top1_precision == 0.5
    assert curve[0].clear_precision == pytest.approx(0.2)
    assert curve[1].clear == 4 and curve[1].clear_precision == 1.0
    assert curve[1].clear_top1_precision == 1.0
    assert curve[2].clear == 0 and curve[2].clear_precision is None
    assert propose_margin(curve, 0.8) == 0.3
    assert propose_margin(curve, 1.01) is None
    with pytest.raises(ValueError):
        margin_curve(refs, preds[:1], rates, [0.0])
