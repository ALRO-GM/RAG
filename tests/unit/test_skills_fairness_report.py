"""The fairness audit report filled from simulated results (#86)."""

import random
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from src.application.skills.fairness_report import (
    AuditResults,
    fill_audit,
    render_audit,
)


def simulated(seed: int = 7) -> Dict[str, Any]:
    """Return a results document with simulated people in three groups.

    Group A is selected 50 % of the time, B 30 %; C has 3 people.
    Speaker errors: A around 0.10, B around 0.12.
    """
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for group, n, rate, error in (("A", 60, 0.5, 0.10),
                                  ("B", 60, 0.3, 0.12), ("C", 3, 0.5, 0.1)):
        for i in range(n):
            unit = f"{group}{i}"
            selected.append({"unit": unit, "group": group,
                             "selected": i < round(rate * n)})
            errors.append({"unit": unit, "group": group,
                           "value": error + rng.uniform(-0.02, 0.02)})
    return {
        "schema_name": "fairness.results.v1",
        "title": "Auditoría simulada",
        "date": "2026-11-16",
        "system": "pipeline de prueba",
        "data": "personas simuladas",
        "groups_source": "simulado",
        "synthetic": True,
        "notes": ["Sólo para probar la plantilla."],
        "audits": [
            {"metric": "selected", "kind": "rate",
             "description": "Quién pasa el corte.",
             "observations": selected},
            {"metric": "speaker_error", "kind": "mean",
             "direction": "lower_is_better", "observations": errors},
            {"metric": "score_above_median", "kind": "rate",
             "scoring": "above_median", "observations": errors},
        ],
    }


def test_fill_audit_gives_one_table_per_measure() -> None:
    results = AuditResults.model_validate(simulated())
    rate, mean, scoring = fill_audit(results)
    assert rate.table.flagged() == ["B"]
    assert rate.table.rows[2].status == "described_only"
    assert mean.table.policy.threshold == 1.25
    assert mean.table.reference_group == "A"
    assert scoring.table.kind == "rate" and scoring.table.units == 123


def test_render_audit_in_spanish_has_the_template_parts() -> None:
    results = AuditResults.model_validate(simulated())
    text = render_audit(results, fill_audit(results))
    assert text.startswith("# Auditoría simulada\n")
    assert "**Datos simulados.**" in text
    assert "- Origen de los grupos: simulado" in text
    assert "- `selected`: Pasan del umbral: B." in text
    assert ("| Grupo | n | % del total | Tasa | IC 95 % | Cociente | "
            "IC del cociente | Estado |") in text
    assert "| A | 60 | 48,780 | 0,500 (30/60) |" in text
    assert "| 1,000 | — | referencia |" in text
    assert "| C | 3 |" in text and "sólo descrito" in text
    assert "Umbral 0,8:" in text and "Umbral 1,25 (nuestro" in text
    assert "menos de 5 personas o menos del 2 %" in text
    assert "Wilson" in text and "semilla 0" in text
    assert "Tasa de puntuación (LL144)" in text
    assert "## Notas\n\n- Sólo para probar la plantilla." in text


def test_render_audit_in_english() -> None:
    document = simulated()
    document["synthetic"] = False
    document["notes"] = []
    results = AuditResults.model_validate(document)
    text = render_audit(results, fill_audit(results), "en")
    assert "Simulated data" not in text and "## Notes" not in text
    assert "| Group | n | % of total | Mean | 95 % CI |" in text
    assert "| A | 60 | 48.780 | 0.500 (30/60) |" in text
    assert "Threshold 1.25 (ours, provisional)" in text


def test_an_uncertain_flag_and_no_reference_are_said() -> None:
    document = simulated()
    document["audits"] = [
        {"metric": "few", "kind": "rate", "min_units": 1,
         "observations": [
             {"unit": f"{g}{i}", "group": g, "selected": i < k}
             for g, k in (("A", 3), ("B", 2)) for i in range(5)]},
        {"metric": "none", "kind": "rate", "observations": [
            {"unit": "x", "group": "A", "selected": True}]},
    ]
    results = AuditResults.model_validate(document)
    text = render_audit(results, fill_audit(results))
    assert "pasa el umbral (el IC cruza el umbral)" in text
    assert "Ningún grupo tiene tamaño suficiente" in text
    assert "- `none`: Ningún grupo pasa del umbral." in text


def test_the_summary_never_calls_a_zero_reference_gap_within() -> None:
    document = simulated()
    document["audits"] = [
        {"metric": "wrong", "kind": "rate", "direction": "lower_is_better",
         "observations": [
             {"unit": f"{g}{i}", "group": g, "selected": i < k}
             for g, k in (("A", 0), ("B", 5)) for i in range(10)]},
        {"metric": "error", "kind": "mean", "direction": "lower_is_better",
         "min_units": 1, "observations": [
             {"unit": "a", "group": "A", "value": 0.0},
             {"unit": "b", "group": "B", "value": 0.3}]},
    ]
    results = AuditResults.model_validate(document)
    text = render_audit(results, fill_audit(results))
    assert "Ningún grupo pasa del umbral" not in text
    assert ("- `wrong`: Sin cociente porque la referencia es 0, peores "
            "que ella: B.") in text
    assert "sin cociente: la referencia es 0 (diferencia +0,500)" in text
    assert "| A | 1 | 50,000 | 0,000 | — |" in text
    assert "dos valores distintos no tiene intervalo" in text
    english = render_audit(results, fill_audit(results), "en")
    assert "No ratio as the reference is 0, worse than it: B." in english
    assert "no ratio: the reference is 0 (difference +0.300)" in english


def test_an_inconsistent_threshold_is_refused() -> None:
    document = simulated()
    document["audits"][1]["threshold"] = 0.8
    with pytest.raises(ValueError):
        fill_audit(AuditResults.model_validate(document))


@pytest.mark.parametrize("change", [
    {"kind": "rate", "observations": [{"unit": "x", "group": "A",
                                       "value": 1.0}]},
    {"kind": "mean", "observations": [{"unit": "x", "group": "A",
                                       "selected": True}]},
    {"kind": "mean", "scoring": "above_median",
     "observations": [{"unit": "x", "group": "A", "value": 1.0}]},
])
def test_fill_audit_refuses_a_measure_of_the_wrong_kind(
    change: Dict[str, Any]
) -> None:
    document = simulated()
    document["audits"] = [{"metric": "m", **change}]
    with pytest.raises(ValueError):
        fill_audit(AuditResults.model_validate(document))


def test_results_document_is_strict() -> None:
    document = simulated()
    document["schema_name"] = "other"
    with pytest.raises(ValidationError):
        AuditResults.model_validate(document)
