"""Expressed customer signals: quoted, ranked, and only in support."""

from typing import Any, Dict, List

from orchestrator.domain.signals import (
    expressed,
    present,
    rank,
    strip_signals,
    weight,
)

TEXTS = {
    "t1.s0": ("S2", "Es la tercera vez que llamo por lo mismo."),
    "t2.s0": ("S1", "Lo siento mucho, lo reviso."),
    "t3.s0": ("S2", "Gracias, así sí."),
}
ROLES = {"S1": "agent", "S2": "customer"}


def suggestion(text: str, signals: List[str], about: List[str]
               ) -> Dict[str, Any]:
    return {"kind": "answer", "text": text, "signals": signals,
            "about": about, "cites": []}


def test_a_signal_is_quoted_from_the_customer_sentence() -> None:
    quoted = expressed(
        suggestion("Ofrece escalar", ["complaint", "urgency", "complaint",
                                      "anger"], ["t1.s0"]),
        TEXTS, ROLES,
    )
    assert quoted == [
        {"signal": "complaint", "utterance": "t1.s0",
         "quote": "Es la tercera vez que llamo por lo mismo."},
        {"signal": "urgency", "utterance": "t1.s0",
         "quote": "Es la tercera vez que llamo por lo mismo."},
    ]


def test_a_signal_without_a_customer_quote_is_dropped() -> None:
    assert expressed(suggestion("x", ["complaint"], ["t2.s0"]),
                     TEXTS, ROLES) == []
    assert expressed(suggestion("x", ["complaint"], ["t9.s9"]),
                     TEXTS, ROLES) == []
    assert expressed(suggestion("x", ["complaint"], []), TEXTS, ROLES) == []


def test_suggestions_are_ranked_by_signals_and_repetition() -> None:
    calm = {"text": "a", "expressed": [{"signal": "satisfaction"}]}
    lost = {"text": "b", "expressed": [{"signal": "confusion"}]}
    angry = {"text": "c", "expressed": [{"signal": "complaint"}]}
    plain = {"text": "d"}
    order = [s["text"] for s in rank([calm, plain, lost, angry], {})]
    assert order == ["c", "b", "a", "d"]
    assert weight(["complaint"], {"complaint": 2}) > weight(
        ["churn_risk"], {})
    assert weight(["complaint"], {}) < weight(["churn_risk"], {})


def test_present_in_support_quotes_ranks_and_counts() -> None:
    history: Dict[str, int] = {}
    shown = present([
        suggestion("Explica el reembolso", ["confusion"], ["t3.s0"]),
        suggestion("Ofrece escalar", ["complaint"], ["t1.s0"]),
        suggestion("Sin señal", ["complaint"], ["t2.s0"]),
    ], TEXTS, ROLES, True, history)
    assert [s["text"] for s in shown] == [
        "Ofrece escalar", "Explica el reembolso", "Sin señal"]
    assert shown[0]["expressed"][0]["quote"].startswith("Es la tercera")
    assert "signals" not in shown[2] and "expressed" not in shown[2]
    assert history == {"complaint": 1, "confusion": 1}


def test_present_outside_support_removes_every_signal() -> None:
    history: Dict[str, int] = {}
    shown = present(
        [suggestion("Pregunta por el proyecto", ["complaint"], ["t1.s0"])],
        TEXTS, ROLES, False, history,
    )
    assert shown == [{"kind": "answer", "text": "Pregunta por el proyecto",
                      "about": ["t1.s0"], "cites": []}]
    assert history == {}
    assert strip_signals({"a": 1, "signals": [], "expressed": []}) == {
        "a": 1}
