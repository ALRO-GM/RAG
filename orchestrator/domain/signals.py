"""Expressed customer signals: quote them and rank suggestions by them.

``docs/platform/ARCHITECTURE.md`` §8.1: in ``live_assist.support`` the
suggester labels what the **customer says** (never what they feel, never
from the voice). Each signal must be backed by the utterances it comes
from (``about``); the page shows it as "El cliente dice…" with the
quoted sentence. A signal that cannot be quoted from a customer's
sentence is dropped. In every other product the signals are removed
before anything is shown or stored.

Ranking: a suggestion's weight is the sum of its signals' weights, and a
signal the customer already expressed earlier in the session counts
more (a third complaint calls for escalation more than the first).
Plain data, no I/O.
"""

from typing import Any, Dict, Iterable, List, Mapping, Tuple

SIGNALS: Tuple[str, ...] = (
    "complaint", "stated_frustration", "urgency", "churn_risk",
    "confusion", "satisfaction",
)
"""The signal vocabulary of ``reasoner.response.v1``."""

WEIGHTS: Dict[str, float] = {
    "churn_risk": 5.0, "complaint": 4.0, "stated_frustration": 3.0,
    "urgency": 3.0, "confusion": 2.0, "satisfaction": 0.0,
}
"""How much each signal pushes a suggestion up (satisfaction: not at all)."""

REPEAT_BONUS = 0.5
"""Extra weight per earlier time the same signal was expressed."""

CUSTOMER_ROLE = "customer"


def strip_signals(suggestion: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``suggestion`` without signals of any kind."""
    return {k: v for k, v in suggestion.items()
            if k not in ("signals", "expressed")}


def expressed(
    suggestion: Mapping[str, Any],
    texts: Mapping[str, Tuple[str, str]],
    roles: Mapping[str, str],
) -> List[Dict[str, str]]:
    """Return the quotable signals of ``suggestion``.

    Args:
        suggestion: One suggestion of ``reasoner.response.v1``.
        texts: Utterance id → ``(speaker, text)`` of the window.
        roles: Speaker id → role.

    Returns:
        ``[{"signal", "utterance", "quote"}]``: each known signal with
        the first customer sentence among ``about``; a signal with no
        such sentence is left out.
    """
    quote = None
    for utterance in suggestion.get("about") or []:
        found = texts.get(str(utterance))
        if found and roles.get(found[0]) == CUSTOMER_ROLE and found[1]:
            quote = (str(utterance), found[1])
            break
    if quote is None:
        return []
    seen: List[str] = []
    for signal in suggestion.get("signals") or []:
        if signal in SIGNALS and signal not in seen:
            seen.append(signal)
    return [{"signal": s, "utterance": quote[0], "quote": quote[1]}
            for s in seen]


def weight(signals: Iterable[str], history: Mapping[str, int]) -> float:
    """Weight of a suggestion with ``signals`` given earlier counts."""
    return sum(
        WEIGHTS.get(s, 0.0) * (1.0 + REPEAT_BONUS * history.get(s, 0))
        for s in signals
    )


def rank(
    suggestions: List[Dict[str, Any]], history: Mapping[str, int]
) -> List[Dict[str, Any]]:
    """Order suggestions by the weight of their expressed signals.

    ``suggestions`` carry ``expressed`` (see :func:`expressed`); the
    sort is stable, so the reasoner's own order breaks ties.
    """
    def key(item: Dict[str, Any]) -> float:
        names = [e["signal"] for e in item.get("expressed") or []]
        return -weight(names, history)

    return sorted(suggestions, key=key)


def present(
    suggestions: List[Dict[str, Any]],
    texts: Mapping[str, Tuple[str, str]],
    roles: Mapping[str, str],
    enabled: bool,
    history: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Prepare a call's suggestions for the page and the log.

    With ``enabled`` off (every product but support) the signals are
    removed. With it on, each suggestion gets ``expressed`` (quoted
    signals, unquotable ones dropped), ``signals`` keeps only those,
    the suggestions are ranked by them, and ``history`` (signal →
    times seen in the session) is updated after ranking.
    """
    if not enabled:
        return [strip_signals(s) for s in suggestions]
    shown: List[Dict[str, Any]] = []
    for suggestion in suggestions:
        item = strip_signals(suggestion)
        quoted = expressed(suggestion, texts, roles)
        if quoted:
            item["expressed"] = quoted
            item["signals"] = [q["signal"] for q in quoted]
        shown.append(item)
    ranked = rank(shown, history)
    for signal in {q["signal"] for s in ranked
                   for q in s.get("expressed") or []}:
        history[signal] = history.get(signal, 0) + 1
    return ranked
