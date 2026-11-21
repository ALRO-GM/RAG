"""Keep only suggestions that rest on what the request contained.

A suggestion may cite knowledge chunks (``cites``) and utterances
(``about``). One that cites an id the request did not carry was made up
by the model and is dropped. The signals of a suggestion (support only)
survive only when every utterance they rest on is the customer's.
Plain functions: no I/O.
"""

from typing import List, Sequence

from src.domain.reasoning.roles import (
    ReasonRequest,
    Signal,
    Suggestion,
)

CUSTOMER_ROLE = "customer"
"""The speaker role whose text signals may be read from."""


def is_grounded(suggestion: Suggestion, request: ReasonRequest) -> bool:
    """Tell whether every id the suggestion cites is in the request."""
    chunks = {r.id for r in request.retrieved}
    lines = {line.id for line in request.context.window}
    return all(c in chunks for c in suggestion.cites) and all(
        a in lines for a in suggestion.about
    )


def customer_only(suggestion: Suggestion, request: ReasonRequest) -> bool:
    """Tell whether the suggestion's ``about`` are all customer lines.

    False when it cites nothing: a signal must be backed by a quote.
    """
    if not suggestion.about:
        return False
    roles = request.context.speaker_roles
    speaker = {line.id: line.speaker for line in request.context.window}
    return all(
        roles.get(speaker.get(a, ""), "") == CUSTOMER_ROLE
        for a in suggestion.about
    )


def ground(
    suggestions: Sequence[Suggestion],
    request: ReasonRequest,
    allow_signals: bool,
) -> List[Suggestion]:
    """Return the grounded suggestions, with signals only where allowed.

    ``allow_signals`` is true for the support template. Even then a
    signal is dropped unless it is backed by customer utterances.
    """
    kept: List[Suggestion] = []
    for suggestion in suggestions:
        if not is_grounded(suggestion, request):
            continue
        signals: List[Signal] = (
            suggestion.signals
            if allow_signals and customer_only(suggestion, request)
            else []
        )
        kept.append(suggestion.model_copy(update={"signals": signals}))
    return kept
