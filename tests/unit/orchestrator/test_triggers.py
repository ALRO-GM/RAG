"""Trigger rules: role, minimum words, debounce, turn end."""

from typing import Optional

from orchestrator.domain.config import Trigger
from orchestrator.domain.events import LiveEvent, Utterance
from orchestrator.domain.triggers import Firing, Skip, TriggerPolicy

ROLES = {"a": "agent", "c": "customer"}


def final(speaker: str, text: str, t: float, n: int = 0) -> LiveEvent:
    return LiveEvent(
        session_id="s", seq=n, type="final", at="x", stream_time=t,
        utterance=Utterance(id=f"t{n}.s0", turn=n, index=0, speaker=speaker,
                            start=0, end=t, text=text, revision=0),
    )


def end(t: float) -> LiveEvent:
    return LiveEvent(session_id="s", seq=9, type="turn_end", at="x",
                     stream_time=t)


def policy(**kw: object) -> TriggerPolicy:
    base = {"on": "final", "speaker_roles": ["customer"], "min_words": 3,
            "debounce_ms": 500}
    base.update(kw)
    return TriggerPolicy(Trigger.model_validate(base), ROLES)


def reason(decision: Optional[object]) -> str:
    assert isinstance(decision, Skip)
    return decision.reason


def test_fires_for_the_configured_role() -> None:
    decision = policy().decide(final("c", "tengo un problema", 5.0))
    assert isinstance(decision, Firing)
    assert decision.role == "customer" and decision.utterance_id == "t0.s0"


def test_other_roles_are_skipped() -> None:
    assert reason(policy().decide(final("a", "buenos días a todos", 1))) \
        == "role"


def test_unknown_speaker_has_unknown_role() -> None:
    assert reason(policy().decide(final("z", "una dos tres", 1))) == "role"


def test_short_sentences_are_skipped() -> None:
    assert reason(policy().decide(final("c", "sí claro", 1))) == "min_words"


def test_debounce_uses_stream_time() -> None:
    p = policy()
    assert isinstance(p.decide(final("c", "una dos tres", 1.0)), Firing)
    assert reason(p.decide(final("c", "cuatro cinco seis", 1.3))) \
        == "debounce"
    assert isinstance(p.decide(final("c", "siete ocho nueve", 1.6)), Firing)


def test_skipped_events_do_not_restart_the_debounce() -> None:
    p = policy()
    p.decide(final("a", "una dos tres", 1.0))
    assert isinstance(p.decide(final("c", "una dos tres", 1.1)), Firing)


def test_zero_debounce_never_skips() -> None:
    p = policy(debounce_ms=0)
    p.decide(final("c", "una dos tres", 1.0))
    assert isinstance(p.decide(final("c", "una dos tres", 1.0)), Firing)


def test_other_events_are_ignored() -> None:
    assert policy().decide(end(3.0)) is None
    partial = final("c", "una dos tres", 1.0).model_copy(
        update={"type": "partial"})
    assert policy().decide(partial) is None


def test_turn_end_trigger_joins_the_turn() -> None:
    p = policy(on="turn_end", min_words=4, debounce_ms=0)
    assert p.decide(final("c", "una dos", 1.0, 0)) is None
    assert p.decide(final("c", "tres cuatro", 2.0, 1)) is None
    decision = p.decide(end(2.5))
    assert isinstance(decision, Firing)
    assert decision.text == "una dos tres cuatro"
    assert decision.utterance_id == "t1.s0"
    assert reason(p.decide(end(3.0))) == "no_turn"


def test_turn_end_resets_per_speaker() -> None:
    p = policy(on="turn_end", min_words=1, debounce_ms=0)
    p.decide(final("a", "hola buenas", 1.0, 0))
    p.decide(final("c", "mi factura", 2.0, 1))
    decision = p.decide(end(3.0))
    assert isinstance(decision, Firing) and decision.text == "mi factura"
