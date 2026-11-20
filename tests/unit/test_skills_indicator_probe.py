"""IndicatorProbe: a judge asked about reference items (#85)."""

from typing import List

from src.application.skills.indicator_probe import IndicatorProbe
from src.domain.skills.judge_reference import ReferenceItem
from tests.fakes import FakeChatModel, make_catalogue

ITEM = ReferenceItem(
    id="x", offered=["listening", "planning"],
    context=["Ana: Ayer hablé con el cliente.", "Luis: Vale."],
    speaker="Ana", text="Entonces te mando los datos el lunes.",
    expect="P1", kind="true",
)


def probe(replies: List[str]) -> IndicatorProbe:
    """A probe whose judge answers ``replies``."""
    return IndicatorProbe(FakeChatModel(replies), make_catalogue())


def test_reversed_order_lists_competencies_and_indicators_backwards() -> None:
    """The reversed listing is the exact mirror of the forward one."""
    p = probe(["{}"])
    forward = [c.id for c in p.offered(ITEM, "forward")]
    backward = p.offered(ITEM, "reversed")
    assert forward == ["listening", "planning"]
    assert [c.id for c in backward] == ["planning", "listening"]
    assert [i.id for i in backward[0].indicators] == ["P2", "P1"]


def test_prompt_carries_context_and_target() -> None:
    """The prompt is the one of ``tag --confirm``."""
    judge = FakeChatModel(['{"indicator": null, "reason": "x"}'])
    IndicatorProbe(judge, make_catalogue()).ask(ITEM, "forward")
    _, user, _ = judge.calls[0]
    assert "Ana: Ayer hablé con el cliente." in user
    assert "TARGET sentence, said by Ana:\nEntonces" in user


def test_named_indicator_none_and_unreadable() -> None:
    """Each kind of reply becomes one outcome."""
    p = probe(['{"indicator": "P1"}', '{"indicator": null}', "no json",
               '{"indicator": "ZZ"}'])
    named = p.ask(ITEM, "forward")
    assert (named.named, named.parsed, named.first_listed) == (
        "P1", True, "L1")
    assert p.ask(ITEM, "forward").named is None
    assert p.ask(ITEM, "forward").parsed is False
    outside = p.ask(ITEM, "reversed")
    assert outside.outside and outside.named is None
    assert outside.first_listed == "P2"


def test_run_answers_every_item() -> None:
    """One outcome per item, in order."""
    p = probe(['{"indicator": "L1"}'])
    outcomes = p.run([ITEM, ITEM.model_copy(update={"id": "y"})], "forward")
    assert [o.item_id for o in outcomes] == ["x", "y"]
