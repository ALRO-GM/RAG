"""Templates of #286: the live tick call and the after-session attribution
also say which concepts the interviewer's questions asked.

Invented lines only.
"""

import json
from typing import Any, Dict

from src.application.reasoning import Reasoner
from src.domain.reasoning.roles import ReasonRequest
from src.domain.reasoning.templates import (
    ATTRIBUTION_V2_MAX_OUTPUT_TOKENS,
    TICKS_V2_MAX_OUTPUT_TOKENS,
    get_template,
)
from tests.fakes import FakeChatModel
from tests.unit.test_reasoning_attribution import POINTS, ROLES, WINDOW


def ticks(payload: Dict[str, Any]) -> ReasonRequest:
    """An ``interview.ticks@2`` request over the fixed window."""
    return ReasonRequest.model_validate({
        "request_id": "r", "role": "structured",
        "template": "interview.ticks@2", "model": "haiku",
        "language": "es", "constraints": {"deadline_ms": 20000},
        "context": {"window": WINDOW, "speaker_roles": ROLES,
                    "focus": "t1.s0"},
        "payload": {**payload, "answers": ["t1.s0"]},
    })


def run(req: ReasonRequest, reply: Any) -> Any:
    model = FakeChatModel([reply if isinstance(reply, str)
                           else json.dumps(reply)])
    return Reasoner(lambda alias: model).reason(req)


def test_the_tick_call_v2_asks_for_asked_concepts_and_stays_cold() -> None:
    template = get_template("interview.ticks@2", "structured")
    assert template.slow and template.yields
    assert template.max_output_tokens == TICKS_V2_MAX_OUTPUT_TOKENS
    system = template.system(ticks(POINTS))
    assert '"asked"' in system and "INTERVIEWER" in system
    assert "poor or missing answer still counts as asked" in system
    assert "Do not judge the answers or the person" in system
    # The user prompt is the same as @1's: the window has the questions.
    assert template.user(ticks(POINTS)) == get_template(
        "interview.ticks@1", "structured").user(ticks(POINTS))


def test_asked_concepts_are_listed_ones_backed_by_an_interviewer_line(
) -> None:
    out = run(ticks(POINTS), {
        "treated": [{"id": "s2.p1", "about": "t1.s0"}],
        "asked": [{"id": "s2.p1", "by": "t0.s0"},
                  {"id": "s2.p1", "by": "t0.s0"},     # repeated
                  {"id": "s2.p2", "by": "t1.s0"},     # the candidate's
                  {"id": "s9.p1", "by": "t0.s0"},     # not listed
                  {"id": " [s5.p2] ", "by": "[t0.s0]"}, "junk", 3]})
    assert out.status == "ok"
    assert out.data == {
        "points": [{"id": "s2.p1", "utterance": "t1.s0"}],
        "asked": [{"id": "s2.p1", "utterance": "t0.s0"},
                  {"id": "s5.p2", "utterance": "t0.s0"}]}


def test_a_malformed_v2_reply_asks_and_ticks_nothing() -> None:
    for reply in ("not json", {"asked": "s2.p1"},
                  {"asked": [{"id": {"x": 1}, "by": ["t0.s0"]}]}):
        out = run(ticks(POINTS), reply)
        assert out.status == "ok"
        assert out.data["asked"] == []


def test_the_attribution_v2_adds_asked_to_v1() -> None:
    v1 = get_template("interview.attribution@1", "structured")
    v2 = get_template("interview.attribution@2", "structured")
    request = ReasonRequest.model_validate({
        "request_id": "r", "role": "structured",
        "template": "interview.attribution@2", "model": "haiku",
        "language": "es", "constraints": {"deadline_ms": 90000},
        "context": {"window": []},
        "payload": {"title": "Tienda", "slots": [], "round": []},
    })
    system = v2.system(request)
    assert system != v1.system(request)
    assert '"asked": [{"id": "concept id", "by": "interviewer utterance ' \
        'id"}]' in system
    assert "whatever the answer was" in system
    assert "the interviewer's for asked" in system
    assert v2.user(request) == v1.user(request)
    assert v2.slow and v2.max_output_tokens == \
        ATTRIBUTION_V2_MAX_OUTPUT_TOKENS
