"""Suggester templates: grounding, support signals, interview never rates."""

import json
from typing import Any, Dict

import pytest

from src.application.reasoning import Reasoner
from src.domain.errors import InvalidInputError
from src.domain.reasoning.roles import ReasonRequest
from src.domain.reasoning.templates import get_template
from tests.fakes import FakeChatModel

WINDOW = [
    {"id": "t0.s0", "speaker": "S1", "text": "Cuénteme su experiencia."},
    {"id": "t1.s0", "speaker": "S2",
     "text": "Migré un monolito a servicios, pero estoy harto de esto."},
]


def request(template: str, roles: Dict[str, str],
            **extra: Any) -> ReasonRequest:
    """Return a suggester request over the fixed window."""
    data: Dict[str, Any] = {
        "request_id": "r-1", "role": "suggester", "template": template,
        "model": "haiku", "language": "es",
        "context": {"window": WINDOW, "speaker_roles": roles,
                    "script_phase": "technical"},
        "retrieved": [{"id": "kb:q1", "text": "¿Cómo gestionas pagos?"}],
    }
    data.update(extra)
    return ReasonRequest.model_validate(data)


INTERVIEW = request("suggester.interview@1",
                    {"S1": "interviewer", "S2": "candidate"})
SUPPORT = request("suggester.support@1", {"S1": "agent", "S2": "customer"})


def answer(request_: ReasonRequest, *items: Any) -> Any:
    """Run the reasoner over a model replying with ``items``."""
    model = FakeChatModel([json.dumps({"suggestions": list(items)})])
    return Reasoner(lambda alias: model).reason(request_), model


def item(**fields: Any) -> Dict[str, Any]:
    """Return one raw suggestion with sensible defaults."""
    base = {"kind": "follow_up", "text": "¿Y las transacciones?",
            "cites": ["kb:q1"], "about": ["t1.s0"]}
    base.update(fields)
    return base


def test_grounded_suggestion_passes() -> None:
    out, model = answer(INTERVIEW, item())
    assert out.status == "ok" and len(out.suggestions) == 1
    assert out.suggestions[0].cites == ["kb:q1"]
    system, user, cap = model.calls[0]
    assert cap <= 120 and "[t1.s0]" in user and "[kb:q1]" in user
    assert "Never evaluate" in system


@pytest.mark.parametrize("bad", [
    item(cites=["kb:made-up"]), item(about=["t9.s9"]),
    item(cites=["kb:q1", "kb:zz"]),
])
def test_suggestions_citing_unknown_ids_are_dropped(
    bad: Dict[str, Any]
) -> None:
    out, _ = answer(INTERVIEW, bad, item(text="buena"))
    assert [s.text for s in out.suggestions] == ["buena"]
    only_bad, _ = answer(INTERVIEW, bad)
    assert only_bad.status == "no_suggestion"


def test_bad_items_and_unparseable_replies_are_ignored() -> None:
    out, _ = answer(INTERVIEW, {"kind": "praise", "text": "x"},
                    {"kind": "question", "text": ""}, "junk")
    assert out.status == "no_suggestion"
    model = FakeChatModel(["no json at all"])
    assert Reasoner(lambda a: model).reason(INTERVIEW).status == (
        "no_suggestion"
    )


def test_max_suggestions_and_text_limits() -> None:
    capped = request("suggester.interview@1", {},
                     constraints={"max_suggestions": 1})
    out, model = answer(capped, item(text="a" * 500), item(text="b"))
    assert len(out.suggestions) == 1 and len(out.suggestions[0].text) == 300
    assert "at most 1" in model.calls[0][0]


def test_output_cap_never_above_the_templates() -> None:
    for name, cap in (("suggester.support@1", 200),
                      ("suggester.interview@1", 120)):
        wide = request(name, {}, constraints={"max_output_tokens": 900})
        _, model = answer(wide, item())
        assert model.calls[0][2] == cap


def test_support_prompt_asks_for_short_replies_without_rationale() -> None:
    system = get_template("suggester.support@1", "suggester").system(SUPPORT)
    assert "25 words" in system and "no rationale" in system


def test_support_reply_with_signals_is_grounded_and_kept() -> None:
    """The real Haiku reply shape that must come out as a suggestion."""
    out, _ = answer(SUPPORT, item(
        kind="question", signals=["stated_frustration", "complaint"],
        about=["t1.s0"],
    ))
    assert out.status == "ok"
    assert out.suggestions[0].signals == [
        "stated_frustration", "complaint"
    ]


def test_interview_never_returns_signals_or_ratings_adversarial() -> None:
    hostile = item(
        signals=["stated_frustration", "churn_risk"], rating=2, score=0.9,
        sentiment="angry", candidate_rating="weak",
        rationale="El candidato parece estresado",
    )
    out, _ = answer(INTERVIEW, hostile)
    suggestion = out.suggestions[0]
    assert suggestion.signals == []
    document = out.to_contract()
    text = json.dumps(document)
    for word in ("rating", "score", "sentiment", "candidate_rating"):
        assert word not in text
    assert all(s["signals"] == [] for s in document["suggestions"])


def test_support_returns_customer_signals_backed_by_customer_text() -> None:
    out, _ = answer(SUPPORT, item(
        signals=["stated_frustration", "made_up"], about=["t1.s0"],
    ))
    assert out.suggestions[0].signals == ["stated_frustration"]
    assert out.to_contract()["suggestions"][0]["signals"] == [
        "stated_frustration"
    ]


def test_support_signals_need_customer_text_and_a_quote() -> None:
    agent_line = item(signals=["complaint"], about=["t0.s0"])
    no_quote = item(signals=["complaint"], about=[])
    out, _ = answer(SUPPORT, agent_line, no_quote)
    assert [s.signals for s in out.suggestions] == [[], []]
    unknown_role, _ = answer(
        request("suggester.support@1", {}), item(signals=["urgency"])
    )
    assert unknown_role.suggestions[0].signals == []


def test_response_validates_against_the_contract_schema() -> None:
    from pathlib import Path
    from jsonschema import Draft202012Validator, FormatChecker
    schema = json.loads((Path(__file__).resolve().parents[2] / "contracts"
                         / "reasoner.response.v1.schema.json").read_text())
    out, _ = answer(SUPPORT, item(signals=["complaint"]))
    Draft202012Validator(
        schema, format_checker=FormatChecker()
    ).validate(out.to_contract())


def test_registry_knows_the_released_templates() -> None:
    for name in ("suggester.interview@1", "suggester.support@1"):
        assert get_template(name, "suggester").id == name
    with pytest.raises(InvalidInputError):
        get_template("suggester.interview@1", "conversational")


def test_interview_system_prompt_forbids_ratings_and_states() -> None:
    system = get_template("suggester.interview@1", "suggester").system(
        INTERVIEW
    )
    for phrase in ("rate", "emotions", "personality"):
        assert phrase in system
    assert "signals" not in system.split("Never evaluate")[1].split(
        "Reply with"
    )[0]
    assert '"signals"' not in system


def test_support_system_prompt_limits_signals_to_the_customers_words() -> None:
    system = get_template("suggester.support@1", "suggester").system(SUPPORT)
    assert "never from the voice" in system and "churn_risk" in system
    assert '"signals"' in system
