"""Reasoner use case: roles, status mapping, usage per role, judge parity."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.application.reasoning import MeteredModel, Reasoner, backend_of
from src.application.skills.tagging import TagConfirmer
from src.domain.errors import (
    InvalidInputError,
    ModelAuthError,
    ModelError,
    ModelTimeoutError,
    QuotaExhaustedError,
)
from src.domain.reasoning.roles import Latency, ReasonRequest
from src.domain.skills.usage import Usage, UsageLedger
from tests.fakes import FakeChatModel, make_catalogue
from tests.unit.test_skills_tagging import answer, candidates


def request(role: str = "conversational", **extra: object) -> ReasonRequest:
    """Return a minimal request of ``role``."""
    data: Dict[str, Any] = {
        "request_id": "r-1", "role": role,
        "template": "conversational.voicebot@1", "model": "haiku",
        "context": {"window": [
            {"id": "t0.s0", "speaker": "S1", "text": "Hola, quiero ayuda."},
        ]},
        "retrieved": [{"id": "kb:1", "text": "Abrimos de 9 a 18."}],
    }
    data.update(extra)
    return ReasonRequest.model_validate(data)


class Metered(FakeChatModel):
    """A fake that reports tokens and a latency breakdown."""

    last_usage: Optional[Usage] = None
    last_latency: Optional[Latency] = None

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer and report usage and latency."""
        self.last_usage = Usage(
            calls=1, input_tokens=10, output_tokens=5, cost_usd=0.001
        )
        self.last_latency = Latency(
            process_start_ms=3.0, first_token_ms=40.0, total_ms=90.0
        )
        return super().complete(system, user, max_tokens)


class Raising(FakeChatModel):
    """A fake whose call fails with ``error``."""

    def __init__(self, error: Exception):
        """Keep the error to raise."""
        super().__init__()
        self.error = error

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Raise."""
        raise self.error


def reasoner(model: FakeChatModel, ledger: Optional[UsageLedger] = None,
             seen: Optional[List[str]] = None) -> Reasoner:
    """Return a reasoner serving ``model`` for every alias."""
    def make(alias: str) -> FakeChatModel:
        if seen is not None:
            seen.append(alias)
        return model
    return Reasoner(make, ledger)


def test_conversational_reply_is_parsed_and_grounded() -> None:
    model = Metered(
        ['{"text": "Abrimos de 9 a 18.", "cites": ["kb:1", "zz"]}'],
        model_name="claude-code:haiku",
    )
    seen: List[str] = []
    ledger = UsageLedger()
    out = reasoner(model, ledger, seen).reason(request())
    assert out.status == "ok" and seen == ["haiku"]
    assert out.reply is not None and out.reply.cites == ["kb:1"]
    assert out.backend == "claude-code" and out.model == "claude-code:haiku"
    assert out.usage.output_tokens == 5
    assert out.latency.first_token_ms == 40.0 and out.latency.total_ms == 90.0
    assert "claude-code:haiku#conversational" in ledger.snapshot().models
    assert model.calls[0][2] == 120


def test_output_cap_follows_constraints() -> None:
    model = FakeChatModel(['{"text": "ok"}'])
    reasoner(model).reason(
        request(constraints={"max_output_tokens": 50})
    )
    assert model.calls[0][2] == 50


def test_unparsed_text_is_taken_as_the_reply_and_empty_is_none() -> None:
    assert reasoner(FakeChatModel(["Claro"])).reason(
        request()
    ).reply is not None
    out = reasoner(FakeChatModel(["  "])).reason(request())
    assert out.status == "no_suggestion"


@pytest.mark.parametrize("error,status,kind", [
    (QuotaExhaustedError("limit", "5pm", datetime(
        2026, 1, 1, tzinfo=timezone.utc)), "quota", "quota"),
    (ModelAuthError("login"), "auth", "auth"),
    (ModelTimeoutError("slow"), "timeout", "timeout"),
    (ModelError("boom"), "error", "transient"),
])
def test_model_failures_become_a_status(
    error: Exception, status: str, kind: str
) -> None:
    out = reasoner(Raising(error)).reason(request())
    assert out.status == status
    assert out.error is not None and out.error.kind == kind
    assert out.suggestions == [] and out.reply is None


def test_quota_error_carries_reset_moment() -> None:
    out = reasoner(Raising(QuotaExhaustedError(
        "limit", "", datetime(2026, 1, 1, tzinfo=timezone.utc)
    ))).reason(request())
    assert out.error is not None
    assert out.error.resets_at == "2026-01-01T00:00:00+00:00"


def test_unknown_template_or_wrong_role_is_the_callers_mistake() -> None:
    with pytest.raises(InvalidInputError):
        reasoner(FakeChatModel()).reason(request(template="nope@1"))
    with pytest.raises(InvalidInputError):
        reasoner(FakeChatModel()).reason(request(role="suggester"))


def test_judge_role_needs_a_prompt_and_returns_the_reply() -> None:
    model = FakeChatModel(["{\"indicator\": null}"])
    out = reasoner(model).reason(request(
        "judge", payload={"system": "S", "user": "U"},
    ))
    assert out.verdict == {"reply": "{\"indicator\": null}"}
    assert model.calls == [("S", "U", 120)]
    with pytest.raises(InvalidInputError):
        reasoner(model).reason(request("judge"))


def test_response_without_backend_latency_uses_the_wall_clock() -> None:
    ticks = iter([1.0, 1.25])
    model = FakeChatModel(['{"text": "ok"}'])
    out = Reasoner(lambda alias: model, clock=lambda: next(ticks)).reason(
        request()
    )
    assert out.latency.total_ms == 250.0 and out.backend == "local"


def test_backend_names() -> None:
    assert backend_of("anthropic:x") == "anthropic-api"
    assert backend_of("opencode:x") == "opencode"
    assert backend_of("qwen") == "local"


def test_metered_model_records_failed_calls_too() -> None:
    ledger = UsageLedger()
    metered = MeteredModel(Raising(ModelError("x")), "judge", ledger)
    with pytest.raises(ModelError):
        metered.complete("s", "u", 1)
    assert metered.calls == 1
    assert ledger.snapshot().models["fake-judge#judge"].calls == 1


def test_judge_confirm_is_identical_to_the_existing_confirm_path() -> None:
    """Regression: the judge role is TagConfirmer, output unchanged."""
    catalogue = make_catalogue()

    def reply(system: str, user: str) -> str:
        return answer("L1" if "nonsense" in user else None)

    direct = TagConfirmer(FakeChatModel(responder=reply)).confirm(
        candidates(), catalogue, "es"
    )
    ledger = UsageLedger()
    via = reasoner(FakeChatModel(responder=reply), ledger).confirm(
        candidates(), catalogue, "es"
    )
    assert [t.model_dump() for t in via] == [t.model_dump() for t in direct]
    assert ledger.snapshot().models["fake-judge#judge"].calls > 0
