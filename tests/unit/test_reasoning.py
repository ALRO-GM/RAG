"""Reasoner use case: roles, status mapping, usage per role, judge parity."""

import time
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


def test_output_cap_overrun_is_invalid_output() -> None:
    from src.domain.errors import OutputTooLongError
    out = reasoner(Raising(OutputTooLongError("too long"))).reason(request())
    assert out.status == "error"
    assert out.error is not None and out.error.kind == "invalid_output"


class Cancellable(FakeChatModel):
    """A fake backend that can give up after a deadline."""

    def __init__(self) -> None:
        """Record the deadlines it was given."""
        super().__init__()
        self.deadlines: List[float] = []

    def complete_within(self, system: str, user: str, max_tokens: int,
                        deadline_s: float) -> str:
        """Give up: the deadline is too short for it."""
        self.deadlines.append(deadline_s)
        raise ModelTimeoutError("cancelled")


class Slow(FakeChatModel):
    """A backend that cannot cancel and answers after the deadline."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer after 20 ms."""
        time.sleep(0.02)
        return super().complete(system, user, max_tokens)


def test_deadline_goes_to_a_cancellable_backend_as_timeout() -> None:
    model = Cancellable()
    out = reasoner(model).reason(request(
        constraints={"deadline_ms": 1800}
    ))
    assert model.deadlines == [1.8]
    assert out.status == "timeout" and out.error is not None
    assert out.error.kind == "timeout" and not model.calls


def test_late_answer_of_a_backend_that_cannot_cancel_is_a_timeout() -> None:
    out = Reasoner(lambda alias: Slow(['{"text": "ok"}'])).reason(
        request(constraints={"deadline_ms": 1})
    )
    assert out.status == "timeout" and out.reply is None


def test_live_roles_use_the_live_factory_and_the_judge_the_other() -> None:
    live, cold = FakeChatModel(['{"text": "hola"}']), FakeChatModel(["x"])
    r = Reasoner(lambda alias: cold, live_models=lambda alias: live)
    assert r.reason(request()).reply is not None
    assert live.calls and not cold.calls
    r.reason(request(role="judge", payload={"system": "s", "user": "u"}))
    assert cold.calls


class Streaming(FakeChatModel):
    """A backend that writes its reply in pieces."""

    def complete_streaming(self, system: str, user: str, max_tokens: int,
                           deadline_s: Optional[float],
                           on_text: Any) -> str:
        """Pass the reply on in three-character pieces."""
        text = self.complete(system, user, max_tokens)
        for at in range(0, len(text), 3):
            on_text(text[at:at + 3])
        return text


def test_a_streaming_backend_passes_each_piece_on() -> None:
    reply = '{"text": "Abrimos de 9 a 18.", "cites": ["kb:1"]}'
    pieces: List[str] = []
    out = reasoner(Streaming([reply])).reason(request(), pieces.append)
    assert len(pieces) > 5 and "".join(pieces) == reply
    assert out.reply is not None and out.reply.text == "Abrimos de 9 a 18."


def test_a_backend_that_does_not_stream_passes_the_whole_reply() -> None:
    pieces: List[str] = []
    out = reasoner(FakeChatModel(['{"text": "Hola."}'])).reason(
        request(), pieces.append)
    assert pieces == ['{"text": "Hola."}'] and out.status == "ok"


def test_conversational_templates_warm_their_own_pool() -> None:
    warmed: List[Any] = []

    class Warm(FakeChatModel):
        def warm(self, system: str, max_tokens: int) -> None:
            warmed.append((system, max_tokens))

    r = Reasoner(lambda alias: Warm())
    assert r.warm_up("conversational.voicebot@1")
    assert r.warm_up()
    assert "voice assistant" in warmed[0][0]
    assert "voice assistant" not in warmed[1][0]


def interview_request(template: str, role: str = "structured",
                      **payload: Any) -> ReasonRequest:
    """Return a request of the AI interviewer's templates (#208)."""
    return request(role, template=template, payload=payload)


def test_the_planner_returns_its_json_as_data() -> None:
    model = FakeChatModel(['Aquí va: {"title": "Mozo", "slots": []} fin'])
    out = reasoner(model).reason(interview_request(
        "interview.planner@1", offer="Oferta de mozo de almacén",
        minutes=20, competencies=[{"id": "COMM", "name": "Comunicación"}],
        role_type="almacen"))
    assert out.status == "ok" and out.data == {"title": "Mozo", "slots": []}
    assert out.reply is None and out.suggestions == []
    system, user, cap = model.calls[0]
    assert "weight" in system and "Never make a slot" in system
    assert "- COMM: Comunicación" in user and "almacen" in user
    assert "Oferta de mozo" in user and cap == 2400
    assert out.to_contract()["data"] == {"title": "Mozo", "slots": []}


def test_structured_output_that_is_not_json_is_no_suggestion() -> None:
    out = reasoner(FakeChatModel(["no sé"])).reason(interview_request(
        "interview.slots@1", slots=[{"id": "s1", "label": "Caja"}],
        target="s1", question="¿Dónde?", answer="En un súper"))
    assert out.status == "no_suggestion" and out.data is None


def test_the_slot_filler_and_report_prompts() -> None:
    model = FakeChatModel(['{"slots": []}', '{"slots": [], "summary": ""}'])
    r = reasoner(model)
    r.reason(interview_request(
        "interview.slots@1", slots=[{"id": "s1", "label": "Caja",
                                     "evidence": "años"}, "junk"],
        target="s1", question="¿Dónde?", answer="En un súper"))
    r.reason(interview_request(
        "interview.report@1", title="Cajero", summary="Súper",
        slots=[{"id": "s1", "label": "Caja", "level": "minimo",
                "status": "filled", "quotes": ["dos años", 3]}, 7],
        other="También sé inventario"))
    slots_user = model.calls[0][1]
    assert "[s1] Caja — evidence: años" in slots_user
    assert "Candidate: En un súper" in slots_user
    report_system, report_user, cap = model.calls[1]
    assert "RELATIVE TO WHAT THIS OFFER REQUIRES" in report_system
    assert "quotes: dos años" in report_user and cap == 2400
    assert "También sé inventario" in report_user


def test_the_interviewer_words_one_step() -> None:
    model = FakeChatModel(['{"text": "Gracias. ¿Dónde trabajaste?"}'])
    out = reasoner(model).reason(request(
        "conversational", template="interviewer@1", payload={
            "title": "Cajero", "summary": "Súper", "minutes": 20,
            "minutes_left": 12.5, "action": "probe", "topic": "Caja",
            "questions": ["¿Dónde?", 3], "evidence": "años",
            "missing": "un dato concreto"}))
    assert out.reply is not None
    assert out.reply.text == "Gracias. ¿Dónde trabajaste?"
    system, user, cap = model.calls[0]
    assert "Never give a score" in system and "clarify" in system
    assert "ACTION: probe" in user and "Missing: un dato concreto" in user
    assert "12.5 left" in user and cap == 160


def test_the_interviewer_words_a_hint_with_the_answer_so_far() -> None:
    model = FakeChatModel(['{"text": "¿Te referías a la caja?"}'])
    reasoner(model).reason(request(
        "conversational", template="interviewer@1", payload={
            "title": "Cajero", "action": "hint", "topic": "Caja",
            "answer_so_far": "Pues en mi último trabajo, eh"}))
    system, user, _ = model.calls[0]
    assert "reask:" in system and "hint:" in system
    assert "ACTION: hint" in user
    assert "Answer so far: Pues en mi último trabajo, eh" in user


def test_the_interviewer_templates_warm_their_pools() -> None:
    warmed: List[Any] = []

    class Warm(FakeChatModel):
        def warm(self, system: str, max_tokens: int) -> None:
            warmed.append((system, max_tokens))

    r = Reasoner(lambda alias: Warm())
    assert r.warm_up("interviewer@1") and r.warm_up("interview.slots@1")
    assert warmed[0][1] == 160 and warmed[1][1] == 160
    with pytest.raises(InvalidInputError):
        r.warm_up("nope@1")


def test_spoken_template_v2_keeps_v1_and_never_mentions_fragments() -> None:
    from src.domain.reasoning.templates import get_template
    v1 = get_template("conversational.voicebot@1", "conversational")
    v2 = get_template("conversational.voicebot@2", "conversational")
    req = request(template="conversational.voicebot@2")
    system = v2.system(req)
    assert v1.system(req) != system and v2.user(req) == v1.user(req)
    assert "never mention" in system and "te paso con" in system
    assert "no está en los fragmentos" not in system
    reply = '{"text": "Abrimos a las nueve.", "cites": ["kb:1"]}'
    parsed = v2.parse(reply, req)
    assert parsed.reply is not None and parsed.reply.cites == ["kb:1"]


# ---- concepts per point (#258) ------------------------------------------

def test_the_planner_v2_asks_for_points_and_v1_stays_as_it_was() -> None:
    model = FakeChatModel(['{"slots": []}', '{"slots": []}'])
    r = reasoner(model)
    for template in ("interview.planner@1", "interview.planner@2"):
        r.reason(interview_request(template, offer="Oferta de dependiente",
                                   minutes=20, competencies=[]))
    (v1, user1, cap1), (v2, user2, cap2) = model.calls
    assert "points" not in v1 and '"weight": 3}]}' in v1
    assert '"points": ["short concrete concept"]' in v2
    assert "tipos de pisada" in v2 and "Never a concept about" in v2
    assert v2.endswith("No other field, no explanation.")
    assert user1 == user2 and (cap1, cap2) == (2400, 3200)


def test_the_slot_filler_v2_lists_each_slot_s_concepts() -> None:
    model = FakeChatModel(['{"slots": []}'])
    reasoner(model).reason(interview_request(
        "interview.slots@2", slots=[
            {"id": "s1", "label": "Running", "evidence": "-",
             "points": [{"id": "p1", "text": "Tipos de pisada"}, "junk"]},
            {"id": "s2", "label": "Caja", "points": "nope"}, 5],
        target="s1", question="¿Zapatillas?", answer="Sé de pisada"))
    system, user, cap = model.calls[0]
    assert "treated|unknown" in system and "do not know it" in system
    assert "Do not judge how good or correct" in system
    assert "[s1] Running — evidence: -\n  concept [p1] Tipos de pisada" \
        in user
    assert "[s2] Caja" in user and "Asked about: s1" in user
    assert cap == 320


def test_the_interviewer_v2_names_the_concepts_of_the_step() -> None:
    model = FakeChatModel(['{"text": "Me has hablado de A. ¿Y de B?"}'] * 2)
    r = reasoner(model)
    out = r.reason(request(
        "conversational", template="interviewer@2", payload={
            "title": "Dependiente", "action": "probe", "topic": "Running",
            "missing": "lo que aún no ha mencionado: B y C",
            "treated_points": ["Tipos de pisada", 4],
            "pending_points": ["Drop", "Talla", "Plantillas", "Extra"]}))
    assert out.reply is not None
    system, user, cap = model.calls[0]
    assert "Me has hablado de A. ¿Y de B y C?" in system
    assert "Never say they are missing" in system and cap == 160
    assert "Concepts already mentioned: Tipos de pisada\n" in user
    assert "Concepts to ask about now: Drop | Talla | Plantillas\n" in user
    assert user.index("Concepts to ask") < user.index("Knowledge:")
    r.reason(request("conversational", template="interviewer@2",
                     payload={"action": "ask", "topic": "Caja"}))
    assert "Concepts" not in model.calls[1][1]
    assert model.calls[1][0] == system  # one warm pool for every step
