"""The «Ajustes» page's live choices: applied to a product when a
session starts, and read from the RAG over HTTP."""

from typing import Dict, List, Optional

import httpx

from orchestrator.adapters.outbound.http_settings import HttpSettings
from orchestrator.application.sessions import LiveSessions
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.live_choices import apply_live_choices
from tests.unit.orchestrator.fixtures import product
from tests.unit.orchestrator.test_sessions import (
    FakeEvents,
    FakeTranscriber,
    SignalReasoner,
    sessions,
)


def mono_support() -> ProductConfig:
    """The support product, attributed by mono diarization."""
    support = product()
    return support.model_copy(update={
        "attribution": support.attribution.model_copy(
            update={"kind": "sortformer_stream"}
        ),
    })


def test_nothing_saved_keeps_the_product() -> None:
    support = product()
    assert apply_live_choices(support, None) is support
    assert apply_live_choices(support, {}) is support
    weird = {"live.suggester_model": "gpt", "live.deadline": "9"}
    assert apply_live_choices(support, weird) is support


def test_model_deadline_and_diarization_are_replaced() -> None:
    mono = mono_support()
    chosen = apply_live_choices(mono, {
        "live.suggester_model": "sonnet", "live.deadline": "1800",
        "live.diarization": "off",
    })
    assert chosen.reasoner is not None
    assert (chosen.reasoner.model, chosen.reasoner.deadline_ms) == (
        "sonnet", 1800
    )
    assert chosen.attribution.kind == "single_speaker"
    kept = apply_live_choices(mono, {"live.deadline": "product",
                                     "live.diarization": "on"})
    assert kept.reasoner is not None and kept.reasoner.deadline_ms == 2800
    assert kept.attribution.kind == "sortformer_stream"


class Saved:
    """The settings source, in memory."""

    def __init__(self, choices: Optional[Dict[str, str]]) -> None:
        """Keep the choices."""
        self.choices = choices
        self.reads = 0

    def current(self) -> Optional[Dict[str, str]]:
        """Return the choices."""
        self.reads += 1
        return self.choices


def with_settings(live: LiveSessions, saved: Saved) -> LiveSessions:
    """``live`` reading ``saved`` at every start."""
    live.settings = saved
    return live


def test_a_session_reads_the_settings_when_it_starts() -> None:
    reasoner = SignalReasoner()
    saved = Saved({"live.suggester_model": "opus", "live.deadline": "2800"})
    live = with_settings(sessions(reasoner=reasoner), saved)
    live.products["live_assist.support"] = mono_support()
    live.start_replay("r-1", "live_assist.support", "t", None)
    assert reasoner.requests and all(
        r["model"] == "opus" and r["constraints"]["deadline_ms"] == 2800
        for r in reasoner.requests
    )
    assert saved.reads == 1
    transcriber = FakeTranscriber()
    live2 = with_settings(
        sessions(transcriber=transcriber, events=FakeEvents([])),
        Saved({"live.diarization": "off"}),
    )
    live2.products["live_assist.support"] = mono_support()
    live2.start_live("live_assist.support", "mic", "l-1")
    body = transcriber.bodies[0]
    assert "diarize" not in body and len(body["speakers"]) == 1


def test_the_adapter_reads_only_saved_choices() -> None:
    answers: List[httpx.Response] = [
        httpx.Response(200, json={"stored": True, "choices": {
            "live.deadline": "1800", "x": 1}}),
        httpx.Response(200, json={"stored": False, "choices": {}}),
        httpx.Response(200, json={"stored": True, "choices": [1]}),
        httpx.Response(200, json=[1]),
        httpx.Response(401, json={}),
        httpx.Response(200, content=b"not json"),
    ]
    seen: List[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answers.pop(0)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    source = HttpSettings("http://rag/", "tok", client=client)
    assert source.current() == {"live.deadline": "1800"}
    assert [source.current() for _ in range(5)] == [None] * 5
    assert str(seen[0].url) == "http://rag/settings"
    assert seen[0].headers["Authorization"] == "Bearer tok"
