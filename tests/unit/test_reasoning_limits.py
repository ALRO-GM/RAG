"""Every reasoner deadline fits the path that serves it, end to end (#267).

Browser → transcriber proxy → orchestrator HTTP client → RAG ``/reason``
→ model backend: a slow template (planner, report) runs on the cold
backend with its deadline up to ``SLOW_MAX_S``; a live one on the warm
backend, whose own time-out caps it. This test fails when a template's
deadline exceeds what its serving path allows, or when any hop gives up
before the next one can answer.
"""

import inspect
from pathlib import Path
from typing import Any, Callable

from orchestrator.adapters.outbound.files import load_product
from orchestrator.adapters.outbound.http_interview import HttpOffers
from orchestrator.adapters.outbound.http_knowledge import HttpKnowledge
from orchestrator.adapters.outbound.http_reasoner import HttpReasoner
from orchestrator.application.interview import (
    CLIENT_MARGIN_S,
    FILLER_DEADLINE_MS,
    PLANNER_DEADLINE_MS,
    PLANNER_TIMEOUT_S,
    REPORT_DEADLINE_MS,
    REPORT_TIMEOUT_S,
)
from orchestrator.application.live_judgement import (
    ATTRIBUTION_DEADLINE_MS,
    ATTRIBUTION_TEMPLATE,
)
from orchestrator.domain.config import InterviewSection
from src.adapters.outbound.claude_code_stream import ClaudeCodeStream
from src.config import SkillsSettings
from src.domain.reasoning.templates import SLOW_MAX_S, get_template
from transcriber.adapters.inbound.live_assist_proxy import (
    PLAN_TIMEOUT,
    TIMEOUT,
)

PRODUCTS = Path(__file__).resolve().parents[2] / "config" / "products"


def default(build: Callable[..., Any]) -> float:
    """The default ``timeout`` of an adapter's constructor."""
    value = inspect.signature(build).parameters["timeout"].default
    assert isinstance(value, (int, float))
    return float(value)


WARM_TIMEOUT_S = default(ClaudeCodeStream)
CLIENT_DEFAULT_S = default(HttpReasoner)


def test_slow_templates_fit_the_cold_path_and_every_hop_waits_for_them(
) -> None:
    section = InterviewSection()
    cold_s = SkillsSettings().claude_timeout
    for template, deadline_ms, client_s in (
        (section.planner_template, PLANNER_DEADLINE_MS, PLANNER_TIMEOUT_S),
        (section.report_template, REPORT_DEADLINE_MS, REPORT_TIMEOUT_S),
        # #268: the live round's attribution goes through the report's
        # client (the live judgement's reasoner).
        (ATTRIBUTION_TEMPLATE, ATTRIBUTION_DEADLINE_MS, REPORT_TIMEOUT_S),
    ):
        assert get_template(template, "structured").slow, template
        assert deadline_ms / 1000.0 <= min(SLOW_MAX_S, cold_s), template
        assert client_s >= deadline_ms / 1000.0 + CLIENT_MARGIN_S
    # The page's «Preparar el plan» waits in the transcriber proxy while
    # the orchestrator reads the offer, queries the corpus and waits for
    # the planner, one after another.
    assert PLAN_TIMEOUT >= (default(HttpOffers) + default(HttpKnowledge)
                            + PLANNER_TIMEOUT_S)
    # The report and the live judgement never hold a proxied request:
    # they run when a session ends, and «Reintentar» answers 202 at once.
    assert TIMEOUT < REPORT_TIMEOUT_S


def test_live_templates_fit_the_warm_path() -> None:
    section = InterviewSection()
    assert not get_template(section.filler_template, "structured").slow
    assert FILLER_DEADLINE_MS / 1000.0 <= WARM_TIMEOUT_S
    products = sorted(PRODUCTS.glob("*.yaml"))
    assert products
    checked = 0
    for path in products:
        reasoner = load_product(path).reasoner
        if reasoner is None or reasoner.deadline_ms is None:
            continue
        assert not get_template(reasoner.template,  # type: ignore
                                reasoner.role).slow, path.name
        assert reasoner.deadline_ms / 1000.0 <= WARM_TIMEOUT_S, path.name
        assert reasoner.deadline_ms / 1000.0 <= CLIENT_DEFAULT_S, path.name
        checked += 1
    assert checked >= 4
