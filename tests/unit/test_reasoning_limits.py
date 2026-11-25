"""Every reasoner deadline fits the path that serves it, end to end (#267).

Orchestrator client time-out → RAG ``/reason`` → model backend: a slow
template (planner, report) runs on the cold backend with its deadline up
to ``SLOW_MAX_S``; a live one on the warm backend, whose own time-out
caps it. This test fails when a template's deadline exceeds what its
serving path allows, or a client gives up before the RAG can answer.
"""

import inspect

from orchestrator.adapters.outbound.http_reasoner import HttpReasoner
from orchestrator.application.interview import (
    CLIENT_MARGIN_S,
    FILLER_DEADLINE_MS,
    PLANNER_DEADLINE_MS,
    PLANNER_TIMEOUT_S,
    REPORT_DEADLINE_MS,
    REPORT_TIMEOUT_S,
)
from orchestrator.domain.config import InterviewSection
from src.adapters.outbound.claude_code_stream import ClaudeCodeStream
from src.config import SkillsSettings
from src.domain.reasoning.templates import SLOW_MAX_S, get_template

WARM_TIMEOUT_S = inspect.signature(ClaudeCodeStream).parameters[
    "timeout"].default
CLIENT_DEFAULT_S = inspect.signature(HttpReasoner).parameters[
    "timeout"].default


def test_slow_templates_fit_the_cold_path_and_the_client_waits_for_them(
) -> None:
    section = InterviewSection()
    for template, deadline_ms, client_s in (
        (section.planner_template, PLANNER_DEADLINE_MS, PLANNER_TIMEOUT_S),
        (section.report_template, REPORT_DEADLINE_MS, REPORT_TIMEOUT_S),
    ):
        assert get_template(template, "structured").slow, template
        assert deadline_ms / 1000.0 <= SLOW_MAX_S, template
        cold_s = SkillsSettings().claude_timeout
        assert deadline_ms / 1000.0 <= cold_s, template
        assert client_s >= deadline_ms / 1000.0 + CLIENT_MARGIN_S
        assert client_s >= deadline_ms / 1000.0 + 20.0


def test_live_templates_fit_the_warm_path() -> None:
    section = InterviewSection()
    assert not get_template(section.filler_template, "structured").slow
    assert FILLER_DEADLINE_MS / 1000.0 <= WARM_TIMEOUT_S
    for product_deadline_ms in (1800, 2800, 4000):  # config/products
        assert product_deadline_ms / 1000.0 <= WARM_TIMEOUT_S
        assert product_deadline_ms / 1000.0 <= CLIENT_DEFAULT_S
