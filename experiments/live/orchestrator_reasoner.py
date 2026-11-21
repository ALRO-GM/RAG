"""In-process reasoner for ``python -m orchestrator replay``.

The orchestrator may not import ``src``, so this glue (outside both
packages) wraps the RAG's :class:`~src.application.reasoning.Reasoner`
and the ``claude -p`` chat model as the orchestrator's reasoner port,
until ``POST /reason`` exists (issue #157)::

    python -m orchestrator replay T.json --product live_assist.support \\
        --reasoner experiments.live.orchestrator_reasoner:build
"""

from typing import Any, Dict

from src.adapters.outbound.claude_code_judge import ClaudeCodeJudge
from src.application.reasoning import Reasoner
from src.domain.reasoning.roles import ReasonRequest


class InProcessReasoner:
    """Answer ``reasoner.request.v1`` documents with the real Reasoner."""

    def __init__(self, timeout: float = 30.0):
        """Build the Reasoner over Claude Code (one process per call)."""
        self.reasoner = Reasoner(
            lambda alias: ClaudeCodeJudge(
                alias or "haiku", timeout=timeout, retries=0
            )
        )

    def reason(self, request: Dict[str, Any], cancel: Any) -> Dict[str, Any]:
        """Run the request; the answer is the response document."""
        if getattr(cancel, "cancelled", False):
            return {
                "schema": "reasoner.response.v1",
                "request_id": request["request_id"],
                "role": request["role"], "status": "cancelled",
                "model": request.get("model", ""), "backend": "claude-code",
                "latency": {"total_ms": 0.0},
            }
        parsed = ReasonRequest.model_validate(request)
        return self.reasoner.reason(parsed).to_contract()


def build() -> InProcessReasoner:
    """Return the reasoner port (the ``--reasoner`` factory)."""
    return InProcessReasoner()
