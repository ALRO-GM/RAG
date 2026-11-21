"""Warm-up of the knowledge domains and of the live reasoner."""

from typing import Any, List, Optional

from src.application.knowledge.warmup import warm_domains
from src.application.reasoning import Reasoner
from src.domain.errors import NotFoundError
from src.domain.knowledge.contract import KnowledgeQuery
from src.domain.knowledge.domains import DomainSummary


def row(domain: str, version: Optional[str]) -> DomainSummary:
    """Return a summary row."""
    return DomainSummary(
        id=domain, title=domain, language="es", documents=1, chunks=2,
        index_version=version, updated_at=None,
    )


class FakeRegistry:
    """Lists fixed rows."""

    def summaries(self) -> List[DomainSummary]:
        """Return an indexed, a broken and an unindexed domain."""
        return [row("a", "v1"), row("broken", "v1"), row("empty", None)]


class FakeQueries:
    """Records queries; one domain fails."""

    def __init__(self) -> None:
        """Start empty."""
        self.seen: List[KnowledgeQuery] = []

    def run(self, query: KnowledgeQuery) -> Any:
        """Record ``query``; ``broken`` is not found."""
        self.seen.append(query)
        if query.domain == "broken":
            raise NotFoundError("no index")
        return None


def test_warm_domains_queries_indexed_domains_only() -> None:
    queries = FakeQueries()
    warmed = warm_domains(FakeRegistry(), queries)  # type: ignore[arg-type]
    assert warmed == ["a"]
    assert [q.domain for q in queries.seen] == ["a", "broken"]


class Warmable:
    """A chat model with a ``warm`` method."""

    model_name = "claude-code:haiku"

    def __init__(self) -> None:
        """Start without calls."""
        self.warmed: List[Any] = []

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Unused."""
        return ""

    def warm(self, system: str, cap: int) -> None:
        """Record the key."""
        self.warmed.append((system, cap))


class Cold:
    """A chat model that cannot be warmed."""

    model_name = "local"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Unused."""
        return ""


def test_reasoner_warms_the_template_key() -> None:
    model = Warmable()
    reasoner = Reasoner(lambda alias: model)
    assert reasoner.warm_up() is True
    system, cap = model.warmed[0]
    # suggester.support@2: the count and the mode are in the user prompt,
    # so one warm process serves every mode and count.
    assert "customer-support agent" in system and "Mode line" in system
    assert cap > 0


def test_reasoner_leaves_a_cold_backend_alone() -> None:
    reasoner = Reasoner(lambda alias: Cold())
    assert reasoner.warm_up() is False
