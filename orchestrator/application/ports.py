"""Ports of the orchestrator: what it needs from the outside.

Knowledge and reasoner are reached with the JSON documents of their
contracts (``knowledge.query.v1`` / ``knowledge.result.v1``,
``reasoner.request.v1`` / ``reasoner.response.v1``), so no adapter has
to import the services' own code.
"""

from typing import Any, Callable, Dict, Protocol

from orchestrator.domain.records import SuggestionRecord

Document = Dict[str, Any]
"""A contract document as plain JSON."""


class CancelToken:
    """A flag a caller sets to say its answer is no longer wanted."""

    def __init__(self) -> None:
        """Start not cancelled."""
        self.cancelled = False

    def cancel(self) -> None:
        """Mark the call as stale."""
        self.cancelled = True


class KnowledgePort(Protocol):
    """Per-domain retrieval."""

    def query(self, query: Document) -> Document:
        """Answer a ``knowledge.query.v1`` with a ``knowledge.result.v1``.

        Raises:
            KnowledgeUnavailable: The service could not answer.
        """
        ...


class ReasonerPort(Protocol):
    """The reasoner service."""

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Answer a ``reasoner.request.v1`` with a ``reasoner.response.v1``.

        A model failure is a ``status`` of the response, not an
        exception; an adapter checks ``cancel`` before it sends.

        Raises:
            ReasonerUnavailable: The service could not be reached.
        """
        ...


class Clock(Protocol):
    """Monotonic time and waiting (replaced by a fake in tests)."""

    def now(self) -> float:
        """Return monotonic seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""
        ...


class Executor(Protocol):
    """Runs a call apart from the event loop."""

    def submit(self, job: Callable[[], None]) -> None:
        """Run ``job`` (inline or on a thread)."""
        ...

    def wait_idle(self) -> None:
        """Return when every submitted job has finished."""
        ...


class SuggestionSink(Protocol):
    """Where the suggestion records go (a JSONL file, a UI)."""

    def write(self, record: SuggestionRecord) -> None:
        """Store one record."""
        ...


class KnowledgeUnavailable(Exception):
    """The knowledge service did not answer."""


class ReasonerUnavailable(Exception):
    """The reasoner service could not be reached."""
