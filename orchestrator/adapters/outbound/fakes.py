"""In-process fakes of the knowledge and reasoner ports.

They let a replay run end to end with no service: deterministic, no
network, no model. The fake reasoner answers with a follow-up question
about the focus utterance, citing it in ``about`` (and the first
retrieved chunk in ``cites``), and can sleep to imitate a service's
latency.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from orchestrator.application.ports import (
    CancelToken,
    Document,
)


class FakeKnowledge:
    """A knowledge service with a fixed list of chunks.

    Args:
        chunks: ``(id, text)`` pairs; a chunk is returned when it shares
            a word of four letters or more with the query (all of them
            when ``everything`` is set).
        everything: Return every chunk whatever the query.
        delay: Seconds to wait per query.
        sleep: Waits (replaced in tests).
    """

    def __init__(
        self,
        chunks: Optional[List[Dict[str, str]]] = None,
        everything: bool = False,
        delay: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Keep the chunks."""
        self.chunks = chunks or []
        self.everything = everything
        self.delay = delay
        self._sleep = sleep
        self.queries: List[Document] = []

    def query(self, query: Document) -> Document:
        """Return the chunks that match ``query['text']``."""
        self.queries.append(query)
        if self.delay:
            self._sleep(self.delay)
        words = {w.lower() for w in query["text"].split() if len(w) >= 4}
        found = [
            c for c in self.chunks
            if self.everything
            or words & {w.lower() for w in c["text"].split()}
        ]
        top_k = int(query.get("top_k", 5))
        return {
            "schema": "knowledge.result.v1",
            "request_id": query["request_id"], "domain": query["domain"],
            "index_version": "fake",
            "chunks": [
                {
                    "id": c["id"], "text": c["text"], "score": 1.0,
                    "source": {"file_path": "fake.md",
                               "first_character_index": 0,
                               "last_character_index": len(c["text"])},
                }
                for c in found[:top_k]
            ],
        }


class FakeReasoner:
    """A reasoner that echoes a follow-up question about the focus.

    Args:
        delay: Seconds to wait per call.
        status: The ``status`` to answer with (``ok`` by default).
        sleep: Waits (replaced in tests).
    """

    def __init__(
        self,
        delay: float = 0.0,
        status: str = "ok",
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Keep the options."""
        self.delay = delay
        self.status = status
        self._sleep = sleep
        self.requests: List[Document] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Answer ``request`` without a model."""
        self.requests.append(request)
        if self.delay:
            self._sleep(self.delay)
        window = request["context"]["window"]
        focus = request["context"].get("focus")
        said = next((w["text"] for w in window if w["id"] == focus), "")
        suggestions: List[Dict[str, Any]] = []
        if self.status == "ok":
            cites = [r["id"] for r in request.get("retrieved", [])[:1]]
            suggestions = [{
                "kind": "follow_up",
                "text": "¿Puedes contar más sobre: " + " ".join(
                    said.split()[:8]) + "?",
                "cites": cites, "about": [focus] if focus else [],
            }]
        return {
            "schema": "reasoner.response.v1",
            "request_id": request["request_id"], "role": request["role"],
            "status": self.status, "suggestions": suggestions,
            "model": "fake", "backend": "local",
            "latency": {"total_ms": self.delay * 1000.0},
        }
