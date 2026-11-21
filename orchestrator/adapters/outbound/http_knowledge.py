"""Knowledge adapter: ``POST /knowledge/query`` of the RAG service."""

from typing import Optional

import httpx

from orchestrator.application.ports import Document, KnowledgeUnavailable


class HttpKnowledge:
    """Query the RAG's knowledge API over HTTP.

    Args:
        base_url: Root of the RAG service (``http://127.0.0.1:8000``).
        token: Bearer token, if the service asks for one.
        timeout: Seconds to wait for an answer.
        client: An ``httpx.Client`` (replaced in tests).
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 2.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address and the client."""
        self.url = base_url.rstrip("/") + "/knowledge/query"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout
        self.client = client or httpx.Client()

    def query(self, query: Document) -> Document:
        """Post ``query``; any failure is :class:`KnowledgeUnavailable`."""
        try:
            response = self.client.post(
                self.url, json=query, headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise KnowledgeUnavailable(type(exc).__name__) from exc
        if not isinstance(body, dict):
            raise KnowledgeUnavailable("unreadable answer")
        return body
