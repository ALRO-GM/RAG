"""Reasoner adapter: ``POST /reason`` of the RAG service.

The endpoint is issue #157; until it exists this adapter has nothing to
talk to, and a replay uses a fake or an in-process reasoner instead.
"""

from typing import Optional

import httpx

from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerUnavailable,
)


class HttpReasoner:
    """Ask the RAG's reasoner over HTTP.

    Args:
        base_url: Root of the RAG service.
        token: Bearer token, if the service asks for one.
        timeout: Seconds to wait for an answer.
        client: An ``httpx.Client`` (replaced in tests).
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address and the client."""
        self.url = base_url.rstrip("/") + "/reason"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout
        self.client = client or httpx.Client()

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Post ``request``; an unreachable service is an exception.

        A call cancelled before it is sent is answered locally with
        ``status: cancelled`` and no network.
        """
        if cancel.cancelled:
            return {
                "schema": "reasoner.response.v1",
                "request_id": request["request_id"],
                "role": request["role"], "status": "cancelled",
                "model": request.get("model", ""), "backend": "local",
                "latency": {"total_ms": 0.0},
            }
        try:
            response = self.client.post(
                self.url, json=request, headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ReasonerUnavailable(type(exc).__name__) from exc
        if not isinstance(body, dict):
            raise ReasonerUnavailable("unreadable answer")
        return body
