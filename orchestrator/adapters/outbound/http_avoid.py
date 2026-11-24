"""Questions better avoided: the RAG's ``/knowledge/interview/avoid`` (LV-17).

``POST /knowledge/interview/avoid`` ``{"text", "language"}`` answers the
categories of protected questions a sentence touches, with why and a
lawful alternative (KN-4's rules, read by the RAG).
"""

from typing import Optional

import httpx

from orchestrator.application.ports import AvoidUnavailable, Document


class HttpAvoid:
    """Check interviewer sentences over HTTP.

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
        timeout: float = 3.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address and the client."""
        self.url = base_url.rstrip("/") + "/knowledge/interview/avoid"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout
        self.client = client or httpx.Client()

    def check(self, text: str, language: str) -> Document:
        """Return the hints of one sentence; any failure is unavailable."""
        try:
            response = self.client.post(
                self.url, json={"text": text, "language": language},
                headers=self.headers, timeout=self.timeout,
            )
            if response.status_code == 404:
                raise AvoidUnavailable("no interview rules", "no_rules")
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AvoidUnavailable(type(exc).__name__) from exc
        if not isinstance(body, dict):
            raise AvoidUnavailable("unreadable answer")
        return body
