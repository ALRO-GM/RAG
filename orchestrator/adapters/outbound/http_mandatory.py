"""Mandatory items adapter: the RAG's ``/knowledge/.../mandatory`` (LV-16).

``GET /knowledge/domains/{domain}/mandatory`` gives a domain's list and
``POST /knowledge/mandatory`` ``{"domain", "text"}`` tells which item a
sentence says (the RAG embeds it with its knowledge model). With
``{"list", "text"}`` the list comes with the sentence, and ``POST
/knowledge/mandatory/review`` ``{"list"}`` names the examples close to
another item's (#294).
"""

from typing import Callable, Optional
from urllib.parse import quote

import httpx

from orchestrator.application.ports import Document, MandatoryUnavailable


class HttpMandatory:
    """Read and check mandatory items over HTTP.

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
        timeout: float = 5.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address and the client."""
        self.base = base_url.rstrip("/") + "/knowledge"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout
        self.client = client or httpx.Client()

    def items(self, domain: str) -> Document:
        """Return the domain's list (no items when it has none)."""
        url = f"{self.base}/domains/{quote(domain, safe='')}/mandatory"
        return self._answer(lambda: self.client.get(
            url, headers=self.headers, timeout=self.timeout
        ))

    def check(self, domain: str, text: str) -> Document:
        """Return each item's similarity to ``text`` and whether it is said."""
        return self._answer(lambda: self.client.post(
            f"{self.base}/mandatory", json={"domain": domain, "text": text},
            headers=self.headers, timeout=self.timeout,
        ))

    def check_list(self, listed: Document, text: str) -> Document:
        """Return each item's similarity to ``text``, for a sent list."""
        return self._answer(lambda: self.client.post(
            f"{self.base}/mandatory", json={"list": listed, "text": text},
            headers=self.headers, timeout=self.timeout,
        ))

    def review(self, listed: Document) -> Document:
        """Return the examples of ``listed`` close to another item's."""
        return self._answer(lambda: self.client.post(
            f"{self.base}/mandatory/review", json={"list": listed},
            headers=self.headers, timeout=self.timeout,
        ))

    @staticmethod
    def _answer(call: Callable[[], httpx.Response]) -> Document:
        """Run ``call``; any failure is :class:`MandatoryUnavailable`."""
        try:
            response = call()
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MandatoryUnavailable(type(exc).__name__) from exc
        if not isinstance(body, dict):
            raise MandatoryUnavailable("unreadable answer")
        return body
