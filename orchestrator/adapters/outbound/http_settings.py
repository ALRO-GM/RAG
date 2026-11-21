"""Settings adapter: ``GET /settings`` of the RAG (``settings.v1``).

The RAG keeps the pipeline settings the web's «Ajustes» page saves; a
live session reads them when it starts. A short wait, and never an
error: without an answer the products are used as written.
"""

import logging
from typing import Dict, Optional

import httpx

LOG = logging.getLogger(__name__)


class HttpSettings:
    """Read the RAG's saved pipeline settings over HTTP.

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
        timeout: float = 2.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address and the client."""
        self.url = base_url.rstrip("/") + "/settings"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout
        self.client = client or httpx.Client()

    def current(self) -> Optional[Dict[str, str]]:
        """Return the saved choices, or ``None`` (none saved, no answer)."""
        try:
            response = self.client.get(
                self.url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            LOG.warning("Pipeline settings not read: %s", type(exc).__name__)
            return None
        if not isinstance(body, dict) or not body.get("stored"):
            return None
        choices = body.get("choices")
        if not isinstance(choices, dict):
            return None
        return {str(k): v for k, v in choices.items() if isinstance(v, str)}
