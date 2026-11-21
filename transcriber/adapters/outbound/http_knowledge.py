"""Knowledge routes: forward ``/knowledge/...`` to the RAG.

As :mod:`~transcriber.adapters.outbound.http_session_analysis`: the RAG
listens on loopback only, the web asks the transcriber and this adapter
asks the RAG with its bearer token, never following a redirect. The
RAG's own refusal message (``detail``) is passed on, because it tells
the person what to fix (a file type, a vocabulary); the token never
appears in a message.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Optional
from urllib.parse import quote

from transcriber.adapters.outbound.no_redirect import OPENER
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    PublishError,
    RagUnavailableError,
    StateError,
    TooLargeError,
    TranscriberError,
    UnsupportedMediaError,
)

ROUTE = "/knowledge/"
MAX_ANSWER = 16 * 1024 * 1024
"""Largest answer read from the RAG."""
MAX_DETAIL = 1000
"""Longest refusal message passed on."""


class HttpKnowledge:
    """Send knowledge requests to the RAG.

    Args:
        base_url: The RAG's base URL (the publisher's).
        token: Returns the RAG's bearer token when a call is made.
        timeout: Seconds to wait for the RAG (uploads are large).
    """

    def __init__(
        self,
        base_url: str,
        token: Callable[[], str],
        timeout: float = 120.0,
    ) -> None:
        """Keep the target; nothing is sent yet."""
        self.url = base_url.rstrip("/") + ROUTE
        self._token = token
        self.timeout = timeout

    def call(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        content_type: Optional[str] = None,
    ) -> Any:
        """Send one request; return the JSON answered (``None``: empty)."""
        try:
            token = self._token()
        except PublishError as exc:
            raise RagUnavailableError(str(exc)) from None
        headers = {"Accept": "application/json",
                   "Authorization": f"Bearer {token}"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        url = self.url + quote(path, safe="/")
        request = urllib.request.Request(
            url, data=body, method=method, headers=headers
        )
        try:
            with OPENER.open(request, timeout=self.timeout) as response:
                raw: bytes = response.read(MAX_ANSWER + 1)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                detail = detail_of(exc.read(64 * 1024))
            finally:
                exc.close()
            raise refusal(status, detail, method, url) from None
        except (socket.timeout, TimeoutError, urllib.error.URLError,
                OSError):
            raise RagUnavailableError(
                f"Cannot reach the RAG at {self.url}."
            ) from None
        if len(raw) > MAX_ANSWER:
            raise RagUnavailableError("The RAG's answer is too large.")
        if not raw.strip():
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RagUnavailableError("The RAG did not answer JSON.") \
                from None


def detail_of(raw: bytes) -> str:
    """Return the RAG's ``detail`` message of an error answer, if any."""
    try:
        detail = json.loads(raw.decode("utf-8")).get("detail")
    except (UnicodeDecodeError, ValueError, AttributeError):
        return ""
    if not isinstance(detail, str):
        return ""
    return detail[:MAX_DETAIL]


def refusal(
    status: int, detail: str, method: str, url: str
) -> TranscriberError:
    """Return the domain error for a RAG answer ``status``."""
    kinds = {
        400: InvalidInputError, 422: InvalidInputError,
        404: NotFoundError, 409: StateError, 413: TooLargeError,
        415: UnsupportedMediaError,
    }
    kind = kinds.get(status)
    if kind is None:
        return RagUnavailableError(f"The RAG answered {status} to {method} "
                                   f"{url}.")
    return kind(detail or f"The RAG refused the request ({status}).")
