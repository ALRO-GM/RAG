"""Reasoner adapter: ``POST /reason`` of the RAG service.

``reason_stream`` uses ``POST /reason/stream`` (NDJSON: the model's
text pieces, then the response) so the voice bot can speak before the
reply ends; a service without that route (404/405) is asked with
``POST /reason`` instead, and the whole reply comes at once.
"""

import json
import re
from typing import Callable, Iterator, Optional

import httpx

from orchestrator.application.ports import (
    CancelToken,
    Document,
    ReasonerRejected,
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
            if response.status_code == 422 and _unknown_template(response):
                raise ReasonerRejected("unknown template")
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ReasonerUnavailable(type(exc).__name__) from exc
        if not isinstance(body, dict):
            raise ReasonerUnavailable("unreadable answer")
        return body

    def reason_stream(
        self, request: Document, cancel: CancelToken,
        on_text: Callable[[str], None],
    ) -> Document:
        """Post ``request`` to the streaming route; pass pieces on.

        Reading stops as soon as ``cancel`` is set: the answer is then
        ``status: cancelled`` and the connection is closed.
        """
        if cancel.cancelled:
            return self.reason(request, cancel)
        try:
            with self.client.stream(
                "POST", self.url + "/stream", json=request,
                headers=self.headers, timeout=self.timeout,
            ) as response:
                if response.status_code in (404, 405):
                    fallback = True
                else:
                    fallback = False
                    response.raise_for_status()
                    answer = self._read_lines(
                        response.iter_lines(), request, cancel, on_text
                    )
        except httpx.HTTPError as exc:
            raise ReasonerUnavailable(type(exc).__name__) from exc
        if fallback:
            return self.reason(request, cancel)
        return answer

    @staticmethod
    def _read_lines(
        lines: Iterator[str], request: Document, cancel: CancelToken,
        on_text: Callable[[str], None],
    ) -> Document:
        """Read the NDJSON lines of a streamed answer."""
        for line in lines:
            if cancel.cancelled:
                return {
                    "schema": "reasoner.response.v1",
                    "request_id": request["request_id"],
                    "role": request["role"], "status": "cancelled",
                    "model": request.get("model", ""), "backend": "local",
                    "latency": {"total_ms": 0.0},
                }
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise ReasonerUnavailable("unreadable line") from exc
            kind = item.get("type") if isinstance(item, dict) else None
            if kind == "text":
                on_text(str(item.get("text", "")))
            elif kind == "response" and isinstance(
                item.get("response"), dict
            ):
                body: Document = item["response"]
                return body
            elif kind == "error":
                raise ReasonerUnavailable(str(item.get("message", "error")))
        raise ReasonerUnavailable("the stream ended without a response")


def _unknown_template(response: httpx.Response) -> bool:
    """Whether a 422 says the template is unknown (#286 review).

    The RAG's detail reads «Unknown structured template …» for a
    revision it does not know; any other 422 stays an error as before.
    """
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return False
    return isinstance(detail, str) and bool(
        re.match(r"Unknown (\w+ )?template\b", detail))
