"""Media transcriber adapter: the transcriber service over HTTP (KN-3a).

The transcriber listens on this machine; its ``/internal/knowledge-media``
routes take the RAG's own bearer token (the one the transcriber
publishes with) and are not forwarded through the tunnel
(``docs/KNOWLEDGE.md`` §8). Requests never follow a redirect (the token
would go with it) and the token never appears in a message. A recording
is streamed from its file (``Content-Length`` set, a block at a time),
never read whole into memory.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import IO, Any, Callable, Dict, Optional, Union
from urllib.parse import quote, urlencode

from src.application.knowledge.ports import MediaJob, StoredFile
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    NotFoundError,
    RagError,
    ServiceUnavailableError,
    UnsupportedDocumentError,
)

ROUTE = "/internal/knowledge-media/"
MAX_ANSWER = 64 * 1024 * 1024
"""Largest answer read (a transcript of hours is a few megabytes)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow any redirect."""

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        """Return ``None``: the ``3xx`` answer is raised as it is."""
        return None


OPENER = urllib.request.build_opener(_NoRedirect())


def message_of(raw: bytes) -> str:
    """Return the transcriber's ``message`` of an error answer, if any."""
    try:
        message = json.loads(raw.decode("utf-8")).get("message")
    except (UnicodeDecodeError, ValueError, AttributeError):
        return ""
    return message[:500] if isinstance(message, str) else ""


def refusal(status: int, message: str) -> RagError:
    """Return the error for a transcriber answer ``status``."""
    kinds = {
        400: InvalidInputError, 422: InvalidInputError,
        404: NotFoundError, 409: BusyError, 413: InvalidInputError,
        415: UnsupportedDocumentError, 507: InvalidInputError,
    }
    kind = kinds.get(status)
    if kind is None:
        return ServiceUnavailableError(
            f"The transcriber answered {status}."
        )
    return kind(message or f"The transcriber refused the request "
                           f"({status}).")


def job_of(answer: Any) -> MediaJob:
    """Read a job answer (``ServiceUnavailableError`` if malformed)."""
    if not isinstance(answer, dict) or not isinstance(
        answer.get("state"), str
    ):
        raise ServiceUnavailableError("The transcriber's answer is not a "
                                      "job.")
    message = answer.get("message")
    transcript = answer.get("transcript")
    return MediaJob(
        state=answer["state"],
        message=message if isinstance(message, str) else None,
        transcript=transcript if isinstance(transcript, dict) else None,
    )


class HttpMediaTranscriber:
    """Send recordings to the transcriber and ask about their jobs.

    Args:
        base_url: The transcriber's base URL (``http://127.0.0.1:8770``).
        token: Returns the RAG's bearer token when a call is made.
        timeout: Seconds to wait for one answer (a file is large).
    """

    def __init__(
        self, base_url: str, token: Callable[[], str], timeout: float = 120.0,
    ) -> None:
        """Keep the target; nothing is sent yet."""
        self.url = base_url.rstrip("/") + ROUTE
        self._token = token
        self.timeout = timeout

    def _call(
        self, method: str, job_id: str,
        body: Union[None, bytes, IO[bytes]] = None,
        query: Optional[Dict[str, str]] = None,
        length: Optional[int] = None,
    ) -> Any:
        """Send one request; return the JSON answered (``None``: empty).

        ``body`` may be an open file of ``length`` bytes, sent in blocks.
        """
        url = self.url + quote(job_id, safe="")
        if query:
            url += "?" + urlencode(query)
        headers = {"Accept": "application/json",
                   "Authorization": f"Bearer {self._token()}"}
        if body is not None:
            headers["Content-Type"] = "application/octet-stream"
        if length is not None:
            headers["Content-Length"] = str(length)
        request = urllib.request.Request(
            url, data=body, method=method, headers=headers
        )
        try:
            with OPENER.open(request, timeout=self.timeout) as response:
                raw: bytes = response.read(MAX_ANSWER + 1)
        except urllib.error.HTTPError as exc:
            try:
                message = message_of(exc.read(64 * 1024))
            finally:
                exc.close()
            raise refusal(exc.code, message) from None
        except (socket.timeout, TimeoutError, urllib.error.URLError,
                OSError):
            raise ServiceUnavailableError(
                f"Cannot reach the transcriber at {self.url}."
            ) from None
        if len(raw) > MAX_ANSWER:
            raise ServiceUnavailableError("The transcriber's answer is too "
                                          "large.")
        if not raw.strip():
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ServiceUnavailableError(
                "The transcriber did not answer JSON."
            ) from None

    def submit(
        self, job_id: str, source: StoredFile, file_format: str,
        language: str, max_seconds: float,
    ) -> MediaJob:
        """Start transcribing the file ``source`` as job ``job_id``."""
        query = {
            "format": file_format, "language": language,
            "max_seconds": f"{max_seconds:g}",
        }
        try:
            handle = open(source.path, "rb")
        except OSError:
            raise NotFoundError("The recording's file is missing.") from None
        with handle:
            return job_of(self._call(
                "PUT", job_id, handle, query, length=source.size
            ))

    def status(self, job_id: str) -> MediaJob:
        """Return the job (with its transcript once done)."""
        return job_of(self._call("GET", job_id))

    def forget(self, job_id: str) -> None:
        """Delete the job at the transcriber."""
        self._call("DELETE", job_id)
