"""Transcript publisher: ``POST`` ``transcript.v1`` documents to the RAG.

The workstation always starts the hand-over (§2 of
``docs/transcriber/ARCHITECTURE.md``): it sends a finished transcript
to ``<url>/transcripts`` with a bearer token. The token is read from
the environment variable named in the settings when a transcript is
published, never from a file in the repository, and it never appears
in a message, a log line or the ``repr`` of the publisher.

The body is the contract document and nothing else: it is checked
against the contract first, and the contract carries no audio, no
words and no voiceprint. Publishing is idempotent by session id: the
RAG replaces an earlier copy of the same session, so a retried or
repeated publish never duplicates it.

Network errors, time-outs and ``5xx`` answers are retried a bounded
number of times with exponential back-off; a ``4xx`` answer (wrong
token, rejected document) is final and is not retried.
"""

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Optional

from transcriber.domain.errors import PublishError
from transcriber.domain.models import Transcript

ROUTE = "/transcripts"


class HttpTranscriptPublisher:
    """Publish transcripts to the RAG's ``POST /transcripts``.

    Args:
        url: Base URL of the RAG (``https://host`` or ``http://host:port``,
            with or without a path prefix); ``/transcripts`` is added.
        token_env: Name of the environment variable holding the token.
        environ: The environment the token is read from.
        timeout: Seconds to wait for a connection or an answer.
        retries: Further attempts after a failed one (network error,
            time-out or ``5xx``).
        backoff_initial: Seconds before the first retry.
        backoff_factor: Multiplier of the delay after every retry.
        sleep: Waits between attempts (replaced in tests).
    """

    def __init__(
        self,
        url: str,
        token_env: str,
        environ: Mapping[str, str],
        timeout: float = 30.0,
        retries: int = 3,
        backoff_initial: float = 1.0,
        backoff_factor: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Keep the target and the policies; nothing is sent yet."""
        if not url.startswith(("http://", "https://")):
            raise ValueError("The RAG url must start with http:// or https://")
        self.endpoint = url.rstrip("/") + ROUTE
        self.token_env = token_env
        self._environ = environ
        self.timeout = timeout
        self.retries = max(retries, 0)
        self.backoff_initial = backoff_initial
        self.backoff_factor = backoff_factor
        self._sleep = sleep

    def __repr__(self) -> str:
        """Describe the publisher without its token."""
        return (
            f"HttpTranscriptPublisher({self.endpoint!r}, "
            f"token from ${self.token_env})"
        )

    def destination(self) -> str:
        """Return where transcripts go, for messages."""
        return self.endpoint

    def publish(self, document: Mapping[str, Any]) -> None:
        """Send one ``transcript.v1`` document; return once it is stored.

        Raises :class:`~transcriber.domain.errors.ContractError` when
        the document does not follow the contract (nothing is sent),
        and :class:`~transcriber.domain.errors.PublishError` when the
        token is missing, the RAG refuses the document, or it cannot be
        reached after every retry.
        """
        Transcript.from_contract(document)
        token = self._environ.get(self.token_env, "").strip()
        if not token:
            raise PublishError(
                f"No token to publish with: set ${self.token_env}."
            )
        body = json.dumps(
            dict(document), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        delay = self.backoff_initial
        for attempt in range(self.retries + 1):
            failure = self._attempt(body, token)
            if failure is None:
                return
            if attempt == self.retries:
                raise PublishError(
                    f"Could not publish to {self.endpoint} after "
                    f"{attempt + 1} attempt(s): {failure}"
                )
            self._sleep(delay)
            delay *= self.backoff_factor

    def _attempt(self, body: bytes, token: str) -> Optional[str]:
        """POST once; return ``None`` on success, else a retryable reason.

        A final refusal (``4xx``) raises at once.
        """
        request = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                response.read()
            return None
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail = _detail(exc)
            exc.close()
            if status >= 500:
                return f"the RAG answered {status}{detail}"
            if status == 401:
                raise PublishError(
                    f"The RAG refused the token in ${self.token_env} "
                    f"(401){detail}."
                ) from None
            raise PublishError(
                f"The RAG refused the transcript ({status}){detail}."
            ) from None
        except (socket.timeout, TimeoutError):
            return f"no answer within {self.timeout:g} s"
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                return f"no answer within {self.timeout:g} s"
            return f"cannot reach it ({exc.reason})"
        except OSError as exc:
            return f"connection failed ({exc})"


def _detail(error: urllib.error.HTTPError) -> str:
    """Return ``": <detail>"`` from a JSON error answer, if there is one."""
    try:
        data = json.loads(error.read(4096).decode("utf-8"))
    except (OSError, ValueError):
        return ""
    if isinstance(data, dict) and isinstance(data.get("detail"), str):
        return f": {data['detail'][:200]}"
    return ""
