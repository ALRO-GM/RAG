"""Transcript publisher: ``POST`` ``transcript.v1`` documents to the RAG.

The workstation always starts the hand-over (§2 of
``docs/transcriber/ARCHITECTURE.md``): it sends a finished transcript
to ``<url>/transcripts`` with a bearer token. The token is read when a
transcript is published, from the environment variable named in the
settings or, on the workstation, from a private file outside the
repository (``make rag-token``; refused unless only its owner may read
it). It never appears in a message, a log line or the ``repr`` of the
publisher.

The body is the contract document and nothing else: it is checked
against the contract first, and the contract carries no audio, no
words and no voiceprint. Publishing is idempotent by session id: the
RAG replaces an earlier copy of the same session, so a retried or
repeated publish never duplicates it.

Network errors, time-outs and ``5xx`` answers are retried a bounded
number of times with exponential back-off; a ``4xx`` answer (wrong
token, rejected document) is final and is not retried.

When the transcriber deletes a session, ``retract`` asks the RAG to
delete its copy (``DELETE <url>/transcripts/<id>``) once, without
retries and with a short time-out: it is best effort, and a copy that
was never there (the RAG's ``404`` with code ``transcript_not_found``)
is not an error.
"""

import json
import os
import re
import socket
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from transcriber.domain.errors import PublishError
from transcriber.domain.models import SESSION_ID_PATTERN, Transcript

ROUTE = "/transcripts"
ABSENT_CODE = "transcript_not_found"
"""``code`` of the RAG's ``404`` when it has no copy to delete."""
TOKEN_PATTERN = r"^[A-Za-z0-9_.~+/=-]{32,256}$"
"""A token file's content: what ``make rag-token`` writes (the RAG's rule)."""


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
        token_file: File holding the token, read instead of
            ``token_env`` when given (``~`` is expanded).
        retract_timeout: Seconds a deletion of the RAG's copy waits at
            most (short: it runs while a session is being deleted).
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
        token_file: Optional[Path] = None,
        retract_timeout: float = 5.0,
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
        self.token_file = token_file
        self.retract_timeout = retract_timeout

    def __repr__(self) -> str:
        """Describe the publisher without its token."""
        return (
            f"HttpTranscriptPublisher({self.endpoint!r}, "
            f"token from {self.token_source()})"
        )

    def token_source(self) -> str:
        """Return where the token is read from, for messages."""
        if self.token_file is not None:
            return f"file {self.token_file}"
        return f"${self.token_env}"

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
        token = self._token()
        body = json.dumps(
            dict(document), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        delay = self.backoff_initial
        for attempt in range(self.retries + 1):
            failure = self._attempt("POST", self.endpoint, body, token)
            if failure is None:
                return
            if attempt == self.retries:
                raise PublishError(
                    f"Could not publish to {self.endpoint} after "
                    f"{attempt + 1} attempt(s): {failure}"
                )
            self._sleep(delay)
            delay *= self.backoff_factor

    def retract(self, session_id: str) -> None:
        """Ask the RAG to delete its copy of ``session_id``, once.

        A copy the RAG does not have (``404``) is not an error. Raises
        :class:`~transcriber.domain.errors.PublishError` when the token
        is missing, the RAG refuses or cannot be reached.
        """
        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            raise PublishError(f"Invalid session id {session_id!r}.")
        token = self._token()
        url = f"{self.endpoint}/{urllib.parse.quote(session_id, safe='')}"
        failure = self._attempt(
            "DELETE", url, None, token, self.retract_timeout, absent_ok=True
        )
        if failure is not None:
            raise PublishError(f"Could not delete {url}: {failure}")

    def _token(self) -> str:
        """Return the token, from the file or the environment."""
        if self.token_file is not None:
            return read_token(self.token_file)
        token = self._environ.get(self.token_env, "").strip()
        if not token:
            raise PublishError(
                f"No token to publish with: set ${self.token_env}."
            )
        return token

    def _attempt(
        self, method: str, url: str, body: Optional[bytes], token: str,
        timeout: Optional[float] = None, absent_ok: bool = False,
    ) -> Optional[str]:
        """Send once; return ``None`` on success, else a retryable reason.

        ``timeout`` defaults to the publisher's. A final refusal
        (``4xx``) raises at once, except, with ``absent_ok`` (deleting),
        a ``404`` whose answer carries the RAG's :data:`ABSENT_CODE`:
        the copy is not there, which is success. Any other ``404`` (a
        route that does not exist, another service) is a refusal.
        """
        wait = self.timeout if timeout is None else timeout
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=body, method=method, headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=wait
            ) as response:
                response.read()
            return None
        except urllib.error.HTTPError as exc:
            status = exc.code
            answer = _answer(exc)
            exc.close()
            detail = _detail(answer)
            if status >= 500:
                return f"the RAG answered {status}{detail}"
            if status == 404 and absent_ok:
                if answer.get("code") == ABSENT_CODE:
                    return None
                raise PublishError(
                    f"{url} answered 404 without saying the transcript"
                    f" is missing{detail}: is the RAG listening there,"
                    " with its transcript routes on?"
                ) from None
            if status == 401:
                raise PublishError(
                    f"The RAG refused the token in {self.token_source()} "
                    f"(401){detail}."
                ) from None
            raise PublishError(
                f"The RAG refused the transcript ({status}){detail}."
            ) from None
        except (socket.timeout, TimeoutError):
            return f"no answer within {wait:g} s"
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                return f"no answer within {wait:g} s"
            return f"cannot reach it ({exc.reason})"
        except OSError as exc:
            return f"connection failed ({exc})"


def read_token(path: Path) -> str:
    """Return the token kept in ``path`` (``~`` is expanded).

    Raises :class:`~transcriber.domain.errors.PublishError` when the
    file is missing, is not a regular file of this user, may be read or
    written by others, or does not hold a token (:data:`TOKEN_PATTERN`,
    the RAG's own rule). No message contains its content.
    """
    path = Path(path).expanduser()
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise PublishError(
            f"{path} does not exist: run `make rag-token`."
        ) from None
    except OSError as exc:
        raise PublishError(f"Cannot read {path}: {exc.strerror}.") from None
    if not stat.S_ISREG(info.st_mode):
        raise PublishError(f"{path} is not a regular file.")
    if info.st_uid != os.getuid():
        raise PublishError(f"{path} belongs to another user.")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o077:
        raise PublishError(
            f"{path} may be read by others (mode {mode:o}): run"
            f" `chmod 600 {path}`."
        )
    try:
        token = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        raise PublishError(f"Cannot read {path}.") from None
    if not re.fullmatch(TOKEN_PATTERN, token):
        raise PublishError(f"{path} does not hold a token.")
    return token


def _answer(error: urllib.error.HTTPError) -> Dict[str, Any]:
    """Return the JSON object of an error answer (``{}`` if none)."""
    try:
        data = json.loads(error.read(4096).decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _detail(answer: Mapping[str, Any]) -> str:
    """Return ``": <detail>"`` from an error answer, if it has one."""
    if isinstance(answer.get("detail"), str):
        return f": {answer['detail'][:200]}"
    return ""
