"""Live sessions of the transcriber: start and stop over HTTP.

``POST /api/live/sessions`` and ``POST /api/live/sessions/{id}/stop``
(``docs/live/LIVE_SESSION.md``), with the transcriber's access code in
the ``X-Access-Code`` header. The code is read from the file
``serve --keep-code`` keeps (``<data_dir>/access_code``) and never
logged.
"""

from pathlib import Path
from typing import Optional

import httpx

from orchestrator.application.ports import Document, TranscriberUnavailable

ACCESS_HEADER = "X-Access-Code"


def read_code(path: Path) -> Optional[str]:
    """Return the access code kept in ``path``, or ``None``."""
    try:
        code = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return code or None


class HttpTranscriber:
    """Start and stop live sessions on the transcriber.

    Args:
        base_url: Root of the transcriber (``http://127.0.0.1:8770``).
        code: The access code.
        timeout: Seconds to wait (starting can wait for the GPU lease).
        client: An ``httpx.Client`` (replaced in tests).
    """

    def __init__(
        self,
        base_url: str,
        code: str,
        timeout: float = 40.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address, the code and the client."""
        self.base = base_url.rstrip("/") + "/api/live/sessions"
        self.headers = {ACCESS_HEADER: code}
        self.timeout = timeout
        self.client = client or httpx.Client()

    def _post(self, url: str, body: Optional[Document]) -> Document:
        try:
            response = self.client.post(
                url, json=body, headers=self.headers, timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise TranscriberUnavailable(type(exc).__name__) from exc
        try:
            answer = response.json()
        except ValueError:
            answer = {}
        if response.status_code >= 400:
            code = answer.get("code") if isinstance(answer, dict) else None
            message = (answer.get("message") if isinstance(answer, dict)
                       else None)
            raise TranscriberUnavailable(
                f"{response.status_code} {code or 'error'}"
                + (f": {message}" if message else "")
            )
        return answer if isinstance(answer, dict) else {}

    def start(self, body: Document) -> Document:
        """Start a live session; return the transcriber's session."""
        return self._post(self.base, body)

    def stop(self, session_id: str) -> None:
        """Stop a live session."""
        self._post(f"{self.base}/{session_id}/stop", None)
