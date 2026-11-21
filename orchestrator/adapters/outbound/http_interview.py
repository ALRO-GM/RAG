"""The RAG's routes the AI interviewer uses (HTTP).

* :class:`HttpOffers`: the text of a job offer uploaded to a knowledge
  domain (``GET /knowledge/domains/{d}/documents/{doc}/text``).
* :class:`HttpHandoff`: the interview's ``transcript.v1``
  (``POST /transcripts``) and the post-call analysis with the slot
  sheet as the job sought (``POST /analysis/sessions/{id}``).
"""

import re
from typing import Optional, Tuple

import httpx

from orchestrator.application.interview import HandoffFailed, OfferUnavailable
from orchestrator.application.ports import Document

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _detail(response: httpx.Response) -> str:
    """The RAG's error message of an answer, short."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    return f"HTTP {response.status_code}: {detail or ''}"[:300]


class _Rag:
    """Common parts of the two adapters."""

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ):
        """Keep the address and the client."""
        self.base = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout
        self.client = client or httpx.Client()


class HttpOffers(_Rag):
    """Read an uploaded job offer's text from the RAG."""

    def document_text(self, domain: str, document: str) -> Tuple[str, str]:
        """Return the document's name and text (``OfferUnavailable``)."""
        if not ID.fullmatch(domain) or not ID.fullmatch(document):
            raise OfferUnavailable("invalid domain or document id")
        url = f"{self.base}/knowledge/domains/{domain}/documents/" \
              f"{document}/text"
        try:
            response = self.client.get(url, headers=self.headers,
                                       timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise OfferUnavailable(type(exc).__name__) from exc
        if response.status_code != 200:
            raise OfferUnavailable(_detail(response))
        try:
            body = response.json()
        except ValueError as exc:
            raise OfferUnavailable("unreadable answer") from exc
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise OfferUnavailable("unreadable answer")
        return str(body.get("name") or document), text


class HttpHandoff(_Rag):
    """Hand an interview to the RAG's post-call analysis."""

    def publish_transcript(self, document: Document) -> None:
        """``POST /transcripts`` (``HandoffFailed`` on failure)."""
        self._post(f"{self.base}/transcripts", document)

    def start_analysis(self, session_id: str, job: str) -> None:
        """``POST /analysis/sessions/{id}`` with ``job``."""
        if not ID.fullmatch(session_id):
            raise HandoffFailed("invalid session id")
        self._post(f"{self.base}/analysis/sessions/{session_id}",
                   {"job": job})

    def _post(self, url: str, body: Document) -> None:
        try:
            response = self.client.post(url, json=body, headers=self.headers,
                                        timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise HandoffFailed(type(exc).__name__) from exc
        if response.status_code >= 300:
            raise HandoffFailed(_detail(response))
