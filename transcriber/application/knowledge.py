"""Use case: the RAG's knowledge domains, forwarded for the web.

The RAG listens on loopback only and wants its bearer token; the web
asks the transcriber (behind the access code) and this use case
forwards the request after checking every id that goes into the path,
so nothing else reaches the RAG. Bodies are passed as they are (JSON
or the multipart upload); the RAG validates them.
"""

import re
from typing import Any, Optional

from transcriber.application.ports import KnowledgePort
from transcriber.domain.errors import InvalidInputError, RagUnavailableError

DOMAIN_ID_PATTERN = r"[a-z0-9][a-z0-9._-]{0,63}"
"""Domain ids (the RAG's ``knowledge.query.v1`` pattern)."""
DOCUMENT_ID_PATTERN = r"[a-z0-9][a-z0-9-]{0,79}"
"""Ids the RAG gives uploaded documents."""
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
"""Largest file the RAG accepts (its ``max_upload_bytes``)."""
UPLOAD_BODY_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
"""Largest upload request: the file plus the multipart envelope."""
JSON_BODY_BYTES = 64 * 1024
"""Largest JSON body forwarded (domain, metadata, query)."""


def checked(value: str, pattern: str, what: str) -> str:
    """Return ``value`` if it matches ``pattern``, else refuse it."""
    if not re.fullmatch(pattern, value or ""):
        raise InvalidInputError(f"Invalid {what} {value!r}.")
    return value


class KnowledgeAccess:
    """List, create and delete domains; upload and manage documents.

    ``port`` is ``None`` when the transcriber has no HTTP link to the
    RAG; every call then says so.
    """

    def __init__(self, port: Optional[KnowledgePort]) -> None:
        """Wire the port (or its absence)."""
        self._port_or_none = port

    def _port(self) -> KnowledgePort:
        """Return the port or say the RAG is not linked."""
        if self._port_or_none is None:
            raise RagUnavailableError(
                "Transcripts are not sent to a RAG over HTTP, so there is "
                "no knowledge base."
            )
        return self._port_or_none

    @staticmethod
    def _domain(domain_id: str) -> str:
        """Check a domain id."""
        return checked(domain_id, DOMAIN_ID_PATTERN, "domain id")

    @staticmethod
    def _document(doc_id: str) -> str:
        """Check a document id."""
        return checked(doc_id, DOCUMENT_ID_PATTERN, "document id")

    def domains(self) -> Any:
        """Return the RAG's list of domains."""
        return self._port().call("GET", "domains")

    def create_domain(self, body: bytes) -> Any:
        """Create a domain from ``{"id", "title", "language"}``."""
        return self._port().call("POST", "domains", body, "application/json")

    def delete_domain(self, domain_id: str) -> None:
        """Delete a domain with its uploads."""
        self._port().call("DELETE", f"domains/{self._domain(domain_id)}")

    def documents(self, domain_id: str) -> Any:
        """Return a domain's documents and their status."""
        return self._port().call(
            "GET", f"domains/{self._domain(domain_id)}/documents"
        )

    def upload(self, domain_id: str, body: bytes, content_type: str) -> Any:
        """Forward a multipart upload (``file`` + optional ``metadata``)."""
        if not content_type.lower().startswith("multipart/form-data"):
            raise InvalidInputError("Send the file as multipart/form-data.")
        return self._port().call(
            "POST", f"domains/{self._domain(domain_id)}/documents", body,
            content_type,
        )

    def relabel(self, domain_id: str, doc_id: str, body: bytes) -> Any:
        """Replace a document's metadata (``{"metadata": {...}}``)."""
        return self._port().call(
            "PATCH",
            f"domains/{self._domain(domain_id)}/documents/"
            f"{self._document(doc_id)}",
            body, "application/json",
        )

    def delete_document(self, domain_id: str, doc_id: str) -> None:
        """Delete a document; its chunks leave the index."""
        self._port().call(
            "DELETE",
            f"domains/{self._domain(domain_id)}/documents/"
            f"{self._document(doc_id)}",
        )

    def query(self, body: bytes) -> Any:
        """Forward a ``knowledge.query.v1``; return the result."""
        return self._port().call("POST", "query", body, "application/json")
