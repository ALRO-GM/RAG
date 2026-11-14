"""Inbound adapter: local HTTP API (FastAPI) over the use cases.

Besides search and answers, the API receives the transcripts the
transcriber publishes (``POST /transcripts``) and gives them back
(``GET /transcripts/{id}``). Those two routes need a bearer token,
compared in constant time; when no token is configured they are not
mounted at all, so they are never open.
"""

import hmac
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.domain.errors import InvalidInputError, NotFoundError, RagError
from src.domain.models import MinimalAnswer, MinimalSource

TRANSCRIPT_MAX_BYTES = 8 * 1024 * 1024
"""Largest ``transcript.v1`` body accepted (text only; hours of speech)."""


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class AnswerRequest(SearchRequest):
    """Body of ``POST /answer``."""


class SearchHit(BaseModel):
    """One retrieved source with its score and text."""

    source: MinimalSource
    score: float
    context: str
    text: str


class SearchResponse(BaseModel):
    """Response of the search endpoints."""

    query: str
    mode: str
    results: List[SearchHit]


def http_error(exc: RagError) -> HTTPException:
    """Map a domain error to an HTTP status code."""
    if isinstance(exc, InvalidInputError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@dataclass(frozen=True)
class TranscriptRoutes:
    """What the transcript routes need: the use cases and the token.

    ``token`` is the bearer token the transcriber sends; an empty one
    leaves the routes unmounted. ``max_bytes`` bounds a request body.
    """

    ingest: IngestTranscript
    show: ShowIngestedTranscript
    token: str
    max_bytes: int = TRANSCRIPT_MAX_BYTES


def create_app(
    search: SearchService,
    answers: AnswerService,
    transcripts: Optional[TranscriptRoutes] = None,
) -> FastAPI:
    """Build the FastAPI application around the use cases.

    The transcript routes are mounted only when ``transcripts`` is
    given with a non-empty token.
    """
    app = FastAPI(
        title="RAG against the machine",
        description="Search and question answering over the vLLM codebase.",
        version="1.0.0",
    )

    def run_search(query: str, k: int) -> SearchResponse:
        """Run a search, mapping domain errors to HTTP errors."""
        if not query.strip():
            raise HTTPException(status_code=400, detail="Empty query.")
        try:
            hits = search.search(query, k)
            mode = search.mode
        except RagError as exc:
            raise http_error(exc)
        return SearchResponse(
            query=query,
            mode=mode,
            results=[
                SearchHit(
                    source=h.chunk.to_source(),
                    score=h.score,
                    context=h.chunk.context,
                    text=h.chunk.text,
                )
                for h in hits
            ],
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Report readiness and index statistics."""
        try:
            snapshot = search.snapshot
        except RagError as exc:
            raise http_error(exc)
        return {
            "status": "ok",
            "chunks": len(snapshot.chunks),
            "mode": search.mode,
            "fingerprint": snapshot.manifest.fingerprint,
        }

    @app.get("/search", response_model=SearchResponse)
    async def search_get(
        q: str = Query(min_length=1), k: int = Query(10, ge=1, le=100)
    ) -> SearchResponse:
        """Search with query parameters."""
        return await run_in_threadpool(run_search, q, k)

    @app.post("/search", response_model=SearchResponse)
    async def search_post(body: SearchRequest) -> SearchResponse:
        """Search with a JSON body."""
        return await run_in_threadpool(run_search, body.query, body.k)

    @app.post("/answer", response_model=MinimalAnswer)
    async def answer(body: AnswerRequest) -> MinimalAnswer:
        """Retrieve context and generate a grounded answer."""
        if not body.query.strip():
            raise HTTPException(status_code=400, detail="Empty query.")
        try:
            return await run_in_threadpool(answers.answer, body.query, body.k)
        except RagError as exc:
            raise http_error(exc)

    if transcripts is not None and transcripts.token:
        mount_transcripts(app, transcripts)
    return app


def authorised(header: str, token: str) -> bool:
    """Tell whether an ``Authorization`` header carries ``token``.

    The comparison takes the same time whatever the header holds.
    """
    scheme, _, presented = header.partition(" ")
    return hmac.compare_digest(
        presented.strip().encode("utf-8"), token.encode("utf-8")
    ) and scheme.lower() == "bearer"


async def read_body(request: Request, limit: int) -> bytes:
    """Read a request body, refusing it (413) beyond ``limit`` bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail="Transcript too large.")
    chunks: List[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=413, detail="Transcript too large."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def mount_transcripts(app: FastAPI, routes: TranscriptRoutes) -> None:
    """Add ``POST /transcripts`` and ``GET /transcripts/{id}`` to ``app``."""

    def require_token(request: Request) -> None:
        """Refuse (401) a request without the right bearer token."""
        header = request.headers.get("authorization", "")
        if not authorised(header, routes.token):
            raise HTTPException(
                status_code=401, detail="Missing or wrong bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    guard = [Depends(require_token)]

    @app.post("/transcripts", dependencies=guard)
    async def put_transcript(request: Request) -> JSONResponse:
        """Store a ``transcript.v1`` document, replacing its session."""
        body = await read_body(request, routes.max_bytes)
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Body is not JSON.")
        try:
            done = await run_in_threadpool(routes.ingest, document)
        except InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RagError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return JSONResponse(
            {"id": done.session_id, "replaced": done.replaced,
             "utterances": done.utterances},
            status_code=200 if done.replaced else 201,
        )

    @app.get("/transcripts/{session_id}", dependencies=guard)
    async def get_transcript(session_id: str) -> Dict[str, Any]:
        """Return the stored ``transcript.v1`` document of a session."""
        try:
            return await run_in_threadpool(routes.show, session_id)
        except InvalidInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except RagError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
