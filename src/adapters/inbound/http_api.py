"""Inbound adapter: local HTTP API (FastAPI) over the use cases."""

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.application.answering import AnswerService
from src.application.search import SearchService
from src.domain.errors import InvalidInputError, NotFoundError, RagError
from src.domain.models import MinimalAnswer, MinimalSource


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


def create_app(search: SearchService, answers: AnswerService) -> FastAPI:
    """Build the FastAPI application around the use cases."""
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

    return app
