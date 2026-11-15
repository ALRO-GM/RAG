"""Inbound adapter: local HTTP API (FastAPI) over the use cases.

Besides search and answers, the API receives the transcripts the
transcriber publishes (``POST /transcripts``), gives them back
(``GET /transcripts/{id}``) and deletes them when the transcriber
deletes the session (``DELETE /transcripts/{id}``, mounted when a
delete use case is given). ``GET/PUT /analysis/judge`` read and
change the judge of the soft-skills analysis (``claude_code``,
``local`` or ``both``) and tell whether Claude's quota is available;
the transcriber's web forwards them. ``/analysis/sessions/{id}`` starts
the whole analysis of an ingested session (``skills analyze``, in a
child process), ``DELETE …/run`` stops it, and the others serve its
step records, step results and person reports. These routes need a
bearer token, compared in constant time; when no token is configured
they are not mounted at all, so they are never open.
"""

import hmac
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.judges import JudgeSelection
from src.application.skills.session_analysis import AnalysisRequests
from src.application.skills.ingest import (
    ForgetIngestedTranscript,
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    LimitError,
    NotFoundError,
    RagError,
)
from src.domain.models import MinimalAnswer, MinimalSource

TRANSCRIPT_MAX_BYTES = 8 * 1024 * 1024
"""Largest ``transcript.v1`` body accepted (text only; hours of speech)."""
TRANSCRIPT_NOT_FOUND = "transcript_not_found"
"""``code`` of ``DELETE /transcripts/{id}``'s ``404``: no copy to delete."""


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class AnswerRequest(SearchRequest):
    """Body of ``POST /answer``."""


REPORT_HEADERS = {
    "Content-Security-Policy": (
        "sandbox; default-src 'none'; style-src 'unsafe-inline'; "
        "img-src data:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}
"""Headers of a served person report: a sandboxed, script-less page
(its CSS and charts are inline), never run in the app's origin."""


class AnalysisStart(BaseModel):
    """Body of ``POST /analysis/sessions/{id}`` (all optional)."""

    restart: bool = False
    synthetic: bool = False


class JudgeChoice(BaseModel):
    """Body of ``PUT /analysis/judge``."""

    mode: str = Field(min_length=1, max_length=32)


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
    ``forget`` adds ``DELETE /transcripts/{id}`` when given, ``judge``
    the ``/analysis/judge`` routes, ``analysis`` the
    ``/analysis/sessions`` ones.
    """

    ingest: IngestTranscript
    show: ShowIngestedTranscript
    token: str
    max_bytes: int = TRANSCRIPT_MAX_BYTES
    forget: Optional[ForgetIngestedTranscript] = None
    judge: Optional[JudgeSelection] = None
    analysis: Optional[AnalysisRequests] = None


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
    """Add ``POST /transcripts`` and ``GET /transcripts/{id}`` to ``app``.

    ``DELETE /transcripts/{id}`` too when ``routes.forget`` is given,
    and ``GET/PUT /analysis/judge`` when ``routes.judge`` is.
    """

    def require_token(request: Request) -> None:
        """Refuse (401) a request without the right bearer token."""
        header = request.headers.get("authorization", "")
        if not authorised(header, routes.token):
            raise HTTPException(
                status_code=401, detail="Missing or wrong bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    guard = [Depends(require_token)]
    if routes.judge is not None:
        mount_judge(app, routes.judge, guard)
    if routes.analysis is not None:
        mount_analysis(app, routes.analysis, guard)

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

    forget = routes.forget
    if forget is None:
        return

    @app.delete(
        "/transcripts/{session_id}", dependencies=guard, status_code=204
    )
    async def delete_transcript(session_id: str) -> Response:
        """Delete the stored document of a session.

        ``204`` once deleted; ``404`` with ``code``
        :data:`TRANSCRIPT_NOT_FOUND` when there is none, so the
        transcriber tells it apart from a route that does not exist.
        """
        try:
            await run_in_threadpool(forget, session_id)
        except InvalidInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except NotFoundError as exc:
            return JSONResponse(
                {"detail": str(exc), "code": TRANSCRIPT_NOT_FOUND},
                status_code=404,
            )
        except RagError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return Response(status_code=204)


def mount_judge(
    app: FastAPI, selection: JudgeSelection, guard: List[Any]
) -> None:
    """Add ``GET`` and ``PUT /analysis/judge`` behind ``guard``."""

    @app.get("/analysis/judge", dependencies=guard)
    async def show_judge() -> Dict[str, Any]:
        """Return the judge in use and Claude's quota status."""
        try:
            status = await run_in_threadpool(selection.status)
        except RagError as exc:
            raise http_error(exc)
        return status.model_dump(mode="json")

    @app.put("/analysis/judge", dependencies=guard)
    async def choose_judge(body: JudgeChoice) -> Dict[str, Any]:
        """Change the judge of the next analyses."""
        try:
            status = await run_in_threadpool(selection.choose, body.mode)
        except InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RagError as exc:
            raise http_error(exc)
        return status.model_dump(mode="json")


def analysis_error(exc: RagError) -> HTTPException:
    """Map an analysis error: 400 input, 404 missing, 409 busy, 429 limit."""
    if isinstance(exc, InvalidInputError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, BusyError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, LimitError):
        return HTTPException(status_code=429, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def mount_analysis(
    app: FastAPI, requests: AnalysisRequests, guard: List[Any]
) -> None:
    """Add the ``/analysis/sessions/{id}`` routes behind ``guard``."""
    results = requests.results

    @app.post(
        "/analysis/sessions/{session_id}", dependencies=guard,
        status_code=202,
    )
    async def start_analysis(
        session_id: str, request: Request
    ) -> Dict[str, Any]:
        """Start the analysis of an ingested session (409: running)."""
        body = await read_body(request, 1024)
        try:
            options = AnalysisStart.model_validate_json(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid body.")
        try:
            return await run_in_threadpool(
                requests.start, session_id, options.restart,
                options.synthetic,
            )
        except RagError as exc:
            raise analysis_error(exc)

    @app.get("/analysis/sessions/{session_id}", dependencies=guard)
    async def show_analysis(session_id: str) -> Dict[str, Any]:
        """Return the steps of a session's analysis and ``active``."""
        try:
            return await run_in_threadpool(requests.status, session_id)
        except RagError as exc:
            raise analysis_error(exc)

    @app.delete(
        "/analysis/sessions/{session_id}/run", dependencies=guard,
        status_code=204,
    )
    async def cancel_analysis(session_id: str) -> Response:
        """Stop the running analysis (404: none; 409: from the terminal)."""
        try:
            await run_in_threadpool(requests.cancel, session_id)
        except RagError as exc:
            raise analysis_error(exc)
        return Response(status_code=204)

    @app.get(
        "/analysis/sessions/{session_id}/steps/{step}", dependencies=guard
    )
    async def step_result(session_id: str, step: str) -> Any:
        """Return the stored result of one step."""
        try:
            return await run_in_threadpool(results.result, session_id, step)
        except RagError as exc:
            raise analysis_error(exc)

    @app.get(
        "/analysis/sessions/{session_id}/reports/{position}",
        dependencies=guard,
    )
    async def person_report(session_id: str, position: int) -> Response:
        """Return one person's HTML report as a download (octet-stream)."""
        try:
            page = await run_in_threadpool(
                results.report, session_id, position
            )
        except RagError as exc:
            raise analysis_error(exc)
        # Bytes to save, never a page of this origin (the CSP sandboxes
        # it too, should a browser render it anyway).
        return Response(
            page.encode("utf-8"), media_type="application/octet-stream",
            headers={
                **REPORT_HEADERS,
                "Content-Disposition": (
                    f'attachment; filename="{session_id}-report-'
                    f'{position}.html"'
                ),
            },
        )
