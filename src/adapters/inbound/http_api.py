"""Inbound adapter: local HTTP API (FastAPI) over the use cases.

Besides search and answers, the API receives the transcripts the
transcriber publishes (``POST /transcripts``), gives them back
(``GET /transcripts/{id}``) and deletes them when the transcriber
deletes the session (``DELETE /transcripts/{id}``, mounted when a
delete use case is given). ``GET/PUT /analysis/judge`` read and
change the judge of the soft-skills analysis (``claude_code``,
``local``, ``both`` or ``opencode``) and tell whether Claude's quota
is available;
the transcriber's web forwards them. ``/analysis/sessions/{id}`` starts
the whole analysis of an ingested session (``skills analyze``, in a
child process), ``DELETE …/run`` stops it, and the others serve its
step records, step results and person reports. ``POST /knowledge/query``
answers a ``knowledge.query.v1`` with a ``knowledge.result.v1`` (PL-8);
``POST /reason`` answers a ``reasoner.request.v1`` with a
``reasoner.response.v1`` (PL-16), and ``POST /reason/stream`` does the
same as NDJSON lines: the model's text pieces as it writes them, then
the response (VB-2, the voice bot speaks before the reply ends);
``/knowledge/domains`` lists, creates and deletes domains and uploads,
relabels and deletes their documents (``docs/KNOWLEDGE.md``); a large
file is sent in pieces with resume (``…/uploads``, #282) and
``GET /knowledge/limits`` says the largest file of each kind;
``GET /knowledge/domains/{id}/mandatory`` gives a domain's mandatory
script items and ``POST /knowledge/mandatory`` tells which of them an
employee's sentence says (LV-16; with ``{"list", "text"}`` the list
comes with the sentence, and ``POST /knowledge/mandatory/review``
``{"list"}`` names the examples close to another item's, #294);
``POST /knowledge/interview/avoid``
lists the protected topics an interviewer's question touches (LV-17).
These routes need a bearer token,
compared in constant time; when no token is configured they are not
mounted at all, so they are never open.
"""

import asyncio
import hmac
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from starlette.datastructures import UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from src.application.answering import AnswerService
from src.application.knowledge.library import KnowledgeLibrary
from src.application.knowledge.uploads import DocumentUploads
from src.application.knowledge.avoid import AvoidQuestions
from src.application.knowledge.mandatory import MandatoryItems
from src.application.knowledge.query import KnowledgeQueryService
from src.application.pipeline_settings import PipelineSettings
from src.application.reasoning import Reasoner
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
    NoSpaceError,
    NotFoundError,
    RagError,
    TooLargeError,
    UnsupportedDocumentError,
    short_reason,
)
from src.domain.knowledge.contract import KnowledgeQuery
from src.domain.knowledge.documents import DocumentMetadata
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.uploads import PIECE_BYTES, PartialUpload
from src.domain.models import MinimalAnswer, MinimalSource
from src.domain.reasoning.roles import ReasonRequest
from src.domain.reasoning.templates import get_template
from src.domain.skills.claude_models import ClaudeModels
from src.domain.skills.synthesis import MAX_JOB, JobSought

TRANSCRIPT_MAX_BYTES = 8 * 1024 * 1024
"""Largest ``transcript.v1`` body accepted (text only; hours of speech)."""
TRANSCRIPT_NOT_FOUND = "transcript_not_found"
"""``code`` of ``DELETE /transcripts/{id}``'s ``404``: no copy to delete."""
KNOWLEDGE_QUERY_MAX_BYTES = 64 * 1024
"""Largest ``knowledge.query.v1`` body (text is at most 4000 characters)."""
KNOWLEDGE_JSON_MAX_BYTES = 64 * 1024
"""Largest JSON body of the domain and document routes."""
REASON_MAX_BYTES = 256 * 1024
"""Largest ``reasoner.request.v1`` body (a window of 40 utterances)."""
MULTIPART_OVERHEAD = 1024 * 1024
"""Room for the multipart envelope and the metadata field of an upload."""


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
    """Body of ``POST /analysis/sessions/{id}`` (all optional).

    ``steps`` runs only those steps again (``["synthesis"]`` after the
    job sought changed); ``job`` (the job sought) and
    ``claude_synthesis`` (leave to use Claude for the profiles when
    the judge is the local model) are kept with the analysis when
    ``job`` is given. ``claude_models`` chooses the Claude model of
    each judged step for this run (``confirmation``, ``assessment``,
    ``synthesis``: an alias or a full id; empty: the configured one).
    """

    restart: bool = False
    synthetic: bool = False
    steps: Optional[List[str]] = Field(default=None, max_length=16)
    job: Optional[str] = Field(default=None, max_length=MAX_JOB)
    claude_synthesis: bool = False
    claude_models: ClaudeModels = Field(default_factory=ClaudeModels)


START_MAX_BYTES = 6 * MAX_JOB + 1024
"""Largest body of an analysis start (the job sought, JSON-escaped)."""


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


class NewDomain(BaseModel):
    """Body of ``POST /knowledge/domains``."""

    model_config = {"extra": "forbid"}

    id: str
    title: str = Field(min_length=1, max_length=200)
    language: str = Field(default="es", min_length=2, max_length=16)


class DocumentChange(BaseModel):
    """Body of ``PATCH …/documents/{doc}``: the new metadata (replaces)."""

    model_config = {"extra": "forbid"}

    metadata: DocumentMetadata


@dataclass(frozen=True)
class KnowledgeRoutes:
    """What the knowledge routes need.

    ``query`` answers ``POST /knowledge/query``; ``library`` (optional)
    adds the domain and document routes; ``mandatory`` (optional) the
    mandatory items of a domain and their check (LV-16); ``avoid``
    (optional) the check of interview questions better avoided (LV-17);
    ``uploads`` (with ``library``) the uploads of a document, in one
    request or in pieces (#282).
    """

    query: KnowledgeQueryService
    library: Optional[KnowledgeLibrary] = None
    mandatory: Optional[MandatoryItems] = None
    avoid: Optional[AvoidQuestions] = None
    uploads: Optional[DocumentUploads] = None


@dataclass(frozen=True)
class TranscriptRoutes:
    """What the transcript routes need: the use cases and the token.

    ``token`` is the bearer token the transcriber sends; an empty one
    leaves the routes unmounted. ``max_bytes`` bounds a request body.
    ``forget`` adds ``DELETE /transcripts/{id}`` when given, ``judge``
    the ``/analysis/judge`` routes, ``analysis`` the
    ``/analysis/sessions`` ones and ``knowledge`` the ``/knowledge``
    ones, ``reasoner`` ``POST /reason`` and ``settings`` ``GET/PUT
    /settings`` (the pipeline settings, ``settings.v1``).
    """

    ingest: IngestTranscript
    show: ShowIngestedTranscript
    token: str
    max_bytes: int = TRANSCRIPT_MAX_BYTES
    forget: Optional[ForgetIngestedTranscript] = None
    judge: Optional[JudgeSelection] = None
    analysis: Optional[AnalysisRequests] = None
    knowledge: Optional[KnowledgeRoutes] = None
    reasoner: Optional[Reasoner] = None
    settings: Optional[PipelineSettings] = None


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


async def read_body(
    request: Request, limit: int, detail: str = "Transcript too large."
) -> bytes:
    """Read a request body, refusing it (413, ``detail``) beyond ``limit``
    bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail=detail)
    chunks: List[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail=detail)
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
    if routes.knowledge is not None:
        mount_knowledge(app, routes.knowledge, guard)
    if routes.reasoner is not None:
        mount_reason(app, routes.reasoner, guard)
    if routes.settings is not None:
        mount_settings(app, routes.settings, guard)

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
        transcriber tells it apart from a route that does not exist;
        ``409`` (nothing deleted) while its analysis runs from the
        command line.
        """
        try:
            await run_in_threadpool(forget, session_id)
        except InvalidInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
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


SETTINGS_MAX_BYTES = 8 * 1024
"""Largest body of ``PUT /settings`` (a few dozen short choices)."""


def mount_settings(
    app: FastAPI, settings: PipelineSettings, guard: List[Any]
) -> None:
    """Add ``GET`` and ``PUT /settings`` (``settings.v1``) behind ``guard``.

    ``PUT`` answers 422 for an unknown step or option and 409 when the
    ``revision`` sent is not the stored one.
    """

    @app.get("/settings", dependencies=guard)
    async def show_settings() -> Dict[str, Any]:
        """Return the pipeline settings and the catalogue of options."""
        try:
            return await run_in_threadpool(settings.show)
        except RagError as exc:
            raise http_error(exc)

    @app.put("/settings", dependencies=guard)
    async def change_settings(request: Request) -> Dict[str, Any]:
        """Store new choices (a preset and/or steps) for the next jobs."""
        body = await read_body(request, SETTINGS_MAX_BYTES)
        try:
            parsed = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid JSON.")
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=422, detail="Expected an object.")
        try:
            return await run_in_threadpool(settings.change, parsed)
        except InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except RagError as exc:
            raise http_error(exc)


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
        body = await read_body(request, START_MAX_BYTES)
        try:
            options = AnalysisStart.model_validate_json(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid body.")
        job = None if options.job is None else JobSought(
            text=options.job, use_claude=options.claude_synthesis
        )
        try:
            return await run_in_threadpool(
                requests.start, session_id, options.restart,
                options.synthetic, options.steps, job,
                options.claude_models,
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


def mount_knowledge(
    app: FastAPI, routes: KnowledgeRoutes, guard: List[Any]
) -> None:
    """Add the ``/knowledge`` routes behind ``guard``."""

    @app.post("/knowledge/query", dependencies=guard)
    async def knowledge_query(request: Request) -> JSONResponse:
        """Answer a ``knowledge.query.v1`` with a ``knowledge.result.v1``.

        ``422`` when the body breaks the contract, ``404`` for an
        unknown domain or one without an index yet.
        """
        body = await read_body(request, KNOWLEDGE_QUERY_MAX_BYTES)
        try:
            query = KnowledgeQuery.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=short_reason(exc))
        try:
            result = await run_in_threadpool(routes.query.run, query)
        except RagError as exc:
            raise analysis_error(exc)
        return JSONResponse(result.model_dump(mode="json", by_alias=True))

    if routes.mandatory is not None:
        mount_mandatory(app, routes.mandatory, guard)
    if routes.avoid is not None:
        mount_avoid(app, routes.avoid, guard)
    if routes.library is not None:
        mount_library(app, routes.library, guard)
        if routes.uploads is not None:
            mount_uploads(app, routes.uploads, guard)


def mount_mandatory(
    app: FastAPI, mandatory: MandatoryItems, guard: List[Any]
) -> None:
    """Add a domain's mandatory items and their check behind ``guard``."""

    @app.get("/knowledge/domains/{domain_id}/mandatory", dependencies=guard)
    async def mandatory_items(domain_id: str) -> JSONResponse:
        """The domain's versioned list (version 0, no items, if none)."""
        try:
            listed = await run_in_threadpool(mandatory.items, domain_id)
        except RagError as exc:
            raise analysis_error(exc)
        return JSONResponse(dict(listed))

    @app.post("/knowledge/mandatory", dependencies=guard)
    async def mandatory_check(request: Request) -> JSONResponse:
        """``{"domain", "text"}``: how close the text is to each item."""
        body = await read_body(request, KNOWLEDGE_QUERY_MAX_BYTES)
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="body is not JSON")
        domain = data.get("domain") if isinstance(data, dict) else None
        text = data.get("text") if isinstance(data, dict) else None
        listed = data.get("list") if isinstance(data, dict) else None
        if isinstance(listed, dict) and isinstance(text, str):
            # #294: an owner's list sent with the sentence.
            try:
                found = await run_in_threadpool(
                    mandatory.check_list, listed, text)
            except RagError as exc:
                raise analysis_error(exc)
            return JSONResponse(dict(found))
        if not isinstance(domain, str) or not isinstance(text, str):
            raise HTTPException(
                status_code=422, detail="domain and text are required"
            )
        try:
            found = await run_in_threadpool(mandatory.check, domain, text)
        except RagError as exc:
            raise analysis_error(exc)
        return JSONResponse(dict(found))

    @app.post("/knowledge/mandatory/review", dependencies=guard)
    async def mandatory_review(request: Request) -> JSONResponse:
        """``{"list"}``: the examples close to another item's (#294)."""
        body = await read_body(request, KNOWLEDGE_QUERY_MAX_BYTES)
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="body is not JSON")
        listed = data.get("list") if isinstance(data, dict) else None
        if not isinstance(listed, dict):
            raise HTTPException(status_code=422, detail="list is required")
        try:
            found = await run_in_threadpool(mandatory.review, listed)
        except RagError as exc:
            raise analysis_error(exc)
        return JSONResponse(dict(found))


def mount_avoid(
    app: FastAPI, avoid: AvoidQuestions, guard: List[Any]
) -> None:
    """Add the check of interview questions better avoided (LV-17)."""

    @app.post("/knowledge/interview/avoid", dependencies=guard)
    async def avoid_check(request: Request) -> JSONResponse:
        """``{"text", "language"?}``: the protected topics it touches.

        ``404`` when there are no interview rules to check with.
        """
        body = await read_body(request, KNOWLEDGE_QUERY_MAX_BYTES)
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            raise HTTPException(status_code=422, detail="body is not JSON")
        text = data.get("text") if isinstance(data, dict) else None
        language = data.get("language", "es") if isinstance(
            data, dict) else "es"
        if not isinstance(text, str) or not isinstance(language, str):
            raise HTTPException(status_code=422, detail="text is required")
        try:
            found = await run_in_threadpool(avoid.check, text, language)
        except RagError as exc:
            raise analysis_error(exc)
        return JSONResponse(dict(found))


def mount_reason(
    app: FastAPI, reasoner: Reasoner, guard: List[Any]
) -> None:
    """Add ``POST /reason`` behind ``guard``."""

    @app.post("/reason", dependencies=guard)
    async def reason(request: Request) -> JSONResponse:
        """Answer a ``reasoner.request.v1`` with a ``reasoner.response.v1``.

        ``422`` when the body breaks the contract or names an unknown
        template; a model failure is not an HTTP error: it comes back
        as the response's ``status`` (``timeout``, ``quota``, ...).
        """
        body = await read_body(request, REASON_MAX_BYTES)
        try:
            asked = ReasonRequest.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=short_reason(exc))
        try:
            answer = await run_in_threadpool(reasoner.reason, asked)
        except InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RagError as exc:
            raise http_error(exc)
        return JSONResponse(answer.to_contract())

    @app.post("/reason/stream", dependencies=guard)
    async def reason_stream(request: Request) -> StreamingResponse:
        """Answer a ``reasoner.request.v1`` as NDJSON, piece by piece.

        Lines: ``{"type": "text", "text": ...}`` for each piece of the
        model's raw output as it is written, then ``{"type":
        "response", "response": <reasoner.response.v1>}`` (or ``{"type":
        "error", "message"}``). ``422`` before any line when the body
        breaks the contract or names an unknown template.
        """
        body = await read_body(request, REASON_MAX_BYTES)
        try:
            asked = ReasonRequest.model_validate_json(body)
            if asked.role != "judge":
                get_template(asked.template, asked.role)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=short_reason(exc))
        except InvalidInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return StreamingResponse(
            reason_lines(reasoner, asked),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )


async def reason_lines(
    reasoner: Reasoner, asked: ReasonRequest
) -> AsyncIterator[str]:
    """Run ``asked`` on a thread and yield its NDJSON lines."""
    loop = asyncio.get_running_loop()
    pieces: "asyncio.Queue[str]" = asyncio.Queue()

    def on_text(text: str) -> None:
        loop.call_soon_threadsafe(pieces.put_nowait, text)

    task = asyncio.ensure_future(
        run_in_threadpool(reasoner.reason, asked, on_text)
    )
    while True:
        getter = asyncio.ensure_future(pieces.get())
        done, _ = await asyncio.wait(
            {getter, task}, return_when=asyncio.FIRST_COMPLETED
        )
        if getter not in done:
            getter.cancel()
            break
        yield _line({"type": "text", "text": getter.result()})
    while not pieces.empty():
        yield _line({"type": "text", "text": pieces.get_nowait()})
    try:
        answer = task.result()
    except RagError as exc:
        yield _line({"type": "error", "message": str(exc)})
        return
    yield _line({"type": "response", "response": answer.to_contract()})


def _line(item: Dict[str, Any]) -> str:
    """One NDJSON line."""
    return json.dumps(item, ensure_ascii=False) + "\n"


def knowledge_error(exc: RagError) -> HTTPException:
    """Map a knowledge error: 404, 409, 413, 415, 422, 507 or 500."""
    for kind, status in (
        (UnsupportedDocumentError, 415), (TooLargeError, 413),
        (InvalidInputError, 422), (NotFoundError, 404), (BusyError, 409),
        (NoSpaceError, 507),
    ):
        if isinstance(exc, kind):
            return HTTPException(status_code=status, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


async def read_model(request: Request, model: Any) -> Any:
    """Read a small JSON body as ``model`` (422 when it does not fit)."""
    body = await read_body(request, KNOWLEDGE_JSON_MAX_BYTES)
    try:
        return model.model_validate_json(body or b"{}")
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=short_reason(exc))


class NewUpload(BaseModel):
    """Body of ``POST …/uploads``: the file announced."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=1000)
    size: int = Field(ge=0)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


def upload_parts(form: Any) -> tuple[UploadFile, DocumentMetadata]:
    """Return the ``file`` and the optional ``metadata`` of a multipart.

    ``422`` without a file or with metadata that is not valid.
    """
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=422,
                            detail="The field 'file' is missing.")
    raw = form.get("metadata")
    if raw is not None and not isinstance(raw, str):
        raise HTTPException(status_code=422,
                            detail="'metadata' must be JSON text.")
    try:
        metadata = DocumentMetadata.model_validate_json(raw or "{}")
    except ValidationError as exc:
        raise HTTPException(status_code=422,
                            detail=f"metadata: {short_reason(exc)}")
    return upload, metadata


def mount_library(
    app: FastAPI, library: KnowledgeLibrary, guard: List[Any]
) -> None:
    """Add the ``/knowledge/domains`` routes behind ``guard``."""

    async def call(func: Any, *args: Any) -> Any:
        """Run a use case in the thread pool, mapping its errors."""
        try:
            return await run_in_threadpool(func, *args)
        except RagError as exc:
            raise knowledge_error(exc)

    @app.get("/knowledge/domains", dependencies=guard)
    async def list_domains() -> List[Dict[str, Any]]:
        """Return every domain: documents, chunks, index version."""
        rows = await call(library.domains)
        return [r.model_dump(mode="json") for r in rows]

    @app.post("/knowledge/domains", dependencies=guard, status_code=201)
    async def create_domain(request: Request) -> Dict[str, Any]:
        """Create an empty domain (409 if the id is taken)."""
        body = await read_model(request, NewDomain)
        try:
            manifest = DomainManifest(
                id=body.id, title=body.title, language=body.language
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=short_reason(exc))
        row = await call(library.create_domain, manifest)
        return dict(row.model_dump(mode="json"))

    @app.delete("/knowledge/domains/{domain_id}", dependencies=guard,
                status_code=204)
    async def delete_domain(domain_id: str) -> Response:
        """Delete a created domain with its uploads and index."""
        await call(library.delete_domain, domain_id)
        return Response(status_code=204)

    @app.get("/knowledge/domains/{domain_id}/documents", dependencies=guard)
    async def list_documents(domain_id: str) -> List[Dict[str, Any]]:
        """Return the domain's documents and their status."""
        documents = await call(library.documents, domain_id)
        return [d.model_dump(mode="json") for d in documents]

    @app.get("/knowledge/domains/{domain_id}/documents/{doc_id}/text",
             dependencies=guard)
    async def document_text(domain_id: str, doc_id: str) -> Dict[str, Any]:
        """Return an indexed document's extracted text (front matter cut).

        ``{"id", "name", "text", "chars"}``; ``409`` while it is not
        indexed. The interview planner reads a job offer this way.
        """
        document, text = await call(library.text, domain_id, doc_id)
        return {"id": document.id, "name": document.name, "text": text,
                "chars": len(text)}

    @app.patch("/knowledge/domains/{domain_id}/documents/{doc_id}",
               dependencies=guard)
    async def relabel_document(
        domain_id: str, doc_id: str, request: Request
    ) -> Dict[str, Any]:
        """Replace a document's metadata; it is re-indexed with it."""
        body = await read_model(request, DocumentChange)
        document = await call(library.update, domain_id, doc_id,
                              body.metadata)
        return dict(document.model_dump(mode="json"))

    @app.delete("/knowledge/domains/{domain_id}/documents/{doc_id}",
                dependencies=guard, status_code=204)
    async def delete_document(domain_id: str, doc_id: str) -> Response:
        """Delete a document; its chunks leave the index in background."""
        await call(library.delete, domain_id, doc_id)
        return Response(status_code=204)


def bounded(request: Request, limit: int) -> Request:
    """Return ``request`` whose body is refused (413) past ``limit`` bytes.

    Counted as it arrives, so a body without ``Content-Length`` (chunked)
    cannot be spooled without end either.
    """
    receive = request.receive
    seen = [0]

    async def counted() -> Any:
        """Pass a message on, counting the body bytes."""
        message = await receive()
        if message.get("type") == "http.request":
            seen[0] += len(message.get("body", b""))
            if seen[0] > limit:
                raise HTTPException(status_code=413, detail=(
                    f"El archivo supera {limit} bytes. / The file is "
                    f"larger than {limit} bytes."
                ))
        return message

    return Request(request.scope, counted)


def mount_uploads(
    app: FastAPI, uploads: DocumentUploads, guard: List[Any]
) -> None:
    """Add the upload routes: one request, or pieces with resume (#282).

    ``GET /knowledge/limits`` says the largest file of each kind and
    the piece size, so the page keeps no copy of the numbers.
    """
    library = uploads.library

    async def call(func: Any, *args: Any) -> Any:
        """Run a use case in the thread pool, mapping its errors."""
        try:
            return await run_in_threadpool(func, *args)
        except RagError as exc:
            raise knowledge_error(exc)

    @app.get("/knowledge/limits", dependencies=guard)
    async def limits() -> Dict[str, Any]:
        """The largest file of each kind, the piece size, the longest
        recording (``null``: audio and video are refused)."""
        media = library.media
        return {
            **library.limits.model_dump(),
            "piece_bytes": PIECE_BYTES,
            "max_media_seconds": (
                media.max_seconds if media is not None else None
            ),
        }

    @app.post("/knowledge/domains/{domain_id}/documents",
              dependencies=guard, status_code=202)
    async def upload_document(
        domain_id: str, request: Request
    ) -> Dict[str, Any]:
        """Accept a file in one request (multipart ``file`` + optional
        ``metadata``).

        ``202`` with the document, status ``queued``: extraction,
        chunking and embedding run in the background. The multipart is
        spooled to a temporary file by the server (in the system's temp
        folder; its body counted as it arrives and refused past the
        largest limit), then read a piece at a time; large files are
        better sent in pieces (below).
        """
        largest = library.limits.largest
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > largest + \
                MULTIPART_OVERHEAD:
            raise HTTPException(status_code=413, detail=(
                f"El archivo supera {largest} bytes. / The file is larger "
                f"than {largest} bytes."
            ))
        if not request.headers.get("content-type", "").startswith(
            "multipart/form-data"
        ):
            raise HTTPException(status_code=422,
                                detail="Send multipart/form-data.")
        try:
            form = await bounded(request, largest + MULTIPART_OVERHEAD).form(
                max_files=1, max_fields=4,
                max_part_size=KNOWLEDGE_JSON_MAX_BYTES,
            )
        except HTTPException:
            raise
        except Exception as exc:  # malformed multipart
            raise HTTPException(status_code=422,
                                detail=f"Invalid multipart body ({exc}).")
        try:
            upload, metadata = upload_parts(form)
            document = await call(
                uploads.receive, domain_id, upload.filename or "",
                upload.size or 0, upload.file.read, metadata,
            )
        finally:
            await form.close()
        return dict(document.model_dump(mode="json"))

    @app.post("/knowledge/domains/{domain_id}/uploads", dependencies=guard,
              status_code=201)
    async def start_upload(
        domain_id: str, request: Request
    ) -> Dict[str, Any]:
        """Announce a file to send in pieces (``{"name", "size",
        "metadata"}``); ``201`` with the upload (``id``, ``received``)."""
        body = await read_model(request, NewUpload)
        upload: PartialUpload = await call(
            uploads.start, domain_id, body.name, body.size, body.metadata
        )
        return upload.view()

    @app.get("/knowledge/domains/{domain_id}/uploads/{upload_id}",
             dependencies=guard)
    async def upload_status(domain_id: str, upload_id: str) -> Dict[str, Any]:
        """How much of the file arrived (to resume after a cut)."""
        upload: PartialUpload = await call(
            uploads.status, domain_id, upload_id
        )
        return upload.view()

    @app.put(
        "/knowledge/domains/{domain_id}/uploads/{upload_id}/pieces/{offset}",
        dependencies=guard,
    )
    async def put_piece(
        domain_id: str, upload_id: str, offset: int, request: Request
    ) -> Dict[str, Any]:
        """Write one piece (raw body) at byte ``offset``.

        ``409`` when it would leave a gap (ask the status and go on
        from ``received``); a piece already received is accepted again.
        """
        data = await read_body(request, PIECE_BYTES, detail=(
            f"Una parte no puede pasar de {PIECE_BYTES} bytes. / A piece "
            f"may not be larger than {PIECE_BYTES} bytes."
        ))
        upload: PartialUpload = await call(
            uploads.put, domain_id, upload_id, offset, data
        )
        return upload.view()

    @app.post(
        "/knowledge/domains/{domain_id}/uploads/{upload_id}/complete",
        dependencies=guard, status_code=202,
    )
    async def complete_upload(
        domain_id: str, upload_id: str
    ) -> Dict[str, Any]:
        """Turn the received file into a document (``202``, ``queued``)."""
        document = await call(uploads.complete, domain_id, upload_id)
        return dict(document.model_dump(mode="json"))

    @app.delete("/knowledge/domains/{domain_id}/uploads/{upload_id}",
                dependencies=guard, status_code=204)
    async def cancel_upload(domain_id: str, upload_id: str) -> Response:
        """Cancel an upload; what arrived of it is deleted."""
        await call(uploads.cancel, domain_id, upload_id)
        return Response(status_code=204)
