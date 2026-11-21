"""Inbound adapter: the HTTP API (FastAPI) the phone and the pages use.

Every route lives under ``/api`` (``docs/transcriber/ARCHITECTURE.md``
§5): the server's Caddy serves the pages and forwards ``/api/*``
through the tunnel to this API, which listens on 127.0.0.1. Every route
but ``GET /api/health`` needs the access code in the ``X-Access-Code``
header (:mod:`transcriber.adapters.inbound.access`). The audio route
alone also accepts the playback cookie that ``POST /api/access/cookie``
sets, because a browser's ``<audio>`` element cannot send a header.

Bodies are raw bytes (audio) or JSON, read with a size limit that is
enforced while the body streams in, so an oversized upload is refused
before it is held in memory. Use cases run in the thread pool, never on
the event loop. Errors are JSON ``{"code": ..., "message": ...}`` with
stable codes (:data:`ERROR_CODES`); unexpected errors give a generic
500 and are logged, never shown.

The audio of a session is served as a WAV rendering with HTTP range
support (:mod:`transcriber.adapters.inbound.playback`), streamed from a
file a block at a time.

Only enrolments wait for the job runner (the models); at most
``api.max_pending_enrolments`` of them may be in progress or waiting,
so waiting requests never fill the thread pool the other routes use.
Every other route returns without waiting for a job.
"""

import logging
import math
import time
from pathlib import PurePath
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar,
)

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from transcriber.adapters.inbound.access import (
    ACCESS_HEADER,
    MIN_CODE_LENGTH,
    PLAYBACK_COOKIE,
    PLAYBACK_COOKIE_PATH,
    FailureLimiter,
    check_code,
    client_key,
    new_playback_token,
)
from transcriber.adapters.inbound.playback import (
    WAV_TYPE,
    Unsatisfiable,
    byte_blocks,
    parse_range,
)
from transcriber.application.analysis import MAX_JOB
from transcriber.application.knowledge import (
    JSON_BODY_BYTES,
    UPLOAD_BODY_BYTES,
)
from transcriber.application.ports import (
    PCM_FORMAT,
    JobStatus,
    PlaybackAudio,
    RecordedAudio,
)
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.bootstrap import TranscriberContainer
from transcriber.domain.consent import ConsentNotice
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    BusyError,
    ConsentRequiredError,
    ContractError,
    InvalidInputError,
    LimitReachedError,
    ModelLoadError,
    NotFoundError,
    RagUnavailableError,
    SampleRejectedError,
    StateError,
    StorageError,
    TooLargeError,
    TranscriberError,
    UnsupportedMediaError,
    VoiceModelMismatchError,
)
from transcriber.domain.models import Session, Transcript
from transcriber.domain.speaker_names import SpeakerNames

LOG = logging.getLogger(__name__)

ERROR_CODES: Tuple[Tuple[Type[TranscriberError], int, str], ...] = (
    (ConsentRequiredError, 400, "consent_required"),
    (SampleRejectedError, 422, "sample_rejected"),
    (VoiceModelMismatchError, 409, "voice_model_mismatch"),
    (TooLargeError, 413, "payload_too_large"),
    (UnsupportedMediaError, 415, "unsupported_media_type"),
    (InvalidInputError, 400, "invalid_input"),
    (NotFoundError, 404, "not_found"),
    (BusyError, 409, "busy"),
    (StateError, 409, "wrong_state"),
    (ModelLoadError, 503, "model_unavailable"),
    (AnalysisUnavailableError, 503, "analysis_unavailable"),
    (RagUnavailableError, 503, "rag_unavailable"),
    (LimitReachedError, 429, "limit_reached"),
    (StorageError, 500, "storage_error"),
    (ContractError, 500, "contract_error"),
)
"""Domain error → status and stable code; the first match wins."""

HTTP_CODES: Dict[int, str] = {
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    416: "range_not_satisfiable",
    422: "invalid_request",
    429: "too_many_attempts",
}
"""Status → stable code for errors that are not domain errors."""

SERVER_MESSAGE = "The transcriber could not complete the request."

M = TypeVar("M", bound=BaseModel)


class ApiError(Exception):
    """An error answered as ``{"code", "message"}`` with ``status``."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Keep what the response needs."""
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


class NewSession(BaseModel):
    """Body of ``POST /api/sessions``.

    ``consent_version`` is the recording notice the page showed;
    ``num_speakers`` and ``expected_speakers`` are kept on the session
    for its transcription. ``type`` and ``goal`` describe the
    conversation (both optional).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    language: str
    title: Optional[str] = None
    consent_version: Optional[str] = Field(default=None, max_length=32)
    num_speakers: Optional[int] = Field(default=None, ge=1)
    expected_speakers: Optional[List[str]] = None
    type: Optional[str] = Field(default=None, max_length=64)
    goal: Optional[str] = Field(default=None, max_length=2000)


class DescribeRequest(BaseModel):
    """Body of ``PATCH /api/sessions/{id}``: the conversation's context.

    A field left out keeps its value; ``null`` clears it.
    """

    model_config = ConfigDict(extra="forbid")

    type: Optional[str] = Field(default=None, max_length=64)
    goal: Optional[str] = Field(default=None, max_length=2000)


class TranscriptionRequest(BaseModel):
    """Body of ``POST /api/sessions/{id}/transcription`` (all optional)."""

    model_config = ConfigDict(extra="forbid")

    language: Optional[str] = None
    num_speakers: Optional[int] = Field(default=None, ge=1)
    expected_speakers: Optional[List[str]] = None


class JudgeRequest(BaseModel):
    """Body of ``PUT /api/analysis/judge``."""

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(max_length=32)


class AnalysisRequest(BaseModel):
    """Body of ``POST /api/analysis/sessions/{id}`` (all optional).

    ``steps`` runs only those steps again; ``job`` (the job sought) and
    ``claude_synthesis`` go to the explanatory profiles (#113);
    ``claude_models`` chooses the Claude model of a judged step
    (``confirmation``, ``assessment``, ``synthesis``).
    """

    model_config = ConfigDict(extra="forbid")

    restart: bool = False
    synthetic: bool = False
    steps: List[str] = Field(default_factory=list, max_length=16)
    job: Optional[str] = Field(default=None, max_length=MAX_JOB)
    claude_synthesis: bool = False
    claude_models: Dict[str, str] = Field(default_factory=dict, max_length=3)


REPORT_HEADERS = {
    "Content-Security-Policy": (
        "sandbox; default-src 'none'; style-src 'unsafe-inline'; "
        "img-src data:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "Cache-Control": "no-store",
}
"""Headers of a person report: a sandboxed page without scripts (its
CSS and charts are inline), sent as a download, never run in the
pages' origin."""


class RenameRequest(BaseModel):
    """Body of ``PATCH /api/sessions/{id}/speakers/{label}``."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    merge_into: Optional[str] = None


class ReassignRequest(BaseModel):
    """Body of ``PATCH /api/sessions/{id}/utterances/{utterance}``."""

    model_config = ConfigDict(extra="forbid")

    speaker: str


def error_response(
    status: int,
    code: str,
    message: str,
    headers: Optional[Mapping[str, str]] = None,
    **extra: Any,
) -> JSONResponse:
    """Return the JSON error body every failure uses."""
    return JSONResponse(
        {"code": code, "message": message, **extra},
        status_code=status, headers=headers,
    )


def domain_error(exc: TranscriberError) -> JSONResponse:
    """Map a domain error to its status and stable code.

    Server-side failures (5xx) answer a generic message: their text may
    name stored files; it is logged instead.
    """
    for kind, status, code in ERROR_CODES:
        if isinstance(exc, kind):
            break
    else:
        status, code = 500, "transcriber_error"
    if status >= 500:
        LOG.warning("%s: %s", type(exc).__name__, exc)
        return error_response(status, code, SERVER_MESSAGE)
    extra: Dict[str, Any] = {}
    if isinstance(exc, SampleRejectedError):
        extra["reason"] = exc.reason
    return error_response(status, code, str(exc), **extra)


def session_json(
    session: Session, status: Optional[JobStatus] = None
) -> Dict[str, Any]:
    """Return what the pages need of a session (and of its job)."""
    data: Dict[str, Any] = session.model_dump(
        mode="json",
        include={
            "id", "title", "language", "started_at", "duration", "state",
            "chunks", "failure", "consent_version", "num_speakers",
            "expected_speakers", "type", "goal",
        },
    )
    if status is not None:
        data.update(
            stage=status.stage, fraction=status.fraction,
            message=status.message,
        )
    return data


def notice_json(notice: ConsentNotice) -> Dict[str, Any]:
    """Return a consent notice as the pages show it."""
    return notice.model_dump(mode="json")


def transcript_json(transcript: Transcript) -> Dict[str, Any]:
    """Return the transcript in the transcriber's own format."""
    return transcript.model_dump(mode="json")


def names_json(
    session_id: str, names: Optional[SpeakerNames], running: bool
) -> Dict[str, Any]:
    """Return the names found in a session as the API shows them.

    ``searched_at`` is ``null`` when names were never searched;
    ``running`` tells that a search (or the transcription that ends
    with one) is under way.
    """
    if names is None:
        return {
            "session": session_id, "running": running,
            "searched_at": None, "model": None, "seconds": None,
            "candidates": 0, "auto": [], "suggestions": [], "note": None,
        }
    return {**names.model_dump(mode="json"), "running": running}


def parse_json(model: Type[M], data: bytes, empty_ok: bool = False) -> M:
    """Validate a JSON body; an invalid one is a 422 ``invalid_request``.

    With ``empty_ok`` an empty body gives the model's defaults.
    """
    try:
        if not data.strip() and empty_ok:
            return model()
        return model.model_validate_json(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "body"
        raise ApiError(
            422, "invalid_request", f"{where}: {first.get('msg', '')}"
        ) from exc


async def read_body(request: Request, limit: int) -> bytes:
    """Read the request body, refusing it once it passes ``limit`` bytes.

    A ``Content-Length`` above the limit is refused before reading; a
    body streamed without one is cut off as soon as it is too long.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)
    body = bytearray()
    async for part in request.stream():
        body += part
        if len(body) > limit:
            raise _too_large(limit)
    return bytes(body)


def _too_large(limit: int) -> ApiError:
    """The error for a body over ``limit`` bytes."""
    return ApiError(
        413, "payload_too_large", f"The body is larger than {limit} bytes."
    )


def recording_format(filename: str, allowed: Tuple[str, ...]) -> str:
    """Return the audio format named by ``filename``'s extension.

    Only the extensions in ``allowed`` are accepted; anything else
    (none, ``.pcm``, ``.txt``, a control character) is a ``400
    unsupported_format``, answered before the body is read.
    """
    kind = PurePath(filename).suffix.lower().lstrip(".")
    if kind not in allowed:
        raise ApiError(
            400, "unsupported_format",
            "Upload an audio or video file: "
            + ", ".join(allowed) + ".",
        )
    return kind


def wav_response(
    rendering: PlaybackAudio, range_header: Optional[str]
) -> Response:
    """Answer a WAV rendering whole or as the one byte range asked for.

    The bytes are read from the rendering a block at a time as they are
    sent; an unsatisfiable range is a ``416`` and closes the rendering.
    """
    size = rendering.size
    headers = {"Accept-Ranges": "bytes"}
    wanted = parse_range(range_header, size)
    if isinstance(wanted, Unsatisfiable):
        rendering.close()
        return error_response(
            416, "range_not_satisfiable",
            f"The audio has {size} bytes.",
            {**headers, "Content-Range": f"bytes */{size}"},
        )
    status = 200
    first, last = 0, size - 1
    if wanted is not None:
        status = 206
        first, last = wanted
        headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    headers["Content-Length"] = str(last - first + 1)
    return StreamingResponse(
        iterate_in_threadpool(byte_blocks(rendering, first, last)),
        status, headers, media_type=WAV_TYPE,
    )


class SecurityHeaders:
    """ASGI middleware: no sniffing anywhere, no caching of the API."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``."""
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Add the headers to every HTTP response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        api = str(scope.get("path", "")).startswith("/api")

        async def sending(message: Message) -> None:
            """Add the headers to the start of the response."""
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                if api:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, sending)


def create_app(
    container: TranscriberContainer,
    access_code: str,
    now: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Build the API over ``container``, guarded by ``access_code``.

    The code must be at least :data:`MIN_CODE_LENGTH` characters.
    ``now`` is the clock of the failed-code throttling (tests move it).
    A new random playback token is made for every app built.
    """
    if len(access_code) < MIN_CODE_LENGTH:
        raise ValueError(
            f"The access code needs at least {MIN_CODE_LENGTH} characters."
        )
    limits = container.settings.api
    limiter = FailureLimiter(
        limits.failed_code_limit, limits.failed_code_window,
        limits.failed_code_cooldown, now,
    )
    enrolments = 0  # in progress or waiting; only touched on the event loop
    playback_token = new_playback_token()
    app = FastAPI(
        title="Transcriber",
        description="Recording, transcription and attribution of sessions.",
        docs_url=None, redoc_url=None, openapi_url=None,
    )
    app.add_middleware(SecurityHeaders)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        """Answer an :class:`ApiError`."""
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(TranscriberError)
    async def transcriber_error(
        request: Request, exc: TranscriberError
    ) -> JSONResponse:
        """Answer a domain error."""
        return domain_error(exc)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Answer a malformed path or query parameter."""
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()))
        return error_response(
            422, "invalid_request", f"{where}: {first.get('msg', '')}"
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        """Answer routing errors (unknown path, wrong method)."""
        code = HTTP_CODES.get(exc.status_code, "http_error")
        return error_response(
            exc.status_code, code, str(exc.detail), exc.headers
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        """Answer anything else without showing it."""
        LOG.error("Unexpected error on %s", request.url.path, exc_info=exc)
        return error_response(500, "internal_error", SERVER_MESSAGE)

    def admit(request: Request, cookie_too: bool) -> None:
        """Let the request through with the code (or, if allowed, cookie).

        A right code (or cookie) always passes, even from a client that
        is being throttled: the code is checked first, so wrong codes
        sent by someone else can never lock the owner out. A request
        with neither counts as one failure of its client, or gets 429
        while that client is blocked.
        """
        allowed = check_code(access_code, request.headers.get(ACCESS_HEADER))
        if not allowed and cookie_too:
            allowed = check_code(
                playback_token, request.cookies.get(PLAYBACK_COOKIE)
            )
        if allowed:
            return
        peer = request.client.host if request.client is not None else None
        key = client_key(peer, request.headers, limits.client_ip_header)
        wait = limiter.retry_after(key)
        if wait > 0:
            raise ApiError(
                429, "too_many_attempts",
                "Too many wrong access codes; try again later.",
                {"Retry-After": str(math.ceil(wait))},
            )
        limiter.failed(key)
        raise ApiError(401, "unauthorized", "A valid access code is required.")

    async def authorise(request: Request) -> None:
        """Let the request through only with the access code header."""
        admit(request, cookie_too=False)

    async def authorise_playback(request: Request) -> None:
        """Let the request through with the code or the playback cookie."""
        admit(request, cookie_too=True)

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        """Tell that the API is up; needs no code, reveals nothing."""
        return {"status": "ok"}

    api = APIRouter(prefix="/api", dependencies=[Depends(authorise)])
    playback = APIRouter(
        prefix="/api", dependencies=[Depends(authorise_playback)]
    )
    c = container

    @api.post("/access/cookie", status_code=204)
    async def playback_cookie() -> Response:
        """Set the cookie that lets the audio player in (needs the code).

        The cookie holds the playback token, not the access code; it is
        ``HttpOnly``, ``SameSite=Strict``, limited to ``/api`` and
        ``Secure`` unless the settings turn that off for local http.
        """
        response = Response(status_code=204)
        response.set_cookie(
            PLAYBACK_COOKIE, playback_token, path=PLAYBACK_COOKIE_PATH,
            httponly=True, secure=limits.cookie_secure, samesite="strict",
        )
        return response

    @api.post("/sessions", status_code=201)
    async def open_session(request: Request) -> Dict[str, Any]:
        """Open a session to record into."""
        body = parse_json(
            NewSession, await read_body(request, limits.json_max_bytes)
        )
        session = await run_in_threadpool(
            c.open_session, body.id, body.language, body.title,
            body.consent_version, body.num_speakers, body.expected_speakers,
            body.type, body.goal,
        )
        return session_json(session)

    @api.put("/sessions/{session_id}/chunks/{number}")
    async def append_chunk(
        session_id: str, number: int, request: Request
    ) -> Dict[str, Any]:
        """Store one numbered chunk of raw PCM (16 kHz, mono, 16-bit).

        ``stored`` is ``false`` when the chunk was the last one again
        (a retried upload whose answer was lost).
        """
        data = await read_body(request, limits.chunk_max_bytes)
        stored = await run_in_threadpool(
            c.append_audio, session_id, number, data
        )
        return {"stored": stored}

    @api.post("/sessions/{session_id}/close")
    async def close_session(session_id: str) -> Dict[str, Any]:
        """End the recording; the session is then ready to transcribe."""
        return session_json(
            await run_in_threadpool(c.close_session, session_id)
        )

    @api.post("/sessions/{session_id}/upload", status_code=201)
    async def upload(
        session_id: str,
        request: Request,
        filename: str = Query(min_length=1, max_length=255),
        language: str = Query(min_length=2, max_length=16),
        title: Optional[str] = Query(default=None, max_length=200),
        consent_version: Optional[str] = Query(default=None, max_length=32),
        conversation: Optional[str] = Query(
            default=None, alias="type", max_length=64
        ),
        goal: Optional[str] = Query(default=None, max_length=2000),
    ) -> Dict[str, Any]:
        """Create a session from a whole recording sent as the body.

        The file name and the size are checked before the session is
        created, so a refused upload leaves nothing behind and can be
        sent again with the same id. ``consent_version`` is the
        recording notice the page showed; ``type`` and ``goal``
        describe the conversation.
        """
        kind = recording_format(filename, limits.upload_formats)
        data = await read_body(request, limits.upload_max_bytes)
        session = await run_in_threadpool(
            c.import_recording, session_id, language,
            RecordedAudio(data, kind), title, consent_version,
            conversation, goal,
        )
        return session_json(session)

    @api.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        """List the sessions with their jobs.

        Each session carries its stage, fraction and the last note of
        its job (``message``: voiceprints skipped...); ``unreadable``
        counts damaged records.
        """

        def listed() -> Dict[str, Any]:
            """Read the sessions and the status of each."""
            listing = c.list_sessions()
            return {
                "sessions": [
                    session_json(s, c.transcription_status.of(s))
                    for s in listing.sessions
                ],
                "unreadable": listing.unreadable,
            }

        return await run_in_threadpool(listed)

    @api.get("/sessions/{session_id}")
    async def show_session(session_id: str) -> Dict[str, Any]:
        """Show a session with its state, stage, fraction and message."""
        view = await run_in_threadpool(c.show_session, session_id)
        return session_json(view.session, view.status)

    @api.patch("/sessions/{session_id}")
    async def describe(session_id: str, request: Request) -> Dict[str, Any]:
        """Set or change the conversation type and goal, in any state."""
        body = parse_json(
            DescribeRequest, await read_body(request, limits.json_max_bytes)
        )
        changes = {key: getattr(body, key) for key in body.model_fields_set}
        return session_json(await run_in_threadpool(
            c.describe_session, session_id, changes
        ))

    @api.post("/sessions/{session_id}/transcription", status_code=202)
    async def queue(session_id: str, request: Request) -> Dict[str, Any]:
        """Queue the transcription; the body's fields are all optional."""
        body = parse_json(
            TranscriptionRequest,
            await read_body(request, limits.json_max_bytes), empty_ok=True,
        )
        options = TranscriptionOptions(
            language=body.language, num_speakers=body.num_speakers,
            expected_speakers=body.expected_speakers,
        )
        session = await run_in_threadpool(
            c.queue_transcription, session_id, options
        )
        return session_json(session)

    @api.get("/sessions/{session_id}/transcript")
    async def transcript(session_id: str) -> Dict[str, Any]:
        """Return the transcript, with words, for the player."""
        return transcript_json(
            await run_in_threadpool(c.show_transcript, session_id)
        )

    @playback.get("/sessions/{session_id}/audio")
    async def audio(session_id: str, request: Request) -> Response:
        """Serve the audio as WAV, whole or as one byte range.

        The only route that also takes the playback cookie. The bytes
        are read from the rendering a block at a time as they are sent.
        """
        rendering = await run_in_threadpool(c.session_audio, session_id)
        return wav_response(rendering, request.headers.get("range"))

    @api.patch("/sessions/{session_id}/speakers/{label}")
    async def rename(
        session_id: str, label: str, request: Request
    ) -> Dict[str, Any]:
        """Name an anonymous voice, or merge it into another speaker."""
        body = parse_json(
            RenameRequest, await read_body(request, limits.json_max_bytes)
        )
        return transcript_json(await run_in_threadpool(
            c.rename_speaker, session_id, label, body.name, body.merge_into
        ))

    @api.patch("/sessions/{session_id}/utterances/{utterance_id}")
    async def reassign(
        session_id: str, utterance_id: str, request: Request
    ) -> Dict[str, Any]:
        """Give one sentence to another speaker."""
        body = parse_json(
            ReassignRequest, await read_body(request, limits.json_max_bytes)
        )
        return transcript_json(await run_in_threadpool(
            c.reassign_utterance, session_id, utterance_id, body.speaker
        ))

    @api.get("/sessions/{session_id}/names")
    async def names(session_id: str) -> Dict[str, Any]:
        """Return the names given and suggested from what people said."""
        found, running = await run_in_threadpool(c.show_names, session_id)
        return names_json(session_id, found, running)

    @api.post("/sessions/{session_id}/names", status_code=202)
    async def search_names(session_id: str) -> Dict[str, Any]:
        """Search a finished session for names again (a background job)."""
        await run_in_threadpool(c.queue_names, session_id)
        found, running = await run_in_threadpool(c.show_names, session_id)
        return names_json(session_id, found, running)

    @api.delete("/sessions/{session_id}/names/{label}")
    async def undo_name(session_id: str, label: str) -> Dict[str, Any]:
        """Undo a name given automatically: the voice is ``S#`` again."""
        return transcript_json(await run_in_threadpool(
            c.undo_auto_name, session_id, label
        ))

    @api.delete("/sessions/{session_id}", status_code=204)
    async def delete(session_id: str) -> Response:
        """Delete the session with its transcript and lab data (409: busy)."""
        await run_in_threadpool(c.delete_session, session_id)
        return Response(status_code=204)

    @api.post("/sessions/{session_id}/publish")
    async def publish(session_id: str) -> Dict[str, Any]:
        """Hand the transcript over to the RAG; return the document."""
        return await run_in_threadpool(c.publish, session_id)

    @api.get("/analysis/judge")
    async def analysis_judge() -> Dict[str, Any]:
        """Return the RAG's analysis judge and Claude's quota status."""
        return await run_in_threadpool(c.analysis_judge.status)

    @api.put("/analysis/judge")
    async def choose_analysis_judge(request: Request) -> Dict[str, Any]:
        """Set the judge of the RAG's next analyses (forwarded on loopback)."""
        body = parse_json(JudgeRequest, await read_body(request, 1024))
        return await run_in_threadpool(c.analysis_judge.choose, body.mode)

    @api.post("/analysis/sessions/{session_id}", status_code=202)
    async def start_analysis(
        session_id: str, request: Request
    ) -> Dict[str, Any]:
        """Start the RAG's whole analysis of a published session."""
        body = parse_json(
            AnalysisRequest, await read_body(request, 4 * MAX_JOB + 1024),
            empty_ok=True,
        )
        return await run_in_threadpool(
            c.session_analysis.start, session_id, body.restart,
            body.synthetic, body.steps, body.job, body.claude_synthesis,
            body.claude_models,
        )

    @api.get("/analysis/sessions/{session_id}")
    async def analysis_status(session_id: str) -> Dict[str, Any]:
        """Return the steps of a session's analysis (forwarded)."""
        return await run_in_threadpool(c.session_analysis.status, session_id)

    @api.delete("/analysis/sessions/{session_id}/run", status_code=204)
    async def cancel_analysis(session_id: str) -> Response:
        """Stop the session's running analysis (forwarded)."""
        await run_in_threadpool(c.session_analysis.cancel, session_id)
        return Response(status_code=204)

    @api.get("/analysis/sessions/{session_id}/steps/{step}")
    async def analysis_step(session_id: str, step: str) -> Any:
        """Return one step's result (forwarded)."""
        return await run_in_threadpool(
            c.session_analysis.result, session_id, step
        )

    @api.get("/analysis/sessions/{session_id}/reports/{position}")
    async def analysis_report(session_id: str, position: int) -> Response:
        """Return one person's HTML report as a download (octet-stream)."""
        page = await run_in_threadpool(
            c.session_analysis.report, session_id, position
        )
        # Bytes to save, never a page of this origin.
        return Response(
            page.encode("utf-8"), media_type="application/octet-stream",
            headers={
                **REPORT_HEADERS,
                "Content-Disposition": (
                    f'attachment; filename="{session_id}-informe-'
                    f'{position}.html"'
                ),
            },
        )

    mount_knowledge(api, c)

    @api.get("/speakers")
    async def speakers() -> Dict[str, Any]:
        """List the people enrolled with the voice model in use."""
        found = await run_in_threadpool(c.list_speakers)
        return {"speakers": [s._asdict() for s in found]}

    @api.get("/enrolment/text")
    async def text(
        language: str = Query(min_length=2, max_length=16),
        name: Optional[str] = Query(default=None, max_length=100),
    ) -> Dict[str, Any]:
        """Return the text to read at enrolment (with the name, if given).

        ``consent_version`` is the consent statement an enrolment made
        now is recorded against, and ``consent`` that statement (title,
        points, sentence to accept) in the language.
        """
        guide = c.enrolment_text(language, name)
        return {
            "language": guide.text.language,
            "version": guide.text.version,
            "template": guide.text.template,
            "text": guide.reading,
            "consent_version": guide.consent_version,
            "consent": notice_json(guide.notice),
        }

    @api.get("/consent")
    async def consent_notice(
        kind: str = Query(min_length=1, max_length=16),
        language: str = Query(min_length=2, max_length=16),
    ) -> Dict[str, Any]:
        """Return the consent notice in force (``recording``/``enrolment``)."""
        return notice_json(c.consent_notice(kind, language))

    @api.post("/speakers/{speaker_id}/enrolment", status_code=201)
    async def enrol(
        speaker_id: str,
        request: Request,
        name: str = Query(min_length=1, max_length=100),
        language: str = Query(min_length=2, max_length=16),
        consent: str = Query(default=""),
        format: str = Query(default=PCM_FORMAT, max_length=32),
        text_version: Optional[str] = Query(default=None, max_length=32),
    ) -> Dict[str, Any]:
        """Enrol a voice from a sample read aloud; needs ``consent=true``.

        Consent is checked before the body is read. The call waits for
        the models (it queues behind a running transcription); when
        ``max_pending_enrolments`` enrolments are already in progress or
        waiting, it is refused at once with ``409 busy`` and a
        ``Retry-After`` header instead of holding a worker thread.
        """
        nonlocal enrolments
        if consent != "true":
            raise ConsentRequiredError(
                "A voice is only enrolled with the person's explicit"
                " consent (consent=true)."
            )
        if enrolments >= limits.max_pending_enrolments:
            raise ApiError(
                409, "busy",
                "The workstation is busy with other enrolments; try"
                " again shortly.",
                {"Retry-After": str(limits.busy_retry_after)},
            )
        enrolments += 1
        try:
            data = await read_body(request, limits.enrolment_max_bytes)
            voiceprint = await run_in_threadpool(
                c.enrol_speaker, speaker_id, name, language,
                RecordedAudio(data, format), True,
                c.enrolment_text.consent_version, text_version,
            )
        finally:
            enrolments -= 1
        return {
            "id": voiceprint.speaker_id,
            "model": voiceprint.model,
            "enrolled_at": voiceprint.enrolled_at.isoformat(),
        }

    @api.delete("/speakers/{speaker_id}", status_code=204)
    async def forget(speaker_id: str) -> Response:
        """Forget a person's voiceprint and consent."""
        await run_in_threadpool(c.forget_speaker, speaker_id)
        return Response(status_code=204)

    if c.lab is not None:
        from transcriber.adapters.inbound.lab_api import add_lab_routes

        add_lab_routes(api, playback, c.lab, c.settings)
    app.include_router(api)
    app.include_router(playback)
    if limits.static_dir is not None:
        app.mount(
            "/", StaticFiles(directory=limits.static_dir, html=True),
            name="pages",
        )
    return app


def mount_knowledge(api: APIRouter, c: TranscriberContainer) -> None:
    """Add ``/api/knowledge/...``: the RAG's knowledge routes, forwarded.

    Same paths, bodies and statuses as the RAG's ``/knowledge/...``
    (``docs/KNOWLEDGE.md``); the access code guards them here and the
    RAG's bearer token is added on the way.
    """
    k = c.knowledge

    @api.get("/knowledge/domains")
    async def knowledge_domains() -> Any:
        """List the knowledge domains."""
        return await run_in_threadpool(k.domains)

    @api.post("/knowledge/domains", status_code=201)
    async def create_knowledge_domain(request: Request) -> Any:
        """Create a domain (``{"id", "title", "language"}``)."""
        body = await read_body(request, JSON_BODY_BYTES)
        return await run_in_threadpool(k.create_domain, body)

    @api.delete("/knowledge/domains/{domain_id}", status_code=204)
    async def delete_knowledge_domain(domain_id: str) -> Response:
        """Delete a domain with its documents and index."""
        await run_in_threadpool(k.delete_domain, domain_id)
        return Response(status_code=204)

    @api.get("/knowledge/domains/{domain_id}/documents")
    async def knowledge_documents(domain_id: str) -> Any:
        """List a domain's documents and their status."""
        return await run_in_threadpool(k.documents, domain_id)

    @api.post("/knowledge/domains/{domain_id}/documents", status_code=202)
    async def upload_knowledge_document(
        domain_id: str, request: Request
    ) -> Any:
        """Upload a file (multipart ``file`` + optional ``metadata``)."""
        body = await read_body(request, UPLOAD_BODY_BYTES)
        return await run_in_threadpool(
            k.upload, domain_id, body,
            request.headers.get("content-type", ""),
        )

    @api.patch("/knowledge/domains/{domain_id}/documents/{doc_id}")
    async def relabel_knowledge_document(
        domain_id: str, doc_id: str, request: Request
    ) -> Any:
        """Replace a document's metadata (``{"metadata": {...}}``)."""
        body = await read_body(request, JSON_BODY_BYTES)
        return await run_in_threadpool(k.relabel, domain_id, doc_id, body)

    @api.delete(
        "/knowledge/domains/{domain_id}/documents/{doc_id}", status_code=204
    )
    async def delete_knowledge_document(
        domain_id: str, doc_id: str
    ) -> Response:
        """Delete a document."""
        await run_in_threadpool(k.delete_document, domain_id, doc_id)
        return Response(status_code=204)

    @api.post("/knowledge/query")
    async def knowledge_query(request: Request) -> Any:
        """Answer a ``knowledge.query.v1`` (forwarded)."""
        body = await read_body(request, JSON_BODY_BYTES)
        return await run_in_threadpool(k.query, body)
