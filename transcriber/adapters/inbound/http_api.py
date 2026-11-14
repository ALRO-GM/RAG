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
from transcriber.application.ports import PCM_FORMAT, JobStatus, RecordedAudio
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.bootstrap import TranscriberContainer
from transcriber.domain.errors import (
    ConsentRequiredError,
    ContractError,
    InvalidInputError,
    ModelLoadError,
    NotFoundError,
    SampleRejectedError,
    StateError,
    StorageError,
    TranscriberError,
    VoiceModelMismatchError,
)
from transcriber.domain.models import Session, Transcript

LOG = logging.getLogger(__name__)

ERROR_CODES: Tuple[Tuple[Type[TranscriberError], int, str], ...] = (
    (ConsentRequiredError, 400, "consent_required"),
    (SampleRejectedError, 422, "sample_rejected"),
    (VoiceModelMismatchError, 409, "voice_model_mismatch"),
    (InvalidInputError, 400, "invalid_input"),
    (NotFoundError, 404, "not_found"),
    (StateError, 409, "wrong_state"),
    (ModelLoadError, 503, "model_unavailable"),
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
    """Body of ``POST /api/sessions``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    language: str
    title: Optional[str] = None


class TranscriptionRequest(BaseModel):
    """Body of ``POST /api/sessions/{id}/transcription`` (all optional)."""

    model_config = ConfigDict(extra="forbid")

    language: Optional[str] = None
    num_speakers: Optional[int] = Field(default=None, ge=1)
    expected_speakers: Optional[List[str]] = None


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
            "chunks", "failure",
        },
    )
    if status is not None:
        data.update(
            stage=status.stage, fraction=status.fraction,
            message=status.message,
        )
    return data


def transcript_json(transcript: Transcript) -> Dict[str, Any]:
    """Return the transcript in the transcriber's own format."""
    return transcript.model_dump(mode="json")


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
            c.open_session, body.id, body.language, body.title
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
    ) -> Dict[str, Any]:
        """Create a session from a whole recording sent as the body.

        The file name and the size are checked before the session is
        created, so a refused upload leaves nothing behind and can be
        sent again with the same id.
        """
        kind = recording_format(filename, limits.upload_formats)
        data = await read_body(request, limits.upload_max_bytes)
        session = await run_in_threadpool(
            c.import_recording, session_id, language,
            RecordedAudio(data, kind), title,
        )
        return session_json(session)

    @api.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        """List the sessions; ``unreadable`` counts damaged records."""
        listing = await run_in_threadpool(c.list_sessions)
        return {
            "sessions": [session_json(s) for s in listing.sessions],
            "unreadable": listing.unreadable,
        }

    @api.get("/sessions/{session_id}")
    async def show_session(session_id: str) -> Dict[str, Any]:
        """Show a session with its state, stage, fraction and message."""
        view = await run_in_threadpool(c.show_session, session_id)
        return session_json(view.session, view.status)

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
        size = rendering.size
        headers = {"Accept-Ranges": "bytes"}
        wanted = parse_range(request.headers.get("range"), size)
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

    @api.delete("/sessions/{session_id}", status_code=204)
    async def delete(session_id: str) -> Response:
        """Delete the session with its audio, words and transcript."""
        await run_in_threadpool(c.delete_session, session_id)
        return Response(status_code=204)

    @api.post("/sessions/{session_id}/publish")
    async def publish(session_id: str) -> Dict[str, Any]:
        """Hand the transcript over to the RAG; return the document."""
        return await run_in_threadpool(c.publish, session_id)

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
        now is recorded against.
        """
        guide = c.enrolment_text(language, name)
        return {
            "language": guide.text.language,
            "version": guide.text.version,
            "template": guide.text.template,
            "text": guide.reading,
            "consent_version": guide.consent_version,
        }

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

    app.include_router(api)
    app.include_router(playback)
    if limits.static_dir is not None:
        app.mount(
            "/", StaticFiles(directory=limits.static_dir, html=True),
            name="pages",
        )
    return app
