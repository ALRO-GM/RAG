"""``python -m orchestrator serve``: the live page's API (port 8772).

Routes (JSON unless said otherwise; ``docs/platform/ORCHESTRATOR.md``):

- ``GET /health``: no token.
- ``GET /products``: the live products and what the page shows of them.
- ``GET /transcripts``: recorded conversations a replay can use.
- ``GET /sessions``, ``GET /sessions/{id}``.
- ``POST /sessions`` ``{"product", "audio": "stereo"|"mic",
  "session_id"?, "employee"?: {"id", "name"}}``: start a live session
  on the transcriber.
- ``POST /sessions/{id}/replay`` ``{"product", "transcript",
  "speed"?, "employee"?}``: replay a transcript as a live session
  (``speed`` 1 is real time, ``null`` as fast as the services answer).
- ``POST /sessions/{id}/stop``.
- ``POST /sessions/{id}/mode`` ``{"mode": "pincelada"|"guiado"}``,
  ``POST /sessions/{id}/speakers/{speaker}`` ``{"name"}`` and
  ``POST /sessions/{id}/mandatory/{item}`` ``{"confirm": true}``: switch
  the assistance mode, name a speaker, tick a mandatory item (LV-16).
- ``POST /sessions/{id}/barge-in``: the user spoke over the voice bot
  (the page's voice detector); ``{"stopped": bool}``.
- ``POST /voicebot/sample`` ``{"choices": {"voicebot.voice": ...}}``:
  the voice bot's fixed sample sentence said with the choices being
  edited on «Ajustes» (unsaved), as ``audio/wav``; the headers
  ``X-Voice-Clone`` (``1`` when the cloned voice spoke),
  ``X-Voice-Length-Scale`` and ``X-Voice-Note`` say what was used.
- ``GET /sessions/{id}/suggestions/stream``: Server-Sent Events, one
  ``message`` per feed message (``id`` is its number; ``?after=n`` or
  the ``Last-Event-ID`` header resumes), a comment every
  ``keepalive`` seconds, and the stream ends with the session.

With a token every route but ``/health`` needs ``Authorization: Bearer
<token>``. The server listens on a loopback address only, like the
other services of the workstation.
"""

import asyncio
import json
import secrets
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from orchestrator.application.sessions import LiveSessions, SessionError

STATUS = {
    "not_found": 404, "busy": 409, "wrong_state": 409,
    "invalid_request": 422, "transcriber": 502, "unavailable": 503,
    "unauthorized": 401,
}
MAX_BODY = 16 * 1024


def create_app(
    sessions: LiveSessions,
    token: Optional[str] = None,
    keepalive: float = 15.0,
) -> FastAPI:
    """Build the API over ``sessions``.

    Args:
        sessions: The use case.
        token: Bearer token required on every route but ``/health``.
        keepalive: Seconds between SSE comments when nothing happens.
    """
    app = FastAPI(
        title="Orchestrator", docs_url=None, redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(SessionError)
    async def session_error(
        request: Request, exc: SessionError
    ) -> JSONResponse:
        """Answer a refused request with its stable code."""
        return JSONResponse(
            {"code": exc.code, "message": str(exc)},
            status_code=STATUS.get(exc.code, 400),
        )

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        """Ask for the bearer token, and keep answers out of caches."""
        if token and request.url.path != "/health":
            given = request.headers.get("authorization", "")
            if not secrets.compare_digest(given.encode(),
                                          f"Bearer {token}".encode()):
                return JSONResponse(
                    {"code": "unauthorized", "message": "bearer token"},
                    status_code=401,
                )
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    async def body(request: Request) -> Dict[str, Any]:
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise SessionError("invalid_request", "body too large")
        try:
            data = json.loads(raw or b"{}")
        except ValueError as exc:
            raise SessionError("invalid_request", "body is not JSON") from exc
        if not isinstance(data, dict):
            raise SessionError("invalid_request", "body is not an object")
        return data

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Say the service is up."""
        return {"status": "ok"}

    @app.get("/products")
    async def products() -> Dict[str, Any]:
        """The live products."""
        return {"products": sessions.list_products()}

    @app.get("/transcripts")
    async def transcripts() -> Dict[str, Any]:
        """Conversations for a replay."""
        found = await run_in_threadpool(sessions.list_transcripts)
        return {"transcripts": found}

    @app.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        """Every session kept."""
        return {"sessions": sessions.list_sessions()}

    @app.get("/sessions/{session_id}")
    async def show(session_id: str) -> Dict[str, Any]:
        """One session."""
        return sessions.get(session_id)

    @app.post("/sessions", status_code=201)
    async def start_live(request: Request) -> Dict[str, Any]:
        """Start a live session."""
        data = await body(request)
        return await run_in_threadpool(
            sessions.start_live, _text(data, "product"),
            _text(data, "audio"), data.get("session_id") or None,
            data.get("employee"),
        )

    @app.post("/sessions/{session_id}/replay", status_code=201)
    async def replay(session_id: str, request: Request) -> Dict[str, Any]:
        """Start a replay."""
        data = await body(request)
        speed = data.get("speed", 1.0)
        if speed is not None and (
            isinstance(speed, bool) or not isinstance(speed, (int, float))
        ):
            raise SessionError("invalid_request", "speed must be a number")
        return await run_in_threadpool(
            sessions.start_replay, session_id, _text(data, "product"),
            _text(data, "transcript"),
            float(speed) if speed is not None else None,
            data.get("employee"),
        )

    @app.post("/sessions/{session_id}/stop")
    async def stop(session_id: str) -> Dict[str, Any]:
        """Stop a session."""
        return await run_in_threadpool(sessions.stop, session_id)

    @app.post("/sessions/{session_id}/mode")
    async def mode(session_id: str, request: Request) -> Dict[str, Any]:
        """Switch the assistance mode of a session."""
        data = await body(request)
        return sessions.set_mode(session_id, _text(data, "mode"))

    @app.post("/sessions/{session_id}/speakers/{speaker}")
    async def rename(
        session_id: str, speaker: str, request: Request
    ) -> Dict[str, Any]:
        """Give a speaker of a session a name."""
        data = await body(request)
        return await run_in_threadpool(
            sessions.rename, session_id, speaker, data.get("name")
        )

    @app.post("/sessions/{session_id}/mandatory/{item}")
    async def tick(
        session_id: str, item: str, request: Request
    ) -> Dict[str, Any]:
        """Tick a mandatory item by hand (``confirm`` must be true)."""
        data = await body(request)
        return sessions.tick(session_id, item, data.get("confirm") is True)

    @app.post("/sessions/{session_id}/barge-in")
    async def barge_in(session_id: str) -> Dict[str, Any]:
        """Stop the voice bot's turn and playback."""
        return await run_in_threadpool(sessions.barge_in, session_id)

    @app.post("/voicebot/sample")
    async def sample(request: Request) -> Response:
        """Say the sample sentence with the voice being chosen."""
        data = await body(request)
        choices = data.get("choices") or {}
        if not isinstance(choices, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and len(v) <= 80
            and k.startswith("voicebot.") for k, v in choices.items()
        ):
            raise SessionError("invalid_request",
                               "choices must be voicebot.* names")
        wav, used = await run_in_threadpool(sessions.sample, choices)
        return Response(wav, media_type="audio/wav", headers={
            "X-Voice-Clone": "1" if used["clone"] else "0",
            "X-Voice-Length-Scale": f"{used['length_scale']:g}",
            "X-Voice-Note": str(used["reason"])[:200].encode(
                "ascii", "replace").decode("ascii"),
        })

    @app.get("/sessions/{session_id}/suggestions/stream")
    async def stream(
        session_id: str, request: Request, after: Optional[int] = None
    ) -> StreamingResponse:
        """The session's feed as Server-Sent Events."""
        feed = sessions.feed(session_id)
        start = after
        if start is None:
            header = request.headers.get("last-event-id", "")
            start = int(header) if header.isdigit() else 0

        async def events() -> AsyncIterator[str]:
            seen = max(0, start)
            yield "retry: 2000\n\n"
            while True:
                found, finished = await run_in_threadpool(
                    feed.read, seen, keepalive
                )
                for number, message in found:
                    seen = number
                    data = json.dumps(message, ensure_ascii=False)
                    yield f"id: {number}\nevent: message\ndata: {data}\n\n"
                if finished:
                    yield "event: end\ndata: {}\n\n"
                    return
                if not found:
                    yield ": keepalive\n\n"
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    return app


def _text(data: Dict[str, Any], key: str) -> str:
    """Return ``data[key]`` as a non-empty string or refuse the body."""
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SessionError("invalid_request", f"{key} is required")
    return value
