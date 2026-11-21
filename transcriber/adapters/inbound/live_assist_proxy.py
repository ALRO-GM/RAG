"""``/api/assist/...``: the orchestrator's live-page API, forwarded.

The web pages are served by the transcriber, and the orchestrator
(``python -m orchestrator serve``, ``docs/platform/ORCHESTRATOR.md``)
listens on loopback only, so the page reaches it through this thin
proxy: the access code guards the route here (the ``api`` router's
dependency), and the orchestrator's bearer token is added on the way
(the RAG token file, shared by the services of the workstation).

Only the orchestrator's own routes exist here (``products``,
``transcripts``, ``sessions``, one session, its ``replay``, ``stop``
and ``suggestions/stream``), and session ids are checked; the stream is passed
through as it comes (Server-Sent Events). It is an inbound adapter that
talks HTTP itself on purpose: it adds no behaviour to the transcriber,
so there is no use case to put behind it.

``TRANSCRIBER_ORCHESTRATOR_URL`` changes the address (default
``http://127.0.0.1:8772``).
"""

import os
import re
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

DEFAULT_URL = "http://127.0.0.1:8772"
URL_VARIABLE = "TRANSCRIBER_ORCHESTRATOR_URL"
_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
ALLOWED = re.compile(
    rf"^(?:products|transcripts|sessions|sessions/{_ID}"
    rf"|sessions/{_ID}/(?:replay|stop|suggestions/stream))$"
)
MAX_BODY = 16 * 1024
TIMEOUT = 45.0
"""Seconds for a plain call (starting a live session waits for the GPU)."""


def read_token(path: Optional[Path]) -> Optional[str]:
    """Return the token kept in ``path`` (``None`` when there is none)."""
    if path is None:
        return None
    try:
        token = path.expanduser().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def unavailable(message: str, status: int = 502) -> JSONResponse:
    """The answer when the orchestrator cannot be reached."""
    return JSONResponse(
        {"code": "orchestrator_unavailable", "message": message},
        status_code=status,
    )


def add_live_assist_proxy(
    api: APIRouter,
    token_file: Optional[Path],
    base_url: Optional[str] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> None:
    """Add the forwarding route to ``api`` (prefix ``/api``).

    Args:
        api: The router guarded by the access code.
        token_file: The orchestrator's bearer token (read per request,
            so a new token needs no restart).
        base_url: The orchestrator (default: the environment, then
            :data:`DEFAULT_URL`).
        transport: An ``httpx`` transport (replaced in tests).
    """
    base = (base_url or os.environ.get(URL_VARIABLE) or DEFAULT_URL)
    base = base.rstrip("/")

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=TIMEOUT)

    @api.get("/assist/products")
    async def assist_products(request: Request) -> Response:
        """The orchestrator's live products."""
        return await forward("products", request)

    @api.get("/assist/transcripts")
    async def assist_transcripts(request: Request) -> Response:
        """Recorded conversations a demo replay can use."""
        return await forward("transcripts", request)

    @api.get("/assist/sessions")
    async def assist_sessions(request: Request) -> Response:
        """The orchestrator's sessions."""
        return await forward("sessions", request)

    @api.post("/assist/sessions")
    async def assist_start(request: Request) -> Response:
        """Start a live session."""
        return await forward("sessions", request)

    @api.get("/assist/sessions/{session_id}")
    async def assist_session(session_id: str, request: Request) -> Response:
        """One session."""
        return await forward(f"sessions/{session_id}", request)

    @api.post("/assist/sessions/{session_id}/replay")
    async def assist_replay(session_id: str, request: Request) -> Response:
        """Start a demo replay."""
        return await forward(f"sessions/{session_id}/replay", request)

    @api.post("/assist/sessions/{session_id}/stop")
    async def assist_stop(session_id: str, request: Request) -> Response:
        """Stop a session."""
        return await forward(f"sessions/{session_id}/stop", request)

    @api.get("/assist/sessions/{session_id}/suggestions/stream")
    async def assist_stream(session_id: str, request: Request) -> Response:
        """The session's Server-Sent Events."""
        return await forward(f"sessions/{session_id}/suggestions/stream",
                             request)

    async def forward(path: str, request: Request) -> Response:
        """Forward one call of the live page to the orchestrator."""
        if not ALLOWED.match(path):
            return JSONResponse(
                {"code": "not_found", "message": "unknown live route"},
                status_code=404,
            )
        headers = {"Content-Type": "application/json"}
        token = read_token(token_file)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        last = request.headers.get("last-event-id")
        if last and last.isdigit():
            headers["Last-Event-ID"] = last
        after = request.query_params.get("after")
        params = {"after": after} if after and after.isdigit() else None
        body = await request.body() if request.method == "POST" else b""
        if len(body) > MAX_BODY:
            return JSONResponse(
                {"code": "too_large", "message": "body too large"},
                status_code=413,
            )
        url = f"{base}/{path}"
        if path.endswith("/suggestions/stream"):
            return await _stream(client(), url, headers, params)
        try:
            async with client() as http:
                answer = await http.request(
                    request.method, url, content=body or None,
                    headers=headers, params=params,
                )
        except httpx.HTTPError as exc:
            return unavailable(type(exc).__name__)
        return Response(
            answer.content, status_code=answer.status_code,
            media_type=answer.headers.get("content-type",
                                          "application/json"),
        )


async def _stream(
    http: httpx.AsyncClient,
    url: str,
    headers: "dict[str, str]",
    params: "Optional[dict[str, str]]",
) -> Response:
    """Pass a Server-Sent Events stream through, chunk by chunk."""
    try:
        upstream = await http.send(
            http.build_request("GET", url, headers=headers, params=params,
                               timeout=httpx.Timeout(TIMEOUT, read=None)),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await http.aclose()
        return unavailable(type(exc).__name__)
    if upstream.status_code != 200:
        content = await upstream.aread()
        await upstream.aclose()
        await http.aclose()
        return Response(content, status_code=upstream.status_code,
                        media_type="application/json")

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except httpx.HTTPError:
            return
        finally:
            await upstream.aclose()
            await http.aclose()

    return StreamingResponse(
        chunks(), media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
