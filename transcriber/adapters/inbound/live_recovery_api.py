"""Inbound adapter: the upload of a live session's lost audio (#265).

Routes (under ``/api``, behind the access code like every other):

``PUT /api/live/sessions/{id}/recovery/chunks/{n}?offset=&channels=``
    Raw body: chunk ``n`` of 16-bit PCM at 16 kHz (``channels``
    interleaved, default 1), starting at stream sample ``offset``. A
    chunk sent again replaces itself (resumable, idempotent). ``413``
    past the size limits (one chunk, one session, every session), ``404``
    for a session this service never ran, ``409`` while the session is
    still on, for a product that keeps no audio, or once the upload is
    closed; ``400`` for audio past the time the session lasted. Answers
    the job.
``POST /api/live/sessions/{id}/recovery/close``
    ``{"count": n, "expires_at"?}``: every chunk was sent; the job is
    queued (``409`` while chunks are missing). Answers the job.
``GET /api/live/sessions/{id}/recovery``
    The job: ``state`` (``uploading``, ``queued``, ``transcribing``,
    ``merging``, ``publishing``, ``analysing``, ``rejudging``, ``done``,
    ``failed``), ``progress``, ``minutes`` recovered, the ``chunks``
    received (to resume), ``expires_at``. No audio, no words.
``DELETE /api/live/sessions/{id}/recovery``
    Delete everything kept for the session now (``204``).
``GET /api/live/recoveries``
    Every job, newest first.
"""

import json
from typing import Any, Dict

from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from transcriber.adapters.inbound.http_api import read_body
from transcriber.application.live_recovery import LiveRecovery
from transcriber.domain.errors import InvalidInputError


def add_recovery_routes(
    api: APIRouter, recovery: LiveRecovery, chunk_limit: int,
    json_limit: int,
) -> None:
    """Add the recovery routes to ``api`` (prefix ``/api``)."""

    @api.put("/live/sessions/{session_id}/recovery/chunks/{number}")
    async def put_chunk(
        session_id: str, number: int, request: Request,
        offset: int = Query(ge=0, le=2 ** 40),
        channels: int = Query(default=1, ge=1, le=2),
    ) -> Dict[str, Any]:
        """Keep one chunk of the session's lost audio."""
        data = await read_body(request, chunk_limit)
        return await run_in_threadpool(
            recovery.put_chunk, session_id, number, offset, data, channels)

    @api.post("/live/sessions/{session_id}/recovery/close")
    async def close(session_id: str, request: Request) -> Dict[str, Any]:
        """Every chunk was sent: queue the recovery."""
        try:
            body = json.loads(await read_body(request, json_limit) or b"{}")
        except ValueError as exc:
            raise InvalidInputError("the body is not JSON") from exc
        count = body.get("count") if isinstance(body, dict) else None
        expires = body.get("expires_at") if isinstance(body, dict) else None
        if not isinstance(count, int) or isinstance(count, bool) or (
                expires is not None and not isinstance(expires, str)):
            raise InvalidInputError("count (and expires_at) are needed")
        return await run_in_threadpool(recovery.close, session_id, count,
                                       expires)

    @api.get("/live/sessions/{session_id}/recovery")
    async def status(session_id: str) -> Dict[str, Any]:
        """Where the session's recovery is (read off the event loop)."""
        found: Dict[str, Any] = await run_in_threadpool(
            recovery.status, session_id)
        return found

    @api.delete("/live/sessions/{session_id}/recovery", status_code=204)
    async def discard(session_id: str) -> Response:
        """Delete everything kept for the session."""
        await run_in_threadpool(recovery.discard, session_id)
        return Response(status_code=204)

    @api.get("/live/recoveries")
    async def recoveries() -> Dict[str, Any]:
        """Every recovery, newest first."""
        return {"recoveries": await run_in_threadpool(recovery.list)}
