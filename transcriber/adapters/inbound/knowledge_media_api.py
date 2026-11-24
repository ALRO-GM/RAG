"""Inbound adapter: ``/internal/knowledge-media``, for the RAG only (KN-3a).

The RAG's worker sends an audio or video file of the knowledge library
here, polls the job and deletes it once it has the transcript
(:mod:`transcriber.application.knowledge_media`). These routes are not
under ``/api``: the server's Caddy forwards only ``/api/*`` through the
tunnel, so they are reachable from this machine only. They also want
the RAG's own bearer token (the one the transcriber publishes with,
compared in constant time) and refuse a request that carries the
client-address header the server's proxy sets (it came through the
tunnel). A wrong token counts as a failed access code of its client.

``PUT /internal/knowledge-media/{job}?format=&language=&max_seconds=``
(raw body: the file) starts the job (``202``); ``GET`` returns it
(``{"id", "state", "stage", "fraction", "message", "duration",
"transcript"}``, the transcript a ``transcript.v1`` once ``done``);
``DELETE`` deletes it (``204``; ``409 wrong_state`` while it runs).
"""

from typing import Any, Callable, Dict, Mapping, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from transcriber.adapters.inbound.access import check_code
from transcriber.adapters.inbound.http_api import ApiError, read_body
from transcriber.application.knowledge_media import KnowledgeMediaJobs
from transcriber.domain.errors import PublishError

PREFIX = "/internal/knowledge-media"

Refuse = Callable[[Optional[str], Mapping[str, str]], None]
"""Counts a failed credential of a client and raises (``401``/``429``)."""


def bearer(request: Request) -> Optional[str]:
    """Return the bearer token of ``request``, if it has one."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def knowledge_media_router(
    jobs: KnowledgeMediaJobs,
    token: Callable[[], str],
    refuse: Refuse,
    client_ip_header: str,
    body_limit: int,
) -> APIRouter:
    """Return the routes of the knowledge library's transcriptions.

    ``token`` returns the RAG's bearer token (read at each request, so a
    renewed token file applies at once); ``refuse`` is the API's
    throttling of wrong credentials; ``body_limit`` the largest file.
    """

    async def authorise(request: Request) -> None:
        """Let only the RAG of this machine through."""
        if request.headers.get(client_ip_header) is not None:
            raise ApiError(403, "forbidden", "Not available through the "
                           "tunnel.")
        try:
            expected = token()
        except PublishError:
            raise ApiError(503, "rag_unavailable",
                           "No RAG token is configured.") from None
        if check_code(expected, bearer(request)):
            return
        peer = request.client.host if request.client is not None else None
        refuse(peer, request.headers)

    router = APIRouter(prefix=PREFIX, dependencies=[Depends(authorise)])

    @router.put("/{job_id}", status_code=202)
    async def submit(
        job_id: str,
        request: Request,
        file_format: str = Query(alias="format", min_length=1,
                                 max_length=16),
        language: str = Query(min_length=2, max_length=16),
        max_seconds: Optional[float] = Query(default=None, gt=0),
    ) -> Dict[str, Any]:
        """Start transcribing the file sent as the body."""
        data = await read_body(request, body_limit)
        job = await run_in_threadpool(
            jobs.submit, job_id, file_format.lower(), data, language,
            max_seconds,
        )
        return job.model_dump(mode="json")

    @router.get("/{job_id}")
    async def status(job_id: str) -> Dict[str, Any]:
        """Return the job, with its transcript once done."""
        job = await run_in_threadpool(jobs.status, job_id)
        return job.model_dump(mode="json")

    @router.delete("/{job_id}", status_code=204)
    async def forget(job_id: str) -> Response:
        """Delete the job and what was stored for it."""
        await run_in_threadpool(jobs.forget, job_id)
        return Response(status_code=204)

    return router
