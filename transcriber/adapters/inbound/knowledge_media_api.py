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
(raw body: the file) starts the job (``202``); the body is written to a
file as it arrives (mode 600, under ``<knowledge-media>/incoming``, the
jobs' disk) and moved into the job, never held in memory (#282); a
disk that would not hold it with ``min_free_bytes`` to spare is refused
first (``507``); ``GET`` returns it
(``{"id", "state", "stage", "fraction", "message", "duration",
"transcript"}``, the transcript a ``transcript.v1`` once ``done``);
``DELETE`` deletes it (``204``; ``409 wrong_state`` while it runs).
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from transcriber.adapters.inbound.access import check_code
from transcriber.adapters.inbound.http_api import ApiError
from transcriber.application.knowledge_media import (
    KnowledgeMediaJobs,
    check_job_id,
)
from transcriber.domain.errors import NoSpaceError, PublishError

PREFIX = "/internal/knowledge-media"
WRITE_BYTES = 1024 * 1024
"""Bytes gathered from the body before one write to disk (in a thread)."""

Refuse = Callable[[Optional[str], Mapping[str, str]], None]
"""Counts a failed credential of a client and raises (``401``/``429``)."""


def bearer(request: Request) -> Optional[str]:
    """Return the bearer token of ``request``, if it has one."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def clear_incoming(incoming: Path) -> int:
    """Delete the files a stopped process was receiving; count them."""
    if not incoming.is_dir():
        return 0
    count = 0
    for entry in incoming.iterdir():
        if entry.is_file() and not entry.is_symlink():
            entry.unlink(missing_ok=True)
            count += 1
    return count


async def receive_file(
    request: Request, target: Path, limit: int, min_free: int,
) -> None:
    """Write the body to ``target`` (mode 600) as it arrives.

    ``413`` beyond ``limit`` bytes, ``507`` when the declared size would
    eat into ``min_free`` bytes of the disk; a partial file is deleted.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise ApiError(413, "payload_too_large",
                       f"The body is larger than {limit} bytes.")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    expected = int(declared) if declared.isdigit() else 0
    if shutil.disk_usage(target.parent).free - expected < min_free:
        raise NoSpaceError(
            "The transcriber's disk has no room for this file."
        )
    size = 0
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                         0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pending: List[bytes] = []
            held = 0
            async for part in request.stream():
                size += len(part)
                if size > limit:
                    raise ApiError(413, "payload_too_large",
                                   f"The body is larger than {limit} bytes.")
                pending.append(part)
                held += len(part)
                if held >= WRITE_BYTES:
                    # the disk is written off the event loop (live sessions
                    # are served by the same loop)
                    await run_in_threadpool(handle.write, b"".join(pending))
                    pending, held = [], 0
            if pending:
                await run_in_threadpool(handle.write, b"".join(pending))
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def knowledge_media_router(
    jobs: KnowledgeMediaJobs,
    token: Callable[[], str],
    refuse: Refuse,
    client_ip_header: str,
    body_limit: int,
    incoming: Path,
    min_free_bytes: int = 0,
) -> APIRouter:
    """Return the routes of the knowledge library's transcriptions.

    ``token`` returns the RAG's bearer token (read at each request, so a
    renewed token file applies at once); ``refuse`` is the API's
    throttling of wrong credentials; ``body_limit`` the largest file;
    ``incoming`` the folder a file is written to while it arrives (on
    the jobs' disk) and ``min_free_bytes`` the disk always left free.
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
        """Start transcribing the file sent as the body (streamed to disk).
        """
        target = incoming / f"{check_job_id(job_id)}-{uuid.uuid4().hex}"
        await receive_file(request, target, body_limit, min_free_bytes)
        try:
            job = await run_in_threadpool(
                jobs.submit, job_id, file_format.lower(), b"", language,
                max_seconds, str(target),
            )
        finally:
            target.unlink(missing_ok=True)  # moved into the job, or unused
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
