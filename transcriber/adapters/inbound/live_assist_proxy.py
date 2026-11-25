"""``/api/assist/...``: the orchestrator's live-page API, forwarded.

The web pages are served by the transcriber, and the orchestrator
(``python -m orchestrator serve``, ``docs/platform/ORCHESTRATOR.md``)
listens on loopback only, so the page reaches it through this thin
proxy: the access code guards the route here (the ``api`` router's
dependency), and the orchestrator's bearer token is added on the way
(the RAG token file, shared by the services of the workstation).

Only the orchestrator's own routes exist here (``products``,
``transcripts``, ``sessions``, one session, its ``replay``, ``stop``,
``barge-in`` (the voice bot), ``mode``, ``speakers/{id}``,
``mandatory/{item}`` (LV-16), ``suggestions/stream`` and
``voicebot/sample``, the «Escuchar ejemplo» WAV of «Ajustes»; a scripted
demo's ``report``, ``voicebot``, ``voices`` and one rendered line clip,
``voices/<line>.wav`` (#254, never cached); the AI
interviewer's ``interview/plans``, one plan and its ``results``; the
candidates of a role, one candidate, their answers to the core questions
and the candidate of a session, #235), and
session, speaker and item ids are checked; the stream is passed through
as it comes (Server-Sent Events, with the voice bot's audio inside). It
is an inbound adapter that talks HTTP itself on purpose: it adds no
behaviour to the transcriber, so there is no use case to put behind it.

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
    rf"^(?:products|transcripts|demos/{_ID}/(?:report|voicebot|voices)"
    rf"|demos/{_ID}/voices/[A-Za-z0-9][A-Za-z0-9._-]{{0,47}}\.wav"
    rf"|sessions|sessions/{_ID}|voicebot/sample"
    rf"|sessions/{_ID}/(?:replay|stop|rejudge|barge-in|mode"
    rf"|suggestions/stream"
    rf"|speakers/{_ID}|mandatory/[a-z0-9][a-z0-9_.-]{{0,47}})"
    rf"|interview/plans|interview/plans/{_ID}"
    rf"|interview/plans/{_ID}/results|interview/plans/{_ID}/results/{_ID}"
    rf"|interview/plans/{_ID}/(?:close|reopen)"
    rf"|interview/plans/{_ID}/candidates"
    rf"|interview/plans/{_ID}/candidates/{_ID}"
    rf"|interview/plans/{_ID}/candidates/{_ID}"
    rf"/(?:answers|proposals)/[a-z][a-z0-9_]{{1,40}}"
    rf"|interview/plans/{_ID}/candidates/{_ID}/confirm"
    rf"|interview/candidates/by-session/{_ID})$"
)
MAX_BODY = 16 * 1024
PLAN_BODY = 96 * 1024
"""Largest plan body (a job offer of 12 000 characters, JSON-escaped)."""
TIMEOUT = 45.0
"""Seconds for a plain call (starting a live session waits for the GPU)."""
PLAN_TIMEOUT = 200.0
"""Seconds for drafting an interview plan: the orchestrator reads the
uploaded offer (30 s), queries the method corpus (5 s) and waits for the
planner (its 120 s deadline plus a 30 s margin, #267)."""
STOP_TIMEOUT = 120.0
"""Seconds for «Terminar» (#264): the orchestrator's stop waits for the
transcriber's (its client gives it 90 s: 15 s for the source to finish
and 60 s for the end), then up to 15 s for the last events."""
SAMPLE_TIMEOUT = 150.0
"""Seconds for «Escuchar ejemplo»: a cloned or natural voice on the GPU
may wait 20 s for the memory and up to 120 s for its model to speak
(the orchestrator's ``WAIT_SECONDS`` and ``SAMPLE_BUDGET_MS``), longer
than :data:`TIMEOUT` (#253)."""


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

    def client(timeout: float = TIMEOUT) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    @api.get("/assist/products")
    async def assist_products(request: Request) -> Response:
        """The orchestrator's live products."""
        return await forward("products", request)

    @api.get("/assist/transcripts")
    async def assist_transcripts(request: Request) -> Response:
        """Recorded conversations a demo replay can use."""
        return await forward("transcripts", request)

    @api.get("/assist/demos/{demo_id}/{part}")
    async def assist_demo_part(
        demo_id: str, part: str, request: Request
    ) -> Response:
        """The report or the voice example of a scripted demo (#250)."""
        return await forward(f"demos/{demo_id}/{part}", request)

    @api.get("/assist/demos/{demo_id}/voices/{clip}")
    async def assist_demo_clip(
        demo_id: str, clip: str, request: Request
    ) -> Response:
        """One rendered line of a scripted demo, ``audio/wav`` (#254)."""
        return await forward(f"demos/{demo_id}/voices/{clip}", request)

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

    @api.post("/assist/sessions/{session_id}/rejudge")
    async def assist_rejudge(session_id: str, request: Request) -> Response:
        """Judge an ended live round again (#262)."""
        return await forward(f"sessions/{session_id}/rejudge", request)

    @api.post("/assist/sessions/{session_id}/barge-in")
    async def assist_barge_in(session_id: str, request: Request) -> Response:
        """The user spoke over the voice bot."""
        return await forward(f"sessions/{session_id}/barge-in", request)

    @api.post("/assist/sessions/{session_id}/mode")
    async def assist_mode(session_id: str, request: Request) -> Response:
        """Switch the assistance mode."""
        return await forward(f"sessions/{session_id}/mode", request)

    @api.post("/assist/sessions/{session_id}/speakers/{speaker}")
    async def assist_rename(
        session_id: str, speaker: str, request: Request
    ) -> Response:
        """Name a speaker of the session."""
        return await forward(f"sessions/{session_id}/speakers/{speaker}",
                             request)

    @api.post("/assist/sessions/{session_id}/mandatory/{item}")
    async def assist_tick(
        session_id: str, item: str, request: Request
    ) -> Response:
        """Tick a mandatory item by hand (after confirming)."""
        return await forward(f"sessions/{session_id}/mandatory/{item}",
                             request)

    @api.post("/assist/voicebot/sample")
    async def assist_sample(request: Request) -> Response:
        """The voice bot's sample sentence with the voice being chosen."""
        return await forward("voicebot/sample", request)

    @api.get("/assist/sessions/{session_id}/suggestions/stream")
    async def assist_stream(session_id: str, request: Request) -> Response:
        """The session's Server-Sent Events."""
        return await forward(f"sessions/{session_id}/suggestions/stream",
                             request)

    @api.get("/assist/interview/plans")
    async def assist_plans(request: Request) -> Response:
        """The interview plans."""
        return await forward("interview/plans", request)

    @api.post("/assist/interview/plans")
    async def assist_new_plan(request: Request) -> Response:
        """Draft an interview plan from a job offer."""
        return await forward("interview/plans", request)

    @api.get("/assist/interview/plans/{plan_id}")
    async def assist_plan(plan_id: str, request: Request) -> Response:
        """One interview plan."""
        return await forward(f"interview/plans/{plan_id}", request)

    @api.put("/assist/interview/plans/{plan_id}")
    async def assist_edit_plan(plan_id: str, request: Request) -> Response:
        """Save the recruiter's edit of a plan."""
        return await forward(f"interview/plans/{plan_id}", request)

    @api.get("/assist/interview/plans/{plan_id}/results")
    async def assist_plan_results(
        plan_id: str, request: Request
    ) -> Response:
        """The candidates interviewed from a plan."""
        return await forward(f"interview/plans/{plan_id}/results", request)

    @api.get("/assist/interview/plans/{plan_id}/results/{session_id}")
    async def assist_plan_report(
        plan_id: str, session_id: str, request: Request
    ) -> Response:
        """One stored interview report of a plan, in full (#262)."""
        return await forward(
            f"interview/plans/{plan_id}/results/{session_id}", request)

    @api.delete("/assist/interview/plans/{plan_id}/results/{session_id}")
    async def assist_delete_plan_report(
        plan_id: str, session_id: str, request: Request
    ) -> Response:
        """Delete a stored report that belongs to no candidate's file."""
        return await forward(
            f"interview/plans/{plan_id}/results/{session_id}", request)

    @api.post("/assist/interview/plans/{plan_id}/close")
    async def assist_close_plan(plan_id: str, request: Request) -> Response:
        """Close the position (no new interviews; nothing deleted)."""
        return await forward(f"interview/plans/{plan_id}/close", request)

    @api.post("/assist/interview/plans/{plan_id}/reopen")
    async def assist_reopen_plan(plan_id: str, request: Request) -> Response:
        """Open the position again."""
        return await forward(f"interview/plans/{plan_id}/reopen", request)

    @api.get("/assist/interview/plans/{plan_id}/candidates")
    async def assist_candidates(plan_id: str, request: Request) -> Response:
        """The candidates of a role (#235)."""
        return await forward(f"interview/plans/{plan_id}/candidates", request)

    @api.post("/assist/interview/plans/{plan_id}/candidates")
    async def assist_new_candidate(
        plan_id: str, request: Request
    ) -> Response:
        """Add a candidate to a role."""
        return await forward(f"interview/plans/{plan_id}/candidates", request)

    @api.get("/assist/interview/plans/{plan_id}/candidates/{candidate}")
    async def assist_candidate(
        plan_id: str, candidate: str, request: Request
    ) -> Response:
        """One candidate and their answers."""
        return await forward(
            f"interview/plans/{plan_id}/candidates/{candidate}", request)

    @api.delete("/assist/interview/plans/{plan_id}/candidates/{candidate}")
    async def assist_delete_candidate(
        plan_id: str, candidate: str, request: Request
    ) -> Response:
        """Delete a candidate's profile."""
        return await forward(
            f"interview/plans/{plan_id}/candidates/{candidate}", request)

    @api.put("/assist/interview/plans/{plan_id}/candidates/{candidate}"
             "/answers/{question}")
    async def assist_correct(
        plan_id: str, candidate: str, question: str, request: Request
    ) -> Response:
        """Keep the interviewer's answer to a core question."""
        return await forward(
            f"interview/plans/{plan_id}/candidates/{candidate}"
            f"/answers/{question}", request)

    @api.delete("/assist/interview/plans/{plan_id}/candidates/{candidate}"
                "/answers/{question}")
    async def assist_clear(
        plan_id: str, candidate: str, question: str, request: Request
    ) -> Response:
        """Forget one answer."""
        return await forward(
            f"interview/plans/{plan_id}/candidates/{candidate}"
            f"/answers/{question}", request)

    @api.post("/assist/interview/plans/{plan_id}/candidates/{candidate}"
              "/confirm")
    async def assist_confirm(
        plan_id: str, candidate: str, request: Request
    ) -> Response:
        """Confirm captured proposals."""
        return await forward(
            f"interview/plans/{plan_id}/candidates/{candidate}/confirm",
            request)

    @api.delete("/assist/interview/plans/{plan_id}/candidates/{candidate}"
                "/proposals/{question}")
    async def assist_discard(
        plan_id: str, candidate: str, question: str, request: Request
    ) -> Response:
        """Discard one captured proposal."""
        return await forward(
            f"interview/plans/{plan_id}/candidates/{candidate}"
            f"/proposals/{question}", request)

    @api.get("/assist/interview/candidates/by-session/{session_id}")
    async def assist_session_candidate(
        session_id: str, request: Request
    ) -> Response:
        """The candidate interviewed in a session."""
        return await forward(f"interview/candidates/by-session/{session_id}",
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
        body = (await request.body()
                if request.method in ("POST", "PUT") else b"")
        limit = PLAN_BODY if path.startswith("interview/") else MAX_BODY
        if len(body) > limit:
            return JSONResponse(
                {"code": "too_large", "message": "body too large"},
                status_code=413,
            )
        url = f"{base}/{path}"
        if path.endswith("/suggestions/stream"):
            return await _stream(client(), url, headers, params)
        wait = (PLAN_TIMEOUT if path == "interview/plans"
                else SAMPLE_TIMEOUT if path == "voicebot/sample"
                else STOP_TIMEOUT if path.startswith("sessions/")
                and path.endswith("/stop") else TIMEOUT)
        try:
            async with client(wait) as http:
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
            headers={k: v for k, v in answer.headers.items()
                     if k.lower().startswith("x-voice-")
                     or k.lower() == "cache-control"},
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
