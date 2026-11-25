"""``python -m orchestrator serve``: the live page's API (port 8772).

Routes (JSON unless said otherwise; ``docs/platform/ORCHESTRATOR.md``):

- ``GET /health``: no token.
- ``GET /products``: the live products and what the page shows of them.
- ``GET /transcripts``: recorded conversations a replay can use, and
  ``demos``: the scripted demos (``live.demo.v1``, LV-17); ``GET
  /demos/{id}/report`` and ``/voicebot``: their report and voice
  example (#250); ``GET /demos/{id}/voices``: which examples can be
  heard and each clip's length, ``GET /demos/{id}/voices/{line}.wav``:
  one rendered clip, ``audio/wav`` (#254; 404 when it may not be played:
  not rendered, text changed, consent withdrawn).
- ``GET /sessions``, ``GET /sessions/{id}``.
- ``POST /sessions`` ``{"product", "audio": "stereo"|"mic",
  "session_id"?, "employee"?: {"id", "name"}, "plan"?, "domain"?,
  "candidate"?}``: start a live session on the transcriber (``plan``:
  the interview plan of a ``voicebot.interview`` session; ``domain``:
  the topic of a voice bot).
- ``GET /interview/plans``, ``POST /interview/plans`` (``{"offer" |
  "document": {"domain", "id"}, "title"?, "role_type"?, "minutes"?}``:
  draft a plan with the planner), ``GET``/``PUT /interview/plans/{id}``
  (the recruiter's edit, with the ``revision`` it was based on; 409 when
  older) and ``GET /interview/plans/{id}/results`` (the candidates
  interviewed from it, for the comparison view); ``GET
  /interview/plans/{id}/board``: the offer page's table (#271), one row
  per person with one comparable figure per interview, best first; ``GET
  /interview/plans/{id}/results/{session}``: one stored interview report
  in full (#262), ``DELETE`` it when it belongs to no candidate's file
  (409 when it does: it goes with the file); ``POST
  /interview/plans/{id}/close`` and ``/reopen``:
  close or reopen the position (closing deletes nothing; a closed
  position starts no interview, 409 ``plan_closed``).
- ``GET``/``POST /interview/plans/{id}/candidates`` (``{"name"}``),
  ``GET``/``DELETE /interview/plans/{id}/candidates/{candidate}``,
  ``PUT``/``DELETE /interview/plans/{id}/candidates/{candidate}/answers/
  {question}`` (``{"text"}``: the interviewer's answer; ``DELETE``
  forgets it), ``POST .../candidates/{candidate}/confirm``
  (``{"questions"?}``: confirm captured proposals, all without a list),
  ``DELETE .../candidates/{candidate}/proposals/{question}`` and ``GET
  /interview/candidates/by-session/{session}``: the candidates of a
  role and their answers to the core questions (#235). ``POST
  /sessions`` takes ``candidate_id`` too.
- ``POST /sessions/{id}/replay`` ``{"product", "transcript",
  "speed"?, "employee"?}``: replay a transcript as a live session
  (``speed`` 1 is real time, ``null`` as fast as the services answer).
  With ``{"demo": id}`` it plays a scripted demo instead (its own
  product and conversation; prepared suggestions, no model call).
- ``POST /sessions/{id}/stop``; ``POST /sessions/{id}/rejudge``: judge
  an ended live round again when its judgement failed (#262; 202, the
  result comes in the session's ``accuracy``).
- ``POST /sessions/{id}/recovered`` ``{"transcript": transcript.v1}``:
  the transcriber recovered audio the round lost (#265); every
  indicator is computed again from the merged transcript
  (:class:`~orchestrator.application.recovered_round.RecoveredRound`)
  and the summary (numbers and states) is answered. Deleting a
  candidate's file also asks the transcriber to delete any recovered
  audio of their rounds.
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
  ``X-Voice-Length-Scale``, ``X-Voice-Engine`` (``piper``,
  ``openvoice``, ``pocket-cpu``, ``pocket-cuda``, ``qwen3-cuda``: who
  said the first sentence) and ``X-Voice-Note`` say what was used.
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

from orchestrator.application.demo_voices import DemoVoiceLibrary
from orchestrator.application.interview import PlanError
from orchestrator.application.recovered_round import (
    RecoveredRound,
    forget_recovery,
)
from orchestrator.application.sessions import LiveSessions, SessionError

STATUS = {
    "not_found": 404, "busy": 409, "wrong_state": 409,
    "invalid_request": 422, "transcriber": 502, "unavailable": 503,
    "unauthorized": 401, "conflict": 409, "offer": 502,
    "plan_closed": 409, "delete_failed": 500,
}
MAX_BODY = 16 * 1024
RECOVERED_BODY = 8 * 1024 * 1024
"""Largest merged transcript the transcriber sends (#265)."""
MAX_PLAN_BODY = 96 * 1024
"""Largest plan body (a job offer of 12 000 characters, JSON-escaped)."""


def create_app(
    sessions: LiveSessions,
    token: Optional[str] = None,
    keepalive: float = 15.0,
    demo_voices: Optional[DemoVoiceLibrary] = None,
) -> FastAPI:
    """Build the API over ``sessions``.

    Args:
        sessions: The use case.
        token: Bearer token required on every route but ``/health``.
        keepalive: Seconds between SSE comments when nothing happens.
        demo_voices: The scripted examples' rendered clips (``None``:
            no example is heard).
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

    @app.exception_handler(PlanError)
    async def plan_error(request: Request, exc: PlanError) -> JSONResponse:
        """Answer a refused plan request with its stable code."""
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

    async def body(
        request: Request, limit: int = MAX_BODY
    ) -> Dict[str, Any]:
        raw = await request.body()
        if len(raw) > limit:
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
        demos = await run_in_threadpool(sessions.list_demos)
        return {"transcripts": found, "demos": demos}

    @app.get("/demos/{demo_id}/voices")
    async def demo_voice_view(demo_id: str) -> Dict[str, Any]:
        """Which examples of a demo can be heard, and their clips."""
        if demo_voices is None:
            raise SessionError("not_found", "no demo voices here")
        try:
            found: Dict[str, Any] = await run_in_threadpool(
                demo_voices.view, demo_id)
        except KeyError as exc:
            raise SessionError("not_found", f"no demo {demo_id!r}") from exc
        return found

    @app.get("/demos/{demo_id}/voices/{clip}")
    async def demo_voice_clip(demo_id: str, clip: str) -> Response:
        """One rendered clip of a demo line, if it may be played."""
        key = clip[:-4] if clip.endswith(".wav") else ""
        if demo_voices is None or not key:
            raise SessionError("not_found", "no such clip")
        try:
            wav = await run_in_threadpool(demo_voices.clip, demo_id, key)
        except KeyError as exc:
            raise SessionError("not_found", "no such clip") from exc
        return Response(wav, media_type="audio/wav",
                        headers={"Cache-Control": "private, no-store"})

    @app.get("/demos/{demo_id}/{part}")
    async def demo_part(demo_id: str, part: str) -> Dict[str, Any]:
        """The report or the voice example of a scripted demo."""
        found: Dict[str, Any] = await run_in_threadpool(
            sessions.demo_part, demo_id, part
        )
        return found

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
            _optional(data, "plan"), _optional(data, "domain"),
            _optional(data, "candidate"), _optional(data, "candidate_id"),
        )

    def planner() -> Any:
        wiring = sessions.interview
        if wiring is None:
            raise SessionError("unavailable", "no interview plans here")
        return wiring.planner

    @app.get("/interview/plans")
    async def list_plans() -> Dict[str, Any]:
        """The interview plans."""
        return {"plans": await run_in_threadpool(planner().list)}

    @app.post("/interview/plans", status_code=201)
    async def create_plan(request: Request) -> Dict[str, Any]:
        """Draft a plan from a job offer (the planner model)."""
        data = await body(request, MAX_PLAN_BODY)
        found: Dict[str, Any] = await run_in_threadpool(
            planner().create, data
        )
        return found

    @app.get("/interview/plans/{plan_id}")
    async def show_plan(plan_id: str) -> Dict[str, Any]:
        """One plan."""
        found: Dict[str, Any] = await run_in_threadpool(
            planner().get, plan_id
        )
        return found

    @app.put("/interview/plans/{plan_id}")
    async def edit_plan(plan_id: str, request: Request) -> Dict[str, Any]:
        """Save the recruiter's edit of a plan."""
        data = await body(request, MAX_PLAN_BODY)
        found: Dict[str, Any] = await run_in_threadpool(
            planner().update, plan_id, data
        )
        return found

    @app.get("/interview/plans/{plan_id}/results")
    async def plan_results(plan_id: str) -> Dict[str, Any]:
        """The candidates interviewed from a plan."""
        found: Dict[str, Any] = await run_in_threadpool(
            planner().results, plan_id
        )
        return found

    @app.get("/interview/plans/{plan_id}/board")
    async def plan_board(plan_id: str) -> Dict[str, Any]:
        """The offer page's table: one row per person, best figure first."""
        found: Dict[str, Any] = await run_in_threadpool(
            planner().board, plan_id
        )
        return found

    @app.get("/interview/plans/{plan_id}/results/{session_id}")
    async def plan_report(plan_id: str, session_id: str) -> Dict[str, Any]:
        """One stored interview report of a plan, in full."""
        found: Dict[str, Any] = await run_in_threadpool(
            planner().report, plan_id, session_id
        )
        return found

    @app.delete("/interview/plans/{plan_id}/results/{session_id}")
    async def delete_plan_report(plan_id: str,
                                 session_id: str) -> Dict[str, Any]:
        """Delete a stored report that belongs to no candidate's file."""
        found: Dict[str, Any] = await run_in_threadpool(
            book().delete_unowned_report, plan_id, session_id
        )
        return found

    def positions() -> Any:
        wiring = sessions.interview
        if wiring is None:
            raise SessionError("unavailable", "no interview plans here")
        return wiring.positions

    @app.post("/interview/plans/{plan_id}/close")
    async def close_plan(plan_id: str) -> Dict[str, Any]:
        """Close the position (no new interviews; nothing is deleted)."""
        found: Dict[str, Any] = await run_in_threadpool(
            positions().close, plan_id
        )
        return found

    @app.post("/interview/plans/{plan_id}/reopen")
    async def reopen_plan(plan_id: str) -> Dict[str, Any]:
        """Open the position again."""
        found: Dict[str, Any] = await run_in_threadpool(
            positions().reopen, plan_id
        )
        return found

    def book() -> Any:
        wiring = sessions.interview
        if wiring is None or wiring.candidates is None:
            raise SessionError("unavailable", "no candidate profiles here")
        return wiring.candidates

    @app.get("/interview/plans/{plan_id}/candidates")
    async def list_candidates(plan_id: str) -> Dict[str, Any]:
        """The candidates of a role."""
        found: Dict[str, Any] = await run_in_threadpool(book().list, plan_id)
        return found

    @app.post("/interview/plans/{plan_id}/candidates", status_code=201)
    async def add_candidate(plan_id: str, request: Request) -> Dict[str, Any]:
        """Add a candidate to a role."""
        data = await body(request)
        found: Dict[str, Any] = await run_in_threadpool(
            book().create, plan_id, data.get("name")
        )
        return found

    @app.get("/interview/plans/{plan_id}/candidates/{candidate_id}")
    async def show_candidate(
        plan_id: str, candidate_id: str
    ) -> Dict[str, Any]:
        """One candidate and their answers."""
        found: Dict[str, Any] = await run_in_threadpool(
            book().get, plan_id, candidate_id
        )
        return found

    @app.delete("/interview/plans/{plan_id}/candidates/{candidate_id}")
    async def delete_candidate(
        plan_id: str, candidate_id: str
    ) -> Dict[str, Any]:
        """Delete a candidate's profile (and any recovered audio, #265)."""
        rounds = await run_in_threadpool(
            book().get, plan_id, candidate_id
        )
        found: Dict[str, Any] = await run_in_threadpool(
            book().delete, plan_id, candidate_id
        )
        await run_in_threadpool(
            forget_recovery, sessions.transcriber,
            [str(r.get("session_id")) for r in rounds.get("rounds") or []
             if r.get("session_id")],
        )
        return found

    @app.put("/interview/plans/{plan_id}/candidates/{candidate_id}"
             "/answers/{question_id}")
    async def correct_answer(
        plan_id: str, candidate_id: str, question_id: str, request: Request
    ) -> Dict[str, Any]:
        """Keep the interviewer's answer to a core question."""
        data = await body(request)
        found: Dict[str, Any] = await run_in_threadpool(
            book().correct, plan_id, candidate_id, question_id,
            data.get("text"),
        )
        return found

    @app.delete("/interview/plans/{plan_id}/candidates/{candidate_id}"
                "/answers/{question_id}")
    async def clear_answer(
        plan_id: str, candidate_id: str, question_id: str
    ) -> Dict[str, Any]:
        """Forget one answer (the question is open again)."""
        found: Dict[str, Any] = await run_in_threadpool(
            book().clear, plan_id, candidate_id, question_id
        )
        return found

    @app.post("/interview/plans/{plan_id}/candidates/{candidate_id}"
              "/confirm")
    async def confirm_answers(
        plan_id: str, candidate_id: str, request: Request
    ) -> Dict[str, Any]:
        """Confirm proposals: ``{"questions": [...]}``, or all of them."""
        data = await body(request)
        wanted = data.get("questions")
        if wanted is not None and not (
            isinstance(wanted, list)
            and all(isinstance(q, str) for q in wanted)
        ):
            raise SessionError("invalid_request",
                               "questions must be a list of ids")
        found: Dict[str, Any] = await run_in_threadpool(
            book().confirm, plan_id, candidate_id, wanted
        )
        return found

    @app.delete("/interview/plans/{plan_id}/candidates/{candidate_id}"
                "/proposals/{question_id}")
    async def discard_proposal(
        plan_id: str, candidate_id: str, question_id: str
    ) -> Dict[str, Any]:
        """Discard one captured proposal."""
        found: Dict[str, Any] = await run_in_threadpool(
            book().discard, plan_id, candidate_id, question_id
        )
        return found

    @app.get("/interview/candidates/by-session/{session_id}")
    async def session_candidate(session_id: str) -> Dict[str, Any]:
        """The candidate interviewed in a session."""
        found: Dict[str, Any] = await run_in_threadpool(
            book().by_session, session_id
        )
        return found

    @app.post("/sessions/{session_id}/replay", status_code=201)
    async def replay(session_id: str, request: Request) -> Dict[str, Any]:
        """Start a replay."""
        data = await body(request)
        speed = data.get("speed", 1.0)
        if speed is not None and (
            isinstance(speed, bool) or not isinstance(speed, (int, float))
        ):
            raise SessionError("invalid_request", "speed must be a number")
        demo = _optional(data, "demo")
        if demo is not None:
            return await run_in_threadpool(
                sessions.start_replay, session_id, "", "",
                float(speed) if speed is not None else None, None, demo,
            )
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

    @app.post("/sessions/{session_id}/recovered")
    async def recovered(session_id: str, request: Request) -> Dict[str, Any]:
        """The transcriber recovered lost audio: rebuild the round."""
        data = await body(request, RECOVERED_BODY)
        transcript = data.get("transcript")
        if not isinstance(transcript, dict):
            raise SessionError("invalid_request", "a transcript is needed")
        try:
            found: Dict[str, Any] = await run_in_threadpool(
                RecoveredRound(sessions).rebuild, session_id, transcript)
        except ValueError as exc:
            raise SessionError("invalid_request", str(exc)) from exc
        return found

    @app.post("/sessions/{session_id}/rejudge", status_code=202)
    async def rejudge(session_id: str) -> Dict[str, Any]:
        """Judge an ended live round again (#262, «Reintentar»)."""
        return await run_in_threadpool(sessions.rejudge, session_id)

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
            "X-Voice-Engine": str(used.get("engine") or "piper")[:32],
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


def _optional(data: Dict[str, Any], key: str) -> Optional[str]:
    """Return ``data[key]`` as a string, ``None`` when absent or empty."""
    value = data.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SessionError("invalid_request", f"{key} must be a string")
    return value
