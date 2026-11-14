"""Inbound adapter: the model lab's routes under ``/api/lab``.

Added to the API by :func:`~transcriber.adapters.inbound.http_api.
create_app` with the same access code, error format and body limits
(``docs/transcriber/LAB.md`` §7 lists every route with examples). The
audio of a run (the original or a denoiser's output) is served like a
session's audio: WAV with byte ranges, and the playback cookie is
accepted on that route only.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from transcriber.adapters.inbound.http_api import (
    parse_json,
    read_body,
    wav_response,
)
from transcriber.application.lab import MAX_PAGE, LabRunView
from transcriber.bootstrap import LabUseCases
from transcriber.config import TranscriberSettings
from transcriber.domain.lab import (
    MAX_LABELS,
    LabRequest,
    LabRun,
    ReferenceLabel,
)


class LabelsBody(BaseModel):
    """Body of ``PUT /api/lab/sessions/{id}/labels``: every label."""

    model_config = ConfigDict(extra="forbid")

    labels: List[ReferenceLabel] = Field(max_length=MAX_LABELS)


def run_json(run: LabRun, state: Optional[str] = None) -> Dict[str, Any]:
    """Return a run as the pages show it (``state`` as reported)."""
    data: Dict[str, Any] = run.model_dump(mode="json", exclude={
        "worker_pid",
    })
    data["state"] = state or run.state
    data["pipeline_count"] = len(run.pipelines())
    for step, shown in zip(run.steps, data["steps"]):
        shown["name"] = step.name
    return data


def view_json(view: LabRunView) -> Dict[str, Any]:
    """Return a run's status: the run, its pipelines and its job."""
    data = run_json(view.run, view.state)
    data["pipelines"] = view.pipelines
    data["done_steps"] = view.done_steps
    data["total_steps"] = view.total_steps
    data["job"] = (
        view.job.model_dump(mode="json", exclude={"session_id"})
        if view.job is not None else None
    )
    return data


def add_lab_routes(
    api: APIRouter, playback: APIRouter, lab: LabUseCases,
    settings: TranscriberSettings,
) -> None:
    """Add the lab's routes to the code-guarded and playback routers."""
    json_limit = settings.api.json_max_bytes
    labels_limit = settings.lab.labels_max_bytes

    @api.get("/lab/catalogue")
    async def catalogue() -> Dict[str, Any]:
        """List the candidate models with their state on this machine."""
        view = await run_in_threadpool(lab.catalogue)
        return {
            "token_present": view.token_present,
            "models": [
                {**item.entry.model_dump(mode="json"), "state": item.state,
                 "reason": item.reason,
                 "weights_cached": item.weights_cached}
                for item in view.items
            ],
        }

    @api.get("/lab/sessions/{session_id}/runs")
    async def runs(session_id: str) -> Dict[str, Any]:
        """List a session's runs, newest first (without their steps)."""
        found = await run_in_threadpool(lab.runs, session_id)
        listed = []
        for run, state in found:
            data = run_json(run, state)
            data["done_steps"] = sum(
                1 for s in run.steps if s.state == "done"
            )
            data["total_steps"] = len(run.steps)
            del data["steps"]
            listed.append(data)
        return {"runs": listed}

    @api.post("/lab/sessions/{session_id}/runs", status_code=202)
    async def start(session_id: str, request: Request) -> Dict[str, Any]:
        """Queue a run of the chosen models on the session."""
        body = parse_json(
            LabRequest, await read_body(request, json_limit)
        )
        run = await run_in_threadpool(lab.start, session_id, body)
        return run_json(run)

    @api.get("/lab/sessions/{session_id}/runs/{run_id}")
    async def status(session_id: str, run_id: str) -> Dict[str, Any]:
        """Show a run: steps, pipelines' states and the job's progress."""
        return view_json(
            await run_in_threadpool(lab.status, session_id, run_id)
        )

    @api.post("/lab/sessions/{session_id}/runs/{run_id}/cancel")
    async def cancel(session_id: str, run_id: str) -> Dict[str, Any]:
        """Ask a queued or running run to stop."""
        await run_in_threadpool(lab.cancel, session_id, run_id)
        return view_json(
            await run_in_threadpool(lab.status, session_id, run_id)
        )

    @api.post(
        "/lab/sessions/{session_id}/runs/{run_id}/resume", status_code=202
    )
    async def resume(session_id: str, run_id: str) -> Dict[str, Any]:
        """Queue a finished, cancelled or interrupted run again."""
        return run_json(
            await run_in_threadpool(lab.resume, session_id, run_id)
        )

    @api.get("/lab/sessions/{session_id}/runs/{run_id}/results")
    async def results(session_id: str, run_id: str) -> Dict[str, Any]:
        """Return the comparison (without fragments: see /fragments)."""
        return await run_in_threadpool(lab.results, session_id, run_id)

    @api.get("/lab/sessions/{session_id}/runs/{run_id}/tracks")
    async def tracks(session_id: str, run_id: str) -> Dict[str, Any]:
        """Return each diarization's named segments and the consensus."""
        return await run_in_threadpool(
            lab.results.tracks, session_id, run_id
        )

    @api.get("/lab/sessions/{session_id}/runs/{run_id}/fragments")
    async def fragments(
        session_id: str,
        run_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=MAX_PAGE),
        sort: str = Query(default="disagreement", max_length=16),
        unlabelled: bool = Query(default=False),
    ) -> Dict[str, Any]:
        """Return a page of fragments, most disputed first by default."""
        return await run_in_threadpool(
            lab.results.fragments, session_id, run_id, offset, limit, sort,
            unlabelled,
        )

    @api.get("/lab/sessions/{session_id}/runs/{run_id}/fragment-spans")
    async def spans(session_id: str, run_id: str) -> Dict[str, Any]:
        """Return every fragment's span and disagreement, in time order."""
        return await run_in_threadpool(
            lab.results.spans, session_id, run_id
        )

    @api.get("/lab/sessions/{session_id}/runs/{run_id}/scores")
    async def scores(session_id: str, run_id: str) -> Dict[str, Any]:
        """Rank the run's pipelines against the reference labels."""
        return await run_in_threadpool(lab.scores, session_id, run_id)

    @api.get("/lab/sessions/{session_id}/labels")
    async def labels(session_id: str) -> Dict[str, Any]:
        """Return the reference labels and who may be named in them."""
        found = await run_in_threadpool(lab.labels, session_id)
        return {
            "labels": [label.model_dump(mode="json") for label in found],
            "speakers": lab.labels.speakers(),
        }

    @api.put("/lab/sessions/{session_id}/labels")
    async def put_labels(
        session_id: str, request: Request
    ) -> Dict[str, Any]:
        """Replace every reference label of the session."""
        body = parse_json(
            LabelsBody, await read_body(request, labels_limit)
        )
        saved = await run_in_threadpool(
            lab.labels.replace, session_id, body.labels
        )
        return {
            "labels": [label.model_dump(mode="json") for label in saved],
            "speakers": lab.labels.speakers(),
        }

    @playback.get(
        "/lab/sessions/{session_id}/runs/{run_id}/audio/{denoiser}"
    )
    async def audio(
        session_id: str, run_id: str, denoiser: str, request: Request
    ) -> Response:
        """Serve the original (``none``) or a denoiser's audio as WAV."""
        rendering = await run_in_threadpool(
            lab.audio, session_id, run_id, denoiser
        )
        return wav_response(rendering, request.headers.get("range"))
