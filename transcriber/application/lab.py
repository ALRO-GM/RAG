"""Model lab use cases: catalogue, runs, results, labels and scores.

A run (``docs/transcriber/LAB.md``) takes one session and a choice of
denoisers, diarizers and recognisers. :class:`StartLabRun` checks the
choice against the catalogue, stores the run with one step per stage
result it needs (denoise once per denoiser; diarize and recognise once
per denoiser and model) and queues :class:`RunLab` on the job runner
(one job at a time, as transcriptions). Each step's result is stored
under a key made from everything it depends on, so a step already
computed by any earlier run is reused and a resumed run computes only
what is missing. When the steps are over, the diarizers' speakers are
named (:func:`~transcriber.application.lab_builtin.map_speakers`) and
the comparison is computed and stored
(:func:`~transcriber.application.lab_compare.compare_run`).

The session's original audio is stored when a run is started or
resumed (and again by the job), so the player's audio route only reads.
A start or a resume checks the session, stores the original and saves
the run while holding the session's lock (the one
:class:`~transcriber.application.record.DeleteSession` holds), so a
deletion either happens first (the start finds no session) or sees the
queued run and is refused; it never removes the lab directory between
the check and the writes.

A run can be cancelled (between steps, and a running subprocess is
stopped) and resumed; a resume checks and queues the run under the
run's lock, so two concurrent resumes queue it once. Reference labels
are per session; scores are computed on request from the stored
results and the labels.
"""

import hashlib
import json
import uuid
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import BaseModel, ConfigDict

from transcriber.application.common import check_id, require_session
from transcriber.application.lab_builtin import builtin_mapping, map_speakers
from transcriber.application.lab_compare import (
    Diarization,
    LabResults,
    Recognition,
    attributed_words,
    compare_run,
)
from transcriber.application.lab_ports import (
    CatalogueSource,
    LabEnvironment,
    LabRepository,
    LabTool,
    ProcessTable,
    ToolRequest,
)
from transcriber.application.ports import (
    Audio,
    AudioDecoder,
    Clock,
    JobRunner,
    JobStatus,
    PlaybackAudio,
    Progress,
    SessionLocks,
    SessionRepository,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.domain.comparison import ComparisonParameters
from transcriber.domain.errors import (
    InvalidInputError,
    LabStepError,
    NotFoundError,
    StateError,
)
from transcriber.domain.lab import (
    BUILTIN_DIARIZER,
    LAB_RUN_ID_PATTERN,
    NO_DENOISER,
    PROJECT_FAMILY,
    Catalogue,
    CatalogueEntry,
    LabRequest,
    LabRun,
    ReferenceLabel,
    SpeakerMapping,
    Stage,
    StageError,
    StageOutcome,
    Step,
    check_labels,
    merged_params,
    named_segments,
    stage_key,
    track_id,
)
from transcriber.domain.lab_scoring import (
    PipelineScore,
    rank_scores,
    score_pipeline,
)
from transcriber.domain.models import SessionState, Voiceprint
from transcriber.domain.parameters import TranscriptionParameters

JOB_PREFIX = "lab:"
"""Job runner key of a run: ``lab:<run id>`` (never a session id)."""

ACTIVE = ("queued", "running")
FINISHED = ("done", "partial", "failed", "cancelled", "interrupted")
MAX_PIPELINES = 600
MAX_PAGE = 200

CatalogueState = Literal["ready", "needs_token", "not_installed"]


def job_key(run_id: str) -> str:
    """Return the job runner key of ``run_id``."""
    return JOB_PREFIX + run_id


def check_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is a valid run id, else raise."""
    return check_id(run_id, LAB_RUN_ID_PATTERN, "lab run")


class CatalogueItem(BaseModel):
    """A catalogue entry with what this machine can do with it.

    ``state``: ``ready``; ``needs_token`` (gated, no token: accept the
    conditions at ``token_url`` and log in); ``not_installed`` (its
    environment is missing or broken: ``reason``). ``weights_cached``
    tells whether its weights are already downloaded (``None``:
    unknown); a first run downloads them.
    """

    model_config = ConfigDict(frozen=True)

    entry: CatalogueEntry
    state: CatalogueState
    reason: Optional[str] = None
    weights_cached: Optional[bool] = None


class CatalogueView(BaseModel):
    """The catalogue as the pages show it."""

    model_config = ConfigDict(frozen=True)

    items: List[CatalogueItem]
    token_present: bool


class ShowCatalogue:
    """List the candidate models with their installed/gated/token state."""

    def __init__(
        self, source: CatalogueSource, environment: LabEnvironment
    ) -> None:
        """Wire the catalogue and the machine's environment."""
        self._source = source
        self._environment = environment

    def __call__(self) -> CatalogueView:
        """Return every entry with its state."""
        token = self._environment.token_present()
        items = []
        problems: Dict[str, Optional[str]] = {}
        for entry in self._source.load().entries:
            if entry.family not in problems:
                problems[entry.family] = (
                    None if entry.family == PROJECT_FAMILY
                    else self._environment.family_problem(entry.family)
                )
            problem = problems[entry.family]
            cached = [self._environment.weights_cached(w)
                      for w in entry.weights]
            known = [c for c in cached if c is not None]
            state: CatalogueState = "ready"
            if problem is not None:
                state = "not_installed"
            elif entry.gated and not token:
                state = "needs_token"
                problem = (
                    "needs token: accept the conditions at"
                    f" {entry.token_url or entry.source_url} and log in"
                    " (huggingface-cli login)"
                )
            items.append(CatalogueItem(
                entry=entry, state=state, reason=problem,
                weights_cached=all(known) if known else None,
            ))
        return CatalogueView(items=items, token_present=token)


class LabRunView(BaseModel):
    """A run as the pages follow it.

    ``state`` is the stored one, except ``interrupted`` for a run whose
    process is gone. ``pipelines`` gives each pipeline's state:
    ``done`` once all its steps are, ``failed`` when one failed (with
    the first ``error``), else the state of its least advanced step.
    """

    model_config = ConfigDict(frozen=True)

    run: LabRun
    state: str
    job: Optional[JobStatus] = None
    pipelines: List[Dict[str, Any]]
    done_steps: int
    total_steps: int


def plan_steps(request: LabRequest) -> List[Step]:
    """Return the steps a request needs, in the order they run.

    Denoisers first; then, per denoised audio, the built-in diarization
    (other diarizers' speakers may be named from it), the other
    diarizers and the recognisers.
    """
    steps = [
        Step(stage="denoise", model=d, denoiser=d)
        for d in request.denoisers if d != NO_DENOISER
    ]
    diarizers = sorted(
        request.diarizers, key=lambda z: z != BUILTIN_DIARIZER
    )
    for d in request.denoisers:
        steps.extend(
            Step(stage="diarize", model=z, denoiser=d) for z in diarizers
        )
        steps.extend(
            Step(stage="recognise", model=r, denoiser=d)
            for r in request.recognisers
        )
    return steps


class StartLabRun:
    """Check a request, store the original audio and the run, queue it."""

    def __init__(
        self,
        sessions: SessionRepository,
        repository: LabRepository,
        catalogue: CatalogueSource,
        runner: JobRunner,
        execute: "RunLab",
        clock: Clock,
        processes: ProcessTable,
        decoder: AudioDecoder,
        locks: SessionLocks,
    ) -> None:
        """Wire the ports; ``locks`` are the sessions' locks."""
        self._sessions = sessions
        self._repository = repository
        self._catalogue = catalogue
        self._runner = runner
        self._execute = execute
        self._clock = clock
        self._processes = processes
        self._decoder = decoder
        self._locks = locks

    def hold(self, session_id: str) -> ContextManager[None]:
        """Return a context holding the lock of ``session_id``."""
        return self._locks.hold(session_id)

    def __call__(self, session_id: str, request: LabRequest) -> LabRun:
        """Store and queue a run of ``request`` on ``session_id``.

        The session must have audio (any state but ``recording``);
        every model must be in the catalogue under its stage and every
        parameter override must be one the model declares. Everything
        from the session check to the queueing happens under the
        session's lock, so a concurrent deletion cannot interleave.
        """
        with self.hold(session_id):
            session = require_session(self._sessions, session_id)
            if session.state is SessionState.RECORDING:
                raise StateError("The session is still recording.")
            catalogue = self._catalogue.load()
            _check_request(catalogue, request)
            self.store_original(session.id)
            run = LabRun(
                id=uuid.uuid4().hex, session_id=session.id,
                created_at=self._clock.now(), request=request,
                steps=plan_steps(request),
                worker_pid=self._processes.current(),
            )
            self._repository.save_run(run)
            self.submit(run)
            return run

    def store_original(self, session_id: str) -> str:
        """Store the session's original audio unless stored; return its key.

        The player's audio route reads it, so it is written here (a
        request that starts or resumes a run) rather than on playback.
        """
        key = self._repository.original_key(session_id)
        if key is None:
            key = self._repository.store_original(
                session_id,
                self._decoder.decode(self._sessions.recording(session_id)),
            )
        return key

    def submit(self, run: LabRun) -> None:
        """Queue the run's job."""
        session_id, run_id = run.session_id, run.id

        def job(progress: Progress) -> None:
            """Run the lab steps of the run."""
            self._execute(session_id, run_id, progress)

        self._runner.submit(job_key(run.id), job)


def _check_request(catalogue: Catalogue, request: LabRequest) -> None:
    """Raise unless every model and override of ``request`` is known."""
    stages: Sequence[Tuple[Stage, Sequence[str]]] = (
        ("denoise", request.denoisers), ("diarize", request.diarizers),
        ("recognise", request.recognisers),
    )
    chosen = []
    for stage, ids in stages:
        for model_id in ids:
            entry = catalogue.require(model_id, stage)
            merged_params(entry, request.params.get(model_id, {}), {})
            chosen.append(model_id)
    strangers = sorted(set(request.params) - set(chosen))
    if strangers:
        raise InvalidInputError(
            f"Parameters for models not chosen: {', '.join(strangers)}."
        )
    count = (
        len(request.denoisers) * max(len(request.diarizers), 1)
        * max(len(request.recognisers), 1)
    )
    if count > MAX_PIPELINES:
        raise InvalidInputError(
            f"{count} pipelines; at most {MAX_PIPELINES} per run."
        )


class ResumeLabRun:
    """Queue a finished, cancelled or interrupted run again.

    Steps already done stay done; the others are tried again (their
    results may also have been computed meanwhile by another run). The
    state check and the save happen under the run's lock (``locks``,
    keyed by :func:`job_key`), so of two concurrent resumes one queues
    the run and the other is refused. The session's lock is held
    around it all (taken first, then the run's), so a concurrent
    deletion of the session cannot interleave with the writes.
    """

    def __init__(
        self, repository: LabRepository, start: StartLabRun,
        status: "LabRunStatus", processes: ProcessTable,
        locks: SessionLocks,
    ) -> None:
        """Wire the ports."""
        self._repository = repository
        self._start = start
        self._status = status
        self._processes = processes
        self._locks = locks

    def __call__(self, session_id: str, run_id: str) -> LabRun:
        """Queue the run again; a queued or running run is refused."""
        with self._start.hold(session_id), self._locks.hold(
            job_key(check_run_id(run_id))
        ):
            view = self._status(session_id, run_id)
            if view.state in ACTIVE:
                raise StateError("The run is still queued or running.")
            self._start.store_original(view.run.session_id)
            self._repository.clear_cancel(session_id, run_id)
            run = view.run.model_copy(update={
                "state": "queued", "finished_at": None, "message": None,
                "worker_pid": self._processes.current(),
                "steps": [
                    s if s.state == "done" else s.model_copy(update={
                        "state": "pending", "error": None,
                    })
                    for s in view.run.steps
                ],
            })
            self._repository.save_run(run)
            self._start.submit(run)
            return run


class CancelLabRun:
    """Ask a run to stop; it stops at the latest at its next step."""

    def __init__(
        self, repository: LabRepository, status: "LabRunStatus"
    ) -> None:
        """Wire the ports."""
        self._repository = repository
        self._status = status

    def __call__(self, session_id: str, run_id: str) -> LabRun:
        """Mark the run to be cancelled; return it as it is now."""
        view = self._status(session_id, run_id)
        if view.state not in ACTIVE:
            raise StateError("The run is not queued or running.")
        self._repository.request_cancel(session_id, run_id)
        return view.run


class LabRunStatus:
    """Report on a run: its steps, each pipeline and the job."""

    def __init__(
        self,
        sessions: SessionRepository,
        repository: LabRepository,
        runner: JobRunner,
        processes: ProcessTable,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._repository = repository
        self._runner = runner
        self._processes = processes

    def run(self, session_id: str, run_id: str) -> LabRun:
        """Return the stored run or raise :class:`NotFoundError`."""
        require_session(self._sessions, session_id)
        found = self._repository.run(session_id, check_run_id(run_id))
        if found is None:
            raise NotFoundError(f"No lab run {run_id}.")
        return found

    def state(self, run: LabRun) -> str:
        """Return the run's state, ``interrupted`` if its process died."""
        if run.state in ACTIVE:
            pid = run.worker_pid
            mine = pid == self._processes.current()
            if mine and self._runner.status(job_key(run.id)) is None:
                return "interrupted"
            if pid is None or (not mine and not self._processes.alive(pid)):
                return "interrupted"
        return run.state

    def __call__(self, session_id: str, run_id: str) -> LabRunView:
        """Return the run with its pipelines' states."""
        run = self.run(session_id, run_id)
        steps = {s.name: s for s in run.steps}
        pipelines = []
        for ref in run.pipelines():
            needed = [
                steps.get(name) for name in _pipeline_steps(ref.denoiser,
                                                            ref.diarizer,
                                                            ref.recogniser)
            ]
            present = [s for s in needed if s is not None]
            failed = [s for s in present if s.state in ("failed",
                                                        "cancelled")]
            if failed:
                state = failed[0].state
            elif all(s.state == "done" for s in present):
                state = "done"
            elif any(s.state == "running" for s in present):
                state = "running"
            else:
                state = "pending"
            pipelines.append({
                "id": ref.id, "denoiser": ref.denoiser,
                "diarizer": ref.diarizer, "recogniser": ref.recogniser,
                "state": state,
                "error": failed[0].error.model_dump()
                if failed and failed[0].error else None,
            })
        return LabRunView(
            run=run, state=self.state(run),
            job=self._runner.status(job_key(run.id)),
            pipelines=pipelines,
            done_steps=sum(1 for s in run.steps if s.state == "done"),
            total_steps=len(run.steps),
        )


def _pipeline_steps(
    denoiser: str, diarizer: Optional[str], recogniser: Optional[str]
) -> List[str]:
    """Return the names of the steps a pipeline needs."""
    names = []
    if denoiser != NO_DENOISER:
        names.append(f"denoise:{denoiser}")
    if diarizer is not None:
        names.append(f"diarize:{track_id(denoiser, diarizer)}")
    if recogniser is not None:
        names.append(f"recognise:{track_id(denoiser, recogniser)}")
    return names


class ListLabRuns:
    """List a session's runs, newest first."""

    def __init__(
        self, sessions: SessionRepository, repository: LabRepository,
        status: LabRunStatus,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._repository = repository
        self._status = status

    def __call__(self, session_id: str) -> List[Tuple[LabRun, str]]:
        """Return each run with its reported state."""
        require_session(self._sessions, session_id)
        return [
            (run, self._status.state(run))
            for run in self._repository.runs(session_id)
        ]


class LabSessionData:
    """What the lab stored for a session, as ``ports.SessionLabData``.

    :class:`~transcriber.application.record.DeleteSession` uses it to
    refuse while a run is active and to delete the lab's data.
    """

    def __init__(
        self, repository: LabRepository, status: LabRunStatus
    ) -> None:
        """Wire the ports."""
        self._repository = repository
        self._status = status

    def active_runs(self, session_id: str) -> List[str]:
        """Return the ids of runs that are queued or running.

        The state is the reported one: a run whose process died is
        ``interrupted``, not active, so it never blocks a deletion.
        """
        return [
            run.id for run in self._repository.runs(session_id)
            if self._status.state(run) in ACTIVE
        ]

    def delete(self, session_id: str) -> None:
        """Remove the session's lab directory."""
        self._repository.delete_session(session_id)


def voiceprint_fingerprint(voiceprints: Sequence[Voiceprint]) -> str:
    """Return a short hash of the voiceprints (ids, models, vectors)."""
    data = json.dumps(
        sorted((v.speaker_id, v.model, v.vector) for v in voiceprints),
        separators=(",", ":"),
    )
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


class RunLab:
    """The job of a run: every step, the speaker mapping, the results."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        decoder: AudioDecoder,
        repository: LabRepository,
        catalogue: CatalogueSource,
        external: LabTool,
        internal: LabTool,
        encoder: VoiceEncoder,
        clock: Clock,
        processes: ProcessTable,
        parameters: TranscriptionParameters = TranscriptionParameters(),
        comparison: ComparisonParameters = ComparisonParameters(),
        release: Callable[[], None] = lambda: None,
    ) -> None:
        """Wire the ports; ``release`` unloads the in-process models."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._decoder = decoder
        self._repository = repository
        self._catalogue = catalogue
        self._external = external
        self._internal = internal
        self._encoder = encoder
        self._clock = clock
        self._processes = processes
        self._parameters = parameters
        self._comparison = comparison
        self._release = release

    def __call__(
        self, session_id: str, run_id: str, progress: Progress
    ) -> LabRun:
        """Run every step not done yet, then map and compare.

        A failing step is recorded and the run goes on; the run ends
        ``done``, ``partial`` (some steps failed), ``failed`` (no
        pipeline finished) or ``cancelled``.
        """
        repo = self._repository
        run = repo.run(session_id, run_id)
        if run is None:
            raise NotFoundError(f"No lab run {run_id}.")
        run = run.model_copy(update={
            "state": "running", "worker_pid": self._processes.current(),
        })
        repo.save_run(run)
        try:
            run = self._steps(run, progress)
        except BaseException as exc:
            reason = str(exc).strip() or type(exc).__name__
            run = run.model_copy(update={
                "state": "failed", "message": reason[:500],
                "finished_at": self._clock.now(),
            })
            repo.save_run(run)
            raise
        finally:
            self._release_all()
        repo.save_run(run)
        return run

    def _release_all(self) -> None:
        """Unload every in-process model."""
        self._internal.release()
        self._release()

    def _cancelled(self, run: LabRun) -> bool:
        """Tell whether the run was asked to stop."""
        return self._repository.cancel_requested(run.session_id, run.id)

    def _steps(self, run: LabRun, progress: Progress) -> LabRun:
        """Run the steps, the mapping and the comparison."""
        repo = self._repository
        session = require_session(self._sessions, run.session_id)
        catalogue = self._catalogue.load()
        audio = self._decoder.decode(self._sessions.recording(session.id))
        original = repo.store_original(session.id, audio)
        # a session imported from a public corpus is never compared with
        # the people enrolled here: its speakers stay anonymous
        anonymous = repo.reference(session.id) is not None
        voiceprints = [] if anonymous else self._candidates(
            session.expected_speakers
        )
        language = run.request.language or session.language
        standard: Dict[str, Any] = {
            "language": language,
            "num_speakers": run.request.num_speakers or session.num_speakers,
            "expected_speakers": list(session.expected_speakers)
            if session.expected_speakers else None,
        }
        audio_keys: Dict[str, Optional[str]] = {NO_DENOISER: original}
        total = len(run.steps) + 2
        for number, step in enumerate(run.steps):
            current = run.step(step.name) or step
            if self._cancelled(run):
                return self._cancel(run)
            progress.report(step.name, number / total)
            if current.state == "done" and current.key and repo.outcome(
                session.id, current.key
            ):
                done = current
            else:
                run = run.with_step(current.model_copy(
                    update={"state": "running", "error": None}
                ))
                repo.save_run(run)
                done = self._step(
                    run, current, catalogue, audio_keys, standard,
                    voiceprints, anonymous,
                )
                run = run.with_step(done)
                repo.save_run(run)
            if done.stage == "denoise":
                audio_keys[done.model] = (
                    done.key if done.state == "done" else None
                )
        if self._cancelled(run):
            return self._cancel(run)
        progress.report("speakers", (total - 2) / total)
        mappings = self._map(run, catalogue, voiceprints, session)
        progress.report("compare", (total - 1) / total)
        results = self._compare(run, catalogue, mappings, audio.duration)
        repo.save_results(session.id, run.id, results.model_dump(
            mode="json"
        ))
        finished = results.pipelines
        failed = any(s.state != "done" for s in run.steps)
        state = "done" if not failed else (
            "partial" if finished else "failed"
        )
        return run.model_copy(update={
            "state": state, "finished_at": self._clock.now(),
            "message": None if finished else "No pipeline finished.",
        })

    def _cancel(self, run: LabRun) -> LabRun:
        """Return the run cancelled, its unfinished steps too."""
        return run.model_copy(update={
            "state": "cancelled", "finished_at": self._clock.now(),
            "steps": [
                s if s.state in ("done", "failed")
                else s.model_copy(update={"state": "cancelled"})
                for s in run.steps
            ],
        })

    def _candidates(
        self, expected: Optional[Sequence[str]]
    ) -> List[Voiceprint]:
        """Return the voiceprints speakers may be named with."""
        model = self._encoder.model_name
        everyone = [v for v in self._voiceprints.all() if v.model == model]
        if expected:
            return [v for v in everyone if v.speaker_id in set(expected)]
        return everyone

    def _step(
        self,
        run: LabRun,
        step: Step,
        catalogue: Catalogue,
        audio_keys: Mapping[str, Optional[str]],
        standard: Mapping[str, Any],
        voiceprints: Sequence[Voiceprint],
        anonymous: bool = False,
    ) -> Step:
        """Run (or find) one step; return it done or failed.

        ``anonymous``: the in-process tools use no voiceprint (a corpus
        session).
        """
        repo = self._repository
        session_id = run.session_id
        source = NO_DENOISER if step.stage == "denoise" else step.denoiser
        audio_key = audio_keys.get(source)
        if audio_key is None:
            return _failed(step, StageError(
                code="failed",
                message=f"The denoiser {step.denoiser} did not finish.",
            ))
        entry = catalogue.require(step.model, step.stage)
        try:
            params = merged_params(
                entry, run.request.params.get(step.model, {}), standard
            )
        except InvalidInputError as exc:
            return _failed(step, StageError(code="bad_input",
                                            message=str(exc)))
        extra: Dict[str, Any] = {}
        if entry.family == PROJECT_FAMILY:
            extra = {
                "parameters": self._parameters.model_dump(mode="json"),
                "voiceprints": voiceprint_fingerprint(voiceprints),
            }
        if entry.revision > 1:
            extra = {**extra, "revision": entry.revision}
        key = stage_key(audio_key, step.stage, step.model, params, extra)
        found = repo.outcome(session_id, key)
        if found is not None:
            return step.model_copy(update={
                "state": "done", "key": key, "cached": True,
                "seconds": found.seconds, "error": None,
            })
        tool = (
            self._internal if entry.family == PROJECT_FAMILY
            else self._external
        )
        workspace = repo.new_workspace(session_id)
        request = ToolRequest(
            entry=entry, stage=step.stage, params=params,
            audio=repo.audio_path(session_id, audio_key),
            workspace=workspace,
            load=lambda: repo.load_audio(session_id, audio_key),
            anonymous=anonymous,
        )
        try:
            outcome = tool.run(request, lambda: self._cancelled(run))
            kept = repo.keep(session_id, key, outcome, workspace)
        except LabStepError as exc:
            repo.discard(session_id, workspace)
            return _failed(step, StageError(
                code=_code(exc.code), message=str(exc)[:1000]
            ))
        except Exception as exc:
            repo.discard(session_id, workspace)
            return _failed(step, StageError(
                code="failed",
                message=(str(exc).strip() or type(exc).__name__)[:1000],
            ))
        finally:
            if entry.family == PROJECT_FAMILY:
                self._internal.release()
        return step.model_copy(update={
            "state": "done", "key": key, "cached": False,
            "seconds": kept.seconds, "error": None,
        })

    def _map(
        self, run: LabRun, catalogue: Catalogue,
        voiceprints: Sequence[Voiceprint], session: Any,
    ) -> Dict[str, List[SpeakerMapping]]:
        """Name every finished diarization's speakers (step name → map)."""
        repo = self._repository
        p = self._parameters
        listed = bool(session.expected_speakers)
        threshold = p.expected_floor if listed else p.match_threshold
        fingerprint = voiceprint_fingerprint(voiceprints)
        builtin: Dict[str, List[Any]] = {}
        out: Dict[str, List[SpeakerMapping]] = {}
        diarized = [
            s for s in run.steps if s.stage == "diarize" and s.state == "done"
        ]
        for step in diarized:
            if step.model == BUILTIN_DIARIZER and step.key:
                outcome = repo.outcome(run.session_id, step.key)
                if outcome is not None:
                    builtin[step.denoiser] = outcome.segments
                    out[step.name] = builtin_mapping(outcome.segments)
        audios: Dict[str, Audio] = {}
        try:
            for step in diarized:
                if step.name in out or not step.key:
                    continue
                outcome = repo.outcome(run.session_id, step.key)
                if outcome is None:
                    continue
                reference = builtin.get(step.denoiser) or builtin.get(
                    NO_DENOISER
                )
                key = hashlib.sha1(json.dumps([
                    step.key, fingerprint, threshold,
                    p.model_dump(mode="json"),
                    stage_key("", "diarize", BUILTIN_DIARIZER,
                              {"segments": [s.model_dump() for s in
                                            reference or []]}),
                ], sort_keys=True, default=str).encode()).hexdigest()
                stored = repo.mapping(run.session_id, key)
                if stored is None:
                    audio = self._input_audio(run, step, outcome, audios)
                    stored = map_speakers(
                        outcome.segments, audio, self._encoder, voiceprints,
                        threshold, p, reference,
                    )
                    repo.save_mapping(run.session_id, key, stored)
                out[step.name] = stored
        finally:
            self._release()
        return out

    def _input_audio(
        self, run: LabRun, step: Step, outcome: StageOutcome,
        cache: Dict[str, Audio],
    ) -> Audio:
        """Return the audio ``step`` read (decoded once per denoiser)."""
        if step.denoiser not in cache:
            if step.denoiser == NO_DENOISER:
                original = self._repository.original_key(run.session_id)
                if original is None:
                    raise StateError("The original audio is missing.")
                key = original
            else:
                denoise = run.step(f"denoise:{step.denoiser}")
                if denoise is None or denoise.key is None:
                    raise StateError("The denoised audio is missing.")
                key = denoise.key
            cache[step.denoiser] = self._repository.load_audio(
                run.session_id, key
            )
        return cache[step.denoiser]

    def _compare(
        self, run: LabRun, catalogue: Catalogue,
        mappings: Mapping[str, List[SpeakerMapping]], duration: float,
    ) -> LabResults:
        """Compare what the run's finished steps produced."""
        return build_results(
            run, catalogue, self._repository, mappings, duration,
            self._comparison,
        )


def build_results(
    run: LabRun,
    catalogue: Catalogue,
    repository: LabRepository,
    mappings: Mapping[str, List[SpeakerMapping]],
    duration: float,
    comparison: ComparisonParameters = ComparisonParameters(),
) -> LabResults:
    """Return the comparison of ``run`` from its stored stage results.

    A session imported from a corpus is also measured against its
    reference (``reference`` of the results).
    """
    diarizations: List[Diarization] = []
    recognitions: List[Recognition] = []
    costs: Dict[str, StageOutcome] = {}
    entries = {e.id: e for e in catalogue.entries}
    for step in run.steps:
        if step.state != "done" or step.key is None:
            continue
        outcome = repository.outcome(run.session_id, step.key)
        entry = entries.get(step.model)
        if outcome is None or entry is None:
            continue
        if step.stage == "denoise":
            costs[step.model] = outcome
        elif step.stage == "diarize":
            mapping = mappings.get(step.name, [])
            diarizations.append(Diarization(
                denoiser=step.denoiser, entry=entry, outcome=outcome,
                mapping=mapping,
                segments=named_segments(outcome.segments, mapping),
            ))
        else:
            recognitions.append(Recognition(
                denoiser=step.denoiser, entry=entry, outcome=outcome,
            ))
    return compare_run(
        run.id, run.session_id, duration, run.pipelines(), diarizations,
        recognitions, entries, costs, comparison,
        repository.reference(run.session_id),
    )


def _failed(step: Step, error: StageError) -> Step:
    """Return ``step`` failed with ``error``."""
    return step.model_copy(update={"state": "failed", "error": error})


ERROR_CODES = (
    "failed", "unknown_model", "bad_input", "needs_token", "out_of_memory",
    "unsupported_language", "not_installed", "timeout", "cancelled",
)


def _code(code: str) -> Any:
    """Return ``code`` if it is a known error code, else ``failed``."""
    return code if code in ERROR_CODES else "failed"


class LabResultsView:
    """The stored comparison of a run, and its pages of fragments."""

    def __init__(
        self, status: LabRunStatus, repository: LabRepository
    ) -> None:
        """Wire the ports."""
        self._status = status
        self._repository = repository

    def __call__(self, session_id: str, run_id: str) -> Dict[str, Any]:
        """Return the results without the fragments (see ``fragments``).

        :class:`NotFoundError` while the run has none yet.
        """
        results = self._results(session_id, run_id)
        return {k: v for k, v in results.items() if k != "fragments"} | {
            "fragment_count": len(results.get("fragments", [])),
        }

    def _results(self, session_id: str, run_id: str) -> Dict[str, Any]:
        """Return the stored results or raise :class:`NotFoundError`."""
        run = self._status.run(session_id, run_id)
        results = self._repository.results(session_id, run.id)
        if results is None:
            raise NotFoundError("The run has no results yet.")
        return results

    def tracks(self, session_id: str, run_id: str) -> Dict[str, Any]:
        """Return every track's named segments and the consensus spans."""
        run = self._status.run(session_id, run_id)
        results = self._results(session_id, run_id)
        mappings = {t["id"]: t["mapping"] for t in results["tracks"]}
        tracks: Dict[str, Any] = {}
        for step in run.steps:
            if step.stage != "diarize" or step.state != "done":
                continue
            outcome = self._outcome(session_id, step)
            mapping = [SpeakerMapping.model_validate(m)
                       for m in mappings.get(step.name.split(":", 1)[1],
                                             [])]
            tracks[track_id(step.denoiser, step.model)] = [
                s.model_dump() for s in
                named_segments(outcome.segments, mapping)
            ]
        return {"tracks": tracks, "consensus": results["consensus"]["spans"]}

    def _outcome(self, session_id: str, step: Step) -> StageOutcome:
        """Return a step's stored outcome or raise NotFoundError."""
        outcome = (
            self._repository.outcome(session_id, step.key)
            if step.key else None
        )
        if outcome is None:
            raise NotFoundError(f"The result of {step.name} is missing.")
        return outcome

    def fragments(
        self, session_id: str, run_id: str, offset: int = 0,
        limit: int = 50, sort: str = "disagreement",
        unlabelled: bool = False,
    ) -> Dict[str, Any]:
        """Return a page of fragments with each pipeline's answer.

        ``sort``: ``disagreement`` (most disputed first, then most
        different texts, then longest) or ``time``. Each fragment
        lists, per pipeline, its speakers and its words grouped by the
        speaker the pipeline gives them, and the reference label that
        covers it, if any. ``unlabelled`` keeps only the fragments no
        label covers (``total`` counts those).
        """
        if sort not in ("disagreement", "time"):
            raise InvalidInputError("sort must be disagreement or time.")
        if offset < 0 or not 1 <= limit <= MAX_PAGE:
            raise InvalidInputError(
                f"offset must be 0 or more and limit 1 to {MAX_PAGE}."
            )
        run = self._status.run(session_id, run_id)
        results = self._results(session_id, run_id)
        labels = self._repository.labels(session_id)
        listed = list(results.get("fragments", []))
        if unlabelled:
            listed = [f for f in listed if _covering(labels, f) is None]
        if sort == "disagreement":
            listed.sort(key=lambda f: (
                -f["disagreement"], -f.get("distinct_texts", 0),
                -(f["end"] - f["start"]), f["start"],
            ))
        page = listed[offset:offset + limit]
        words, segments = self._sources(session_id, run, results)
        shown = []
        for fragment in page:
            pipelines: Dict[str, Any] = {}
            for row in results["pipelines"]:
                pipelines[row["id"]] = _pipeline_answer(
                    row, fragment, words, segments
                )
            shown.append({
                **fragment, "pipelines": pipelines,
                "label": _covering(labels, fragment),
            })
        return {"total": len(listed), "offset": offset, "limit": limit,
                "sort": sort, "fragments": shown}

    def spans(self, session_id: str, run_id: str) -> Dict[str, Any]:
        """Return every fragment in time order, without the answers.

        Each item: ``index``, ``start``, ``end``, ``disagreement`` and
        ``distinct_answers``: enough for a page to count labelled
        fragments and to draw where the disputes are.
        """
        self._status.run(session_id, run_id)
        results = self._results(session_id, run_id)
        return {"spans": [
            {"index": f["index"], "start": f["start"], "end": f["end"],
             "disagreement": f["disagreement"],
             "distinct_answers": f["distinct_answers"]}
            for f in sorted(results.get("fragments", []),
                            key=lambda f: f["start"])
        ]}

    def _sources(
        self, session_id: str, run: LabRun, results: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return each transcript's words and each track's segments."""
        mappings = {t["id"]: t["mapping"] for t in results["tracks"]}
        words: Dict[str, Any] = {}
        segments: Dict[str, Any] = {}
        for step in run.steps:
            if step.state != "done" or step.stage == "denoise":
                continue
            ident = track_id(step.denoiser, step.model)
            if step.stage == "recognise":
                words[ident] = self._outcome(session_id, step).words
            elif ident in mappings:
                segments[ident] = named_segments(
                    self._outcome(session_id, step).segments,
                    [SpeakerMapping.model_validate(m)
                     for m in mappings[ident]],
                )
        return words, segments


def _pipeline_answer(
    row: Mapping[str, Any], fragment: Mapping[str, Any],
    words: Mapping[str, Any], segments: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return what one pipeline says about one fragment."""
    track = str(row.get("track") or "")
    transcript = str(row.get("transcript") or "")
    speakers = fragment["answers"].get(track, []) if track else []
    groups: List[Dict[str, Any]] = []
    if transcript in words:
        inside = [
            w for w in words[transcript]
            if fragment["start"] <= (w.start + w.end) / 2 < fragment["end"]
        ]
        if track in segments:
            inside = attributed_words(inside, segments[track])
        for word in inside:
            if groups and groups[-1]["speaker"] == word.speaker:
                groups[-1]["text"] += " " + word.text.strip()
            else:
                groups.append({"speaker": word.speaker,
                               "text": word.text.strip()})
    return {
        "speakers": speakers,
        "text": " ".join(g["text"] for g in groups),
        "words_by_speaker": groups,
    }


def _covering(
    labels: Sequence[ReferenceLabel], fragment: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return the label holding the fragment's middle, if any."""
    middle = (fragment["start"] + fragment["end"]) / 2
    for label in labels:
        if label.start <= middle < label.end:
            return label.model_dump(mode="json")
    return None


class LabLabels:
    """Read and replace a session's reference labels."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        repository: LabRepository,
        decoder: AudioDecoder,
        clock: Clock,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._repository = repository
        self._decoder = decoder
        self._clock = clock

    def __call__(self, session_id: str) -> List[ReferenceLabel]:
        """Return the labels in time order."""
        require_session(self._sessions, session_id)
        return self._repository.labels(session_id)

    def speakers(self) -> List[str]:
        """Return the enrolled ids a label may name."""
        return sorted(v.speaker_id for v in self._voiceprints.all())

    def replace(
        self, session_id: str, labels: Sequence[ReferenceLabel]
    ) -> List[ReferenceLabel]:
        """Replace every label of the session (checked as a set).

        Labels without ``updated_at`` get the current time.
        """
        session = require_session(self._sessions, session_id)
        duration = session.duration
        if duration <= 0:
            duration = self._decoder.decode(
                self._sessions.recording(session.id)
            ).duration
        now = self._clock.now()
        stamped = [
            label if label.updated_at else label.model_copy(
                update={"updated_at": now}
            )
            for label in labels
        ]
        checked = check_labels(stamped, duration, self.speakers())
        self._repository.save_labels(session.id, checked)
        return checked


class LabScores:
    """Score every finished pipeline of a run against the labels."""

    def __init__(
        self, status: LabRunStatus, repository: LabRepository,
        comparison: ComparisonParameters = ComparisonParameters(),
    ) -> None:
        """Wire the ports."""
        self._status = status
        self._repository = repository
        self._comparison = comparison

    def __call__(self, session_id: str, run_id: str) -> Dict[str, Any]:
        """Return the ranked scores and the labelled time they rest on."""
        run = self._status.run(session_id, run_id)
        results = self._repository.results(session_id, run.id)
        if results is None:
            raise NotFoundError("The run has no results yet.")
        labels = self._repository.labels(session_id)
        view = LabResultsView(self._status, self._repository)
        words, segments = view._sources(session_id, run, results)
        scores: List[PipelineScore] = []
        for row in results["pipelines"]:
            track = segments.get(row.get("track") or "", [])
            text = list(words.get(row.get("transcript") or "", []))
            if track:
                text = attributed_words(text, track)
            scores.append(score_pipeline(
                row["id"], track, text, labels, results["duration"],
                self._comparison,
            ))
        ranked = rank_scores(scores)
        scored = [lb for lb in labels if lb.kind != "unsure"]
        return {
            "labels": len(labels),
            "labelled_seconds": round(sum(lb.end - lb.start
                                          for lb in scored), 3),
            "unsure_seconds": round(sum(
                lb.end - lb.start for lb in labels if lb.kind == "unsure"
            ), 3),
            "text_labels": sum(1 for lb in scored if lb.text is not None),
            "scores": [s.model_dump(mode="json") for s in ranked],
        }


class LabAudio:
    """The original or a denoised audio of a run, for the player.

    It only reads: the original is stored when a run is started or
    resumed (:meth:`StartLabRun.store_original`).
    """

    def __init__(
        self, status: LabRunStatus, repository: LabRepository,
    ) -> None:
        """Wire the ports."""
        self._status = status
        self._repository = repository

    def __call__(
        self, session_id: str, run_id: str, denoiser: str
    ) -> PlaybackAudio:
        """Return the WAV the run's ``denoiser`` wrote (``none``: original).

        Raises :class:`NotFoundError` when that audio is not stored.
        """
        run = self._status.run(session_id, run_id)
        if denoiser == NO_DENOISER:
            key = self._repository.original_key(session_id)
            if key is None:
                raise NotFoundError("The original audio is not stored.")
            return self._repository.playback(session_id, key)
        step = run.step(f"denoise:{denoiser}")
        if step is None or step.state != "done" or step.key is None:
            raise NotFoundError(f"No denoised audio of {denoiser!r}.")
        return self._repository.playback(session_id, step.key)
