"""Use case: the whole soft-skills flow on one session, step by step.

``skills analyze --session <id>`` (and ``POST /analysis/sessions/<id>``
on the RAG) runs, in order, on the session's ``transcript.v1``:

a. ``sentences``: the sentences of each person with their times;
b. ``interaction``: who spoke, how much, how evenly;
c. ``candidates``: rubric indicators each sentence may show, found by
   retrieval (the RAG step; no language model);
d. ``acts``: dialogue acts per sentence (#82), and
e. ``events``: decisions, action items, unanswered questions (#83),
   both behind ports and marked ``unavailable`` while those modules
   do not exist;
f. ``confirmation``: the selected judge confirms or rejects the
   candidates (Claude Code, the local model behind it, or both);
g. ``assessment``: evidence per person and scorecards, with a
   provisional estimate and observations where there is some evidence
   but not enough for a firm level (#103, using step f's
   confirmations; no extra judge call);
h. ``reports``: one HTML page per person and the session summary;
i. ``synthesis``: per person, an explanatory profile written by Claude
   Code from everything above (what they stand out in, with cited
   quotes; what could not be observed; how it could help in the job
   sought and what to ask next; how confident), checked against the
   analysis and added on top of each report (#113). The job sought is
   kept with the run (``job.json``); changing it reruns only this step.

Every step's result is stored under the session
(:class:`~src.application.skills.ports.AnalysisStore`) together with
its record: status, duration, who ran it and counts
(:mod:`src.domain.skills.analysis_run`). The record is saved after
every step, so a failure keeps the earlier results and the next run
resumes at the step that failed.
"""

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple,
)

from src.application.skills.assessment import AssessmentService
from src.application.skills.device_notes import join_notes, new_notes
from src.application.skills.judges import (
    CallBudget,
    compare_confirmations,
    release,
)
from src.application.skills.person_report import build_person_report
from src.application.skills.person_report_html import render_person_html
from src.application.skills.person_report_texts import TEXTS
from src.application.skills.ports import (
    AnalysisLauncher,
    AnalysisLock,
    AnalysisStore,
    DialogueActTagger,
    EventExtractor,
    StartLog,
    StepOutput,
    TranscriptRepository,
    TranscriptStore,
)
from src.application.skills.report import render_scorecards
from src.application.skills.synthesis import ProfileSynthesizer, person_facts
from src.application.skills.tagging import (
    SentenceTagger,
    TagConfirmer,
    TaggedSentence,
)
from src.domain.errors import (
    BusyError,
    InvalidInputError,
    LimitError,
    ModelError,
    NotFoundError,
    short_reason,
)
from src.domain.skills.analysis_run import (
    STEP_OUTPUTS,
    STEPS,
    AnalysisRun,
    StepName,
    StepRecord,
    new_run,
    shown_status,
    steps_to_run,
)
from src.domain.skills.assessment import Observation, SessionAssessment
from src.domain.skills.interaction import describe
from src.domain.skills.judging import JudgeMode
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.speech import Sentence, transcript_sentences
from src.domain.skills.synthesis import JobSought, PersonSynthesis, skipped
from src.domain.skills.transcript import Transcript
from src.domain.skills.transcript_contract import is_session_id
from src.domain.skills.usage import StepUsage, UsageLedger

RUN_FILE = "run.json"
"""The run's record, next to the step outputs."""
SCORECARD_FILE = "scorecard.md"
"""The technical scorecard of the whole session (step h)."""
JOB_FILE = "job.json"
"""The job sought (:class:`~src.domain.skills.synthesis.JobSought`),
kept across restarts; step i reads it."""

SYNTHESIS_INPUTS = frozenset(
    {"acts", "events", "confirmation", "assessment"}
)
"""Steps step i reads, whose rerun makes the stored profiles stale:
a done step i goes back to pending, so the reports never show profiles
of earlier acts, events or assessment (#115, #121)."""

UNAVAILABLE = {
    "acts": "Not available: no dialogue-act tagger is configured (its "
            "settings card is missing).",
    "events": "Not available yet: decisions, action items and open "
              "questions (#83) are not connected.",
    "events_without_acts": "Not available: the events are derived from "
                           "the dialogue acts of step d, which did not "
                           "run.",
    "synthesis": "Not available: no judge is configured for the "
                 "explanatory profiles.",
}
"""Why a step behind a missing module is skipped."""


@dataclass(frozen=True)
class JudgeSetup:
    """The judges of steps f and g, built only when they are needed.

    ``second`` is the local judge's confirmer in ``both`` mode;
    ``budget``, when given, is the first judge's call limit for the whole
    analysis (steps f, g and i share it). ``synthesis`` builds the
    writer of step i given the owner's leave to use Claude when the
    run's judge is the local model (``None``: step i is unavailable).
    ``usage`` is the ledger the judges' tokens and cache hits go to;
    each step records what it added.
    """

    mode: JudgeMode
    first: TagConfirmer
    assessment: AssessmentService
    second: Optional[TagConfirmer] = None
    budget: Optional[CallBudget] = None
    synthesis: Optional[Callable[[bool], ProfileSynthesizer]] = None
    usage: Optional[UsageLedger] = None


def report_file(position: int) -> str:
    """Return the file name of the ``position``-th person's report."""
    return f"report-{position}.html"


def dumps(data: Any) -> str:
    """Return ``data`` as the JSON every result file uses."""
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def judge_models(
    tagged: Sequence[TaggedSentence], second: bool = False
) -> List[str]:
    """Return the models that made the decisions of ``tagged``."""
    decisions = [t.second_decision if second else t.decision for t in tagged]
    return sorted({d.model for d in decisions if d is not None and d.model})


def confirmed_observations(
    tagged: Sequence[TaggedSentence],
) -> List[Observation]:
    """Return step f's confirmed sentences as observations.

    Only the first judge's decisions count (in ``both`` mode the second
    one is a comparison). The judge's reason is left out: nothing a
    model writes reaches the reports.
    """
    found: List[Observation] = []
    for item in tagged:
        decision = item.decision
        if (
            decision is None or decision.status != "confirmed"
            or not decision.indicator_id or decision.polarity is None
        ):
            continue
        found.append(Observation(
            turn=item.sentence.turn, quote=item.sentence.text,
            indicator_id=decision.indicator_id, polarity=decision.polarity,
            source="confirmation", start=item.sentence.start,
            sentence_id=item.sentence.id,
        ))
    return found


def most_promising(
    tagged: Sequence[TaggedSentence], limit: int
) -> List[int]:
    """Return the positions of the sentences the judge should read.

    All sentences with candidates when ``limit`` is 0; otherwise the
    ``limit`` with the best candidate score, in speaking order.
    """
    with_tags = [n for n, t in enumerate(tagged) if t.tags]
    if limit <= 0 or len(with_tags) <= limit:
        return with_tags
    best = sorted(
        with_tags, key=lambda n: (-max(t.score for t in tagged[n].tags), n)
    )[:limit]
    return sorted(best)


def chosen_steps(names: Sequence[str]) -> List[StepName]:
    """Return the named steps in running order; refuse unknown names."""
    wanted = {str(n).strip() for n in names if str(n).strip()}
    unknown = sorted(wanted - set(STEPS))
    if unknown or not wanted:
        raise InvalidInputError(
            f"Unknown step(s) {', '.join(unknown) or '(none)'}; choose "
            f"from {', '.join(STEPS)}."
        )
    return [name for name in STEPS if name in wanted]


def read_job(store: AnalysisStore, session_id: str) -> Optional[JobSought]:
    """Return the job sought stored for ``session_id``, if any."""
    try:
        return JobSought.model_validate_json(store.read(session_id, JOB_FILE))
    except NotFoundError:
        return None
    except ValueError:
        return None


def check_session(session_id: str) -> None:
    """Refuse a session id the contract does not allow."""
    if not is_session_id(session_id):
        raise InvalidInputError(f"Invalid session id {session_id!r}.")


class AnalysisResults:
    """Read what an analysis stored: its run, step results, reports.

    It needs only the store, so serving the results never builds a
    model. A step's result is served only while that step is ``done``
    in the stored run, so a restarted analysis never shows the files of
    the run before it. Between a restart's launch and the moment its
    process clears the old run, that run is :meth:`hide`-den: no step
    result or report of it is served (#99).
    """

    def __init__(self, store: AnalysisStore):
        """Wire the store."""
        self.store = store
        self._hidden: Dict[str, datetime] = {}
        self._guard = threading.Lock()

    def hide(self, session_id: str) -> None:
        """Serve no result of the run stored now (it is being replaced)."""
        run = self.store.load(session_id)
        if run is not None:
            with self._guard:
                self._hidden[session_id] = run.started_at

    def show(self, session_id: str) -> None:
        """Serve the stored run's results again (nothing replaces it)."""
        with self._guard:
            self._hidden.pop(session_id, None)

    def _replaced(self, run: AnalysisRun) -> bool:
        """Tell whether ``run`` is a hidden run not yet cleared."""
        with self._guard:
            hidden = self._hidden.get(run.session_id)
            if hidden is None:
                return False
            if hidden == run.started_at:
                return True
            del self._hidden[run.session_id]
            return False

    def status(self, session_id: str) -> AnalysisRun:
        """Return the stored run of ``session_id``."""
        check_session(session_id)
        run = self.store.load(session_id)
        if run is None:
            raise NotFoundError(f"No analysis of session {session_id!r}.")
        return run

    def _done(self, session_id: str, step: StepName) -> None:
        """Refuse (not found) a step that is not done in the stored run."""
        run = self.status(session_id)
        if self._replaced(run) or run.step(step).status != "done":
            raise NotFoundError(
                f"Step {step} of {session_id!r} has no result now."
            )

    def result(self, session_id: str, step: str) -> Any:
        """Return the stored result of one step, as JSON data."""
        check_session(session_id)
        if step not in STEP_OUTPUTS:
            raise InvalidInputError(f"Unknown step {step!r}.")
        name: StepName = step
        self._done(session_id, name)
        return json.loads(self.store.read(session_id, STEP_OUTPUTS[name]))

    def job(self, session_id: str) -> Optional[JobSought]:
        """Return the job sought stored with the analysis, if any."""
        check_session(session_id)
        return read_job(self.store, session_id)

    def report(self, session_id: str, position: int) -> str:
        """Return the HTML report of the ``position``-th person (1-based)."""
        check_session(session_id)
        if position < 1:
            raise InvalidInputError("Reports are numbered from 1.")
        self._done(session_id, "reports")
        return self.store.read(session_id, report_file(position))


@dataclass(frozen=True)
class StartLimits:
    """How often analyses may be started through the API.

    ``min_interval`` seconds between two starts of one session and at
    most ``daily_starts`` starts in any 24 hours; they bound what a
    page left open (or someone with the access code) can spend of a
    hosted judge's quota.
    """

    min_interval: float = 300.0
    daily_starts: int = 20


class MemoryStartLog:
    """Starts kept in memory only (a restart of the process forgets them)."""

    def __init__(self) -> None:
        """No start yet."""
        self.starts: List[Tuple[float, str]] = []

    def load(self) -> List[Tuple[float, str]]:
        """Return the starts kept."""
        return list(self.starts)

    def save(self, starts: Sequence[Tuple[float, str]]) -> None:
        """Keep ``starts``."""
        self.starts = list(starts)


class AnalysisRequests:
    """Start, follow, cancel and forget analyses (the API).

    The launcher runs ``skills analyze`` apart from the server, so a
    long analysis neither blocks the API nor keeps its models loaded
    in it; the run's record is what both share. Starts are kept in
    ``starts`` to apply :class:`StartLimits` (in memory unless a
    persistent log is given).

    A start (check, job write, launch, start log) and an erase run one
    at a time, so two simultaneous requests can neither both pass the
    limits nor let a refused one touch the job of an accepted one (#99).
    """

    def __init__(
        self,
        results: AnalysisResults,
        transcripts: TranscriptStore,
        launcher: AnalysisLauncher,
        lock: AnalysisLock,
        limits: StartLimits = StartLimits(),
        clock: Callable[[], float] = time.time,
        starts: Optional[StartLog] = None,
    ):
        """Wire the stored results, the transcripts, launcher and lock."""
        self.results = results
        self.transcripts = transcripts
        self.launcher = launcher
        self.lock = lock
        self.limits = limits
        self.clock = clock
        self.starts: StartLog = starts if starts is not None else (
            MemoryStartLog()
        )
        self._serial = threading.Lock()

    def _active(self, session_id: str) -> bool:
        """Tell whether any process (API child or CLI) analyses it now."""
        return (
            self.launcher.running(session_id)
            or self.lock.holder(session_id) is not None
        )

    def _check_limits(
        self, session_id: str, now: float
    ) -> List[Tuple[float, str]]:
        """Refuse a start too soon after the last one, or one too many.

        Return the starts of the last 24 hours.
        """
        starts = [(t, s) for t, s in self.starts.load() if now - t < 86400]
        if len(starts) >= self.limits.daily_starts:
            raise LimitError(
                f"{self.limits.daily_starts} analyses were started in the "
                "last 24 hours, the most allowed; try again later."
            )
        last = max((t for t, s in starts if s == session_id), default=None)
        if last is not None and now - last < self.limits.min_interval:
            wait = int(self.limits.min_interval - (now - last)) + 1
            raise LimitError(
                f"This session's analysis was started {int(now - last)} s "
                f"ago; wait {wait} s before starting it again."
            )
        return starts

    def start(
        self,
        session_id: str,
        restart: bool = False,
        synthetic: bool = False,
        steps: Optional[Sequence[str]] = None,
        job: Optional[JobSought] = None,
    ) -> Dict[str, Any]:
        """Start the analysis of ``session_id``; return its status.

        ``steps`` runs only those steps again (e.g. ``synthesis`` after
        the job sought changed); ``job``, when given, replaces the job
        stored with the analysis before the run starts. It is written
        before the launch, so the run always reads it, and put back when
        the launcher refuses: the earlier job, or, when there was none,
        the empty default job (no text, no leave to use Claude), which
        every reader takes as no job (#115, #121).

        Raises :class:`~src.domain.errors.NotFoundError` when the
        session has no transcript,
        :class:`~src.domain.errors.BusyError` when its analysis is
        already running and :class:`~src.domain.errors.LimitError` when
        a start limit or the launcher's concurrency limit refuses it.
        """
        check_session(session_id)
        chosen = None if steps is None else chosen_steps(steps)
        if restart and chosen is not None:
            raise InvalidInputError(
                "Choose either a restart or the steps to run again."
            )
        self.transcripts.get(session_id)
        with self._serial:
            return self._start(session_id, restart, synthetic, chosen, job)

    def _start(
        self,
        session_id: str,
        restart: bool,
        synthetic: bool,
        chosen: Optional[List[StepName]],
        job: Optional[JobSought],
    ) -> Dict[str, Any]:
        """Check, write the job and launch (one start at a time).

        The transcript is looked up again here: a deletion of the
        session holds the same in-process lock, so a start that waited
        for it finds no transcript and writes nothing (#121).
        """
        self.transcripts.get(session_id)
        if self._active(session_id):
            raise BusyError(
                f"The analysis of {session_id!r} is already running."
            )
        now = self.clock()
        starts = self._check_limits(session_id, now)
        store = self.results.store
        before: Optional[str] = None
        if job is not None:
            try:
                before = store.read(session_id, JOB_FILE)
            except NotFoundError:
                before = JobSought().model_dump_json()
            store.write(session_id, JOB_FILE, job.cleaned().model_dump_json())
        if restart:
            self.results.hide(session_id)
        try:
            self.launcher.start(session_id, restart, synthetic, chosen or [])
        except Exception:
            self.results.show(session_id)
            if before is not None:
                store.write(session_id, JOB_FILE, before)
            raise
        self.starts.save(starts + [(now, session_id)])
        return {"session_id": session_id, "active": True}

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return the stored run with ``active``: is it running now.

        Just after a start, before the worker has written its record,
        the answer is an active run with no steps. A run left
        ``running`` by a process that died reads ``interrupted``;
        starting again resumes it.
        """
        check_session(session_id)
        active = self._active(session_id)
        if not active and self._serial.acquire(blocking=False):
            # Only between starts: a start in progress has hidden the
            # old run and not launched yet (#121).
            try:
                if not self._active(session_id):
                    self.results.show(session_id)
            finally:
                self._serial.release()
        try:
            run = self.results.status(session_id)
        except NotFoundError:
            if not active:
                raise
            return {"session_id": session_id, "status": "running",
                    "steps": [], "open_issues": [], "active": True}
        job = self.results.job(session_id)
        return {
            **run.model_dump(mode="json"),
            "status": shown_status(run, active),
            "active": active,
            "job": job.model_dump() if job is not None else None,
        }

    def cancel(self, session_id: str) -> None:
        """Stop the analysis this server started for ``session_id``.

        Raises :class:`~src.domain.errors.BusyError` when it runs from
        the command line (stop it there) and
        :class:`~src.domain.errors.NotFoundError` when none runs.
        """
        check_session(session_id)
        if self.launcher.stop(session_id):
            return
        if self.lock.holder(session_id) is not None:
            raise BusyError(
                "This analysis runs from the command line; stop it there."
            )
        raise NotFoundError(f"No analysis of {session_id!r} is running.")

    def erase(self, session_id: str) -> None:
        """Stop and delete everything analysed of ``session_id``.

        Called when the session's transcript is deleted: its sentences,
        quotes and reports must not outlive it. See :meth:`erasing`.
        """
        with self.erasing(session_id):
            pass

    @contextmanager
    def erasing(self, session_id: str) -> Iterator[None]:
        """Hold ``session_id`` for the block, then delete its analysis.

        The analysis this server started is stopped first (up to the
        launcher's grace period, about 10 s, during which other starts
        and erases of this server wait); the session's lock is then
        held while the block runs (the transcript's deletion) and its
        files are deleted, so no run can read the transcript and write
        them meanwhile (#121). The files go last because the lock file
        is among them. Raises :class:`~src.domain.errors.BusyError`,
        deleting nothing, while the analysis runs from the command line
        (it would write its files again): stop it there, then delete
        (#99).
        """
        check_session(session_id)
        with self._serial:
            self.launcher.stop(session_id)
            if self.lock.acquire(session_id) is None:
                holder = self.lock.holder(session_id)
                raise BusyError(
                    "The analysis of this session runs from the command "
                    "line" + (f" (process {holder})" if holder else "")
                    + "; stop it there before deleting the session."
                )
            try:
                yield
                self.results.store.erase(session_id)
            finally:
                self.lock.release(session_id)
            self.results.show(session_id)


class SessionAnalysis:
    """Run, record and resume the whole analysis of a session.

    Args:
        transcripts: Where the session's transcript is found by id.
        store: Where every step's record and result are kept.
        rubric: The rubric for a transcript language.
        report_catalogue: The catalogue ``name`` in a language (the
            person report's wording).
        tagger: Candidate indicators by retrieval (step c).
        judges: Builds the judges of steps f and g when first needed.
        clock: The time steps start at (replaced in tests).
        lock: Keeps one run per session (CLI and API alike).
        acts: The dialogue-act tagger, or ``None`` until it exists.
        events: The event extractor, or ``None`` until it exists.
        max_confirmations: Most sentences the judge reads in step f
            (the most promising candidates first; 0: all). It bounds
            the calls made on a hosted judge's quota.
        embedding_model: Name of the retrieval's embedding model, for
            step c's record (``""``: words only).
        device_notes: Reads the notes of the shared models that left
            the GPU (out of memory, none visible); a step that added
            one has it in its note (#110).
    """

    def __init__(
        self,
        transcripts: TranscriptRepository,
        store: AnalysisStore,
        rubric: Callable[[str], RubricCatalogue],
        report_catalogue: Callable[[str, str], RubricCatalogue],
        tagger: SentenceTagger,
        judges: Callable[[], JudgeSetup],
        clock: Callable[[], datetime],
        lock: AnalysisLock,
        acts: Optional[DialogueActTagger] = None,
        events: Optional[EventExtractor] = None,
        max_confirmations: int = 0,
        embedding_model: str = "",
        device_notes: Callable[[], List[str]] = list,
    ):
        """Wire the steps; nothing runs yet."""
        self.transcripts = transcripts
        self.store = store
        self.rubric = rubric
        self.report_catalogue = report_catalogue
        self.tagger = tagger
        self.judges = judges
        self.clock = clock
        self.lock = lock
        self.acts = acts
        self.events = events
        self.max_confirmations = max(max_confirmations, 0)
        self.embedding_model = embedding_model
        self.device_notes = device_notes
        self._judges: Optional[JudgeSetup] = None
        self._base: Optional[StepUsage] = None

    def job(self, session_id: str) -> Optional[JobSought]:
        """Return the job sought stored with the analysis, if any."""
        check_session(session_id)
        return read_job(self.store, session_id)

    def run(
        self,
        session_id: str,
        restart: bool = False,
        synthetic: bool = False,
        on_step: Optional[Callable[[StepRecord], None]] = None,
        only: Optional[Sequence[str]] = None,
        job: Optional[JobSought] = None,
    ) -> AnalysisRun:
        """Run the steps still to do; return the run as stored.

        A failing step is recorded as ``failed`` with its reason and
        stops the run; the earlier results stay, and a later call
        starts again at that step. ``restart`` runs every step anew.
        ``on_step`` is told each record once its step ends. ``only``
        runs exactly those steps again (in running order) and keeps
        the others as they are, e.g. the acts and events once a tagger
        exists, without asking the judge again; rerunning step d, e, f
        or g without step i sets step i back to pending (its profiles
        were written on the earlier results). ``job`` replaces the
        job sought stored with the analysis; when the stored profiles
        were written for another job, step i runs again (only it, if
        everything else is done). Raises
        :class:`~src.domain.errors.BusyError` while another process
        analyses the same session (or its deletion holds it) and
        :class:`~src.domain.errors.NotFoundError` when it has no
        transcript (read once the session is held).
        """
        check_session(session_id)
        chosen = None if only is None else chosen_steps(only)
        if restart and chosen is not None:
            raise InvalidInputError(
                "Choose either a restart or the steps to run again, not "
                "both: a restart clears every earlier result."
            )
        worker = self.lock.acquire(session_id)
        if worker is None:
            holder = self.lock.holder(session_id)
            raise BusyError(
                f"The analysis of {session_id!r} is already running"
                + (f" (process {holder})." if holder else ".")
            )
        try:
            # Read under the lock: a deletion holds it until the
            # transcript is gone, so a run never revives an erased
            # analysis (#121).
            try:
                transcript = self.transcripts.find(session_id)
            except NotFoundError:
                self.lock.discard(session_id)
                raise
            if job is not None:
                self.store.write(
                    session_id, JOB_FILE, job.cleaned().model_dump_json()
                )
            return self._run(
                session_id, transcript, restart, synthetic, on_step, worker,
                chosen,
            )
        finally:
            self.lock.release(session_id)

    def _run(
        self,
        session_id: str,
        transcript: Transcript,
        restart: bool,
        synthetic: bool,
        on_step: Optional[Callable[[StepRecord], None]],
        worker: int,
        chosen: Optional[List[StepName]] = None,
    ) -> AnalysisRun:
        """Run the steps still to do while holding the session."""
        if restart:
            kept = read_job(self.store, session_id)
            self.store.clear(session_id)
            if kept is not None:
                self.store.write(session_id, JOB_FILE, kept.model_dump_json())
        previous = None if restart else self.store.load(session_id)
        if previous is not None and self._job_changed(session_id, previous):
            previous = previous.with_step(StepRecord(name="synthesis"))
        todo = steps_to_run(previous, restart) if chosen is None else chosen
        run = previous if previous is not None else new_run(
            session_id, self.clock(), synthetic
        )
        run = run.model_copy(update={
            "status": "running", "finished_at": None,
            "synthetic": synthetic or run.synthetic, "worker_pid": worker,
        })
        stale = (
            "synthesis" not in todo and bool(SYNTHESIS_INPUTS & set(todo))
            and run.step("synthesis").status == "done"
        )
        for name in todo:
            run = run.with_step(StepRecord(name=name))
        if stale:
            run = run.with_step(StepRecord(name="synthesis"))
        self.store.save(run)
        context = _Context(transcript, run)
        context.job = read_job(self.store, session_id) or JobSought()
        for name in todo:
            started = self.clock()
            run = run.with_step(
                StepRecord(name=name, status="running", started_at=started)
            )
            self.store.save(run)
            record = self._step(name, session_id, context, started)
            run = run.with_step(record)
            if record.status == "failed":
                run = run.model_copy(update={
                    "status": "failed", "finished_at": self.clock(),
                })
            self.store.save(run)
            if on_step is not None:
                on_step(record)
            if record.status == "failed":
                return run
        finished = not steps_to_run(run)
        run = run.model_copy(update={
            "status": "done" if finished else "partial",
            "finished_at": self.clock(),
        })
        self.store.save(run)
        return run

    def _step(
        self, name: StepName, session_id: str, context: "_Context",
        started: datetime,
    ) -> StepRecord:
        """Run one step and return its record (never raises).

        Notes the shared models added meanwhile (a move to the CPU)
        are appended to the step's note.
        """
        begin = time.perf_counter()
        before = self.device_notes()
        used_before = self._usage() if self._judges is not None else None
        try:
            step: Callable[[str, _Context], StepRecord] = getattr(
                self, f"_{name}"
            )
            record = step(session_id, context)
        except Exception as exc:  # any failure is recorded, not lost
            record = StepRecord(
                name=name, status="failed", error=short_reason(exc),
            )
        moved = new_notes(before, self.device_notes())
        used = self._usage()
        if used_before is None:
            # The judges were built during this step: what the ledger
            # held then (other work of the process) is not the step's.
            used_before = self._base
        spent = (
            used.minus(used_before) if used is not None
            and used_before is not None else used
        )
        return record.model_copy(update={
            "usage": spent if used is not None else None,
            "started_at": started,
            "seconds": round(time.perf_counter() - begin, 3),
            "note": join_notes(record.note, *moved),
        })

    def _job_changed(self, session_id: str, run: AnalysisRun) -> bool:
        """Tell whether the stored profiles were written for another job."""
        if run.step("synthesis").status != "done":
            return False
        stored = self._stored(session_id, "synthesis")
        if not isinstance(stored, dict):
            return False
        job = read_job(self.store, session_id) or JobSought()
        return (
            str(stored.get("job", "")) != job.text
            or bool(stored.get("use_claude")) != job.use_claude
        )

    def _write(self, session_id: str, name: StepName, data: Any) -> str:
        """Store a step's result; return its file name."""
        self.store.write(session_id, STEP_OUTPUTS[name], dumps(data))
        return STEP_OUTPUTS[name]

    def _usage(self) -> Optional[StepUsage]:
        """Return what the judges have used so far (``None``: not tracked).

        Empty before the judges are built.
        """
        if self._judges is None:
            return StepUsage()
        ledger = self._judges.usage
        return ledger.snapshot() if ledger is not None else None

    def _setup(self) -> JudgeSetup:
        """Return the judges, built once per analysis."""
        if self._judges is None:
            self._judges = self.judges()
            ledger = self._judges.usage
            self._base = ledger.snapshot() if ledger is not None else None
        return self._judges

    # ----------------------------------------------------------- steps

    def _sentences(self, session_id: str, c: "_Context") -> StepRecord:
        """Step a: the sentences of each person with their times."""
        people: Dict[str, Dict[str, int]] = {}
        for s in c.sentences:
            entry = people.setdefault(s.speaker, {"sentences": 0, "words": 0})
            entry["sentences"] += 1
            entry["words"] += s.n_words()
        output = self._write(session_id, "sentences", {
            "session_id": session_id,
            "conversation_type": c.transcript.conversation_type,
            "conversation_goal": c.transcript.conversation_goal,
            "speakers": [{"name": k, **v} for k, v in people.items()],
            "sentences": [s.model_dump(exclude_none=True)
                          for s in c.sentences],
        })
        return StepRecord(
            name="sentences", status="done", output=output,
            counts={"sentences": len(c.sentences), "speakers": len(people)},
        )

    def _interaction(self, session_id: str, c: "_Context") -> StepRecord:
        """Step b: turns, words, share of speech, overlaps."""
        team = describe(c.transcript)
        output = self._write(
            session_id, "interaction", team.model_dump(mode="json")
        )
        return StepRecord(
            name="interaction", status="done", output=output,
            counts={"speakers": team.speakers, "turns": team.turns,
                    "words": team.words},
        )

    def _candidates(self, session_id: str, c: "_Context") -> StepRecord:
        """Step c: candidate indicators per sentence by retrieval."""
        rubric = c.rubric(self)
        tagged = self.tagger.tag(c.transcript, rubric)
        output = self._write(session_id, "candidates", {
            "session_id": session_id, "catalogue": rubric.name,
            "catalogue_language": rubric.language,
            "sentences": [t.model_dump(exclude_none=True) for t in tagged],
        })
        return StepRecord(
            name="candidates", status="done", output=output,
            model=self.embedding_model or "bm25",
            counts={
                "sentences": len(tagged),
                "with_candidates": sum(1 for t in tagged if t.tags),
                "candidates": sum(len(t.tags) for t in tagged),
            },
        )

    def _optional(
        self,
        name: StepName,
        session_id: str,
        produce: Callable[[], StepOutput],
    ) -> StepRecord:
        """Run an optional step and store what it produced."""
        made = produce()
        output = self._write(session_id, name, made.data)
        return StepRecord(
            name=name, status="done", output=output, model=made.model,
            counts=dict(made.counts), note=made.note,
        )

    def _acts(self, session_id: str, c: "_Context") -> StepRecord:
        """Step d: dialogue acts, when the tagger exists."""
        tagger = self.acts
        if tagger is None:
            return StepRecord(
                name="acts", status="unavailable", note=UNAVAILABLE["acts"]
            )
        return self._optional(
            "acts", session_id,
            lambda: tagger.tag(c.transcript, c.sentences),
        )

    def _events(self, session_id: str, c: "_Context") -> StepRecord:
        """Step e: derived events, when the extractor exists."""
        extractor = self.events
        if extractor is None:
            return StepRecord(
                name="events", status="unavailable",
                note=UNAVAILABLE["events"],
            )
        acts = self._stored(session_id, "acts")
        if acts is None:
            return StepRecord(
                name="events", status="unavailable",
                note=UNAVAILABLE["events_without_acts"],
            )
        return self._optional(
            "events", session_id,
            lambda: extractor.extract(c.transcript, c.sentences, acts),
        )

    def _stored(self, session_id: str, name: StepName) -> Optional[Any]:
        """Return a step's stored data, or ``None`` when it has none."""
        try:
            return json.loads(self.store.read(session_id, STEP_OUTPUTS[name]))
        except NotFoundError:
            return None

    def _confirmation(self, session_id: str, c: "_Context") -> StepRecord:
        """Step f: the judge confirms or rejects the candidates."""
        stored = self._stored(session_id, "candidates")
        if stored is None:
            raise NotFoundError("The candidates of step c are missing.")
        tagged = [
            TaggedSentence.model_validate(item)
            for item in stored["sentences"]
        ]
        rubric = c.rubric(self)
        setup = self._setup()
        cap = self.max_confirmations
        before = setup.budget.calls if setup.budget else 0
        if setup.budget is not None:
            # Keep enough calls for step g (one per person and
            # competency) and step i (one per person).
            people = len(c.transcript.speakers())
            reserve = people * len(rubric.competencies)
            if setup.synthesis is not None:
                reserve += people
            room = max(setup.budget.remaining - reserve, 0)
            cap = min(cap, room) if cap else room
            if not cap:
                cap = -1
        chosen = (
            set() if cap < 0 else set(most_promising(tagged, cap))
        )
        offered = [
            t if n in chosen else t.model_copy(update={"tags": []})
            for n, t in enumerate(tagged)
        ]
        record: Dict[str, Any] = {}
        if setup.mode == "both" and setup.second is not None:
            decided, comparison = compare_confirmations(
                offered, rubric, setup.first, setup.second, rubric.language
            )
            record["comparison"] = comparison.model_dump()
        else:
            decided = setup.first.confirm(offered, rubric, rubric.language)
        merged = [
            d.model_copy(update={"tags": t.tags})
            for d, t in zip(decided, tagged)
        ]
        reason = str(getattr(setup.first.judge, "fallback_reason", ""))
        models = judge_models(merged)
        judge_info: Dict[str, Any] = {"mode": setup.mode, "models": models}
        if reason:
            judge_info["fallback_reason"] = reason
        if setup.mode == "both":
            judge_info["second_models"] = judge_models(merged, second=True)
        output = self._write(session_id, "confirmation", {
            "session_id": session_id, "catalogue": rubric.name,
            "catalogue_language": rubric.language, "judge": judge_info,
            **record,
            "max_confirmations": self.max_confirmations,
            "read_at_most": max(cap, 0),
            "sentences": [t.model_dump(exclude_none=True) for t in merged],
        })
        counts = {
            "sent": len(chosen),
            "not_sent": sum(1 for t in tagged if t.tags) - len(chosen),
        }
        if setup.budget is not None:
            counts["judge_calls"] = setup.budget.calls - before
        for status in ("confirmed", "rejected", "unparsed"):
            counts[status] = sum(
                1 for t in merged
                if t.decision is not None and t.decision.status == status
            )
        return StepRecord(
            name="confirmation", status="done", output=output,
            model=", ".join(models + judge_models(merged, second=True)),
            judge=setup.mode, fallback_reason=reason, counts=counts,
        )

    def _assessment(self, session_id: str, c: "_Context") -> StepRecord:
        """Step g: evidence per person and scorecards."""
        setup = self._setup()
        rubric = c.rubric(self)
        service = setup.assessment
        before = setup.budget.calls if setup.budget else 0
        stored = self._stored(session_id, "confirmation") or {}
        confirmed = confirmed_observations([
            TaggedSentence.model_validate(item)
            for item in stored.get("sentences", [])
        ])
        try:
            result = service.assess(c.transcript, rubric, confirmed=confirmed)
        finally:
            release(service.judge)
        output = self._write(
            session_id, "assessment", result.model_dump(mode="json")
        )
        self.store.write(
            session_id, SCORECARD_FILE,
            render_scorecards(result, rubric, c.transcript),
        )
        reason = str(getattr(service.judge, "fallback_reason", ""))
        cards = result.scorecards
        calls = (
            {"judge_calls": setup.budget.calls - before}
            if setup.budget is not None else {}
        )
        return StepRecord(
            name="assessment", status="done", output=output,
            model=result.judge_model, judge=setup.mode,
            fallback_reason=reason,
            counts={
                "participants": len(cards),
                "judgements": sum(len(card.assessments) for card in cards),
                "rated": sum(card.tiers()[0] for card in cards),
                "provisional": sum(card.tiers()[1] for card in cards),
                "evidence": sum(
                    len(a.evidence) for card in cards
                    for a in card.assessments
                ),
                "observations": sum(card.tiers()[2] for card in cards),
                **calls,
            },
        )

    def _reports(self, session_id: str, c: "_Context") -> StepRecord:
        """Step h: one HTML page per person and the session summary."""
        stored = self._stored(session_id, "assessment")
        if stored is None:
            raise NotFoundError("The assessment of step g is missing.")
        result = SessionAssessment.model_validate(stored)
        language = result.language if result.language in TEXTS else "es"
        catalogue = self.report_catalogue(result.catalogue, language)
        profiles = (
            self._profiles(session_id) if "synthesis" in c.done else {}
        )
        people: List[Dict[str, Any]] = []
        for position, card in enumerate(result.scorecards, start=1):
            report = build_person_report(
                result, catalogue, card.participant, c.transcript,
                c.synthetic,
            )
            name = report_file(position)
            self.store.write(session_id, name, render_person_html(
                report, profiles.get(card.participant), c.job.text,
            ))
            firm, provisional, observed = card.tiers()
            people.append({
                "position": position, "name": card.participant,
                "report": name,
                "rated": firm,
                "provisional": provisional,
                "observations": observed,
                "why": report.why,
                "words": card.interaction.words,
                "competencies": len(report.rows),
                "strengths": [p.name for p in report.strengths],
                "to_improve": [p.name for p in report.weaknesses],
                "signals": sum(len(p.quotes) for p in report.signals),
                "ratings": {
                    a.competency_id: a.rating for a in card.assessments
                },
                "estimates": {
                    a.competency_id: a.estimate.model_dump()
                    for a in card.assessments if a.estimate is not None
                },
            })
        output = self._write(session_id, "reports", {
            "session_id": session_id, "language": language,
            "synthetic": c.synthetic, "judge_model": result.judge_model,
            "scorecard": SCORECARD_FILE, "people": people,
            "min_incidents": result.min_incidents,
            "open_issues": [i.model_dump() for i in c.open_issues],
        })
        return StepRecord(
            name="reports", status="done", output=output,
            counts={"reports": len(people)},
        )

    def _profiles(self, session_id: str) -> Dict[str, PersonSynthesis]:
        """Return the stored profiles of step i by person."""
        stored = self._stored(session_id, "synthesis")
        people = stored.get("people", []) if isinstance(stored, dict) else []
        found: Dict[str, PersonSynthesis] = {}
        for item in people:
            profile = PersonSynthesis.model_validate(item)
            found[profile.participant] = profile
        return found

    def _synthesis(self, session_id: str, c: "_Context") -> StepRecord:
        """Step i: the explanatory profile of each person (#113).

        One judge call per person with something quoted, counted in
        the run's budget; a person the model fails on is skipped with
        the reason (a budget refusal fails the step, to resume later).
        The reports of step h are written again with the profile on
        top.
        """
        stored = self._stored(session_id, "assessment")
        if stored is None:
            raise NotFoundError("The assessment of step g is missing.")
        setup = self._setup()
        if setup.synthesis is None:
            return StepRecord(
                name="synthesis", status="unavailable",
                note=UNAVAILABLE["synthesis"],
            )
        job = c.job
        writer = setup.synthesis(job.use_claude)
        result = SessionAssessment.model_validate(stored)
        language = result.language if result.language in TEXTS else "es"
        catalogue = self.report_catalogue(result.catalogue, language)
        acts = self._stored(session_id, "acts")
        events = self._stored(session_id, "events")
        before = setup.budget.calls if setup.budget else 0
        profiles: List[PersonSynthesis] = []
        try:
            for card in result.scorecards:
                facts = person_facts(
                    result, catalogue, card.participant, acts, events
                )
                try:
                    profiles.append(writer.write(facts, job.text))
                except LimitError:
                    raise
                except ModelError as exc:
                    profiles.append(skipped(
                        facts, short_reason(exc), writer.judge.model_name
                    ))
        finally:
            release(writer.judge)
        reason = str(getattr(writer.judge, "fallback_reason", ""))
        models = sorted({p.model for p in profiles if p.model})
        note = (
            "The run's judge is the local model and Claude was not "
            "allowed for the synthesis: the local model wrote it."
            if setup.mode == "local" and not job.use_claude else ""
        )
        output = self._write(session_id, "synthesis", {
            "session_id": session_id, "language": language,
            "synthetic": c.synthetic, "job": job.text,
            "use_claude": job.use_claude,
            "judge": {"mode": setup.mode, "models": models,
                      "fallback_reason": reason},
            "people": [p.model_dump(mode="json") for p in profiles],
            "open_issues": [i.model_dump() for i in c.open_issues],
        })
        if self._stored(session_id, "reports") is not None:
            by_name = {p.participant: p for p in profiles}
            for position, card in enumerate(result.scorecards, start=1):
                report = build_person_report(
                    result, catalogue, card.participant, c.transcript,
                    c.synthetic,
                )
                self.store.write(
                    session_id, report_file(position),
                    render_person_html(
                        report, by_name.get(card.participant), job.text
                    ),
                )
        counts = {
            "people": sum(1 for p in profiles if p.status == "done"),
            "skipped": sum(1 for p in profiles if p.status == "skipped"),
            "points": sum(len(p.strengths) + len(p.fit) for p in profiles),
            "dropped": sum(sum(p.dropped.values()) for p in profiles),
        }
        if setup.budget is not None:
            counts["judge_calls"] = setup.budget.calls - before
        return StepRecord(
            name="synthesis", status="done", output=output,
            model=", ".join(models), judge=setup.mode,
            fallback_reason=reason, counts=counts, note=note,
        )


class _Context:
    """What the steps of one run share: the transcript and its sentences.

    ``done`` are the steps done when the run started (kept from an
    earlier run); ``job`` is the job sought stored with the analysis.
    """

    def __init__(self, transcript: Transcript, run: AnalysisRun):
        """Split the transcript once; the rubric is loaded on demand."""
        self.transcript = transcript
        self.synthetic = run.synthetic
        self.open_issues = run.open_issues
        self.done = {s.name for s in run.steps if s.status == "done"}
        self.job = JobSought()
        self.sentences: List[Sentence] = transcript_sentences(transcript)
        self._rubric: Optional[RubricCatalogue] = None

    def rubric(self, analysis: SessionAnalysis) -> RubricCatalogue:
        """Return the rubric for the transcript's language."""
        if self._rubric is None:
            language = analysis.tagger.retriever.language_of(self.transcript)
            self._rubric = analysis.rubric(language)
        return self._rubric


def step_table(run: AnalysisRun) -> List[Tuple[str, str, str, str, str]]:
    """Return one printable row per step: name, status, time, who, counts."""
    rows: List[Tuple[str, str, str, str, str]] = []
    for s in run.steps:
        who = s.model + (f" ({s.judge})" if s.judge else "")
        counts = ", ".join(f"{k} {v}" for k, v in s.counts.items())
        detail = s.error or "; ".join(p for p in (counts, s.note) if p)
        if s.fallback_reason:
            detail += f"; fell back to local: {s.fallback_reason}"
        rows.append((
            s.name, s.status,
            "" if s.seconds is None else f"{s.seconds:.1f}s",
            who, detail,
        ))
    return rows
