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
g. ``assessment``: evidence per person and scorecards;
h. ``reports``: one HTML page per person and the session summary.

Every step's result is stored under the session
(:class:`~src.application.skills.ports.AnalysisStore`) together with
its record: status, duration, who ran it and counts
(:mod:`src.domain.skills.analysis_run`). The record is saved after
every step, so a failure keeps the earlier results and the next run
resumes at the step that failed.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.application.skills.assessment import AssessmentService
from src.application.skills.judges import (
    FallbackJudge,
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
    StepOutput,
    TranscriptRepository,
    TranscriptStore,
)
from src.application.skills.report import render_scorecards
from src.application.skills.tagging import (
    SentenceTagger,
    TagConfirmer,
    TaggedSentence,
)
from src.domain.errors import (
    BusyError,
    InvalidInputError,
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
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.interaction import describe
from src.domain.skills.judging import JudgeMode
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.speech import Sentence, transcript_sentences
from src.domain.skills.transcript import Transcript
from src.domain.skills.transcript_contract import is_session_id

RUN_FILE = "run.json"
"""The run's record, next to the step outputs."""
SCORECARD_FILE = "scorecard.md"
"""The technical scorecard of the whole session (step h)."""

UNAVAILABLE = {
    "acts": "Not available: no dialogue-act tagger is configured (its "
            "settings card is missing).",
    "events": "Not available yet: decisions, action items and open "
              "questions (#83) are not connected.",
    "events_without_acts": "Not available: the events are derived from "
                           "the dialogue acts of step d, which did not "
                           "run.",
}
"""Why a step behind a missing module is skipped."""


@dataclass(frozen=True)
class JudgeSetup:
    """The judges of steps f and g, built only when they are needed.

    ``second`` is the local judge's confirmer in ``both`` mode.
    """

    mode: JudgeMode
    first: TagConfirmer
    assessment: AssessmentService
    second: Optional[TagConfirmer] = None


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


def check_session(session_id: str) -> None:
    """Refuse a session id the contract does not allow."""
    if not is_session_id(session_id):
        raise InvalidInputError(f"Invalid session id {session_id!r}.")


class AnalysisResults:
    """Read what an analysis stored: its run, step results, reports.

    It needs only the store, so serving the results never builds a
    model.
    """

    def __init__(self, store: AnalysisStore):
        """Wire the store."""
        self.store = store

    def status(self, session_id: str) -> AnalysisRun:
        """Return the stored run of ``session_id``."""
        check_session(session_id)
        run = self.store.load(session_id)
        if run is None:
            raise NotFoundError(f"No analysis of session {session_id!r}.")
        return run

    def result(self, session_id: str, step: str) -> Any:
        """Return the stored result of one step, as JSON data."""
        check_session(session_id)
        if step not in STEP_OUTPUTS:
            raise InvalidInputError(f"Unknown step {step!r}.")
        name = STEP_OUTPUTS[step]
        return json.loads(self.store.read(session_id, name))

    def report(self, session_id: str, position: int) -> str:
        """Return the HTML report of the ``position``-th person (1-based)."""
        check_session(session_id)
        if position < 1:
            raise InvalidInputError("Reports are numbered from 1.")
        return self.store.read(session_id, report_file(position))


class AnalysisRequests:
    """Start an analysis in the background and report on it (the API).

    The launcher runs ``skills analyze`` apart from the server, so a
    long analysis neither blocks the API nor keeps its models loaded
    in it; the run's record is what both share.
    """

    def __init__(
        self,
        results: AnalysisResults,
        transcripts: TranscriptStore,
        launcher: AnalysisLauncher,
        lock: AnalysisLock,
    ):
        """Wire the stored results, the transcripts, launcher and lock."""
        self.results = results
        self.transcripts = transcripts
        self.launcher = launcher
        self.lock = lock

    def _active(self, session_id: str) -> bool:
        """Tell whether any process (API child or CLI) analyses it now."""
        return (
            self.launcher.running(session_id)
            or self.lock.holder(session_id) is not None
        )

    def start(
        self, session_id: str, restart: bool = False, synthetic: bool = False
    ) -> Dict[str, Any]:
        """Start the analysis of ``session_id``; return its status.

        Raises :class:`~src.domain.errors.NotFoundError` when the
        session has no transcript and
        :class:`~src.domain.errors.BusyError` when its analysis is
        already running.
        """
        check_session(session_id)
        self.transcripts.get(session_id)
        if self._active(session_id):
            raise BusyError(
                f"The analysis of {session_id!r} is already running."
            )
        self.launcher.start(session_id, restart, synthetic)
        return {"session_id": session_id, "active": True}

    def status(self, session_id: str) -> Dict[str, Any]:
        """Return the stored run with ``active``: is it running now.

        A run left ``running`` by a process that died reads
        ``interrupted``; starting again resumes it.
        """
        run = self.results.status(session_id)
        active = self._active(session_id)
        return {
            **run.model_dump(mode="json"),
            "status": shown_status(run, active),
            "active": active,
        }


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
        self._judges: Optional[JudgeSetup] = None

    def run(
        self,
        session_id: str,
        restart: bool = False,
        synthetic: bool = False,
        on_step: Optional[Callable[[StepRecord], None]] = None,
        only: Optional[Sequence[str]] = None,
    ) -> AnalysisRun:
        """Run the steps still to do; return the run as stored.

        A failing step is recorded as ``failed`` with its reason and
        stops the run; the earlier results stay, and a later call
        starts again at that step. ``restart`` runs every step anew.
        ``on_step`` is told each record once its step ends. ``only``
        runs exactly those steps again (in running order) and keeps
        the others as they are, e.g. the acts and events once a tagger
        exists, without asking the judge again. Raises
        :class:`~src.domain.errors.BusyError` while another process
        analyses the same session.
        """
        check_session(session_id)
        chosen = None if only is None else chosen_steps(only)
        transcript = self.transcripts.find(session_id)
        worker = self.lock.acquire(session_id)
        if worker is None:
            holder = self.lock.holder(session_id)
            raise BusyError(
                f"The analysis of {session_id!r} is already running"
                + (f" (process {holder})." if holder else ".")
            )
        try:
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
        previous = None if restart else self.store.load(session_id)
        todo = steps_to_run(previous, restart) if chosen is None else chosen
        run = previous if previous is not None else new_run(
            session_id, self.clock(), synthetic
        )
        run = run.model_copy(update={
            "status": "running", "finished_at": None,
            "synthetic": synthetic or run.synthetic, "worker_pid": worker,
        })
        for name in todo:
            run = run.with_step(StepRecord(name=name))
        self.store.save(run)
        context = _Context(transcript, run)
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
        run = run.model_copy(update={
            "status": "done", "finished_at": self.clock(),
        })
        self.store.save(run)
        return run

    def _step(
        self, name: StepName, session_id: str, context: "_Context",
        started: datetime,
    ) -> StepRecord:
        """Run one step and return its record (never raises)."""
        begin = time.perf_counter()
        try:
            step: Callable[[str, _Context], StepRecord] = getattr(
                self, f"_{name}"
            )
            record = step(session_id, context)
        except Exception as exc:  # any failure is recorded, not lost
            record = StepRecord(
                name=name, status="failed", error=short_reason(exc),
            )
        return record.model_copy(update={
            "started_at": started,
            "seconds": round(time.perf_counter() - begin, 3),
        })

    def _write(self, session_id: str, name: StepName, data: Any) -> str:
        """Store a step's result; return its file name."""
        self.store.write(session_id, STEP_OUTPUTS[name], dumps(data))
        return STEP_OUTPUTS[name]

    def _setup(self) -> JudgeSetup:
        """Return the judges, built once per analysis."""
        if self._judges is None:
            self._judges = self.judges()
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
        chosen = set(most_promising(tagged, self.max_confirmations))
        offered = [
            t if n in chosen else t.model_copy(update={"tags": []})
            for n, t in enumerate(tagged)
        ]
        setup = self._setup()
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
        judge = setup.first.judge
        reason = (
            judge.fallback_reason if isinstance(judge, FallbackJudge) else ""
        )
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
            "sentences": [t.model_dump(exclude_none=True) for t in merged],
        })
        counts = {
            "sent": len(chosen),
            "not_sent": sum(1 for t in tagged if t.tags) - len(chosen),
        }
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
        try:
            result = service.assess(c.transcript, rubric)
        finally:
            release(service.judge)
        output = self._write(
            session_id, "assessment", result.model_dump(mode="json")
        )
        self.store.write(
            session_id, SCORECARD_FILE,
            render_scorecards(result, rubric, c.transcript),
        )
        judge = service.judge
        reason = (
            judge.fallback_reason if isinstance(judge, FallbackJudge) else ""
        )
        cards = result.scorecards
        return StepRecord(
            name="assessment", status="done", output=output,
            model=result.judge_model, judge=setup.mode,
            fallback_reason=reason,
            counts={
                "participants": len(cards),
                "judgements": sum(len(card.assessments) for card in cards),
                "rated": sum(
                    1 for card in cards for a in card.assessments
                    if a.rating is not None
                ),
                "evidence": sum(
                    len(a.evidence) for card in cards
                    for a in card.assessments
                ),
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
        people: List[Dict[str, Any]] = []
        for position, card in enumerate(result.scorecards, start=1):
            report = build_person_report(
                result, catalogue, card.participant, c.transcript,
                c.synthetic,
            )
            name = report_file(position)
            self.store.write(session_id, name, render_person_html(report))
            people.append({
                "position": position, "name": card.participant,
                "report": name,
                "rated": len(report.rated),
                "competencies": len(report.rows),
                "strengths": [p.name for p in report.strengths],
                "to_improve": [p.name for p in report.weaknesses],
                "signals": sum(len(p.quotes) for p in report.signals),
                "ratings": {
                    a.competency_id: a.rating for a in card.assessments
                },
            })
        output = self._write(session_id, "reports", {
            "session_id": session_id, "language": language,
            "synthetic": c.synthetic, "judge_model": result.judge_model,
            "scorecard": SCORECARD_FILE, "people": people,
            "open_issues": [i.model_dump() for i in c.open_issues],
        })
        return StepRecord(
            name="reports", status="done", output=output,
            counts={"reports": len(people)},
        )


class _Context:
    """What the steps of one run share: the transcript and its sentences."""

    def __init__(self, transcript: Transcript, run: AnalysisRun):
        """Split the transcript once; the rubric is loaded on demand."""
        self.transcript = transcript
        self.synthetic = run.synthetic
        self.open_issues = run.open_issues
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
        detail = s.error or s.note or counts
        if s.fallback_reason:
            detail += f"; fell back to local: {s.fallback_reason}"
        rows.append((
            s.name, s.status,
            "" if s.seconds is None else f"{s.seconds:.1f}s",
            who, detail,
        ))
    return rows
