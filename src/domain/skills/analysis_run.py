"""The record of one whole analysis of a session, step by step.

``skills analyze`` runs the steps of the soft-skills flow in a fixed
order on one ``transcript.v1`` document (``docs/skills/DESIGN.md``
§12): sentences per person, interaction metrics, candidate rubric
indicators found by retrieval, dialogue acts, derived events,
confirmation by the judge, evidence and scorecards, reports and the
explanatory profile of each person (#113). Each step
leaves a :class:`StepRecord` (status, time, who ran it, counts) in an
:class:`AnalysisRun`, which is stored after every step, so a failure
keeps what was done before it and a later run resumes from the first
step that is not done. Plain data and rules: no I/O, no model.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, get_args

from pydantic import BaseModel, Field, model_validator

StepName = Literal[
    "sentences", "interaction", "candidates", "acts", "events",
    "confirmation", "assessment", "reports", "synthesis",
]
"""The steps, in the order they run."""

STEPS: Tuple[StepName, ...] = get_args(StepName)
"""Every step name, in running order."""

StepStatus = Literal[
    "pending", "running", "done", "failed", "unavailable",
]
"""``unavailable``: the module behind the step does not exist yet; the
step is skipped and the run goes on."""

RunStatus = Literal["running", "done", "failed", "partial"]
"""``partial``: the steps asked for ran, earlier ones are still to do."""

FINISHED: Tuple[StepStatus, ...] = ("done", "unavailable")
"""Statuses a resumed run does not repeat."""

STEP_OUTPUTS: Dict[StepName, str] = {
    "sentences": "sentences.json",
    "interaction": "interaction.json",
    "candidates": "candidates.json",
    "acts": "acts.json",
    "events": "events.json",
    "confirmation": "confirmation.json",
    "assessment": "assessment.json",
    "reports": "summary.json",
    "synthesis": "synthesis.json",
}
"""The file each step writes in the session's analysis directory."""

OPEN_ISSUES: Tuple[Tuple[str, str], ...] = (
    ("consent", "Consent, retention and who sees the results must be "
                "decided before analysing real people (#19)."),
    ("human_review", "The labels and levels are a model's draft: a "
                     "trained person must review them before any use."),
    ("situations", "One conversation is one situation; a profile needs "
                   "several situations per person (#84)."),
)
"""Steps the flow does not do yet and every result must say so of."""


class StepRecord(BaseModel):
    """What one step did.

    ``model`` names who ran it (the embedding model, the judge, the
    dialogue-act tagger); ``judge`` is the judge mode of a step that
    asked one and ``fallback_reason`` why the local judge took over.
    ``counts`` are what the step produced (sentences, candidates,
    confirmed...). ``note`` explains a skipped step; ``error`` is the
    short reason of a failure.
    """

    name: StepName
    status: StepStatus = "pending"
    started_at: Optional[datetime] = None
    seconds: Optional[float] = None
    model: str = ""
    judge: str = ""
    fallback_reason: str = ""
    counts: Dict[str, int] = Field(default_factory=dict)
    note: str = ""
    error: str = ""
    output: str = ""


class OpenIssue(BaseModel):
    """A step the flow leaves to people, stated with every result."""

    id: str
    text: str


class AnalysisRun(BaseModel):
    """Every step of the analysis of one session, in order."""

    session_id: str
    status: RunStatus = "running"
    started_at: datetime
    finished_at: Optional[datetime] = None
    synthetic: bool = False
    worker_pid: Optional[int] = None
    steps: List[StepRecord]
    open_issues: List[OpenIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def every_step(self) -> "AnalysisRun":
        """Give a run stored before a step existed that step, pending.

        A run recorded before step i (``synthesis``) was added reads
        with it pending, so resuming it runs only that step.
        """
        known = {s.name: s for s in self.steps}
        if len(known) < len(STEPS) and set(known) <= set(STEPS):
            self.steps = [
                known.get(name) or StepRecord(name=name) for name in STEPS
            ]
        return self

    def step(self, name: str) -> StepRecord:
        """Return the record of step ``name``."""
        for record in self.steps:
            if record.name == name:
                return record
        raise KeyError(name)

    def with_step(self, record: StepRecord) -> "AnalysisRun":
        """Return a copy with ``record`` in place of its step's record."""
        return self.model_copy(update={"steps": [
            record if s.name == record.name else s for s in self.steps
        ]})


def new_run(
    session_id: str, now: datetime, synthetic: bool = False
) -> AnalysisRun:
    """Return a run with every step pending."""
    return AnalysisRun(
        session_id=session_id, started_at=now, synthetic=synthetic,
        steps=[StepRecord(name=name) for name in STEPS],
        open_issues=[OpenIssue(id=i, text=t) for i, t in OPEN_ISSUES],
    )


def steps_to_run(
    run: Optional[AnalysisRun], restart: bool = False
) -> List[StepName]:
    """Return the steps a run must still do, in order.

    Everything from the first step that is not finished on (a step
    after a failed one depends on it); all of them for a new run or a
    restart.
    """
    if run is None or restart:
        return list(STEPS)
    for position, name in enumerate(STEPS):
        if run.step(name).status not in FINISHED:
            return list(STEPS[position:])
    return []


def shown_status(run: AnalysisRun, alive: bool) -> str:
    """Return the run's status as people should read it.

    A run recorded as ``running`` whose worker no longer holds the
    session (it was killed, the machine stopped) is ``interrupted``:
    the next run resumes it at the step that was running.
    """
    if run.status == "running" and not alive:
        return "interrupted"
    return run.status


def is_step(name: str) -> bool:
    """Tell whether ``name`` is one of the steps."""
    return name in STEPS
