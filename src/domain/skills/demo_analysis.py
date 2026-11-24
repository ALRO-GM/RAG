"""A prepared analysis result for the live page's scripted demo (#232).

The live page's demo interview (``examples/demo/*.demo.json``) ends on
the analysis of its fictitious candidate. That result is not computed:
it is written by hand in an ``analysis.demo.v1`` file and installed in
the analysis store under a ``demo-`` session id, so the usual result
pages show it. Besides the synthesis it carries the few step results
the result page needs (``files``: the sentences, for the transcript and
the length, and an assessment with one scorecard per summarised person
and no levels, so only the candidate is a person of it). ``demo`` is
always true, the stored run is marked ``synthetic`` and every step
carries :data:`DEMO_NOTE` (the ones not given read ``unavailable``), so
it can never pass for a real analysis.
Plain data and rules: no I/O.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.skills.analysis_run import (
    STEP_OUTPUTS,
    STEPS,
    AnalysisRun,
    StepRecord,
)
from src.domain.skills.assessment import SessionAssessment
from src.domain.skills.synthesis import PersonSynthesis

DEMO_NOTE = "Demo data: prepared by hand, not computed."
EXTRA_STEPS = ("sentences", "assessment")
"""Step results a demo may carry besides the synthesis."""
"""The note of every step a demo result did not run."""


class DemoAnalysis(BaseModel):
    """An ``analysis.demo.v1`` file: the stored result of a demo session.

    ``started_at`` is the run's time and its version: a file with a new
    time replaces the installed result.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["analysis.demo.v1"] = Field(alias="schema")
    demo: Literal[True]
    session_id: str = Field(pattern=r"^demo-[A-Za-z0-9._-]{1,59}$")
    note: str = Field(min_length=1)
    started_at: datetime
    title: str = ""
    job: str = ""
    files: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    synthesis: Dict[str, Any]

    @model_validator(mode="after")
    def _a_marked_synthesis(self) -> "DemoAnalysis":
        """The synthesis is marked demo, names this session and parses."""
        if self.synthesis.get("demo") is not True:
            raise ValueError("the synthesis must say demo: true")
        if self.synthesis.get("session_id") != self.session_id:
            raise ValueError("the synthesis names another session")
        people = self.synthesis.get("people")
        if not isinstance(people, list) or not people:
            raise ValueError("the synthesis has no people")
        named = {PersonSynthesis.model_validate(p).participant
                 for p in people}
        unknown = set(self.files) - set(EXTRA_STEPS)
        if unknown:
            raise ValueError(f"unexpected step results {sorted(unknown)}")
        for name, content in self.files.items():
            if content.get("demo") is not True:
                raise ValueError(f"the {name} result must say demo: true")
        if "assessment" in self.files:
            stored = SessionAssessment.model_validate(
                self.files["assessment"])
            cards = {c.participant for c in stored.scorecards}
            if cards != named or stored.session_id != self.session_id:
                raise ValueError("the assessment must summarise the same "
                                 "people of this session")
        return self

    def people(self) -> List[PersonSynthesis]:
        """Return the synthesis of each person."""
        return [PersonSynthesis.model_validate(p)
                for p in self.synthesis["people"]]

    def run(self) -> AnalysisRun:
        """Return the stored run: the steps given are done, others not."""
        given = set(self.files) | {"synthesis"}
        steps = [
            StepRecord(
                name=name, status="done", seconds=0.0, model="demo",
                judge="demo" if name == "synthesis" else "",
                output=STEP_OUTPUTS[name], note=DEMO_NOTE,
                counts={"people": len(self.people())}
                if name == "synthesis" else {},
            ) if name in given else
            StepRecord(name=name, status="unavailable", note=DEMO_NOTE)
            for name in STEPS
        ]
        return AnalysisRun(
            session_id=self.session_id, status="done",
            started_at=self.started_at, finished_at=self.started_at,
            synthetic=True, steps=steps,
        )
