"""A prepared analysis result for the live page's scripted demo (#232).

The live page's demo interview (``examples/demo/*.demo.json``) ends on
the analysis of its fictitious candidate. That result is not computed:
it is written by hand in an ``analysis.demo.v1`` file and installed in
the analysis store under a ``demo-`` session id, so the usual result
pages show it. ``demo`` is always true, the stored run is marked
``synthetic`` and every step but the synthesis reads ``unavailable``
with :data:`DEMO_NOTE`, so it can never pass for a real analysis.
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
from src.domain.skills.synthesis import PersonSynthesis

DEMO_NOTE = "Demo data: prepared by hand, not computed."
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
        for person in people:
            PersonSynthesis.model_validate(person)
        return self

    def people(self) -> List[PersonSynthesis]:
        """Return the synthesis of each person."""
        return [PersonSynthesis.model_validate(p)
                for p in self.synthesis["people"]]

    def run(self) -> AnalysisRun:
        """Return the stored run: only the synthesis is done."""
        steps = [
            StepRecord(
                name="synthesis", status="done", seconds=0.0,
                model="demo", judge="demo", output=STEP_OUTPUTS["synthesis"],
                counts={"people": len(self.people())}, note=DEMO_NOTE,
            ) if name == "synthesis" else
            StepRecord(name=name, status="unavailable", note=DEMO_NOTE)
            for name in STEPS
        ]
        return AnalysisRun(
            session_id=self.session_id, status="done",
            started_at=self.started_at, finished_at=self.started_at,
            synthetic=True, steps=steps,
        )
