"""How far a lab run got: its steps counted by what is left to do.

The "Pruebas" tab's overview (``docs/transcriber/LAB.md`` §7) shows,
for each session, a bar of its newest run's steps: done, running,
pending and failed (failed split by error code). A step reported
``running`` by a run that is no longer active (interrupted, cancelled)
is not running any more: it counts as pending, like a ``cancelled``
step, since a resume would try both again.
"""

from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from transcriber.domain.lab import Step

UNKNOWN_ERROR = "unknown"
"""Error code counted for a failed step that kept no error."""


class OpenStep(BaseModel):
    """A step that is not done: what it is, where it stands and why."""

    model_config = ConfigDict(frozen=True)

    name: str
    stage: str
    model: str
    denoiser: str
    state: str
    code: Optional[str] = None
    message: Optional[str] = None


class StepTally(BaseModel):
    """A run's steps counted by progress state.

    ``failed_codes`` maps each error code to its failed steps; ``open``
    lists every step that is not done, in the run's order.
    """

    model_config = ConfigDict(frozen=True)

    done: int = 0
    running: int = 0
    pending: int = 0
    failed: int = 0
    failed_codes: Dict[str, int] = {}
    open: List[OpenStep] = []

    @property
    def total(self) -> int:
        """Return the number of steps counted."""
        return self.done + self.running + self.pending + self.failed


def progress_state(step: Step, active: bool) -> str:
    """Return ``done``, ``running``, ``pending`` or ``failed`` for a step.

    ``active`` tells whether the run is queued or running now: only
    then is a ``running`` step really running.
    """
    if step.state == "running":
        return "running" if active else "pending"
    if step.state == "cancelled":
        return "pending"
    return step.state


def tally_steps(steps: Sequence[Step], active: bool) -> StepTally:
    """Count ``steps`` by :func:`progress_state` and list the open ones."""
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    codes: Dict[str, int] = {}
    open_steps: List[OpenStep] = []
    for step in steps:
        state = progress_state(step, active)
        counts[state] += 1
        if state == "done":
            continue
        code = step.error.code if step.error is not None else None
        if state == "failed":
            key = code or UNKNOWN_ERROR
            codes[key] = codes.get(key, 0) + 1
        open_steps.append(OpenStep(
            name=step.name, stage=step.stage, model=step.model,
            denoiser=step.denoiser, state=state, code=code,
            message=step.error.message if step.error is not None else None,
        ))
    return StepTally(
        done=counts["done"], running=counts["running"],
        pending=counts["pending"], failed=counts["failed"],
        failed_codes=dict(sorted(codes.items())), open=open_steps,
    )
