"""Apply a runtime plan to this process and to PyTorch."""

import json
import os
from pathlib import Path
from typing import Optional

from src.domain.runtime import HardwareProfile, RuntimePlan

_applied: Optional[RuntimePlan] = None


def apply_plan(plan: RuntimePlan) -> RuntimePlan:
    """Pin the process and size PyTorch's thread pool.

    Called by every model adapter just before loading its model, so
    commands that load no model never touch the scheduler. The first
    plan applied in a process stays in force: thread pools cannot be
    resized reliably once models are running. Pinning is skipped
    where the platform does not support it or refuses it.
    """
    global _applied
    if _applied is not None:
        return _applied
    if plan.cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(plan.cpus))
        except (OSError, ValueError):
            pass
    import torch

    torch.set_num_threads(plan.threads)
    _applied = plan
    return plan


def prepare(plan: Optional[RuntimePlan]) -> str:
    """Apply ``plan`` (or the historical default) and return the device.

    Without a plan the process keeps the old behaviour: one thread
    per logical CPU, on the CPU.
    """
    if plan is None:
        plan = RuntimePlan(threads=os.cpu_count() or 1)
    return apply_plan(plan).device


def applied_plan() -> Optional[RuntimePlan]:
    """Return the plan in force in this process, if any."""
    return _applied


def reset_plan() -> None:
    """Forget the applied plan (for tests)."""
    global _applied
    _applied = None


def load_measured(
    path: str, profile: HardwareProfile
) -> Optional[RuntimePlan]:
    """Return the plan measured for this hardware, if one was saved."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("signature") != profile.signature():
            return None
        plan = RuntimePlan.model_validate(data.get("plan"))
    except (OSError, ValueError, AttributeError):
        return None
    allowed = set(profile.logical_cpus())
    return plan if set(plan.cpus) <= allowed else None


def save_measured(
    path: str, profile: HardwareProfile, plan: RuntimePlan
) -> None:
    """Remember the fastest plan measured on this hardware."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "signature": profile.signature(),
        "plan": plan.model_dump(),
    }, indent=2), encoding="utf-8")
