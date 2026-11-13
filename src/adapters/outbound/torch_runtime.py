"""Apply a runtime plan to this process and to PyTorch."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.application.runtime_tuning import RuntimeTuning
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


HALF_PRECISION_DEVICES = ("cuda",)
"""Devices where models run in float16.

Measured on the RTX 3070 Ti with bge-base (SciFact, 300 queries): 3x
faster than float32, document vectors within cosine 0.9995, the same
top-1 for every query and 99.7% of the top-10 shared. CPUs keep
float32 (float16 is slow there).
"""


def model_kwargs(device: str) -> Dict[str, Any]:
    """Keyword arguments for loading a transformer on ``device``."""
    if device not in HALF_PRECISION_DEVICES:
        return {}
    import torch

    return {"torch_dtype": torch.float16}


def precision_tag(device: str) -> str:
    """Suffix that keeps float16 and float32 vectors in separate caches."""
    return "|fp16" if device in HALF_PRECISION_DEVICES else ""


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


def tuning_record(
    profile: HardwareProfile,
    tuning: RuntimeTuning,
    torch_version: str = "",
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Describe one benchmark run as a JSON-ready record."""
    return {
        "created": (now or datetime.now(timezone.utc))
        .replace(microsecond=0).isoformat(),
        "signature": tuning.signature,
        "gpus": [g.model_dump() for g in profile.gpus],
        "driver": ", ".join(
            f"{c.vendor}: {c.driver or 'none'}" for c in profile.cards
        ),
        "torch": torch_version,
        "results": [r.model_dump() for r in tuning.results],
    }


def append_tuning(
    path: str, profile: HardwareProfile, tuning: RuntimeTuning
) -> None:
    """Append a benchmark run to the JSON Lines history at ``path``.

    The history is versioned with the experiments, so that the results
    site can show how the hardware and its speed changed over time.
    """
    try:
        import torch

        version = str(torch.__version__)
    except ImportError:  # pragma: no cover - torch is a dependency
        version = ""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(tuning_record(profile, tuning, version)))
        handle.write("\n")
