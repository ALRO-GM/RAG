"""Use case: find the fastest runtime plan by measuring on this machine."""

from typing import List, Optional, Protocol

from pydantic import BaseModel

from src.domain.errors import InvalidInputError
from src.domain.runtime import (
    HardwareProfile,
    RuntimePlan,
    candidate_plans,
)


class PlanMeasurement(BaseModel):
    """Speeds obtained under one plan."""

    embed_seconds: float
    tokens_per_second: float


class HardwareProbe(Protocol):
    """Describe the machine."""

    def profile(self) -> HardwareProfile:
        """Return the hardware profile of this process."""
        ...


class PlanBenchmark(Protocol):
    """Run a fixed workload under a plan."""

    def measure(self, plan: RuntimePlan) -> PlanMeasurement:
        """Return the speeds obtained under ``plan``."""
        ...


class PlanResult(BaseModel):
    """A measured plan; ``cost`` is seconds for the whole workload."""

    plan: RuntimePlan
    embed_seconds: float
    tokens_per_second: float
    cost: float


class RuntimeTuning(BaseModel):
    """Every measured plan, fastest first, and the hardware measured."""

    signature: str
    results: List[PlanResult]

    def best(self) -> RuntimePlan:
        """Return the fastest plan."""
        return self.results[0].plan

    def speedup(self, name: str) -> Optional[float]:
        """Return how many times faster the best plan is than ``name``."""
        for result in self.results:
            if result.plan.name == name and self.results[0].cost > 0:
                return result.cost / self.results[0].cost
        return None


class RuntimeTuner:
    """Measure the candidate CPU plans and rank them.

    Each plan is measured ``repeats`` times and its best run kept,
    because the first run of a process pays for cold caches. The cost
    of a plan is the time to embed the texts plus the time to generate
    ``tokens`` tokens, so both stages of the pipeline count.
    """

    def __init__(
        self,
        probe: HardwareProbe,
        benchmark: PlanBenchmark,
        repeats: int = 2,
        tokens: int = 48,
    ):
        """Wire the probe and the benchmark."""
        if repeats < 1:
            raise InvalidInputError("repeats must be at least 1.")
        self.probe = probe
        self.benchmark = benchmark
        self.repeats = repeats
        self.tokens = tokens

    def tune(self) -> RuntimeTuning:
        """Measure every candidate plan of this machine."""
        profile = self.probe.profile()
        results: List[PlanResult] = []
        for plan in candidate_plans(profile):
            runs = [self.benchmark.measure(plan) for _ in range(self.repeats)]
            embed = min(r.embed_seconds for r in runs)
            speed = max(r.tokens_per_second for r in runs)
            results.append(PlanResult(
                plan=plan,
                embed_seconds=round(embed, 3),
                tokens_per_second=round(speed, 2),
                cost=round(embed + self.tokens / max(speed, 1e-9), 3),
            ))
        results.sort(key=lambda r: r.cost)
        return RuntimeTuning(signature=profile.signature(), results=results)
