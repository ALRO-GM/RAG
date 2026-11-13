"""Plan benchmark adapter: one subprocess per measured plan."""

import json
import subprocess
import sys

from src.application.runtime_tuning import PlanMeasurement
from src.domain.errors import ModelError
from src.domain.runtime import RuntimePlan


class SubprocessBenchmark:
    """Measure a plan with the real models in a fresh interpreter."""

    def __init__(
        self,
        embedding_model: str,
        llm: str,
        texts: int = 256,
        tokens: int = 48,
        timeout: int = 600,
    ):
        """Configure the workload of each measurement."""
        self.embedding_model = embedding_model
        self.llm = llm
        self.texts = texts
        self.tokens = tokens
        self.timeout = timeout

    def measure(self, plan: RuntimePlan) -> PlanMeasurement:
        """Return the speeds obtained under ``plan``."""
        request = json.dumps({
            "plan": plan.model_dump(),
            "embedding_model": self.embedding_model,
            "llm": self.llm,
            "texts": self.texts,
            "tokens": self.tokens,
        })
        try:
            done = subprocess.run(
                [sys.executable, "-m",
                 "src.adapters.outbound.benchmark_worker", request],
                capture_output=True, text=True, timeout=self.timeout,
            )
            last = done.stdout.strip().splitlines()[-1]
            return PlanMeasurement.model_validate_json(last)
        except (OSError, subprocess.SubprocessError, IndexError,
                ValueError) as exc:
            raise ModelError(
                f"Benchmark of {plan.name!r} failed: "
                f"{type(exc).__name__}"
            )
