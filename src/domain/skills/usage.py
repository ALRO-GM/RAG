"""Tokens a judge used, per model and per step of an analysis.

``claude -p --output-format json`` reports the tokens of every call;
the adapters turn them into a :class:`Usage`, the call budget adds
them to a :class:`UsageLedger`, and the analysis stores the difference
of the ledger over each step as a :class:`StepUsage` in ``run.json``.
A run stored before this existed has no usage: ``None`` means unknown,
never zero. Plain data and sums: no I/O.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """What one model used: calls and tokens.

    ``calls`` counts calls that reached the model; a call whose output
    reported no tokens still counts. ``cost_usd`` is what Claude Code
    reports as the equivalent API price (not charged to a subscription).
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0

    def plus(self, other: "Usage") -> "Usage":
        """Return the sum of both."""
        return Usage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens
                + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )

    def minus(self, other: "Usage") -> "Usage":
        """Return what this has beyond ``other`` (an earlier reading)."""
        return Usage(
            calls=self.calls - other.calls,
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens
                - other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens - other.cache_read_input_tokens
            ),
            cost_usd=round(self.cost_usd - other.cost_usd, 6),
        )

    @property
    def total_tokens(self) -> int:
        """Return every token counted (input, output and both caches)."""
        return (
            self.input_tokens + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def is_empty(self) -> bool:
        """Tell whether nothing was used."""
        return self == Usage()


class StepUsage(BaseModel):
    """The usage of one step (or of a whole run): per model, plus cache.

    ``cache_hits`` are answers served from the reply cache: they cost
    nothing and are not calls.
    """

    models: Dict[str, Usage] = Field(default_factory=dict)
    cache_hits: int = 0

    def total(self) -> Usage:
        """Return the usage summed over the models."""
        result = Usage()
        for usage in self.models.values():
            result = result.plus(usage)
        return result

    def plus(self, other: "StepUsage") -> "StepUsage":
        """Return the sum of both, model by model."""
        models = dict(self.models)
        for name, usage in other.models.items():
            models[name] = models.get(name, Usage()).plus(usage)
        return StepUsage(
            models=models, cache_hits=self.cache_hits + other.cache_hits
        )

    def minus(self, earlier: "StepUsage") -> "StepUsage":
        """Return what was used since the ``earlier`` reading."""
        models = {
            name: usage.minus(earlier.models.get(name, Usage()))
            for name, usage in self.models.items()
        }
        return StepUsage(
            models={n: u for n, u in models.items() if not u.is_empty()},
            cache_hits=self.cache_hits - earlier.cache_hits,
        )

    def is_empty(self) -> bool:
        """Tell whether the step used no model and hit no cache."""
        return not self.models and not self.cache_hits


def total_usage(steps: List[Optional[StepUsage]]) -> Optional[StepUsage]:
    """Return the sum of the known step usages (``None``: none known)."""
    known = [s for s in steps if s is not None]
    if not known:
        return None
    result = StepUsage()
    for usage in known:
        result = result.plus(usage)
    return result


class UsageLedger:
    """Running usage of one process, per model, with its cache hits."""

    def __init__(self) -> None:
        """Start empty."""
        self._models: Dict[str, Usage] = {}
        self._cache_hits = 0

    def record(self, model: str, usage: Optional[Usage]) -> None:
        """Count one call to ``model`` with the tokens it reported, if any."""
        used = usage or Usage()
        self._models[model] = self._models.get(model, Usage()).plus(
            used.model_copy(update={"calls": 1})
        )

    def hit(self) -> None:
        """Count one answer served from the cache."""
        self._cache_hits += 1

    def snapshot(self) -> StepUsage:
        """Return a copy of what has been counted so far."""
        return StepUsage(
            models={n: u.model_copy() for n, u in self._models.items()},
            cache_hits=self._cache_hits,
        )
