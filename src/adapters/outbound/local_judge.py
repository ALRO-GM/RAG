"""Chat model adapter: the local judge on a shared GPU.

The GPU of the workstation is shared with the live transcriber, the
RAG and another project's worker, so how much memory is free changes
from job to job. When a job first needs the judge, this adapter reads
the free memory and loads the largest model of the ladder that fits it
(``docs/skills/DESIGN.md`` §4.2), quantised as the ladder says; it
frees the memory again when the job ends (:meth:`LocalJudge.release`).
On the CPU it loads the model chosen for the machine.
"""

from typing import Callable, Optional, Protocol, Sequence

from src.adapters.outbound.transformers_chat_model import (
    TransformersChatModel,
)
from src.domain.errors import ModelError
from src.domain.runtime import (
    ModelOption,
    RuntimePlan,
    largest_fitting,
    model_memory_gb,
)


class LoadableModel(Protocol):
    """What the local judge needs of a model adapter."""

    bits: int

    @property
    def model_name(self) -> str:
        """Return the Hugging Face name of the model."""
        ...

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply."""
        ...

    def release(self) -> None:
        """Free the model's memory."""
        ...


GPU_MARGIN_GB = 0.4
"""Free memory left untouched, for the CUDA context and fragmentation."""


def cuda_free_gb() -> Optional[float]:
    """Return the free memory of GPU 0 in GB (``None`` without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info(0)
        return float(free) / 1e9
    except (ImportError, RuntimeError):
        return None


class LocalJudge:
    """The local chat model, picked by the memory free when a job starts.

    Args:
        ladder: The candidate models with their size and precision.
        plan: Device and threads (a CPU plan loads ``cpu_model``).
        cpu_model: The model used on the CPU (or when ``fixed``).
        fixed: Load this model whatever the memory says.
        free_gb: Reads the free GPU memory (replaced in tests).
        build: Makes the model adapter (replaced in tests).
    """

    def __init__(
        self,
        ladder: Sequence[ModelOption],
        plan: Optional[RuntimePlan],
        cpu_model: str,
        fixed: Optional[str] = None,
        free_gb: Callable[[], Optional[float]] = cuda_free_gb,
        build: Callable[..., LoadableModel] = TransformersChatModel,
    ):
        """Keep the choices; nothing is measured or loaded yet."""
        self.ladder = list(ladder)
        self.plan = plan
        self.cpu_model = cpu_model
        self.fixed = fixed
        self._free_gb = free_gb
        self._build = build
        self._model: Optional[LoadableModel] = None
        self.choice_reason = ""

    def _pick(self) -> LoadableModel:
        """Choose and build the model for the memory free now."""
        if self._model is not None:
            return self._model
        on_gpu = self.plan is not None and self.plan.device == "cuda"
        if self.fixed or not on_gpu:
            name = self.fixed or self.cpu_model
            bits = next(
                (m.bits for m in self.ladder if m.name == name), 16
            ) if on_gpu else 16
            self.choice_reason = "fixed" if self.fixed else "CPU plan"
        else:
            free = self._free_gb()
            budget = (free or 0.0) - GPU_MARGIN_GB
            option = largest_fitting(self.ladder, "cuda", budget)
            if option is None:
                smallest = min(self.ladder, key=lambda m: m.params_billion)
                raise ModelError(
                    f"The GPU has {max(free or 0.0, 0.0):.1f} GB free; the "
                    f"smallest local judge ({smallest.name}) needs "
                    f"{model_memory_gb(smallest, 'cuda'):.1f} GB."
                )
            name, bits = option.name, option.bits
            self.choice_reason = (
                f"largest that fits {free:.1f} GB free"
            )
        self._model = self._build(name, plan=self.plan, bits=bits)
        return self._model

    @property
    def model_name(self) -> str:
        """Return the identifier of the model this job uses."""
        model = self._pick()
        suffix = f"@{model.bits}bit" if model.bits < 16 else ""
        return f"{model.model_name}{suffix}"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the local model's reply."""
        return self._pick().complete(system, user, max_tokens)

    def release(self) -> None:
        """Free the model; the next job measures the memory again."""
        if self._model is not None:
            self._model.release()
        self._model = None
