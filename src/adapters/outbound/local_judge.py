"""Chat model adapter: the local judge on a shared GPU.

The GPU of the workstation is shared with the live transcriber, the
RAG and another project's worker, so how much memory is free changes
from job to job. Three policies (``docs/skills/DESIGN.md`` §4.2):

- ``preferred`` (default, owner decision on #74): an ordered list of
  models (Qwen3-8B, then Qwen3-4B, both at 4 bits); the first that
  fits the free memory now is loaded; when none fits, the job waits
  (up to a time-out) until the last one fits, and never takes a model
  outside the list;
- ``largest_fitting``: the largest model of the ladder that fits the
  free memory now;
- ``fixed``: always load one model, waiting (up to a time-out) until
  enough memory is free instead of taking a smaller one.

Quantised weights need ``bitsandbytes`` and ``accelerate`` (``uv sync
--group local-judge``). Without them the judge logs a warning and uses
half precision, choosing among the models that fit at 16 bits, so a
missing optional package never stops a job (nor the fallback from
Claude). The model is freed when the job ends
(:meth:`LocalJudge.release`).

On the CPU (#110) it never waits for a GPU: with a CPU plan, when the
plan wanted the GPU but no CUDA device is visible, or when the GPU runs
out of memory while loading or answering, it loads the CPU model
(``preferred``: the last model of the preference, so never one outside
the list; otherwise the model chosen for the machine; an explicit model
always) and adds a "slow" line to :attr:`LocalJudge.device_notes`.
``on_cpu = "fail"`` refuses instead, at once and with the reason.
Waiting stays only for a GPU that exists but is busy, up to
``wait_seconds``.
"""

import importlib.util
import logging
import time
from typing import Callable, List, Literal, Optional, Protocol, Sequence

from src.adapters.outbound.gpu_fallback import is_cuda_oom
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

LOG = logging.getLogger(__name__)

LocalPolicy = Literal["preferred", "largest_fitting", "fixed"]
OnCpu = Literal["run", "fail"]

NO_CUDA = "no CUDA device is visible"
CUDA_OOM = "the GPU ran out of memory"


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

QUANTISATION_PACKAGES = ("bitsandbytes", "accelerate")


def cuda_total_gb() -> Optional[float]:
    """Return the total memory of GPU 0 in GB (``None`` without CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        _, total = torch.cuda.mem_get_info(0)
        return float(total) / 1e9
    except (ImportError, RuntimeError):
        return None


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


def quantisation_installed() -> bool:
    """Tell whether 4- and 8-bit loading is possible here."""
    return all(
        importlib.util.find_spec(name) is not None
        for name in QUANTISATION_PACKAGES
    )


class LocalJudge:
    """The local chat model, picked when a job starts.

    Args:
        ladder: The candidate models with their size and precision.
        plan: Device and threads (a CPU plan loads ``cpu_model``).
        cpu_model: The model used on the CPU.
        fixed: The model to load (forced with ``largest_fitting``,
            pinned with ``fixed``); ``None`` lets the policy choose.
        policy: ``preferred``, ``largest_fitting`` or ``fixed`` (see
            the module).
        preference: The ordered models of ``preferred`` (ladder names).
        wait_seconds: How long ``preferred`` and ``fixed`` wait for
            free memory.
        poll_seconds: How often it looks again meanwhile.
        free_gb: Reads the free GPU memory (replaced in tests).
        total_gb: Reads the GPU's total memory (replaced in tests).
        build: Makes the model adapter (replaced in tests).
        quantisable: Tells whether quantised loading is installed.
        sleep: Waits between looks (replaced in tests).
        clock: Monotonic seconds (replaced in tests).
        on_cpu: ``run`` the CPU model when the GPU cannot be used
            (slow, said in :attr:`device_notes`) or ``fail`` at once.
        is_oom: Recognises a CUDA out-of-memory error (replaced in
            tests).
    """

    def __init__(
        self,
        ladder: Sequence[ModelOption],
        plan: Optional[RuntimePlan],
        cpu_model: str,
        fixed: Optional[str] = None,
        policy: LocalPolicy = "preferred",
        preference: Sequence[str] = ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"),
        wait_seconds: float = 600.0,
        poll_seconds: float = 15.0,
        free_gb: Callable[[], Optional[float]] = cuda_free_gb,
        total_gb: Callable[[], Optional[float]] = cuda_total_gb,
        build: Callable[..., LoadableModel] = TransformersChatModel,
        quantisable: Callable[[], bool] = quantisation_installed,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_cpu: OnCpu = "run",
        is_oom: Callable[[BaseException], bool] = is_cuda_oom,
    ):
        """Keep the choices; nothing is measured or loaded yet."""
        self.ladder = list(ladder)
        self.plan = plan
        self.cpu_model = cpu_model
        self.fixed = fixed
        self.policy = policy
        self.preference = list(preference)
        self._total_gb = total_gb
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self._free_gb = free_gb
        self._build = build
        self._quantisable = quantisable
        self._sleep = sleep
        self._clock = clock
        self._model: Optional[LoadableModel] = None
        self.choice_reason = ""
        self.on_cpu = on_cpu
        self._is_oom = is_oom
        self._cpu_reason = ""
        self.device_notes: List[str] = []

    def _options(self) -> List[ModelOption]:
        """Return the ladder, at 16 bits when quantising is impossible."""
        if all(m.bits >= 16 for m in self.ladder) or self._quantisable():
            return self.ladder
        LOG.warning(
            "bitsandbytes/accelerate are not installed (uv sync --group "
            "local-judge): the local judge runs in half precision."
        )
        return [m.model_copy(update={"bits": 16}) for m in self.ladder]

    def _cpu_choice(self) -> str:
        """Return the model the judge runs on the CPU."""
        if self.fixed:
            return self.fixed
        if self.policy == "preferred" and self.preference:
            return self.preference[-1]
        return self.cpu_model

    def _on_cpu(self, reason: str) -> LoadableModel:
        """Build the CPU model, or refuse at once when told to."""
        name = self._cpu_choice()
        if self.on_cpu == "fail":
            raise ModelError(
                f"The local judge cannot use a GPU ({reason}) and "
                "local_on_cpu is 'fail', so it does not run on the CPU. "
                "Make a GPU visible or set local_on_cpu to 'run' (slow)."
            )
        self.choice_reason = reason
        LOG.warning("Local judge on the CPU (%s): %s, slow.", reason, name)
        self.device_notes.append(
            f"The local judge ran {name} on the CPU ({reason}): much "
            "slower than on the GPU."
        )
        plan = (self.plan or RuntimePlan()).model_copy(
            update={"device": "cpu"}
        )
        self._model = self._build(name, plan=plan, bits=16)
        return self._model

    def _pick(self) -> LoadableModel:
        """Choose and build the model for the memory free now."""
        if self._model is not None:
            return self._model
        on_gpu = self.plan is not None and self.plan.device == "cuda"
        if on_gpu and not self._cpu_reason and self._total_gb() is None:
            self._cpu_reason = NO_CUDA
        if not on_gpu:
            return self._on_cpu("CPU plan")
        if self._cpu_reason:
            return self._on_cpu(self._cpu_reason)
        if self.policy == "fixed":
            name, bits = self._wait_for_fixed()
        elif self.policy == "preferred" and not self.fixed:
            name, bits = self._preferred()
        elif self.fixed:
            options = self._options()
            bits = next((m.bits for m in options if m.name == self.fixed), 16)
            name = self.fixed
            self.choice_reason = "forced"
        else:
            name, bits = self._largest()
        self._model = self._build(name, plan=self.plan, bits=bits)
        return self._model

    def _largest(self) -> "tuple[str, int]":
        """Return the largest ladder model that fits the free memory."""
        options = self._options()
        free = self._free_gb() or 0.0
        option = largest_fitting(options, "cuda", free - GPU_MARGIN_GB)
        if option is None:
            smallest = min(options, key=lambda m: m.params_billion)
            raise ModelError(
                f"The GPU has {max(free, 0.0):.1f} GB free; the smallest "
                f"local judge ({smallest.name}) needs "
                f"{model_memory_gb(smallest, 'cuda'):.1f} GB."
            )
        self.choice_reason = f"largest that fits {free:.1f} GB free"
        return option.name, option.bits

    def _wait_for_fixed(self) -> "tuple[str, int]":
        """Wait until the pinned model fits, up to ``wait_seconds``."""
        if not self.fixed:
            raise ModelError(
                "local_policy fixed needs the model to pin (--local_model)."
            )
        options = self._options()
        option = next((m for m in options if m.name == self.fixed), None)
        if option is None:
            self.choice_reason = "fixed (not in the ladder: size unknown)"
            return self.fixed, 16
        self._wait_for(option, "fixed")
        return option.name, option.bits

    def _preferred(self) -> "tuple[str, int]":
        """Return the first preferred model that fits, else wait for the last.

        Never returns a model outside :attr:`preference`.
        """
        options = {m.name: m for m in self._options()}
        missing = [name for name in self.preference if name not in options]
        if not self.preference or missing:
            raise ModelError(
                "Every model of local_preference must be in model_ladder "
                f"(its size is needed); missing: {', '.join(missing) or '-'}."
            )
        wanted = [options[name] for name in self.preference]
        free = self._free_gb() or 0.0
        for option in wanted:
            if model_memory_gb(option, "cuda") + GPU_MARGIN_GB <= free:
                self.choice_reason = f"preferred, {free:.1f} GB free"
                return option.name, option.bits
        last = wanted[-1]
        self._wait_for(last, "preferred")
        return last.name, last.bits

    def _wait_for(self, option: ModelOption, why: str) -> None:
        """Wait until ``option`` fits the free memory, or fail clearly."""
        need = model_memory_gb(option, "cuda") + GPU_MARGIN_GB
        hint = "" if option.bits < 16 or all(
            m.bits >= 16 for m in self.ladder
        ) else (
            " (4-bit loading is not installed: uv sync --group local-judge)"
        )
        total = self._total_gb()
        if total is not None and need > total:
            raise ModelError(
                f"{option.name} needs {need:.1f} GB and the GPU has "
                f"{total:.1f} GB in all{hint}."
            )
        deadline = self._clock() + self.wait_seconds
        waited = False
        while True:
            reading = self._free_gb()
            if reading is None:
                raise ModelError(
                    f"The GPU disappeared while the local judge waited "
                    f"for {need:.1f} GB for {option.name}."
                )
            free = reading
            if free >= need:
                self.choice_reason = f"{why}, {free:.1f} GB free"
                if waited:
                    LOG.warning("GPU memory is free; loading %s.",
                                option.name)
                return
            if self._clock() >= deadline:
                raise ModelError(
                    f"Waited {self.wait_seconds:g} s for {need:.1f} GB of "
                    f"GPU memory for {option.name}; {free:.1f} GB free"
                    f"{hint}. No smaller model is used."
                )
            LOG.warning(
                "Local judge: waiting for %.1f GB of free GPU memory for "
                "%s (%.1f GB free now).", need, option.name, free,
            )
            waited = True
            self._sleep(self.poll_seconds)

    @property
    def model_name(self) -> str:
        """Return the identifier of the model this job uses."""
        model = self._pick()
        suffix = f"@{model.bits}bit" if model.bits < 16 else ""
        return f"{model.model_name}{suffix}"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the local model's reply.

        When the GPU runs out of memory (loading or answering), the
        model is freed and the question asked again of the CPU model.
        """
        model = self._pick()
        try:
            return model.complete(system, user, max_tokens)
        except Exception as exc:
            if self._cpu_reason or not self._is_oom(exc) or (
                self.plan is None or self.plan.device != "cuda"
            ):
                raise
        self.release()
        self._cpu_reason = CUDA_OOM
        return self._pick().complete(system, user, max_tokens)

    def release(self) -> None:
        """Free the model; the next job measures the memory again."""
        if self._model is not None:
            self._model.release()
        self._model = None
        self._cpu_reason = ""
