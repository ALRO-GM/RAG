"""Runtime planning: which device, cores and model fit this machine.

Pure decision logic. What the machine has is described by a
:class:`HardwareProfile` (filled by an adapter); what to do with it is
a :class:`RuntimePlan`. Nothing here touches the operating system.
"""

from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from src.domain.errors import InvalidInputError

Device = Literal["cpu", "cuda", "mps"]

BYTES_PER_PARAM = {"cpu": 4.0, "cuda": 2.0, "mps": 2.0}
MEMORY_OVERHEAD = 1.3
RESERVED_GB = 2.0
CPU_TOKENS_PER_CORE = 1.4
MIN_GPU_GB = 1.5


class Gpu(BaseModel):
    """An accelerator visible to the deep-learning framework."""

    name: str
    memory_gb: float
    backend: Literal["cuda", "mps"] = "cuda"


class GraphicsCard(BaseModel):
    """A graphics card present in the machine, usable or not.

    ``driver`` is the kernel driver bound to it (empty when none).
    """

    vendor: str
    driver: str = ""


class HardwareProfile(BaseModel):
    """What the machine offers to this process.

    ``cores`` maps every usable physical core to its logical CPUs
    (hardware threads). ``performance_cores`` lists the cores of the
    fastest class on a hybrid processor; it equals all the cores on a
    homogeneous one. Only CPUs this process is allowed to run on are
    listed, so a container limit is respected.
    """

    cores: Dict[int, List[int]]
    performance_cores: List[int]
    memory_gb: float
    available_memory_gb: float
    gpus: List[Gpu] = Field(default_factory=list)
    cards: List[GraphicsCard] = Field(default_factory=list)
    torch_build: str = ""
    torch_cuda: bool = False

    def logical_cpus(self) -> List[int]:
        """Return every usable logical CPU."""
        return sorted(c for cpus in self.cores.values() for c in cpus)

    def is_hybrid(self) -> bool:
        """Tell whether some cores are slower than others."""
        return 0 < len(self.performance_cores) < len(self.cores)

    def one_cpu_per_core(self, cores: Sequence[int]) -> List[int]:
        """Return the first hardware thread of each given core."""
        return sorted(min(self.cores[c]) for c in cores)

    def signature(self) -> str:
        """Identify the hardware a measured plan was tuned for."""
        gpus = ",".join(f"{g.backend}:{g.name}" for g in self.gpus)
        return (
            f"{len(self.cores)}c/{len(self.logical_cpus())}t/"
            f"{len(self.performance_cores)}p/{round(self.memory_gb)}g/{gpus}"
        )


class RuntimePlan(BaseModel):
    """How to run the models on this machine.

    ``cpus`` is the set of logical CPUs the process is pinned to
    (empty = leave the scheduler alone) and ``threads`` the size of
    the framework's thread pool.
    """

    device: Device = "cpu"
    threads: int = Field(default=1, ge=1)
    cpus: List[int] = Field(default_factory=list)
    name: str = ""
    reason: str = ""
    measured: bool = False


class ModelOption(BaseModel):
    """A language model the analyzer may pick, with its size."""

    name: str
    params_billion: float = Field(gt=0)


class ModelChoice(BaseModel):
    """The model picked for a plan, and why the others were not."""

    model: str
    memory_gb: float
    tokens_per_second: Optional[float]
    reason: str
    rejected: Dict[str, str] = Field(default_factory=dict)


def gpu_diagnosis(profile: HardwareProfile) -> List[str]:
    """Explain why a graphics card that is present cannot be used.

    Returns nothing when a GPU is usable or when the machine has no
    card that PyTorch could drive. Each line names one missing piece,
    in the order it has to be fixed.
    """
    if profile.gpus:
        return []
    problems: List[str] = []
    nvidia = [c for c in profile.cards if c.vendor == "NVIDIA"]
    for card in nvidia:
        if card.driver != "nvidia":
            problems.append(
                f"the NVIDIA card runs on the kernel driver "
                f"{card.driver or 'none'!r}; CUDA needs the NVIDIA "
                "driver (install it and reboot)"
            )
    if nvidia and not profile.torch_cuda:
        problems.append(
            f"the installed PyTorch ({profile.torch_build or 'unknown'}) "
            "is a CPU-only build; install the CUDA build"
        )
    if nvidia and not problems:
        problems.append(
            "the driver and PyTorch look right but CUDA reports no "
            "device; check that the driver version supports the card"
        )
    return problems


def candidate_plans(profile: HardwareProfile) -> List[RuntimePlan]:
    """Return the configurations worth comparing on this machine.

    Every usable GPU backend is a candidate too, so that the GPU is
    chosen because it was measured faster, not because it exists.

    Matrix code gains little from two hardware threads of one core,
    and on a hybrid processor a thread pool waits for its slowest
    member, so the candidates are: every logical CPU, one thread per
    core, and (hybrid only) the performance cores with and without
    their second hardware thread.
    """
    logical = profile.logical_cpus()
    every_core = profile.one_cpu_per_core(list(profile.cores))
    plans = [
        RuntimePlan(threads=len(logical), cpus=logical,
                    name="all logical CPUs"),
        RuntimePlan(threads=len(every_core), cpus=every_core,
                    name="one thread per core"),
    ]
    if profile.is_hybrid():
        fast = profile.performance_cores
        threads = sorted(c for core in fast for c in profile.cores[core])
        plans.append(RuntimePlan(
            threads=len(threads), cpus=threads,
            name="performance cores, all threads",
        ))
        single = profile.one_cpu_per_core(fast)
        plans.append(RuntimePlan(
            threads=len(single), cpus=single,
            name="performance cores, one thread per core",
        ))
    unique: Dict[str, RuntimePlan] = {}
    for plan in plans:
        unique.setdefault(f"{plan.threads}:{plan.cpus}", plan)
    host = plan_runtime(profile, device="cpu")
    for backend in sorted({
        g.backend for g in profile.gpus if g.memory_gb >= MIN_GPU_GB
    }):
        unique[backend] = host.model_copy(update={
            "device": backend, "name": f"GPU ({backend})", "reason": "",
        })
    return list(unique.values())


def plan_runtime(
    profile: HardwareProfile,
    device: str = "auto",
    threads: Optional[int] = None,
    cores: str = "auto",
    measured: Optional[RuntimePlan] = None,
) -> RuntimePlan:
    """Choose the device, the CPUs and the thread count.

    Explicit values win. Otherwise a plan measured on this machine
    (``measured``) decides both the device and the CPUs, as long as
    its device is still available. Without a measurement, a GPU with
    at least ``MIN_GPU_GB`` of memory is used when there is one, and
    the CPU side gets one thread per performance core.
    """
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise InvalidInputError(
            f"Unknown device {device!r}; choose auto, cpu, cuda or mps."
        )
    if cores not in ("auto", "all", "performance"):
        raise InvalidInputError(
            f"Unknown cpu_cores {cores!r}; choose auto, all or "
            "performance."
        )
    usable = [g for g in profile.gpus if g.memory_gb >= MIN_GPU_GB]
    if device in ("cuda", "mps") and not any(
        g.backend == device for g in profile.gpus
    ):
        raise InvalidInputError(
            f"Device {device!r} was requested but is not available."
        )
    chosen: Device = "cpu"
    reason = "no usable GPU"
    if measured is not None and device == "auto" and cores == "auto" and (
        measured.device == "cpu"
        or any(g.backend == measured.device for g in usable)
    ):
        chosen = measured.device
        reason = "fastest measured on this machine"
    elif device in ("cuda", "mps"):
        chosen = "cuda" if device == "cuda" else "mps"
        reason = "requested"
    elif device == "auto" and usable:
        best = max(usable, key=lambda g: g.memory_gb)
        chosen = best.backend
        reason = f"{best.name} ({best.memory_gb:.0f} GB)"
    elif device == "cpu":
        reason = "requested"
    if cores == "all":
        base = RuntimePlan(
            threads=len(profile.logical_cpus()), cpus=[],
            name="all logical CPUs", reason="requested",
        )
    elif cores == "auto" and measured is not None and threads is None:
        base = measured.model_copy(update={
            "measured": True,
            "name": measured.name if measured.device == "cpu"
            else "host threads on the performance cores",
        })
    else:
        fast = profile.performance_cores or list(profile.cores)
        cpus = profile.one_cpu_per_core(fast)
        base = RuntimePlan(
            threads=len(cpus),
            cpus=cpus if profile.is_hybrid() else [],
            name="performance cores, one thread per core"
            if profile.is_hybrid() else "one thread per core",
        )
    if threads is not None:
        if threads < 1:
            raise InvalidInputError("threads must be at least 1.")
        base = base.model_copy(update={"threads": threads})
    return base.model_copy(update={
        "device": chosen,
        "reason": f"device: {reason}; CPU: {base.name}"
        + (" (measured)" if base.measured else ""),
    })


def model_memory_gb(option: ModelOption, device: str) -> float:
    """Estimate the memory a model needs on a device, in GB.

    Weights are 4 bytes per parameter on the CPU (float32) and 2 on a
    GPU (half precision), plus 30% for activations and the cache.
    """
    return option.params_billion * BYTES_PER_PARAM[device] * MEMORY_OVERHEAD


def estimate_cpu_speed(option: ModelOption, plan: RuntimePlan) -> float:
    """Estimate generated tokens per second on the CPU.

    A rule of thumb calibrated on one machine (0.6B parameters at
    about 19 tokens/s on 8 performance cores): proportional to the
    threads, inversely to the size. Good to a factor of two.
    """
    return CPU_TOKENS_PER_CORE * plan.threads / option.params_billion


def choose_model(
    profile: HardwareProfile,
    plan: RuntimePlan,
    ladder: Sequence[ModelOption],
    min_tokens_per_second: float = 4.0,
) -> ModelChoice:
    """Pick the largest model that fits the memory and is fast enough.

    On a GPU only memory limits the choice. On the CPU the estimated
    speed must also reach ``min_tokens_per_second``. When nothing
    qualifies the smallest model is returned, with the reason.
    """
    if not ladder:
        raise InvalidInputError("The model ladder is empty.")
    if plan.device == "cpu":
        budget = profile.available_memory_gb - RESERVED_GB
    else:
        budget = max(
            (g.memory_gb for g in profile.gpus if g.backend == plan.device),
            default=0.0,
        ) - 0.5
    ordered = sorted(ladder, key=lambda m: m.params_billion)
    rejected: Dict[str, str] = {}
    fitting: List[ModelOption] = []
    for option in ordered:
        need = model_memory_gb(option, plan.device)
        speed = (
            estimate_cpu_speed(option, plan) if plan.device == "cpu"
            else None
        )
        if need > budget:
            rejected[option.name] = (
                f"needs {need:.1f} GB, {max(budget, 0):.1f} GB free"
            )
        elif speed is not None and speed < min_tokens_per_second:
            rejected[option.name] = (
                f"about {speed:.1f} tokens/s, below "
                f"{min_tokens_per_second:g}"
            )
        else:
            fitting.append(option)
    best = fitting[-1] if fitting else ordered[0]
    speed = (
        round(estimate_cpu_speed(best, plan), 1)
        if plan.device == "cpu" else None
    )
    rejected.pop(best.name, None)
    return ModelChoice(
        model=best.name,
        memory_gb=round(model_memory_gb(best, plan.device), 1),
        tokens_per_second=speed,
        reason="largest model that fits and is fast enough" if fitting
        else "nothing qualifies; smallest model as a fallback",
        rejected=rejected,
    )
