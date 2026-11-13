"""Runtime planning: device, cores, threads and model choice."""

from typing import List, Optional

import pytest

from src.application.runtime_tuning import (
    PlanMeasurement,
    RuntimeTuner,
)
from src.domain.errors import InvalidInputError
from src.domain.runtime import (
    Gpu,
    HardwareProfile,
    ModelOption,
    RuntimePlan,
    candidate_plans,
    choose_model,
    estimate_cpu_speed,
    model_memory_gb,
    plan_runtime,
)

LADDER = [
    ModelOption(name="small", params_billion=0.6),
    ModelOption(name="medium", params_billion=1.7),
    ModelOption(name="large", params_billion=4.0),
    ModelOption(name="huge", params_billion=14.0),
]


def hybrid(
    memory: float = 32.0, gpus: Optional[List[Gpu]] = None
) -> HardwareProfile:
    """Return 2 performance cores with 2 threads and 2 efficiency cores."""
    return HardwareProfile(
        cores={0: [0, 1], 1: [2, 3], 2: [4], 3: [5]},
        performance_cores=[0, 1],
        memory_gb=memory, available_memory_gb=memory, gpus=gpus or [],
    )


def uniform(cores: int = 4, threads: int = 2) -> HardwareProfile:
    """Return a homogeneous processor."""
    layout = {
        c: [c * threads + t for t in range(threads)] for c in range(cores)
    }
    return HardwareProfile(
        cores=layout, performance_cores=list(layout),
        memory_gb=16.0, available_memory_gb=16.0,
    )


def test_profile_describes_hybrid_layout() -> None:
    """Logical CPUs, hybrid flag and one CPU per core are derived."""
    profile = hybrid()
    assert profile.logical_cpus() == [0, 1, 2, 3, 4, 5]
    assert profile.is_hybrid()
    assert profile.one_cpu_per_core([0, 1]) == [0, 2]
    assert not uniform().is_hybrid()
    assert profile.signature() != uniform().signature()


def test_hybrid_default_pins_one_thread_per_performance_core() -> None:
    """Efficiency cores and second hardware threads are left out."""
    plan = plan_runtime(hybrid())
    assert (plan.device, plan.threads, plan.cpus) == ("cpu", 2, [0, 2])
    assert not plan.measured


def test_uniform_default_uses_physical_cores_without_pinning() -> None:
    """A homogeneous CPU gets one thread per core and no affinity."""
    plan = plan_runtime(uniform(cores=4, threads=2))
    assert (plan.threads, plan.cpus) == (4, [])


def test_gpu_is_preferred_when_large_enough() -> None:
    """The largest usable GPU wins; a tiny one is ignored."""
    big = Gpu(name="big", memory_gb=24.0)
    tiny = Gpu(name="tiny", memory_gb=1.0)
    assert plan_runtime(hybrid(gpus=[tiny, big])).device == "cuda"
    assert plan_runtime(hybrid(gpus=[tiny])).device == "cpu"
    mps = Gpu(name="apple", memory_gb=10.0, backend="mps")
    assert plan_runtime(hybrid(gpus=[mps])).device == "mps"


def test_explicit_choices_win() -> None:
    """Device, threads and cores given by the user are obeyed."""
    gpu = [Gpu(name="g", memory_gb=24.0)]
    assert plan_runtime(hybrid(gpus=gpu), device="cpu").device == "cpu"
    assert plan_runtime(hybrid(), threads=3).threads == 3
    everything = plan_runtime(hybrid(), cores="all")
    assert (everything.threads, everything.cpus) == (6, [])
    fast = plan_runtime(hybrid(), cores="performance")
    assert fast.cpus == [0, 2]


def test_measured_plan_replaces_rule_of_thumb() -> None:
    """A saved measurement is used, unless the user overrides it."""
    measured = RuntimePlan(threads=4, cpus=[0, 1, 2, 3], name="measured")
    plan = plan_runtime(hybrid(), measured=measured)
    assert (plan.threads, plan.cpus, plan.measured) == (4, [0, 1, 2, 3], True)
    assert plan_runtime(hybrid(), cores="all", measured=measured).cpus == []
    assert plan_runtime(hybrid(), threads=1, measured=measured).threads == 1


@pytest.mark.parametrize("kwargs", [
    {"device": "tpu"}, {"device": "cuda"}, {"device": "mps"},
    {"threads": 0}, {"cores": "some"},
])
def test_invalid_requests_are_rejected(kwargs: dict[str, object]) -> None:
    """Unknown or unavailable devices and bad values raise."""
    with pytest.raises(InvalidInputError):
        plan_runtime(hybrid(), **kwargs)  # type: ignore[arg-type]


def test_candidates_cover_the_relevant_configurations() -> None:
    """Hybrid CPUs get four candidates, homogeneous ones two."""
    names = [p.name for p in candidate_plans(hybrid())]
    assert names == [
        "all logical CPUs", "one thread per core",
        "performance cores, all threads",
        "performance cores, one thread per core",
    ]
    assert len(candidate_plans(uniform())) == 2
    assert len(candidate_plans(uniform(cores=4, threads=1))) == 1


def test_model_choice_on_cpu_is_bound_by_speed_and_memory() -> None:
    """The largest model that fits and is fast enough is chosen."""
    profile = hybrid(memory=32.0)
    plan = RuntimePlan(threads=8)
    choice = choose_model(profile, plan, LADDER, min_tokens_per_second=4.0)
    assert choice.model == "medium"
    assert "tokens/s" in choice.rejected["large"]
    assert "GB" in choice.rejected["huge"]
    assert choice.tokens_per_second == round(
        estimate_cpu_speed(LADDER[1], plan), 1
    )
    slow_ok = choose_model(profile, plan, LADDER, min_tokens_per_second=1.0)
    assert slow_ok.model == "large"


def test_model_choice_on_gpu_is_bound_by_memory_only() -> None:
    """Half precision on a GPU: 24 GB holds the 4B model, not the 14B."""
    profile = hybrid(gpus=[Gpu(name="g", memory_gb=24.0)])
    choice = choose_model(profile, RuntimePlan(device="cuda"), LADDER)
    assert choice.model == "large"
    assert choice.tokens_per_second is None
    assert model_memory_gb(LADDER[2], "cuda") < model_memory_gb(
        LADDER[2], "cpu"
    )


def test_model_choice_falls_back_to_the_smallest() -> None:
    """With too little memory the smallest model is returned anyway."""
    choice = choose_model(hybrid(memory=2.0), RuntimePlan(threads=2), LADDER)
    assert choice.model == "small"
    assert "fallback" in choice.reason
    with pytest.raises(InvalidInputError):
        choose_model(hybrid(), RuntimePlan(), [])


class FakeProbe:
    """A probe returning a fixed profile."""

    def profile(self) -> HardwareProfile:
        """Return the hybrid sample."""
        return hybrid()


class FakeBenchmark:
    """A benchmark where fewer threads are faster, noisy on first run."""

    def __init__(self) -> None:
        """Record the measured plans."""
        self.calls: List[str] = []

    def measure(self, plan: RuntimePlan) -> PlanMeasurement:
        """Return speeds that favour the two-thread plan."""
        first = plan.name not in self.calls
        self.calls.append(plan.name)
        return PlanMeasurement(
            embed_seconds=plan.threads * (2.0 if first else 1.0),
            tokens_per_second=48.0 / plan.threads,
        )


def test_tuner_ranks_plans_by_measured_cost() -> None:
    """Each plan is measured ``repeats`` times; the best run counts."""
    benchmark = FakeBenchmark()
    tuning = RuntimeTuner(FakeProbe(), benchmark, repeats=2).tune()
    assert len(benchmark.calls) == 8
    assert tuning.best().name == "performance cores, one thread per core"
    assert tuning.results[0].embed_seconds == 2.0
    assert [r.cost for r in tuning.results] == sorted(
        r.cost for r in tuning.results
    )
    assert tuning.speedup("all logical CPUs") == pytest.approx(3.0)
    assert tuning.speedup("unknown") is None
    with pytest.raises(InvalidInputError):
        RuntimeTuner(FakeProbe(), benchmark, repeats=0)


def test_gpu_diagnosis_names_each_missing_piece() -> None:
    """A present but unusable NVIDIA card is explained step by step."""
    from src.domain.runtime import GraphicsCard, gpu_diagnosis

    broken = hybrid().model_copy(update={
        "cards": [GraphicsCard(vendor="NVIDIA", driver="nouveau")],
        "torch_build": "2.14.1+cpu", "torch_cuda": False,
    })
    problems = gpu_diagnosis(broken)
    assert len(problems) == 2
    assert "nouveau" in problems[0] and "CPU-only" in problems[1]
    driver_ok = broken.model_copy(update={
        "cards": [GraphicsCard(vendor="NVIDIA", driver="nvidia")],
    })
    assert len(gpu_diagnosis(driver_ok)) == 1
    both_ok = driver_ok.model_copy(update={"torch_cuda": True})
    assert "reports no device" in gpu_diagnosis(both_ok)[0]
    working = both_ok.model_copy(update={
        "gpus": [Gpu(name="RTX", memory_gb=8.0)],
    })
    assert gpu_diagnosis(working) == []
    assert gpu_diagnosis(hybrid()) == []
    intel = hybrid().model_copy(update={
        "cards": [GraphicsCard(vendor="Intel", driver="i915")],
    })
    assert gpu_diagnosis(intel) == []


def test_usable_gpu_is_a_benchmark_candidate() -> None:
    """The GPU is measured against the CPU plans, not assumed faster."""
    profile = hybrid(gpus=[Gpu(name="RTX", memory_gb=8.0)])
    plans = candidate_plans(profile)
    gpu = [p for p in plans if p.device == "cuda"]
    assert len(gpu) == 1 and gpu[0].name == "GPU (cuda)"
    assert gpu[0].threads == 2 and gpu[0].cpus == [0, 2]
    assert len(plans) == 5
    assert all(p.device == "cpu" for p in candidate_plans(hybrid()))


def test_measured_device_decides_between_gpu_and_cpu() -> None:
    """With a GPU present, the measured winner is used either way."""
    profile = hybrid(gpus=[Gpu(name="RTX", memory_gb=8.0)])
    assert plan_runtime(profile).device == "cuda"
    cpu_won = RuntimePlan(threads=2, cpus=[0, 2], name="fast cpu")
    assert plan_runtime(profile, measured=cpu_won).device == "cpu"
    gpu_won = RuntimePlan(device="cuda", threads=2, cpus=[0, 2])
    plan = plan_runtime(profile, measured=gpu_won)
    assert (plan.device, plan.measured) == ("cuda", True)
    assert plan_runtime(hybrid(), measured=gpu_won).device == "cpu"
    assert plan_runtime(profile, device="cpu", measured=gpu_won).device == (
        "cpu"
    )
