"""GPU models of the analysis leave a full or missing GPU for the CPU (#110).

No real model and no GPU: out-of-memory is simulated by raising
``torch.OutOfMemoryError`` from fakes, and "no CUDA device" by a GPU
reading of ``None``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch

from src.adapters.outbound.encoder_act_tagger import (
    CARD,
    EncoderActTagger,
    EncoderCard,
)
from src.adapters.outbound.gpu_fallback import (
    cpu_fallback_note,
    free_cuda_memory,
    is_cuda_oom,
)
from src.adapters.outbound.local_judge import LocalJudge
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.adapters.outbound import torch_runtime
from src.adapters.outbound.torch_runtime import prepare
from src.config import SkillsSettings
from src.domain.errors import ModelError
from src.domain.runtime import ModelOption, RuntimePlan
from src.domain.skills.act_tagging import ActInput
from src.domain.skills.dialogue_acts import CATEGORIES
from src.skills_bootstrap import SkillsContainer

GPU = RuntimePlan(device="cuda", threads=torch.get_num_threads())
"""A GPU plan that leaves the thread pool as the tests found it."""
QWEN = [
    ModelOption(name="Qwen/Qwen3-4B", params_billion=4.0, bits=4),
    ModelOption(name="Qwen/Qwen3-8B", params_billion=8.2, bits=4),
]


def oom() -> torch.OutOfMemoryError:
    """Return the error PyTorch raises when the GPU is full."""
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate")


def never_sleep(seconds: float) -> None:
    """Fail the test: nothing may wait here."""
    raise AssertionError(f"waited {seconds} s")


# ------------------------------------------------------------ helpers


def test_out_of_memory_is_recognised_by_type_or_message() -> None:
    assert is_cuda_oom(oom())
    assert is_cuda_oom(RuntimeError("CUDA out of memory (re-raised)"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))
    assert not is_cuda_oom(ValueError("CUDA out of memory"))
    free_cuda_memory()
    assert "the embeddings" in cpu_fallback_note("the embeddings")


def test_an_explicit_plan_keeps_its_own_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU plan after a GPU one is still the CPU (threads stay)."""
    monkeypatch.setattr(torch_runtime, "_applied", GPU)
    assert prepare(RuntimePlan(device="cpu", threads=1)) == "cpu"
    assert prepare(None) == "cuda", "no plan: the one in force, as before"
    assert torch_runtime.applied_plan() is GPU


# -------------------------------------------------------- local judge


class DeviceModel:
    """A built local model that records its device; may run out of memory."""

    built: List[Dict[str, Any]] = []
    full_gpu = False

    def __init__(
        self, name: str, plan: Optional[RuntimePlan] = None, bits: int = 16
    ) -> None:
        """Record what was asked for."""
        self.model_name = name
        self.bits = bits
        self.device = plan.device if plan is not None else "cpu"
        self.released = False
        DeviceModel.built.append(
            {"name": name, "bits": bits, "device": self.device}
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer with the device, or run out of GPU memory."""
        if self.device == "cuda" and DeviceModel.full_gpu:
            raise oom()
        return f"{self.model_name} on {self.device}"

    def release(self) -> None:
        """Mark the model released."""
        self.released = True


def judge(**kwargs: Any) -> LocalJudge:
    """A preferred local judge (8B, else 4B) over :class:`DeviceModel`."""
    DeviceModel.built = []
    DeviceModel.full_gpu = False
    options: Dict[str, Any] = {
        "build": DeviceModel, "quantisable": lambda: True,
        "total_gb": lambda: 8.0, "free_gb": lambda: 7.5,
        "sleep": never_sleep,
    }
    options.update(kwargs)
    plan = options.pop("plan", GPU)
    return LocalJudge(QWEN, plan, "Qwen/Qwen3-1.7B", **options)


def test_without_a_visible_gpu_the_judge_runs_the_4b_on_the_cpu() -> None:
    """A GPU plan but no CUDA device: no wait, the 4B on the CPU, a note."""
    no_gpu = judge(total_gb=lambda: None, free_gb=lambda: None)
    assert no_gpu.complete("s", "u", 5) == "Qwen/Qwen3-4B on cpu"
    assert DeviceModel.built == [
        {"name": "Qwen/Qwen3-4B", "bits": 16, "device": "cpu"}
    ]
    assert no_gpu.choice_reason == "no CUDA device is visible"
    assert len(no_gpu.device_notes) == 1
    assert "on the CPU (no CUDA device is visible)" in no_gpu.device_notes[0]
    assert "slower" in no_gpu.device_notes[0]


def test_a_cpu_plan_runs_the_last_preferred_model_and_says_it_is_slow() -> (
    None
):
    cpu = judge(plan=RuntimePlan(threads=2))
    assert cpu.model_name == "Qwen/Qwen3-4B"
    assert cpu.choice_reason == "CPU plan"
    assert "slower" in cpu.device_notes[0]
    forced = judge(plan=RuntimePlan(threads=2), fixed="Qwen/Qwen3-8B")
    assert forced.model_name == "Qwen/Qwen3-8B", "an explicit model wins"


def test_fail_on_cpu_refuses_at_once_with_the_reason() -> None:
    for plan, reason in ((GPU, "no CUDA device is visible"),
                         (RuntimePlan(threads=2), "CPU plan")):
        refusing = judge(plan=plan, total_gb=lambda: None,
                         free_gb=lambda: None, on_cpu="fail")
        with pytest.raises(ModelError, match=reason) as error:
            refusing.complete("s", "u", 5)
        assert "local_on_cpu" in str(error.value)
        assert DeviceModel.built == [], "nothing loaded"


def test_out_of_memory_on_the_gpu_retries_on_the_cpu() -> None:
    """The 8B runs out of memory: freed, the 4B answers on the CPU."""
    busy = judge()
    DeviceModel.full_gpu = True
    assert busy.complete("s", "u", 5) == "Qwen/Qwen3-4B on cpu"
    assert [b["device"] for b in DeviceModel.built] == ["cuda", "cpu"]
    assert DeviceModel.built[0]["name"] == "Qwen/Qwen3-8B"
    assert "the GPU ran out of memory" in busy.device_notes[0]
    assert busy.complete("s", "u", 5) == "Qwen/Qwen3-4B on cpu"
    assert len(DeviceModel.built) == 2, "stays on the CPU for this job"
    busy.release()
    DeviceModel.full_gpu = False
    assert busy.complete("s", "u", 5) == "Qwen/Qwen3-8B on cuda", (
        "the next job measures the GPU again"
    )
    assert len(busy.device_notes) == 1


def test_out_of_memory_with_fail_on_cpu_or_other_errors_are_raised() -> None:
    refusing = judge(on_cpu="fail")
    DeviceModel.full_gpu = True
    with pytest.raises(ModelError, match="the GPU ran out of memory"):
        refusing.complete("s", "u", 5)

    class Broken(DeviceModel):
        """A model that fails for another reason."""

        def complete(self, system: str, user: str, max_tokens: int) -> str:
            """Fail."""
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        judge(build=Broken).complete("s", "u", 5)


def test_a_busy_gpu_still_waits_and_a_vanished_one_stops_the_wait() -> None:
    """Waiting is kept for a GPU that exists; ``None`` ends it at once."""
    readings: List[Optional[float]] = [1.0, 1.0, None]
    slept: List[float] = []
    vanishing = judge(free_gb=lambda: readings.pop(0), sleep=slept.append,
                      clock=lambda: 0.0, poll_seconds=5)
    with pytest.raises(ModelError, match="disappeared"):
        vanishing.complete("s", "u", 5)
    assert slept == [5], "waited while the GPU was busy, not after"


def test_the_container_passes_the_cpu_setting_and_collects_notes(
    tmp_path: Path,
) -> None:
    settings = SkillsSettings(local_on_cpu="fail", device="cpu",
                              act_card=str(tmp_path / "none.json"))
    app = SkillsContainer(settings)
    assert app.device_notes() == [], "nothing built, nothing noted"
    local = app.local_judge
    assert isinstance(local, LocalJudge) and local.on_cpu == "fail"
    local.device_notes.append("moved")
    assert app.device_notes() == ["moved"]
    assert app.session_analysis.device_notes() == ["moved"]


# --------------------------------------------------------- embeddings


class FakeSentenceModel:
    """A sentence-transformers stand-in that may run out of GPU memory."""

    def __init__(self, device: str, full: bool) -> None:
        """Remember the device and whether the GPU is full."""
        self.device = device
        self.full = full

    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any:
        """Return one unit vector per text, or run out of GPU memory."""
        if self.device == "cuda" and self.full:
            raise oom()
        return np.ones((len(texts), 4), dtype=np.float32) / 2

    def get_sentence_embedding_dimension(self) -> int:
        """Return the vector size."""
        return 4


def embedder(full_on: str) -> "tuple[SentenceTransformerEmbedder, List[str]]":
    """An embedder on a GPU plan whose GPU is full at ``full_on``."""
    loads: List[str] = []

    def load(device: str) -> FakeSentenceModel:
        """Build the fake, or run out of memory while loading."""
        loads.append(device)
        if device == "cuda" and full_on == "load":
            raise oom()
        return FakeSentenceModel(device, full_on == "encode")

    return SentenceTransformerEmbedder(
        "fake-model", plan=GPU, load=load, batch_size=2
    ), loads


@pytest.mark.parametrize("full_on", ["load", "encode"])
def test_embeddings_move_to_the_cpu_when_the_gpu_is_full(
    full_on: str,
) -> None:
    model, loads = embedder(full_on)
    vectors = model.encode(["a", "b", "c"])
    assert vectors.shape == (3, 4)
    assert loads == ["cuda", "cpu"]
    assert model.device_notes == [
        cpu_fallback_note("the embeddings (fake-model)")
    ]
    model.encode(["d"])
    assert loads == ["cuda", "cpu"] and len(model.device_notes) == 1


def test_embeddings_raise_other_errors_and_keep_the_gpu_when_it_fits() -> (
    None
):
    model, loads = embedder("never")
    assert model.encode(["a"]).shape == (1, 4)
    assert model.encode([]).shape == (0, 4)
    assert loads == ["cuda"] and model.device_notes == []

    def broken(device: str) -> Any:
        """Fail for another reason."""
        raise RuntimeError("bad weights")

    failing = SentenceTransformerEmbedder("x", plan=GPU, load=broken)
    with pytest.raises(RuntimeError, match="bad weights"):
        failing.encode(["a"])


# ---------------------------------------------------------- act tagger


class FakeBatch(dict):  # type: ignore[type-arg, unused-ignore]
    """Tokenised inputs that run out of memory when sent to a full GPU."""

    full = False

    def to(self, device: str) -> "FakeBatch":
        """Move the batch, or run out of GPU memory."""
        if device.startswith("cuda") and FakeBatch.full:
            raise oom()
        return self


class FakeTokenizer:
    """Turns pairs into a batch of their count."""

    def __call__(self, contexts: List[str], texts: List[str],
                 **kwargs: Any) -> FakeBatch:
        """Return the batch."""
        return FakeBatch(n=len(texts))


class FakeClassifier:
    """Gives the first category the highest logit."""

    def __call__(self, n: int) -> Any:
        """Return an object with the logits."""
        logits = torch.zeros((n, len(CATEGORIES)))
        logits[:, 0] = 3.0

        class Out:
            """The model's output."""

        out = Out()
        setattr(out, "logits", logits)
        return out


def write_encoder(directory: Path) -> Path:
    """Write a card for a fake fine-tuned encoder."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CARD).write_text(EncoderCard(
        version="v1", base_model="b", categories=list(CATEGORIES),
        max_length=64, temperature=1.0).model_dump_json(), encoding="utf-8")
    return directory


def test_the_act_tagger_moves_to_the_cpu_when_the_gpu_is_full(
    tmp_path: Path,
) -> None:
    loads: List[str] = []

    def load(directory: str, device: str) -> Any:
        """Return the fake tokenizer and classifier."""
        loads.append(device)
        return FakeTokenizer(), FakeClassifier()

    tagger = EncoderActTagger(str(write_encoder(tmp_path / "enc")), "cuda",
                              batch_size=2, load=load)
    items = [ActInput(id=f"s{n}", speaker="A", text="hola")
             for n in range(3)]
    FakeBatch.full = True
    try:
        probabilities = tagger.predict(items)
    finally:
        FakeBatch.full = False
    assert len(probabilities) == 3
    assert max(probabilities[0], key=probabilities[0].__getitem__) == (
        CATEGORIES[0]
    )
    assert loads == ["cuda", "cpu"] and tagger.device == "cpu"
    assert tagger.device_notes == [
        cpu_fallback_note("the act tagger (XLM-R)")
    ]
    tagger.release()
    assert tagger.device_notes, "the note outlives the release"


def test_the_act_tagger_raises_errors_that_are_not_out_of_memory(
    tmp_path: Path,
) -> None:
    def load(directory: str, device: str) -> Any:
        """Fail for another reason."""
        raise RuntimeError("missing weights")

    tagger = EncoderActTagger(str(write_encoder(tmp_path / "enc")), "cuda",
                              load=load)
    with pytest.raises(RuntimeError, match="missing weights"):
        tagger.predict([ActInput(id="s1", speaker="A", text="hola")])
    assert tagger.device_notes == []
