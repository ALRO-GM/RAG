"""Chat model adapter: the local Qwen3 that reads sentences for names.

The transcriber's own copy of the loading rules of the RAG's local
judge (``src/adapters/outbound/local_judge.py``, policy ``preferred``;
the transcriber never imports ``src``):

- an ordered list of models, Qwen3-8B then Qwen3-4B, at 4 bits on the
  GPU; the first that fits the free GPU memory now is loaded;
- when none fits, wait (up to ``wait_seconds``) until the last one
  fits, then give up with :class:`ModelLoadError`; a smaller model is
  never taken: only the models of :data:`ALLOWED_MODELS` can be
  configured (never Qwen3-1.7B, the owner's rule on #102);
- the model is freed after use (:meth:`LocalNameModel.release`).

The GPU is shared with the RAG and another project's worker, so how
much is free changes from job to job; nothing here ever stops another
process. On the CPU the last model of the list is loaded in float32
(slow, but it runs). Nothing is sent outside the machine.

4-bit weights need ``bitsandbytes`` and ``accelerate`` (``uv sync
--group local-judge``); without them the GPU needs the 16-bit sizes,
which an 8 GB card does not have, and the step is skipped with the
reason.
"""

import gc
import importlib.util
import logging
import re
import time
import traceback
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from transcriber.adapters.outbound.torch_runtime import pick_device
from transcriber.config import Device
from transcriber.domain.errors import ModelLoadError
from transcriber.domain.speaker_names import NAME_MODELS

LOG = logging.getLogger(__name__)

ALLOWED_MODELS: Tuple[str, ...] = NAME_MODELS
"""The models that may read transcripts for names, in preference order."""

PARAMS_BILLION: Dict[str, float] = {
    "Qwen/Qwen3-8B": 8.2,
    "Qwen/Qwen3-4B": 4.0,
}
"""Parameters of each allowed model (their model cards)."""

MEASURED_GPU_PEAK_GB: Dict[Tuple[str, int], float] = {
    ("Qwen/Qwen3-8B", 4): 6.31,
    ("Qwen/Qwen3-4B", 4): 2.87,
}
"""Peak GPU memory at 4 bits measured on the RTX 3070 Ti for the RAG's
local judge (``src/domain/runtime.py``, ``MEASURED_GPU_PEAK_GB``)."""

MEMORY_OVERHEAD = 1.3
"""Activations and cache on top of the weights, when not measured."""

GPU_MARGIN_GB = 0.4
"""Free memory left untouched, for the CUDA context and fragmentation."""

QUANTISATION_PACKAGES = ("bitsandbytes", "accelerate")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def memory_gb(name: str, bits: int) -> float:
    """Return the GPU memory ``name`` needs at ``bits``, in GB."""
    measured = MEASURED_GPU_PEAK_GB.get((name, bits))
    if measured is not None:
        return measured
    return PARAMS_BILLION[name] * bits / 8 * MEMORY_OVERHEAD


def check_models(models: Sequence[str]) -> Tuple[str, ...]:
    """Return ``models`` if every one is allowed, else raise ValueError."""
    chosen = tuple(models)
    wrong = [m for m in chosen if m not in ALLOWED_MODELS]
    if not chosen or wrong:
        raise ValueError(
            "The models for names must be among "
            f"{', '.join(ALLOWED_MODELS)}; not {', '.join(wrong) or '-'}."
        )
    return chosen


def cuda_memory_gb() -> Optional[Tuple[float, float]]:
    """Return the free and total memory of GPU 0 in GB (``None``: none)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return float(free) / 1e9, float(total) / 1e9
    except (ImportError, RuntimeError):
        return None


def quantisation_installed() -> bool:
    """Tell whether 4-bit loading is possible here."""
    return all(
        importlib.util.find_spec(name) is not None
        for name in QUANTISATION_PACKAGES
    )


class ChatBackend(Protocol):
    """A loaded chat model."""

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply."""
        ...

    def release(self) -> None:
        """Free its memory."""
        ...


class TransformersChat:
    """A Hugging Face causal LM answering with greedy decoding.

    Greedy decoding makes the same transcript always get the same
    answer; Qwen3's thinking is switched off.
    """

    def __init__(
        self, name: str, device: str, bits: int,
        max_input_tokens: int = 4000,
    ) -> None:
        """Load the tokenizer and the weights on ``device``.

        A load that fails part-way (CUDA out of memory while the weights
        are copied, for instance) leaves nothing behind: the frames that
        held the partial weights are cleared, they are collected and
        the CUDA cache is emptied, so the GPU memory is free again for
        a smaller model or the next job. The failure is a
        :class:`ModelLoadError` when it is an expected one.
        """
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.tokenizer: Any = None
        self.model: Any = None
        try:
            self._load(name, device, bits)
        except ImportError as exc:  # pragma: no cover - environment
            raise ModelLoadError(f"Cannot load {name}: {exc}") from exc
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)
            self.release()
            if isinstance(exc, (OSError, ValueError, RuntimeError)):
                raise ModelLoadError(f"Cannot load {name}: {exc}") from exc
            raise

    def _load(self, name: str, device: str, bits: int) -> None:
        """Load the tokenizer, then the weights (4-bit on CUDA if asked)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        options: Dict[str, Any] = {
            "dtype": torch.float32 if device == "cpu" else torch.float16,
        }
        quantised = device == "cuda" and bits < 16
        if quantised:
            from transformers import BitsAndBytesConfig

            config: Any = BitsAndBytesConfig
            options["quantization_config"] = config(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            options["device_map"] = {"": 0}
        model: Any = AutoModelForCausalLM.from_pretrained(name, **options)
        if not quantised:
            model.to(device)
        model.eval()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the reply to ``user`` under ``system``."""
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        size = int(inputs["input_ids"].shape[1])
        if size > self.max_input_tokens:
            raise ModelLoadError(
                f"The prompt has {size} tokens, more than "
                f"{self.max_input_tokens}."
            )
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output[0][size:], skip_special_tokens=True
        )
        return _THINK.sub("", str(text)).strip()

    def release(self) -> None:
        """Drop the weights and give the GPU memory back."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - environment
            pass


class LocalNameModel:
    """The local model for names, chosen when it is first asked.

    Args:
        models: The models in preference order (each in
            :data:`ALLOWED_MODELS`).
        device: ``auto`` (CUDA when available), ``cuda`` or ``cpu``.
        wait_seconds: How long to wait for GPU memory for the last
            model before giving up.
        poll_seconds: How often to look again meanwhile.
        memory: Reads the free and total GPU memory (tests replace it).
        resolve: Turns ``device`` into the torch device.
        load: Loads a model ``(name, device, bits)``.
        quantisable: Tells whether 4-bit loading is installed.
        sleep: Waits between looks.
        clock: Monotonic seconds.
    """

    def __init__(
        self,
        models: Sequence[str] = ALLOWED_MODELS,
        device: Device = "auto",
        wait_seconds: float = 600.0,
        poll_seconds: float = 15.0,
        memory: Callable[[], Optional[Tuple[float, float]]] = (
            cuda_memory_gb
        ),
        resolve: Callable[[Device], str] = pick_device,
        load: Callable[[str, str, int], ChatBackend] = TransformersChat,
        quantisable: Callable[[], bool] = quantisation_installed,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep the choices; nothing is measured or loaded yet."""
        self.models = check_models(models)
        self.device = device
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self._memory = memory
        self._resolve = resolve
        self._load = load
        self._quantisable = quantisable
        self._sleep = sleep
        self._clock = clock
        self._backend: Optional[ChatBackend] = None
        self._chosen: Optional[Tuple[str, int]] = None

    @property
    def model_name(self) -> str:
        """Return the loaded model with its precision (the first
        preferred one while nothing is loaded)."""
        if self._chosen is None:
            return self.models[0]
        name, bits = self._chosen
        return f"{name}@{bits}bit" if bits < 16 else name

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the reply of the model, loading it first if needed."""
        return self._pick().complete(system, user, max_tokens)

    def release(self) -> None:
        """Free the model; the next use measures the memory again."""
        if self._backend is not None:
            self._backend.release()
        self._backend = None
        self._chosen = None

    def _pick(self) -> ChatBackend:
        """Return the loaded model, choosing and loading it if needed."""
        if self._backend is not None:
            return self._backend
        device = self._resolve(self.device)
        if device == "cpu":
            name, bits = self.models[-1], 16
        else:
            name, bits = self._fitting()
        LOG.info("Names: loading %s (%d bits) on %s", name, bits, device)
        self._backend = self._load(name, device, bits)
        self._chosen = (name, bits)
        return self._backend

    def _fitting(self) -> Tuple[str, int]:
        """Return the first model that fits now, or wait for the last."""
        bits = 4 if self._quantisable() else 16
        free, total = self._memory() or (0.0, 0.0)
        for name in self.models:
            if memory_gb(name, bits) + GPU_MARGIN_GB <= free:
                return name, bits
        last = self.models[-1]
        need = memory_gb(last, bits) + GPU_MARGIN_GB
        hint = "" if bits < 16 else (
            " (4-bit loading is not installed: uv sync --group local-judge)"
        )
        if need > total:
            raise ModelLoadError(
                f"{last} needs {need:.1f} GB and the GPU has {total:.1f} GB"
                f" in all{hint}."
            )
        deadline = self._clock() + self.wait_seconds
        while free < need:
            if self._clock() >= deadline:
                raise ModelLoadError(
                    f"Waited {self.wait_seconds:g} s for {need:.1f} GB of"
                    f" GPU memory for {last}; {free:.1f} GB free{hint}."
                    " No smaller model is used."
                )
            LOG.warning(
                "Names: waiting for %.1f GB of free GPU memory for %s"
                " (%.1f GB free now).", need, last, free,
            )
            self._sleep(self.poll_seconds)
            free, total = self._memory() or (0.0, total)
        return last, bits
