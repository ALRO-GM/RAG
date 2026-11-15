"""Chat model adapter: a Hugging Face causal LM running locally.

On a GPU the weights may be quantised to 4 or 8 bits with
``bitsandbytes`` (``uv sync --group local-judge``), which lets an 8 B
model fit next to the other models of a shared 8 GB card. The model is
loaded on first use and :meth:`TransformersChatModel.release` frees it,
so a job holds the memory only while it runs.
"""

import gc
import re
from typing import Any, Dict, Optional

from src.adapters.outbound.torch_runtime import prepare
from src.domain.errors import ModelError
from src.domain.runtime import RuntimePlan

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class TransformersChatModel:
    """Answer one system + user message with a local chat model.

    Nothing leaves the machine, which is the reason to use it for
    recordings of people. Decoding is greedy, so the same transcript
    always gets the same judgement.
    """

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int = 6000,
        plan: Optional[RuntimePlan] = None,
        bits: int = 16,
    ):
        """Configure the model; the weights load on first use.

        ``bits`` 4 or 8 quantises the weights on a GPU (ignored on the
        CPU, which keeps float32).
        """
        if bits not in (4, 8, 16):
            raise ModelError(f"bits must be 4, 8 or 16, not {bits}.")
        self.bits = bits
        self.plan = plan
        self._device = "cpu"
        self._model_name = model_name
        self.max_input_tokens = max_input_tokens
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the language model."""
        return self._model_name

    def _load(self) -> None:
        """Load the tokenizer and the weights."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = prepare(self.plan)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            options: Dict[str, Any] = {
                "dtype": torch.float32 if self._device == "cpu"
                else torch.float16,
            }
            quantised = self._device == "cuda" and self.bits < 16
            if quantised:
                options["quantization_config"] = quantisation(self.bits)
                options["device_map"] = {"": 0}
            model: Any = AutoModelForCausalLM.from_pretrained(
                self._model_name, **options
            )
            if not quantised:
                model.to(self._device)
            model.eval()
            self._model = model
        except ImportError as exc:
            raise ModelError(
                f"Cannot load model {self._model_name} at {self.bits} bits:"
                f" {exc}. Quantised weights need `uv sync --group "
                "local-judge`."
            )
        except (OSError, ValueError) as exc:
            raise ModelError(f"Cannot load model {self._model_name}: {exc}")

    def release(self) -> None:
        """Drop the weights and give the GPU memory back."""
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply to ``user`` under ``system``."""
        self._load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = inputs.to(self._device)
        size = int(inputs["input_ids"].shape[1])
        if size > self.max_input_tokens:
            raise ModelError(
                f"The prompt has {size} tokens, more than the "
                f"{self.max_input_tokens} allowed for {self._model_name}; "
                "lower the number of candidate turns."
            )
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(
            output[0][size:], skip_special_tokens=True
        )
        return _THINK.sub("", str(text)).strip()


def quantisation(bits: int) -> Any:
    """Return the ``bitsandbytes`` configuration for 4 or 8 bits.

    4 bits uses NF4 with float16 compute and double quantisation, the
    usual setting for inference.
    """
    import torch
    from transformers import BitsAndBytesConfig

    config: Any = BitsAndBytesConfig
    if bits == 4:
        return config(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return config(load_in_8bit=True)
