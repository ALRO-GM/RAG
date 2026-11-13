"""Chat model adapter: a Hugging Face causal LM running locally."""

import re
from typing import Any, Optional

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
    ):
        """Configure the model; the weights load on first use."""
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
            model: Any = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                dtype=torch.float32 if self._device == "cpu"
                else torch.float16,
            )
            model.to(self._device)
            model.eval()
            self._model = model
        except (OSError, ValueError) as exc:
            raise ModelError(f"Cannot load model {self._model_name}: {exc}")

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
