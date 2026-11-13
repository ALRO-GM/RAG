"""Generator adapter: Hugging Face causal LM running locally on CPU."""

import re
from typing import Any, Optional, Sequence

from src.adapters.outbound.torch_runtime import prepare
from src.domain.errors import ModelError
from src.domain.prompting import (
    NO_ANSWER,
    SYSTEM_PROMPT,
    build_context,
    user_message,
)
from src.domain.models import Chunk
from src.domain.runtime import RuntimePlan

DEFAULT_LLM = "Qwen/Qwen3-0.6B"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class TransformersGenerator:
    """Generate grounded answers with a Hugging Face chat model on CPU.

    Retrieved chunks are packed into the prompt within
    ``context_tokens`` tokens. Qwen3's thinking mode is disabled: the
    0.6B model answers more faithfully and much faster without it.
    Decoding is greedy so answers are reproducible.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM,
        context_tokens: int = 2000,
        max_new_tokens: int = 256,
        plan: Optional[RuntimePlan] = None,
    ):
        """Configure the generator; the model loads on first use."""
        self.plan = plan
        self._device = "cpu"
        self._model_name = model_name
        self.context_tokens = context_tokens
        self.max_new_tokens = max_new_tokens
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the language model."""
        return self._model_name

    def _load(self) -> None:
        """Load the tokenizer and model weights."""
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

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens of ``text``."""
        assert self._tokenizer is not None
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def generate(self, question: str, context: Sequence[Chunk]) -> str:
        """Return an answer to ``question`` grounded in ``context``."""
        if not context:
            return NO_ANSWER
        self._load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        excerpts = build_context(
            context, self._count_tokens, self.context_tokens
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(question, excerpts)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = inputs.to(self._device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = _THINK.sub("", str(text)).strip()
        if not answer or answer.startswith(NO_ANSWER):
            # Small models sometimes repeat the refusal; keep it once.
            return NO_ANSWER
        return answer
