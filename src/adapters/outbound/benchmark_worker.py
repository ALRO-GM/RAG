"""Benchmark worker: time one runtime plan in a fresh process.

Run as ``python -m src.adapters.outbound.benchmark_worker '<json>'``.
Pinning and thread pools are per process and cannot be changed
reliably once models have run, so every plan gets its own process.
"""

import json
import sys
import time
from typing import Any

from src.adapters.outbound.torch_runtime import apply_plan
from src.domain.runtime import RuntimePlan

SENTENCE = (
    "The team agreed on the plan and split the remaining work so that "
    "everyone knows what to deliver and when. "
)


def main() -> None:
    """Measure embedding and generation speed under the given plan."""
    request = json.loads(sys.argv[1])
    plan = RuntimePlan.model_validate(request["plan"])
    device = apply_plan(plan).device
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = [(SENTENCE * 4)[:400]] * int(request["texts"])
    embedder = SentenceTransformer(request["embedding_model"], device=device)
    embedder.max_seq_length = 64
    embedder.encode(texts[:64], batch_size=64, show_progress_bar=False)
    start = time.perf_counter()
    embedder.encode(texts, batch_size=64, show_progress_bar=False)
    embed_seconds = time.perf_counter() - start

    tokens = int(request["tokens"])
    tokenizer = AutoTokenizer.from_pretrained(request["llm"])
    model: Any = AutoModelForCausalLM.from_pretrained(
        request["llm"],
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    model.to(device)
    model.eval()
    inputs = tokenizer(SENTENCE * 40, return_tensors="pt").to(device)
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=4, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
        start = time.perf_counter()
        model.generate(**inputs, max_new_tokens=tokens,
                       min_new_tokens=tokens, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
        generation_seconds = time.perf_counter() - start
    print(json.dumps({
        "embed_seconds": embed_seconds,
        "tokens_per_second": tokens / generation_seconds,
    }))


if __name__ == "__main__":
    main()
