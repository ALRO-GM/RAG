"""Prompt construction for grounded answer generation."""

from typing import Callable, List, Sequence

from src.domain.models import Chunk

SYSTEM_PROMPT = (
    "You are an expert assistant for the vLLM codebase. Answer the "
    "question using ONLY the numbered context excerpts below. Rules:\n"
    "1. Base every statement on the context; never invent APIs, flags, "
    "files or numbers.\n"
    "2. Quote exact identifiers, commands and values from the context "
    "when relevant.\n"
    "3. Cite the excerpts you used as [1], [2], ...\n"
    "4. If the context does not contain the answer, reply exactly: "
    "\"The retrieved sources do not contain enough information to "
    "answer this question.\"\n"
    "5. Be concise: at most a few sentences or a short list."
)

NO_ANSWER = (
    "The retrieved sources do not contain enough information to answer "
    "this question."
)


def format_excerpt(index: int, chunk: Chunk, body: str) -> str:
    """Render one numbered context excerpt."""
    header = f"[{index}] {chunk.file_path}"
    if chunk.context:
        header += f" ({chunk.context})"
    return f"{header}\n{body}"


def build_context(
    chunks: Sequence[Chunk],
    count_tokens: Callable[[str], int],
    budget: int,
) -> List[str]:
    """Render as many excerpts as fit in ``budget`` tokens.

    Excerpts are added in rank order; the first excerpt that does not
    fit is truncated to the remaining budget and the rest are dropped,
    so the best-ranked evidence always reaches the model.
    """
    excerpts: List[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        excerpt = format_excerpt(i, chunk, chunk.text)
        cost = count_tokens(excerpt)
        if used + cost <= budget:
            excerpts.append(excerpt)
            used += cost
            continue
        remaining = budget - used
        if remaining > 64:
            ratio = remaining / max(cost, 1)
            cut = chunk.text[: int(len(chunk.text) * ratio * 0.9)]
            excerpts.append(format_excerpt(i, chunk, cut + "\n[...]"))
        break
    return excerpts


def user_message(question: str, excerpts: Sequence[str]) -> str:
    """Return the user turn: the context followed by the question."""
    context = "\n\n".join(excerpts)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
