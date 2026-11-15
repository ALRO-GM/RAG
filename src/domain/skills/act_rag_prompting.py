"""Prompt and reply of the retrieval + judge act tagger (RG, #94).

The judge sees the target sentence marked as such, the sentences before
it marked as context, and the cards retrieved for it **split by
position**: the category cards that match the target alone, then the
sequence chunks that match the pair and the window, each saying which
category it gives the target position and what the earlier positions
were. It answers up to three labels with weights and a one-line reason
that cites a card.
"""

import json
import re
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.act_prompting import ACT_DEFINITIONS, ACT_IDS
from src.domain.skills.act_retrieval import CardChunk, CategoryCard
from src.domain.skills.act_tagging import ActInput, Probabilities
from src.domain.skills.dialogue_acts import CATEGORIES

RAG_SYSTEM_PROMPT = (
    "Etiquetas frases de conversaciones de trabajo en español según lo "
    "que hacen en la conversación. Recibes UNA frase OBJETIVO, las "
    "frases dichas antes (CONTEXTO) y fichas de teoría recuperadas para "
    "cada posición.\n"
    "Categorías:\n{categories}\n"
    "Reglas:\n"
    "1. Etiqueta solo la frase OBJETIVO. El contexto sirve para saber a "
    "qué responde, nunca para copiar su etiqueta: si la frase anterior "
    "es una pregunta, la objetivo suele ser su respuesta, no otra "
    "pregunta.\n"
    "2. Las fichas de 'frase sola' describen el objetivo; en las de "
    "'par' y 'ventana' solo cuenta la categoría de la última posición "
    "(la del objetivo); las posiciones anteriores describen el "
    "contexto.\n"
    "3. Las fichas son pistas, no votos: decide con las definiciones y "
    "las reglas de frontera ('no es').\n"
    "4. Hasta tres etiquetas, la principal primero, con pesos enteros "
    "que sumen 100; una sola si la frase hace una sola cosa.\n"
    "5. Responde con un único objeto JSON y nada más:\n"
    '{{"labels": [{{"act": "<id>", "weight": <0-100>}}], '
    '"reason": "<una línea que cite la ficha>"}}'
)


def rag_system_prompt() -> str:
    """Return the system prompt of the retrieval + judge tagger."""
    lines = "\n".join(f"- {act}: {text}" for act, text in ACT_DEFINITIONS)
    return RAG_SYSTEM_PROMPT.format(categories=lines)


def _card_text(card: CategoryCard, shown: Sequence[str]) -> str:
    """Return a category card as prompt lines (contrasts with ``shown``)."""
    lines = [f"* {card.category}: {card.definition}"]
    if card.cues:
        lines.append(f"  pistas: {'; '.join(card.cues[:3])}")
    for note in card.not_:
        if note.category in shown:
            lines.append(f"  no es {note.category}: {note.contrast}")
    return "\n".join(lines)


def _chunk_text(chunk: CardChunk) -> str:
    """Return a pair or window chunk as one prompt line, by position."""
    n = len(chunk.texts)
    parts = []
    for i, text in enumerate(chunk.texts):
        offset = i - (n - 1)
        label = chunk.context[i] if i < n - 1 else chunk.category
        who = ""
        if i:
            who = "otro " if chunk.changes[i - 1] else "mismo "
        parts.append(f"[{offset}] {who}{label}: «{text}»")
    return f"- {' → '.join(parts)}  ⇒ objetivo: {chunk.category}"


def rag_user_prompt(
    item: ActInput,
    cards: Sequence[CategoryCard],
    single: Sequence[CardChunk],
    pair: Sequence[CardChunk],
    window: Sequence[CardChunk],
) -> str:
    """Return the user turn: context, target, then evidence by position.

    ``cards`` are the candidate categories' cards; ``single`` the
    example chunks that matched the target alone; ``pair`` and
    ``window`` the sequence chunks that matched with context.
    """
    n = len(item.context)
    context = "\n".join(
        f"[{i - n}] {c.speaker}: {c.text}" for i, c in enumerate(item.context))
    changed = bool(item.context) and item.context[-1].speaker != item.speaker
    turn = ("otro hablante que la frase anterior" if changed
            else "el mismo hablante que la frase anterior" if item.context
            else "inicio de la conversación")
    shown = [c.category for c in cards]
    blocks = [
        f"CONTEXTO (no se etiqueta):\n{context or '(inicio)'}",
        f"OBJETIVO [0], dicha por {item.speaker} ({turn}):\n{item.text}",
        "FICHAS DE LAS CATEGORÍAS CANDIDATAS:\n"
        + "\n".join(_card_text(c, shown) for c in cards),
    ]
    if single:
        blocks.append(
            "FRASE SOLA — ejemplos parecidos al objetivo:\n" + "\n".join(
                f"- {c.category}: «{c.texts[-1]}»" for c in single))
    if pair:
        blocks.append("PAR [-1]→[0] — secuencias parecidas:\n"
                      + "\n".join(_chunk_text(c) for c in pair))
    if window:
        blocks.append("VENTANA — secuencias parecidas:\n"
                      + "\n".join(_chunk_text(c) for c in window))
    blocks.append("JSON:")
    return "\n\n".join(blocks)


class JudgedActs(BaseModel):
    """The judge's labels (normalised to 100) and its reason."""

    model_config = ConfigDict(frozen=True)

    labels: List[Tuple[str, int]] = Field(min_length=1, max_length=3)
    reason: str = ""


_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _weights(raw: object) -> List[Tuple[str, float]]:
    """Return ``(act, weight)`` pairs of a ``labels`` value (known acts)."""
    out: List[Tuple[str, float]] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        act = str(entry.get("act", "")).strip().lower()
        try:
            weight = float(entry.get("weight", 100))
        except (TypeError, ValueError):
            continue
        if act in ACT_IDS and weight > 0 and act not in dict(out):
            out.append((act, weight))
    return out


def parse_weighted_acts(reply: str) -> Optional[JudgedActs]:
    """Return the labels in a reply, or ``None`` if none is usable.

    Reads the outermost JSON object; ``{"act": ...}`` alone is accepted
    as one label of 100. Unknown ids are dropped, at most three labels
    are kept (heaviest first) and the weights rescaled to sum 100.
    """
    match = _OBJECT.search(reply)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    pairs = _weights(data.get("labels"))
    if not pairs and isinstance(data.get("act"), str):
        pairs = _weights([{"act": data["act"], "weight": 100}])
    if not pairs:
        return None
    pairs = sorted(pairs, key=lambda p: -p[1])[:3]
    total = sum(w for _, w in pairs)
    labels = [(a, int(round(100 * w / total))) for a, w in pairs]
    labels[0] = (labels[0][0], labels[0][1] + 100 - sum(w for _, w in labels))
    return JudgedActs(labels=labels, reason=str(data.get("reason", ""))[:300])


def judged_distribution(judged: JudgedActs) -> Probabilities:
    """Return the judge's weights as a distribution over the 13."""
    p = {c: 0.0 for c in CATEGORIES}
    for act, weight in judged.labels:
        p[act] += weight / 100.0
    return p
