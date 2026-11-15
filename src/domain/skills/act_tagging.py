"""What an automatic dialogue-act tagger reads and writes (#82).

A tagger reads one sentence with the sentences said just before it
(:class:`ActInput`) and returns a probability for each of the 13
categories. Every approach (rules, a language model, a fine-tuned
encoder, a combination) has this shape, so they are measured with the
same functions (:mod:`src.domain.skills.act_scoring`) and shown in the
same weighted format as a human annotation
(:func:`~src.domain.skills.act_agreement.tagger_labels`).

This module also holds what is plain data or string handling around
them: inputs built from imported records or from a transcript, the text
an encoder reads (context with speaker changes, then the sentence), the
reference a prediction is scored against (a compatible set of
categories, plus the weights of our own set), and the routing of a
combined tagger.
"""

import hashlib
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.dialogue_acts import (
    CATEGORIES,
    ActRecord,
    ContextLine,
    WeightedLabel,
)

Probabilities = Dict[str, float]
"""A tagger's output for one sentence: category → probability (0–1)."""

SAME_SPEAKER = "[=]"
"""Marker of a context line said by the speaker of the target sentence."""

OTHER_SPEAKER = "[>]"
"""Marker of a context line said by somebody else."""

CONTEXT_CHARS = 400
"""Most characters of context an encoder input keeps (the latest)."""


Split = Literal["train", "dev", "test"]


def split_of(dialogue: str, test: float = 0.15, dev: float = 0.10) -> Split:
    """Return the split of a public-corpus dialogue (by meeting).

    Our own deterministic split, not a corpus's published one: the
    first 8 hex digits of the SHA-256 of the dialogue id, as a share of
    their range, below ``test`` → test, below ``test + dev`` → dev,
    else train. A whole meeting always falls in one split.
    """
    if test < 0 or dev < 0 or test + dev >= 1:
        raise ValueError("test and dev shares must be >= 0 and sum < 1")
    share = int(hashlib.sha256(dialogue.encode()).hexdigest()[:8], 16) / 16**8
    if share < test:
        return "test"
    return "dev" if share < test + dev else "train"


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ActInput(_Frozen):
    """One sentence to tag, with the sentences said before it.

    ``context`` is oldest first; speakers are whatever the source
    calls them (only whether they change matters to the taggers).
    """

    id: str = Field(min_length=1)
    speaker: str
    text: str
    context: List[ContextLine] = []


class Reference(_Frozen):
    """What a prediction for one sentence is scored against.

    ``categories`` is the set of categories counted as right (public
    corpora: the categories the source label is compatible with; our
    set: the primary label). ``labels`` are the weighted labels when
    the source gives them (our set), used for the log-loss.
    """

    categories: List[str] = Field(min_length=1)
    labels: List[WeightedLabel] = []


def record_input(record: ActRecord) -> ActInput:
    """Return the tagger input of an imported record."""
    return ActInput(id=f"{record.dialogue}/{record.id}",
                    speaker=record.speaker, text=record.text,
                    context=list(record.context))


def record_reference(record: ActRecord) -> Optional[Reference]:
    """Return the reference of a record (``None``: not usable)."""
    if not record.categories:
        return None
    return Reference(categories=list(record.categories),
                     labels=list(record.labels))


def sentence_inputs(
    sentences: Sequence[Tuple[str, str, str]], context: int = 3
) -> List[ActInput]:
    """Return inputs for ``(id, speaker, text)`` sentences in spoken order.

    Each carries the ``context`` sentences said just before it.
    """
    if context < 0:
        raise ValueError("context must be >= 0")
    inputs: List[ActInput] = []
    for n, (sid, speaker, text) in enumerate(sentences):
        before = sentences[max(n - context, 0):n] if context else []
        inputs.append(ActInput(
            id=sid, speaker=speaker, text=text,
            context=[ContextLine(speaker=s, text=t) for _, s, t in before]))
    return inputs


def speaker_changes(item: ActInput) -> bool:
    """Tell whether the sentence opens a new turn (another speaker)."""
    return bool(item.context) and item.context[-1].speaker != item.speaker


def encoder_texts(item: ActInput) -> Tuple[str, str]:
    """Return the pair of texts an encoder reads for ``item``.

    First the context, each line prefixed :data:`SAME_SPEAKER` or
    :data:`OTHER_SPEAKER` (who said it relative to the target sentence,
    so no speaker name reaches the model), cut to the last
    :data:`CONTEXT_CHARS` characters; then the target sentence prefixed
    by the marker of whether the speaker changed just before it.
    """
    lines = [
        f"{SAME_SPEAKER if c.speaker == item.speaker else OTHER_SPEAKER} "
        f"{c.text}" for c in item.context
    ]
    context = " ".join(lines)[-CONTEXT_CHARS:]
    marker = OTHER_SPEAKER if speaker_changes(item) else SAME_SPEAKER
    return context, f"{marker} {item.text}"


def check_probabilities(probabilities: Mapping[str, float]) -> Probabilities:
    """Return a full, normalised distribution over the 13 categories.

    Categories left out get 0; the values are rescaled to sum to 1.
    Raises :class:`ValueError` on an unknown category, a negative value
    or an all-zero distribution.
    """
    for category, p in probabilities.items():
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        if p < 0:
            raise ValueError(f"negative probability for {category}")
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("probabilities add up to 0")
    return {c: float(probabilities.get(c, 0.0)) / total for c in CATEGORIES}


def top_category(probabilities: Mapping[str, float]) -> str:
    """Return the most probable category (ties: category order)."""
    return max(CATEGORIES, key=lambda c: (probabilities.get(c, 0.0),
                                          -CATEGORIES.index(c)))


class Route(_Frozen):
    """One step of a combined tagger: whom to trust for which acts.

    When ``tagger``'s most probable category is one of ``categories``
    (and, if ``keys`` is not empty, its rule key is one of them), its
    distribution is the answer; otherwise the next route is tried, and
    the last fallback answers everything else.
    """

    tagger: str = Field(min_length=1)
    categories: List[str] = []
    keys: List[str] = []


def route(
    routes: Sequence[Route],
    fallback: str,
    outputs: Mapping[str, Probabilities],
    keys: Optional[Mapping[str, str]] = None,
) -> Tuple[str, Probabilities]:
    """Return which tagger answers one sentence and its distribution.

    ``outputs`` holds every tagger's distribution for the sentence;
    ``keys`` the rule key of taggers that have one (rules).
    """
    keys = keys or {}
    for step in routes:
        out = outputs[step.tagger]
        if top_category(out) not in step.categories:
            continue
        if step.keys and keys.get(step.tagger) not in step.keys:
            continue
        return step.tagger, out
    return fallback, outputs[fallback]
