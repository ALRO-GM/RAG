"""Label dialogue acts by retrieval over a theory corpus (#94).

The corpus (``knowledge/dialogue_acts/theory``) has two kinds of card:

- a :class:`CategoryCard` per category: definition, what it does in the
  conversation, cues, what it is *not* (with the nearest neighbours),
  boundary cases and examples;
- a :class:`SequenceCard` per pattern of two or more sentences
  (pregunta → respuesta, propuesta → rechazo → propuesta...), each
  position with its category and speaker.

Both are cut into :class:`CardChunk` objects for three **views** of a
sentence: ``single`` (the sentence alone), ``pair`` (the sentence
before it, then the sentence) and ``window`` (two or three sentences
before it, then the sentence). A chunk keeps one text per position,
oldest first, and the category of **its last position only**: that is
the one it votes for, so what describes an earlier sentence (the
question an answer follows) never votes for the target. The categories
of earlier positions are kept in ``context`` to be shown to a judge.

Matching is position by position (:func:`aligned_scores`): the target
is compared with each chunk's last text, the previous sentence with the
chunk's previous text, and so on, so a chunk only scores high when the
target itself looks like the position it describes. Votes of the best
chunks of each view become a distribution (:func:`view_votes`) and the
views are fused (:func:`fuse_views`) before calibration.
"""

import math
import re
import unicodedata
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.skills.act_tagging import ActInput, Probabilities
from src.domain.skills.dialogue_acts import CATEGORIES, ActRecord

FloatArray = npt.NDArray[np.float32]

View = Literal["single", "pair", "window"]
VIEWS: Tuple[View, ...] = ("single", "pair", "window")
"""The three views of a sentence, from no context to the most."""

WINDOW = 4
"""Most positions of a window chunk or query (3 before + the target)."""

ChunkKind = Literal["definition", "cue", "boundary", "example",
                    "sequence", "dialogue"]


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _known(category: str) -> str:
    """Return ``category`` if it is one of the 13, else raise."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    return category


class Contrast(_Frozen):
    """What a category is not: the neighbour, the rule and its examples.

    ``examples`` belong to the neighbour ``category``.
    """

    category: str
    contrast: str = Field(min_length=1)
    examples: List[str] = []

    @model_validator(mode="after")
    def _check(self) -> "Contrast":
        """Refuse an unknown neighbour."""
        _known(self.category)
        return self


class CategoryCard(_Frozen):
    """The theory card of one category."""

    model_config = ConfigDict(extra="forbid", frozen=True,
                              populate_by_name=True)

    category: str
    definition: str = Field(min_length=1)
    function: str = Field(min_length=1)
    cues: List[str] = []
    not_: List[Contrast] = Field(default=[], alias="not")
    boundary: List[str] = []
    examples: List[str] = []

    @model_validator(mode="after")
    def _check(self) -> "CategoryCard":
        """Refuse an unknown category or a contrast with itself."""
        _known(self.category)
        if any(c.category == self.category for c in self.not_):
            raise ValueError(f"{self.category}: contrast with itself")
        return self


class SequencePosition(_Frozen):
    """One position of a sequence: its category, speaker and role."""

    category: str
    speaker: str = Field(min_length=1)
    role: str = ""

    @model_validator(mode="after")
    def _check(self) -> "SequencePosition":
        """Refuse an unknown category."""
        _known(self.category)
        return self


class SequenceCard(_Frozen):
    """A pattern of two or more sentences, each position tagged."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    positions: List[SequencePosition] = Field(min_length=2)
    examples: List[List[str]] = []

    @model_validator(mode="after")
    def _check(self) -> "SequenceCard":
        """Every example has one text per position."""
        for example in self.examples:
            if len(example) != len(self.positions):
                raise ValueError(
                    f"{self.id}: an example has {len(example)} texts for "
                    f"{len(self.positions)} positions")
        return self


class TheoryCorpus(_Frozen):
    """Every card: one per category (all 13) and the sequences."""

    cards: List[CategoryCard]
    sequences: List[SequenceCard] = []

    @model_validator(mode="after")
    def _check(self) -> "TheoryCorpus":
        """One card per category, every category covered, unique ids."""
        names = [c.category for c in self.cards]
        if sorted(names) != sorted(CATEGORIES):
            raise ValueError("need exactly one card per category")
        ids = [s.id for s in self.sequences]
        if len(ids) != len(set(ids)):
            raise ValueError("sequence ids must be unique")
        return self

    def card(self, category: str) -> CategoryCard:
        """Return the card of ``category``."""
        for c in self.cards:
            if c.category == category:
                return c
        raise KeyError(category)

    def example_texts(self) -> List[str]:
        """Return every example sentence of every card and sequence."""
        texts = [e for c in self.cards for e in c.examples]
        texts += [e for c in self.cards for n in c.not_ for e in n.examples]
        texts += [t for s in self.sequences for e in s.examples for t in e]
        return texts


class CardChunk(_Frozen):
    """One retrievable piece, tied to the position it describes.

    ``texts`` holds one text per position, oldest first; the last is the
    position the chunk describes and ``category`` is its category, the
    only one the chunk votes for. ``changes`` tells, for every position
    after the first, whether the speaker changed before it. ``context``
    holds the categories of the earlier positions (shown, never voted).
    """

    id: str = Field(min_length=1)
    view: View
    kind: ChunkKind
    texts: List[str] = Field(min_length=1, max_length=WINDOW)
    changes: List[bool] = []
    category: str
    context: List[str] = []
    source: str = ""

    @model_validator(mode="after")
    def _check(self) -> "CardChunk":
        """Positions, speaker changes and context categories agree."""
        _known(self.category)
        for c in self.context:
            _known(c)
        n = len(self.texts)
        if len(self.changes) != n - 1 or len(self.context) != n - 1:
            raise ValueError(f"{self.id}: changes/context must have "
                             f"{n - 1} entries")
        expected = 1 if self.view == "single" else 2 if (
            self.view == "pair") else None
        if expected is not None and n != expected:
            raise ValueError(f"{self.id}: a {self.view} chunk has "
                             f"{expected} position(s)")
        if self.view == "window" and n < 3:
            raise ValueError(f"{self.id}: a window chunk has 3+ positions")
        return self


def _single(cid: str, kind: ChunkKind, text: str, category: str,
            source: str) -> CardChunk:
    """Return a one-position chunk."""
    return CardChunk(id=cid, view="single", kind=kind, texts=[text],
                     category=category, source=source)


def card_chunks(card: CategoryCard, theory: bool = True) -> List[CardChunk]:
    """Return the single-view chunks of a category card.

    With ``theory`` the definition, cues and boundary cases are chunks
    too (they describe the category in words); examples always are,
    and the examples of a contrast vote for the neighbour they belong
    to.
    """
    c = card.category
    chunks: List[CardChunk] = []
    if theory:
        chunks.append(_single(f"{c}/def", "definition",
                              f"{card.definition} {card.function}", c, c))
        chunks += [_single(f"{c}/cue{n}", "cue", cue, c, c)
                   for n, cue in enumerate(card.cues)]
        chunks += [_single(f"{c}/edge{n}", "boundary", b, c, c)
                   for n, b in enumerate(card.boundary)]
    chunks += [_single(f"{c}/ex{n}", "example", e, c, c)
               for n, e in enumerate(card.examples)]
    for m, note in enumerate(card.not_):
        chunks += [_single(f"{c}/not{m}.{n}", "example", e, note.category, c)
                   for n, e in enumerate(note.examples)]
    return chunks


def _windowed(cid: str, kind: ChunkKind, texts: Sequence[str],
              speakers: Sequence[str], categories: Sequence[str],
              source: str) -> List[CardChunk]:
    """Return the pair and window chunks ending at every position ≥ 1."""
    chunks: List[CardChunk] = []
    spans: Tuple[Tuple[View, int], ...] = (("pair", 2), ("window", WINDOW))
    for end in range(1, len(texts)):
        for view, size in spans:
            start = max(0, end + 1 - size)
            if view == "window" and end - start < 2:
                continue
            span = range(start, end + 1)
            chunks.append(CardChunk(
                id=f"{cid}@{end}/{view}", view=view, kind=kind,
                texts=[texts[i] for i in span],
                changes=[speakers[i] != speakers[i - 1]
                         for i in span if i > start],
                category=categories[end],
                context=[categories[i] for i in span if i < end],
                source=source))
    return chunks


def sequence_chunks(sequence: SequenceCard) -> List[CardChunk]:
    """Return the pair and window chunks of a sequence card's examples."""
    speakers = [p.speaker for p in sequence.positions]
    categories = [p.category for p in sequence.positions]
    chunks: List[CardChunk] = []
    for n, example in enumerate(sequence.examples):
        chunks += _windowed(f"{sequence.id}/{n}", "sequence", example,
                            speakers, categories, sequence.id)
    return chunks


def corpus_chunks(corpus: TheoryCorpus, theory: bool = True
                  ) -> List[CardChunk]:
    """Return every chunk of the theory corpus (all views)."""
    chunks = [c for card in corpus.cards for c in card_chunks(card, theory)]
    for sequence in corpus.sequences:
        chunks += sequence_chunks(sequence)
    return chunks


def dialogue_chunks(records: Sequence[ActRecord]) -> List[CardChunk]:
    """Return chunks of labelled dialogues (all views), by primary label.

    Used to add the training folds of the Spanish set as examples: the
    caller must never pass the dialogues being tested.
    """
    by_dialogue: Dict[str, List[ActRecord]] = {}
    for r in records:
        if r.categories:
            by_dialogue.setdefault(r.dialogue, []).append(r)
    chunks: List[CardChunk] = []
    for dialogue, rows in sorted(by_dialogue.items()):
        rows = sorted(rows, key=lambda r: r.order)
        texts = [r.text for r in rows]
        speakers = [r.speaker for r in rows]
        categories = [_primary(r) for r in rows]
        chunks += [_single(f"{dialogue}/{r.id}", "dialogue", r.text,
                           categories[n], dialogue)
                   for n, r in enumerate(rows)]
        chunks += _windowed(dialogue, "dialogue", texts, speakers,
                            categories, dialogue)
    return chunks


def _primary(record: ActRecord) -> str:
    """Return a record's primary category (the heaviest label)."""
    if record.labels:
        return max(record.labels, key=lambda lab: lab.weight).label
    return record.categories[0]


class ViewQuery(_Frozen):
    """What one view of a sentence compares with the chunks.

    ``texts`` oldest first, the target last; ``changes`` as in
    :class:`CardChunk`.
    """

    view: View
    texts: List[str] = Field(min_length=1)
    changes: List[bool] = []


def view_query(item: ActInput, view: View) -> Optional[ViewQuery]:
    """Return the query of ``item`` for ``view`` (``None``: no context).

    ``pair`` needs one sentence before, ``window`` two.
    """
    size = {"single": 1, "pair": 2, "window": WINDOW}[view]
    before = item.context[-(size - 1):] if size > 1 else []
    if view == "pair" and len(before) < 1:
        return None
    if view == "window" and len(before) < 2:
        return None
    speakers = [c.speaker for c in before] + [item.speaker]
    return ViewQuery(
        view=view, texts=[c.text for c in before] + [item.text],
        changes=[speakers[i] != speakers[i - 1]
                 for i in range(1, len(speakers))])


_WORD = re.compile(r"\w+")


def act_terms(text: str) -> List[str]:
    """Return the lexical terms of a sentence for BM25.

    Lower case without accents, every word kept (in dialogue acts the
    short words are the cues: «vale», «sí», «no», «yo»), plus ``?`` when
    the sentence has a question mark.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    terms = _WORD.findall(folded)
    if "?" in text or "¿" in text:
        terms.append("?")
    return terms


def offset_weights(target: float, positions: int) -> List[float]:
    """Return the weight of each offset (0 = target) for a view.

    The target takes ``target``; the earlier positions share the rest
    equally. One position: ``[1.0]``.
    """
    if not 0 < target <= 1:
        raise ValueError("target weight must be in (0, 1]")
    if positions <= 1:
        return [1.0]
    rest = (1.0 - target) / (positions - 1)
    return [target] + [rest] * (positions - 1)


def aligned_scores(
    similarity: Sequence[FloatArray],
    query_mask: Sequence[npt.NDArray[np.bool_]],
    chunk_mask: Sequence[npt.NDArray[np.bool_]],
    change_match: Optional[FloatArray],
    weights: Sequence[float],
    speaker_weight: float = 0.0,
) -> FloatArray:
    """Return the position-aligned score of every (query, chunk).

    ``similarity[k]`` is the queries × chunks similarity at offset
    ``k`` (0 = the target, 1 = the sentence before...);
    ``query_mask[k]`` / ``chunk_mask[k]`` tell which queries and chunks
    have that offset. The score is the weighted mean over the offsets
    both have (always at least the target), plus ``speaker_weight``
    times ``change_match`` (the share of aligned speaker changes that
    agree; queries × chunks).
    """
    if not similarity:
        raise ValueError("need the target offset")
    total = weights[0] * similarity[0].astype(np.float64)
    norm = np.full(total.shape, weights[0], dtype=np.float64)
    for k in range(1, len(similarity)):
        both = np.outer(query_mask[k], chunk_mask[k]).astype(np.float64)
        total += weights[k] * similarity[k] * both
        norm += weights[k] * both
    out = total / norm
    if change_match is not None and speaker_weight:
        out = out + speaker_weight * change_match
    result: FloatArray = out.astype(np.float32)
    return result


def change_agreement(queries: Sequence[ViewQuery],
                     chunks: Sequence[CardChunk]) -> FloatArray:
    """Return the share of aligned speaker changes that agree.

    Aligned from the target backwards; 1.0 when neither has a change to
    compare (single view).
    """
    out = np.ones((len(queries), len(chunks)), dtype=np.float32)
    for i, q in enumerate(queries):
        for j, c in enumerate(chunks):
            n = min(len(q.changes), len(c.changes))
            if n:
                agree = sum(a == b for a, b in
                            zip(q.changes[-n:], c.changes[-n:]))
                out[i, j] = agree / n
    return out


class Hit(_Frozen):
    """A retrieved chunk and its score for one query."""

    chunk: int = Field(ge=0)
    score: float


def top_hits(scores: FloatArray, k: int) -> List[List[Hit]]:
    """Return the ``k`` best chunks of every query row, best first."""
    if k <= 0:
        raise ValueError("k must be > 0")
    rows: List[List[Hit]] = []
    k = min(k, scores.shape[1]) if scores.ndim == 2 else 0
    for row in scores:
        if not k:
            rows.append([])
            continue
        part = np.argpartition(-row, k - 1)[:k]
        order = part[np.argsort(-row[part], kind="stable")]
        rows.append([Hit(chunk=int(j), score=float(row[j])) for j in order])
    return rows


def view_votes(hits: Sequence[Hit], chunks: Sequence[CardChunk],
               sharpness: float = 20.0) -> Probabilities:
    """Return the category distribution of one view's hits.

    Each hit votes ``exp(sharpness · (score − best))`` for its chunk's
    category (the last position's), so near-ties share the vote and
    distant hits count little. Empty hits: an empty dict.
    """
    if not hits:
        return {}
    best = max(h.score for h in hits)
    votes = {c: 0.0 for c in CATEGORIES}
    for h in hits:
        votes[chunks[h.chunk].category] += math.exp(
            sharpness * (h.score - best))
    total = sum(votes.values())
    return {c: v / total for c, v in votes.items()}


class FusionWeights(_Frozen):
    """How the three views are fused and calibrated.

    ``views`` weighs each view's distribution; a view without a query
    (no context) is left out and the rest renormalised; when no
    weighed view has votes, the single view answers. ``floor`` mixes
    in the uniform distribution before the logarithm and
    ``temperature`` divides the log-probabilities.
    """

    views: Dict[str, float] = {"single": 1.0, "pair": 0.0, "window": 0.0}
    floor: float = Field(default=0.02, gt=0, lt=1)
    temperature: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> "FusionWeights":
        """Known views, non-negative weights, at least one weighed."""
        for v, w in self.views.items():
            if v not in VIEWS or w < 0:
                raise ValueError(f"bad view weight {v}={w}")
        if not any(w > 0 for w in self.views.values()):
            raise ValueError("at least one view needs a weight > 0")
        return self


def fuse_views(votes: Mapping[str, Probabilities],
               weights: FusionWeights) -> Probabilities:
    """Return the calibrated fusion of one sentence's view votes.

    ``p ∝ exp(ln((1 − floor) · Σ w_v p_v / Σ w_v + floor / 13) / T)``.
    """
    mixed = {c: 0.0 for c in CATEGORIES}
    norm = 0.0
    for view, p in votes.items():
        w = weights.views.get(view, 0.0)
        if not p or w <= 0:
            continue
        norm += w
        for c in CATEGORIES:
            mixed[c] += w * p.get(c, 0.0)
    if norm <= 0:
        single = votes.get("single") or {}
        if not single:
            return {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
        mixed = {c: single.get(c, 0.0) for c in CATEGORIES}
        norm = 1.0
    floor = weights.floor / len(CATEGORIES)
    logs = {c: math.log((1 - weights.floor) * mixed[c] / norm + floor)
            / weights.temperature for c in CATEGORIES}
    top = max(logs.values())
    exps = {c: math.exp(v - top) for c, v in logs.items()}
    total = sum(exps.values())
    return {c: v / total for c, v in exps.items()}


def closest_pairs(names: Sequence[str], vectors: FloatArray,
                  top: int = 10) -> List[Tuple[str, str, float]]:
    """Return the ``top`` most similar pairs of L2-normalised vectors."""
    if len(names) != len(vectors):
        raise ValueError("one vector per name")
    sims = vectors @ vectors.T
    pairs = [(names[i], names[j], float(sims[i, j]))
             for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs.sort(key=lambda p: -p[2])
    return pairs[:top]
