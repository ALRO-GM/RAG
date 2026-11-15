"""Use cases: tag dialogue acts by retrieval over the theory cards (#94).

Two taggers behind :class:`~src.application.skills.ports.ActTagger`:

- :class:`RetrievalActTagger` (R): each view of the sentence (alone,
  with the sentence before, with two or three before) retrieves its
  best chunks from :class:`ActCardIndex`; every chunk votes only for
  the category of the position the target sits in, and the views are
  fused and calibrated (:class:`~src.domain.skills.act_retrieval.
  FusionWeights`);
- :class:`RetrievalJudgeTagger` (RG): the same evidence, split by
  position, goes into a prompt for a judge (Claude Code or a local
  model), which names up to three labels with weights and a reason;
  its weights are mixed with R's distribution.

The index compares texts position by position with an
:class:`~src.application.ports.Embedder` and, optionally, BM25
(:class:`~src.application.ports.LexicalIndexBuilder`): the hybrid of
the RAG core, with the lexical score max-scaled per query.
"""

from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.application.caching import cache_key
from src.application.ports import (
    Cache,
    Embedder,
    FloatArray,
    LexicalIndex,
    LexicalIndexBuilder,
)
from src.application.skills.ports import ChatModel
from src.domain.skills.act_rag_prompting import (
    JudgedActs,
    judged_distribution,
    parse_weighted_acts,
    rag_system_prompt,
    rag_user_prompt,
)
from src.domain.skills.act_retrieval import (
    VIEWS,
    WINDOW,
    CardChunk,
    FusionWeights,
    Hit,
    TheoryCorpus,
    View,
    ViewQuery,
    act_terms,
    aligned_scores,
    change_agreement,
    fuse_views,
    offset_weights,
    top_hits,
    view_query,
    view_votes,
)
from src.domain.skills.act_tagging import (
    ActInput,
    Probabilities,
    check_probabilities,
    top_category,
)
from src.domain.skills.dialogue_acts import CATEGORIES


Mask = npt.NDArray[np.bool_]


class RetrievalSettings(BaseModel):
    """How the index retrieves and votes (fixed before measuring).

    ``k`` chunks per view; ``target_weight`` the share of the target
    position in a pair/window score; ``speaker_weight`` the bonus for
    the same speaker changes; ``sharpness`` of the votes;
    ``lexical_weight`` the share of BM25 in each position's similarity
    (0: embeddings only).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(default=10, gt=0)
    target_weight: float = Field(default=0.6, gt=0, le=1)
    speaker_weight: float = Field(default=0.05, ge=0)
    sharpness: float = Field(default=20.0, gt=0)
    lexical_weight: float = Field(default=0.0, ge=0, le=1)


class Evidence(BaseModel):
    """What one sentence retrieved: hits and votes per view."""

    model_config = ConfigDict(frozen=True)

    hits: Dict[str, List[Hit]]
    votes: Dict[str, Probabilities]


class _ViewIndex:
    """The chunks of one view, by offset from the target."""

    def __init__(self, chunks: Sequence[CardChunk],
                 vectors: Mapping[str, int], matrix: FloatArray,
                 lexical: Optional[LexicalIndexBuilder]):
        """Stack each offset's chunk vectors (zeros where absent)."""
        self.chunks = list(chunks)
        width = WINDOW if chunks and chunks[0].view == "window" else (
            2 if chunks and chunks[0].view == "pair" else 1)
        self.width = width
        dim = matrix.shape[1]
        self.offsets: List[FloatArray] = []
        self.masks: List[Mask] = []
        self.lexical: List[LexicalIndex] = []
        for k in range(width):
            rows = np.zeros((len(chunks), dim), dtype=np.float32)
            mask = np.zeros(len(chunks), dtype=bool)
            counts: List[Dict[str, int]] = []
            for j, c in enumerate(chunks):
                if k < len(c.texts):
                    text = c.texts[-1 - k]
                    rows[j] = matrix[vectors[text]]
                    mask[j] = True
                    counts.append(dict(Counter(act_terms(text))))
                else:
                    counts.append({})
            self.offsets.append(rows)
            self.masks.append(mask)
            if lexical is not None:
                self.lexical.append(lexical.build(counts))


class ActCardIndex:
    """Chunks of the theory corpus (and optional examples), searchable."""

    def __init__(self, chunks: Sequence[CardChunk], embedder: Embedder,
                 settings: Optional[RetrievalSettings] = None,
                 lexical: Optional[LexicalIndexBuilder] = None):
        """Embed every chunk text once and build one index per view."""
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = list(chunks)
        self.embedder = embedder
        self.settings = settings or RetrievalSettings()
        if self.settings.lexical_weight > 0 and lexical is None:
            raise ValueError("lexical_weight needs a lexical index builder")
        texts = sorted({t for c in self.chunks for t in c.texts})
        self._vectors = {t: n for n, t in enumerate(texts)}
        matrix = embedder.encode(texts)
        self.views: Dict[View, _ViewIndex] = {}
        for view in VIEWS:
            mine = [c for c in self.chunks if c.view == view]
            if mine:
                self.views[view] = _ViewIndex(
                    mine, self._vectors, matrix,
                    lexical if self.settings.lexical_weight > 0 else None)

    def _similarity(self, index: _ViewIndex, k: int, texts: Sequence[str],
                    encoded: Mapping[str, FloatArray]) -> FloatArray:
        """Return queries × chunks similarity at offset ``k``."""
        q = np.stack([encoded[t] for t in texts]) if texts else np.zeros(
            (0, index.offsets[k].shape[1]), dtype=np.float32)
        dense: FloatArray = (q @ index.offsets[k].T).astype(np.float32)
        weight = self.settings.lexical_weight
        if weight <= 0 or not index.lexical:
            return dense
        lexical = np.stack([index.lexical[k].scores(act_terms(t))
                            for t in texts])
        top = lexical.max(axis=1, keepdims=True)
        scaled = np.divide(lexical, top, out=np.zeros_like(lexical),
                           where=top > 0)
        mixed: FloatArray = ((1 - weight) * dense
                             + weight * scaled).astype(np.float32)
        return mixed

    def evidence(self, items: Sequence[ActInput]) -> List[Evidence]:
        """Return each item's hits and votes for every view."""
        s = self.settings
        queries = {v: [view_query(item, v) for item in items] for v in VIEWS}
        texts = sorted({t for qs in queries.values() for q in qs if q
                        for t in q.texts})
        matrix = self.embedder.encode(texts) if texts else np.zeros((0, 1))
        encoded = {t: matrix[n] for n, t in enumerate(texts)}
        hits: Dict[View, List[List[Hit]]] = {}
        for view, index in self.views.items():
            present = [n for n, q in enumerate(queries[view]) if q]
            rows: List[List[Hit]] = [[] for _ in items]
            if present:
                found = self._view_hits(
                    index, [q for q in queries[view] if q], encoded)
                for n, h in zip(present, found):
                    rows[n] = h
            hits[view] = rows
        out: List[Evidence] = []
        for n in range(len(items)):
            per: Dict[str, List[Hit]] = {v: hits[v][n] for v in hits}
            votes: Dict[str, Probabilities] = {
                v: view_votes(hits[v][n], self.views[v].chunks, s.sharpness)
                for v in hits if hits[v][n]}
            out.append(Evidence(hits=per, votes=votes))
        return out

    def _view_hits(self, index: _ViewIndex, queries: Sequence[ViewQuery],
                   encoded: Mapping[str, FloatArray]
                   ) -> List[List[Hit]]:
        """Return the best chunks of one view for its queries."""
        s = self.settings
        sims: List[FloatArray] = []
        qmasks: List[Mask] = []
        for k in range(index.width):
            texts = [q.texts[-1 - k] if k < len(q.texts) else q.texts[-1]
                     for q in queries]
            sims.append(self._similarity(index, k, texts, encoded))
            qmasks.append(np.array([k < len(q.texts) for q in queries],
                                   dtype=bool))
        changes = (change_agreement(queries, index.chunks)
                   if index.width > 1 else None)
        scores = aligned_scores(
            sims, qmasks, index.masks, changes,
            offset_weights(s.target_weight, index.width), s.speaker_weight)
        return top_hits(scores, s.k)

    def chunk(self, view: View, hit: Hit) -> CardChunk:
        """Return the chunk a hit of ``view`` points to."""
        return self.views[view].chunks[hit.chunk]


class RetrievalActTagger:
    """Approach R: retrieval only, a calibrated vote of the views."""

    def __init__(self, index: ActCardIndex, weights: FusionWeights,
                 version: str = "theory-cards"):
        """Keep the index and the fitted fusion weights."""
        self.index = index
        self.weights = weights
        self._version = version

    @property
    def name(self) -> str:
        """Return ``rag``."""
        return "rag"

    @property
    def version(self) -> str:
        """Return the corpus version and the embedding model."""
        return f"{self._version}+{self.index.embedder.model_name}"

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return the fused distribution of each item."""
        return [check_probabilities(fuse_views(e.votes, self.weights))
                for e in self.index.evidence(items)]


class JudgeVerdict(BaseModel):
    """What the judge said about one sentence (``None``: not understood)."""

    id: str
    judged: Optional[JudgedActs]
    model: str


def candidate_categories(evidence: Evidence, index: ActCardIndex,
                         fused: Probabilities, most: int = 4) -> List[str]:
    """Return the categories whose cards go into the prompt.

    The top 3 of the fused vote, then the target category of the best
    pair and window chunk, up to ``most``.
    """
    ranked = sorted(CATEGORIES, key=lambda c: -fused.get(c, 0.0))[:3]
    for view in ("pair", "window"):
        hits = evidence.hits.get(view) or []
        if hits:
            c = index.chunk(view, hits[0]).category
            if c not in ranked:
                ranked.append(c)
    return ranked[:most]


class RetrievalJudgeTagger:
    """Approach RG: retrieved cards by position, then a judge decides.

    ``mix`` is the share of R's distribution in the output; a reply
    that names no category leaves R's distribution alone.
    """

    def __init__(self, retrieval: RetrievalActTagger, corpus: TheoryCorpus,
                 judge: ChatModel, mix: float = 0.2,
                 cache: Optional[Cache] = None, examples: int = 4,
                 max_tokens: int = 160):
        """Wire retrieval, cards, judge and the optional reply cache."""
        if not 0 <= mix <= 1:
            raise ValueError("mix must be in [0, 1]")
        self.retrieval = retrieval
        self.corpus = corpus
        self.judge = judge
        self.mix = mix
        self.cache = cache
        self.examples = examples
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        """Return ``rag-judge``."""
        return "rag-judge"

    @property
    def version(self) -> str:
        """Return the retrieval version and the judge's model."""
        return f"{self.retrieval.version}+{self.judge.model_name}"

    def prompts(self, items: Sequence[ActInput]
                ) -> List[Tuple[str, Probabilities]]:
        """Return each item's user prompt and R's distribution for it."""
        index = self.retrieval.index
        out = []
        for item, ev in zip(items, index.evidence(items)):
            fused = check_probabilities(
                fuse_views(ev.votes, self.retrieval.weights))
            cards = [self.corpus.card(c) for c in
                     candidate_categories(ev, index, fused)]
            by_view = {v: [index.chunk(v, h) for h in
                           (ev.hits.get(v) or [])[:self.examples]]
                       for v in VIEWS}
            out.append((rag_user_prompt(
                item, cards, by_view["single"], by_view["pair"],
                by_view["window"]), fused))
        return out

    def verdicts(self, items: Sequence[ActInput]
                 ) -> Tuple[List[JudgeVerdict], List[Probabilities]]:
        """Ask the judge; return its verdicts and R's distributions."""
        system = rag_system_prompt()
        verdicts: List[JudgeVerdict] = []
        fused: List[Probabilities] = []
        for item, (user, p) in zip(items, self.prompts(items)):
            key = cache_key("rag-act", self.judge.model_name, system, user)
            reply = self.cache.get(key) if self.cache else None
            if reply is None:
                reply = self.judge.complete(system, user, self.max_tokens)
                if self.cache:
                    self.cache.set(key, reply)
            verdicts.append(JudgeVerdict(
                id=item.id, judged=parse_weighted_acts(reply),
                model=self.judge.model_name))
            fused.append(p)
        return verdicts, fused

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return the judge's weights mixed with R's distribution."""
        verdicts, fused = self.verdicts(items)
        return [mix_judged(v.judged, p, self.mix)
                for v, p in zip(verdicts, fused)]


def mix_judged(judged: Optional[JudgedActs], retrieval: Probabilities,
               mix: float) -> Probabilities:
    """Return ``(1 − mix) · judge + mix · retrieval`` (retrieval alone if
    the judge said nothing usable)."""
    if judged is None:
        return check_probabilities(retrieval)
    j = judged_distribution(judged)
    return check_probabilities({c: (1 - mix) * j[c] + mix * retrieval.get(
        c, 0.0) for c in CATEGORIES})


def answerer_as_question(
    previous: Sequence[Optional[str]], gold: Sequence[str],
    changed: Sequence[bool], predictions: Sequence[Probabilities],
) -> Tuple[int, int]:
    """Count the error the owner fears: an answerer labelled pregunta.

    Among sentences said right after a ``pregunta`` of another speaker
    whose own primary label is not ``pregunta``, return how many were
    predicted ``pregunta`` and how many there were.
    """
    wrong = total = 0
    for prev, g, c, p in zip(previous, gold, changed, predictions):
        if prev == "pregunta" and c and g != "pregunta":
            total += 1
            wrong += top_category(p) == "pregunta"
    return wrong, total
