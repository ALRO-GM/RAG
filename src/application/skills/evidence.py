"""Use case: find the turns of a participant that match a competency."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from src.application.ports import (
    Embedder,
    FloatArray,
    LexicalIndex,
    LexicalIndexBuilder,
)
from src.domain.languages import analyzer_for, detect_language
from src.domain.ranking import linear_scores, top_k
from src.domain.skills.rubric import Competency, Indicator
from src.domain.skills.transcript import Transcript, Turn


class Candidate(BaseModel):
    """A turn retrieved for a competency, with its best indicator."""

    turn: int
    score: float
    indicator_id: str


@dataclass
class TranscriptIndex:
    """A transcript prepared for evidence retrieval."""

    transcript: Transcript
    language: str
    lexical: LexicalIndex
    embeddings: Optional[FloatArray]
    eligible: npt.NDArray[np.bool_]
    by_speaker: Dict[str, npt.NDArray[np.bool_]] = field(
        default_factory=dict
    )

    def mask(self, speaker: str) -> npt.NDArray[np.bool_]:
        """Return the eligible turns of ``speaker`` as a boolean mask."""
        if speaker not in self.by_speaker:
            own = np.asarray(
                [t.speaker == speaker for t in self.transcript.turns],
                dtype=np.bool_,
            )
            self.by_speaker[speaker] = own & self.eligible
        return self.by_speaker[speaker]


class EvidenceRetriever:
    """Rank the turns of a transcript against rubric indicators.

    Each indicator (its text plus example utterances) is a query. A
    turn is scored by ``bm25 / max + semantic_weight * cosine``, the
    same fusion as the document search, and keeps its best indicator.
    Turns shorter than ``min_words`` are never evidence on their own
    ("yes", "ok"). When the rubric and the transcript are in different
    languages only the dense part contributes, which is why the
    embedding model must be multilingual.
    """

    def __init__(
        self,
        lexical_builder: LexicalIndexBuilder,
        embedder: Optional[Embedder] = None,
        semantic_weight: float = 1.0,
        min_words: int = 4,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        """Wire the lexical builder and the optional embedder."""
        self.lexical_builder = lexical_builder
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.min_words = min_words
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._queries: Dict[str, FloatArray] = {}

    def language_of(self, transcript: Transcript) -> str:
        """Return the declared or detected language of a transcript."""
        return (transcript.language or detect_language(transcript.text()))[:2]

    def index(self, transcript: Transcript) -> TranscriptIndex:
        """Analyse and embed every turn of ``transcript``."""
        language = self.language_of(transcript)
        analyzer = analyzer_for(language)
        counts = [dict(Counter(analyzer(t.text))) for t in transcript.turns]
        embeddings: Optional[FloatArray] = None
        if self.embedder is not None:
            embeddings = self.embedder.encode(
                [self.passage_prefix + t.text for t in transcript.turns]
            )
        eligible = np.asarray(
            [t.n_words() >= self.min_words for t in transcript.turns],
            dtype=np.bool_,
        )
        return TranscriptIndex(
            transcript, language, self.lexical_builder.build(counts),
            embeddings, eligible,
        )

    def _query_vectors(self, queries: List[str]) -> FloatArray:
        """Embed indicator queries once per retriever."""
        assert self.embedder is not None
        missing = [q for q in queries if q not in self._queries]
        if missing:
            vectors = self.embedder.encode(
                [self.query_prefix + q for q in missing]
            )
            for query, vector in zip(missing, vectors):
                self._queries[query] = vector
        return np.vstack([self._queries[q] for q in queries])

    def candidates(
        self,
        index: TranscriptIndex,
        participant: str,
        competency: Competency,
        k: int,
        rubric_language: Optional[str] = None,
    ) -> List[Candidate]:
        """Return the ``k`` turns of ``participant`` closest to the rubric."""
        mask = index.mask(participant)
        if k <= 0 or not mask.any():
            return []
        analyzer = analyzer_for(rubric_language or index.language)
        queries = [i.query() for i in competency.indicators]
        dense: Optional[FloatArray] = None
        if self.embedder is not None and index.embeddings is not None:
            dense = index.embeddings @ self._query_vectors(queries).T
        n_turns = len(index.transcript.turns)
        best: FloatArray = np.full(n_turns, -np.inf, dtype=np.float32)
        which: npt.NDArray[np.int64] = np.zeros(n_turns, dtype=np.int64)
        for column, query in enumerate(queries):
            lexical = index.lexical.scores(analyzer(query))
            scores = lexical if dense is None else linear_scores(
                lexical, dense[:, column], self.semantic_weight
            )
            better = scores > best
            best = np.where(better, scores, best).astype(np.float32)
            which = np.where(better, column, which)
        best = np.where(mask, best, -np.inf).astype(np.float32)
        return [
            Candidate(
                turn=position,
                score=float(best[position]),
                indicator_id=competency.indicators[int(which[position])].id,
            )
            for position in top_k(best, min(k, int(mask.sum())))
        ]

    def similarity(
        self, index: TranscriptIndex, turn: int, indicator: Indicator
    ) -> Optional[float]:
        """Return the cosine between a turn and an indicator.

        ``None`` without an embedding model: there is then no second
        opinion on the judge's classification.
        """
        if self.embedder is None or index.embeddings is None:
            return None
        vector = self._query_vectors([indicator.query()])[0]
        return float(index.embeddings[turn] @ vector)

    def turns(
        self, index: TranscriptIndex, candidates: List[Candidate]
    ) -> List[Turn]:
        """Return the turns named by ``candidates``."""
        return [index.transcript.turns[c.turn] for c in candidates]
