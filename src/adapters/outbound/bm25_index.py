"""Lexical index adapter: Okapi BM25 on a precomputed sparse matrix."""

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse

FloatArray = npt.NDArray[np.float32]
# scipy-stubs model sparse shapes too narrowly for column fancy indexing,
# so sparse matrices are typed loosely.
SparseMatrix = Any


class BM25Index:
    """BM25 scoring with precomputed per-term document weights.

    The weight ``idf(t) * tf * (k1 + 1) / (tf + k1 * norm(d))`` of every
    (document, term) pair is precomputed into a sparse matrix, so scoring
    a query is a sum of a few matrix columns.
    """

    def __init__(
        self,
        weights: SparseMatrix,
        vocabulary: Dict[str, int],
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """Wrap a precomputed weight matrix (documents x terms)."""
        self.weights = weights
        self.vocabulary = vocabulary
        self.k1 = k1
        self.b = b

    @classmethod
    def build(
        cls,
        documents: Sequence[Mapping[str, int]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "BM25Index":
        """Build the index from per-document term counts."""
        vocabulary: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        for doc_id, counts in enumerate(documents):
            for term, count in counts.items():
                term_id = vocabulary.setdefault(term, len(vocabulary))
                rows.append(doc_id)
                cols.append(term_id)
                data.append(count)
        n_docs = max(len(documents), 1)
        tf: SparseMatrix = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(documents), max(len(vocabulary), 1)),
        )
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = float(doc_len.mean()) if doc_len.size else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        norm = k1 * (1.0 - b + b * doc_len / max(avg_len, 1e-9))
        row_ids = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        tf_vals = tf.data
        tf.data = (
            idf[tf.indices] * tf_vals * (k1 + 1.0)
            / (tf_vals + norm[row_ids])
        ).astype(np.float32)
        return cls(tf.tocsc(), vocabulary, k1, b)

    def scores(self, query_tokens: Sequence[str]) -> FloatArray:
        """Return the BM25 score of every document for the query."""
        counts = Counter(t for t in query_tokens if t in self.vocabulary)
        n_docs = self.weights.shape[0]
        if not counts:
            return np.zeros(n_docs, dtype=np.float32)
        ids = [self.vocabulary[t] for t in counts]
        qtf = np.asarray([counts[t] for t in counts], dtype=np.float32)
        sub = self.weights[:, ids]
        return np.asarray(sub @ qtf, dtype=np.float32).ravel()

    def state(self) -> Tuple[SparseMatrix, Dict[str, int]]:
        """Return what must be persisted to rebuild this index."""
        return self.weights, self.vocabulary


class BM25Builder:
    """Build :class:`BM25Index` instances with fixed parameters."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Store the BM25 term-saturation and length-normalisation knobs."""
        self.k1 = k1
        self.b = b

    def build(self, term_counts: Sequence[Mapping[str, int]]) -> BM25Index:
        """Return a BM25 index over the given documents."""
        return BM25Index.build(term_counts, self.k1, self.b)
