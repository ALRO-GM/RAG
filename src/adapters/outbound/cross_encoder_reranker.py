"""Reranking adapter: sentence-transformers cross-encoder on CPU."""

from typing import Any, List, Optional, Sequence, Tuple

from src.adapters.outbound.torch_runtime import prepare
from src.domain.errors import ModelError
from src.domain.runtime import RuntimePlan

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Score (query, passage) pairs with a cross-encoder.

    Unlike the bi-encoder used for embeddings, the cross-encoder reads
    the query and the passage together, which is more precise but too
    slow for the whole corpus: it only re-scores a few candidates. The
    model is loaded lazily.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        max_length: int = 256,
        batch_size: int = 64,
        plan: Optional[RuntimePlan] = None,
    ):
        """Configure the reranker without loading the model yet."""
        self.plan = plan
        self._model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the reranking model."""
        return self._model_name

    def _load(self) -> Any:
        """Load the cross-encoder on first use."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = prepare(self.plan)
            try:
                self._model = CrossEncoder(
                    self._model_name,
                    max_length=self.max_length,
                    device=device,
                )
            except (OSError, ValueError) as exc:
                raise ModelError(
                    f"Cannot load reranking model {self._model_name}: {exc}"
                )
        return self._model

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return one relevance score (a logit) per pair."""
        if not pairs:
            return []
        scores = self._load().predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]
