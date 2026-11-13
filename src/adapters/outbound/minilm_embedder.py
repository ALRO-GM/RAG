"""Embedding adapter: sentence-transformers model (CPU or GPU)."""

from typing import Any, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.adapters.outbound.torch_runtime import model_kwargs, prepare
from src.domain.errors import ModelError
from src.domain.runtime import RuntimePlan

FloatArray = npt.NDArray[np.float32]

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BLOCK_BATCHES = 32
"""Batches handed to the model per call (it length-sorts within a call)."""


class SentenceTransformerEmbedder:
    """Encode texts into L2-normalised vectors (cosine = dot product).

    The model is loaded lazily so that purely lexical commands never pay
    the start-up cost of the deep-learning stack.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        max_seq_length: int = 128,
        plan: Optional[RuntimePlan] = None,
    ):
        """Configure the embedder without loading the model yet."""
        self._model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.plan = plan
        self._model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the embedding model."""
        return self._model_name

    def _load(self) -> Any:
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = prepare(self.plan)
            try:
                model = SentenceTransformer(
                    self._model_name, device=device,
                    model_kwargs=model_kwargs(device),
                )
            except (OSError, ValueError) as exc:
                raise ModelError(
                    f"Cannot load embedding model {self._model_name}: {exc}"
                )
            model.max_seq_length = self.max_seq_length
            self._model = model
        return self._model

    def encode(
        self, texts: Sequence[str], progress: bool = False
    ) -> FloatArray:
        """Return one normalised embedding row per text."""
        model = self._load()
        if not texts:
            get_dim = getattr(
                model, "get_embedding_dimension",
                model.get_sentence_embedding_dimension,
            )
            dim = int(get_dim() or 0)
            return np.zeros((0, dim), dtype=np.float32)
        # sentence-transformers sorts each call's texts by length, so
        # passing large blocks (not single batches) wastes far less
        # padding; blocks only exist to report progress.
        block = self.batch_size * BLOCK_BATCHES
        batches: List[FloatArray] = []
        starts = range(0, len(texts), block)
        for start in tqdm(
            starts, desc="Embedding chunks", unit="block", disable=not progress
        ):
            batch = list(texts[start:start + block])
            vectors = model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batches.append(np.asarray(vectors, dtype=np.float32))
        return np.vstack(batches)
