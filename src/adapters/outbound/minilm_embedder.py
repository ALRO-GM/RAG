"""Embedding adapter: sentence-transformers model (CPU or GPU).

On a GPU that runs out of memory (loading or encoding) the model moves
to the CPU for the rest of its life and says so in
:attr:`SentenceTransformerEmbedder.device_notes` (#110).
"""

import logging
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.adapters.outbound.gpu_fallback import (
    cpu_fallback_note,
    free_cuda_memory,
    is_cuda_oom,
)
from src.adapters.outbound.torch_runtime import model_kwargs, prepare
from src.domain.errors import ModelError
from src.domain.runtime import RuntimePlan

LOG = logging.getLogger(__name__)

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
        load: Optional[Callable[[str], Any]] = None,
        is_oom: Callable[[BaseException], bool] = is_cuda_oom,
    ):
        """Configure the embedder without loading the model yet.

        ``load`` builds the model on a device and ``is_oom`` recognises
        running out of GPU memory (both replaced in tests).
        """
        self._model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.plan = plan
        self._model: Optional[Any] = None
        self._loader = load or self._sentence_transformer
        self._is_oom = is_oom
        self._device = ""
        self.device_notes: List[str] = []

    @property
    def model_name(self) -> str:
        """Return the identifier of the embedding model."""
        return self._model_name

    def _sentence_transformer(self, device: str) -> Any:
        """Build the sentence-transformers model on ``device``."""
        from sentence_transformers import SentenceTransformer

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
        return model

    def _load(self) -> Any:
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            planned = prepare(self.plan)
            self._device = self._device or planned
            try:
                self._model = self._loader(self._device)
            except Exception as exc:
                if not self._out_of_gpu_memory(exc):
                    raise
                return self._to_cpu()
        return self._model

    def _out_of_gpu_memory(self, exc: BaseException) -> bool:
        """Tell whether ``exc`` is a GPU running out of memory."""
        return self._device != "cpu" and self._is_oom(exc)

    def _to_cpu(self) -> Any:
        """Free the GPU copy and load the model on the CPU."""
        LOG.warning("CUDA out of memory: the embeddings move to the CPU.")
        self._model = None
        free_cuda_memory()
        self._device = "cpu"
        self.device_notes.append(cpu_fallback_note(
            f"the embeddings ({self._model_name})"))
        self._model = self._loader("cpu")
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
            try:
                vectors = self._encode(model, batch)
            except Exception as exc:
                if not self._out_of_gpu_memory(exc):
                    raise
                model = self._to_cpu()
                vectors = self._encode(model, batch)
            batches.append(np.asarray(vectors, dtype=np.float32))
        return np.vstack(batches)

    def _encode(self, model: Any, batch: List[str]) -> Any:
        """Return the normalised vectors of one block."""
        return model.encode(
            batch,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
