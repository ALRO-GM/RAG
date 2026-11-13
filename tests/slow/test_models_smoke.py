"""Smoke tests with the real models (run with ``pytest -m slow``)."""

import os
from pathlib import Path

import numpy as np
import pytest

from src.adapters.outbound.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.minilm_embedder import (
    SentenceTransformerEmbedder,
)
from src.adapters.outbound.transformers_generator import (
    TransformersGenerator,
)
from src.application.indexing import IndexingService
from src.application.search import SearchService
from src.domain.errors import ModelError
from src.domain.prompting import NO_ANSWER
from src.domain.text_analysis import tokenize
from tests.fakes import make_chunk
from tests.integration.conftest import real_indexing
from tests.samples import MD_PATH, TXT_PATH

pytestmark = pytest.mark.slow

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """Return the MiniLM embedder (loaded once per module)."""
    return SentenceTransformerEmbedder()


def test_embedder_shape_and_norm(
    embedder: SentenceTransformerEmbedder
) -> None:
    """Vectors are 384-d, float32 and L2-normalised."""
    vectors = embedder.encode(["load a LoRA adapter", "FP8 quantization"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0,
                               rtol=1e-4)
    assert embedder.encode([]).shape == (0, 384)


def test_embedder_similarity(embedder: SentenceTransformerEmbedder) -> None:
    """A paraphrase is closer than an unrelated sentence."""
    q, para, other = embedder.encode([
        "How do I reduce GPU memory with quantization?",
        "FP8 quantization lowers memory usage.",
        "The cat sat on the mat.",
    ])
    assert float(q @ para) > float(q @ other)


def test_unknown_embedding_model() -> None:
    """A model that cannot be loaded raises ModelError."""
    with pytest.raises(ModelError):
        SentenceTransformerEmbedder("no-such-org/no-such-model").encode(["x"])


def test_semantic_index_and_hybrid_search(
    project: Path, embedder: SentenceTransformerEmbedder
) -> None:
    """Index with embeddings; auto mode becomes hybrid and finds docs."""
    service: IndexingService = real_indexing()
    service.embedder = embedder
    report = service.run(max_chunk_size=300)
    snapshot = FileSystemIndexRepository("data/processed").load()
    assert snapshot.embeddings is not None
    assert snapshot.embeddings.shape == (report.chunks, 384)
    search = SearchService(FileSystemIndexRepository("data/processed"),
                           tokenize, embedder)
    assert search.mode == "hybrid"
    hits = search.search("draft model proposes tokens", 2)
    assert hits[0].chunk.file_path == TXT_PATH
    semantic = SearchService(FileSystemIndexRepository("data/processed"),
                             tokenize, embedder, mode="semantic")
    assert semantic.search("reduce memory usage", 1)[0].chunk.file_path == (
        MD_PATH
    )


def test_generator_answers_from_context() -> None:
    """Qwen3 answers a trivial grounded question."""
    generator = TransformersGenerator(max_new_tokens=32)
    chunk = make_chunk(
        "docs/serving.md", 0,
        "The vLLM server listens on port 8000 by default.",
        context="Serving",
    )
    answer = generator.generate("Which port does the server use?", [chunk])
    assert answer and "<think>" not in answer
    assert "8000" in answer
    assert generator.generate("anything", []) == NO_ANSWER


def test_cross_encoder_prefers_relevant_passage() -> None:
    """The real cross-encoder scores the answering passage higher."""
    reranker = CrossEncoderReranker()
    query = "How do I enable FP8 quantization?"
    relevant, other = reranker.score([
        (query, "Pass quantization='fp8' to enable FP8 quantization."),
        (query, "Speculative decoding uses a small draft model."),
    ])
    assert relevant > other


def test_unknown_rerank_model() -> None:
    """A missing reranking model is reported as a ModelError."""
    with pytest.raises(ModelError):
        CrossEncoderReranker("no/such-reranker-model").score([("q", "p")])
