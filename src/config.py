"""Settings: every tunable of the system in one validated object."""

from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.adapters.outbound.cross_encoder_reranker import (
    DEFAULT_RERANK_MODEL,
)
from src.adapters.outbound.fs_corpus import DEFAULT_EXTENSIONS

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "Qwen/Qwen3-0.6B"


class Settings(BaseModel):
    """Configuration consumed by the composition root.

    The retrieval defaults (field weights, ``bm25_b``, routing, test
    prior, file boost, table context) are the consensus profile found
    by the experiment lab with 5-fold cross-validation; see
    ``experiments/REPORT.md``.
    """

    raw_dir: str = "data/raw"
    index_dir: str = "data/processed"
    extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS
    max_chunk_size: int = 2000
    target_chunk_size: Optional[int] = None
    semantic: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_max_tokens: int = Field(default=64, ge=8)
    mode: str = "auto"
    semantic_weight: float = 0.4
    fusion: str = "linear"
    route_strength: float = Field(default=0.1, ge=0.0)
    router_path: str = "models/query_router.json"
    test_weight: float = Field(default=0.6, ge=0.0)
    file_boost: float = Field(default=0.2, ge=0.0)
    query_prefix: str = ""
    passage_prefix: str = ""
    path_weight: int = Field(default=3, ge=0)
    context_weight: int = Field(default=3, ge=0)
    table_context: bool = True
    bigrams: bool = False
    embed_style: Literal["full", "body"] = "full"
    embedding_cache_dir: Optional[str] = None
    rerank: bool = False
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_depth: int = Field(default=10, ge=1)
    rerank_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rerank_max_tokens: int = Field(default=256, ge=16)
    bm25_k1: float = 1.2
    bm25_b: float = 0.5
    llm: str = DEFAULT_LLM
    context_tokens: int = Field(default=2000, ge=64)
    max_new_tokens: int = Field(default=256, ge=1)
    use_cache: bool = True
    workers: Optional[int] = None
    progress: bool = True
    lab_dir: str = "experiments"
    lab_work_dir: str = "data/experiments"
    lab_cold: bool = False
    device: str = "auto"
    torch_threads: Optional[int] = Field(default=None, ge=1)
    cpu_cores: Literal["auto", "all", "performance"] = "auto"
    runtime_plan_path: str = "data/runtime_plan.json"
    runtime_history_path: str = "experiments/hardware/runtime.jsonl"
    benchmark_dir: str = "data/benchmarks"


BGE_SMALL: Dict[str, Any] = {
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "embedding_max_tokens": 512,
    "embed_style": "body",
    "query_prefix": (
        "Represent this sentence for searching relevant passages: "
    ),
}
"""A stronger embedding model of the same size class (33M parameters)."""

BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "system as shipped (hybrid)": {},
    "BM25 only, as shipped": {"mode": "bm25"},
    "BM25 only, plain (no field weights, k1 0.9, b 0.4)": {
        "mode": "bm25", "path_weight": 0, "context_weight": 1,
        "bm25_k1": 0.9, "bm25_b": 0.4,
    },
    "embeddings only, 64 tokens (as shipped)": {"mode": "semantic"},
    "embeddings only, 256 tokens": {
        "mode": "semantic", "embedding_max_tokens": 256,
    },
    "hybrid, 256 tokens": {"embedding_max_tokens": 256},
    "hybrid, 256 tokens, semantic weight 1.0": {
        "embedding_max_tokens": 256, "semantic_weight": 1.0,
    },
    "hybrid, 256 tokens, semantic weight 2.0": {
        "embedding_max_tokens": 256, "semantic_weight": 2.0,
    },
    "hybrid, 256 tokens, RRF fusion": {
        "embedding_max_tokens": 256, "fusion": "rrf",
    },
    "hybrid, 256 tokens, weight 2.0 + reranker": {
        "embedding_max_tokens": 256, "semantic_weight": 2.0,
        "rerank": True,
    },
    "embeddings only, 256 tokens, path not embedded": {
        "mode": "semantic", "embedding_max_tokens": 256,
        "embed_style": "body",
    },
    "bge-small embeddings only, 512 tokens": {
        "mode": "semantic", **BGE_SMALL,
    },
    "bge-small hybrid, weight 1.0": {**BGE_SMALL, "semantic_weight": 1.0},
    "bge-small hybrid, weight 2.0": {**BGE_SMALL, "semantic_weight": 2.0},
    "bge-small 256 tokens, embeddings only": {
        "mode": "semantic", **BGE_SMALL, "embedding_max_tokens": 256,
    },
    "bge-small 256 tokens, hybrid weight 2.0": {
        **BGE_SMALL, "embedding_max_tokens": 256, "semantic_weight": 2.0,
    },
    "hybrid, 256 tokens + reranker": {
        "embedding_max_tokens": 256, "rerank": True,
    },
}
"""Configurations compared on public benchmarks."""


INDEX_FIELDS = (
    "raw_dir", "extensions", "max_chunk_size", "target_chunk_size",
    "semantic", "embedding_model", "embedding_max_tokens", "embed_style",
    "passage_prefix",
    "bm25_k1", "bm25_b", "path_weight", "context_weight", "table_context",
    "bigrams",
)
"""Settings that change the index; the others only change the search."""


DEFAULT_SKILLS_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_SKILLS_LLM = "auto"
SKILLS_MODEL_LADDER: Tuple[Tuple[str, float], ...] = (
    ("Qwen/Qwen3-0.6B", 0.6),
    ("Qwen/Qwen3-1.7B", 1.7),
    ("Qwen/Qwen3-4B", 4.0),
    ("Qwen/Qwen3-8B", 8.2),
    ("Qwen/Qwen3-14B", 14.8),
)
"""Local judge candidates: (Hugging Face model, billions of parameters)."""
DEFAULT_SKILLS_API_MODEL = "claude-opus-5-5"


class SkillsSettings(BaseModel):
    """Configuration of the soft-skills analyzer.

    ``rubric_language`` is the language of the rubric matched against
    the transcript: ``auto`` picks the transcript's own language and
    falls back to English when the catalogue is not translated.
    ``judge`` chooses where the language model runs: ``local`` keeps
    the transcript on the machine, ``api`` sends it to Anthropic.
    With ``llm = "auto"`` the local judge is the largest model of
    ``model_ladder`` that fits this machine's memory and, on a CPU,
    is estimated to reach ``min_tokens_per_second``.

    A rating needs ``min_incidents`` verified turns from
    ``min_episodes`` separate exchanges (turns within ``episode_gap``
    of each other are one exchange); a participant with fewer than
    ``min_participant_words`` words is not rated at all. These rules
    follow ``docs/skills/THEORY.md`` §4.3.

    Those thresholds, ``min_similarity``, ``semantic_weight``,
    ``candidates`` and ``min_words`` are provisional: they were set
    on synthetic sessions and must be tuned with ``skills lab`` once
    human-annotated reference data exists. ``min_similarity`` depends
    on the embedding model and has to be set again whenever that
    model changes.
    """

    rubric_dir: str = "knowledge/rubrics"
    catalogue: str = "core"
    rubric_language: str = "auto"
    transcripts_dir: str = "data/transcripts"
    output_dir: str = "data/output/skills"
    semantic: bool = True
    embedding_model: str = DEFAULT_SKILLS_EMBEDDING_MODEL
    embedding_max_tokens: int = Field(default=128, ge=8)
    semantic_weight: float = Field(default=1.0, ge=0.0)
    query_prefix: str = ""
    passage_prefix: str = ""
    min_words: int = Field(default=4, ge=1)
    bm25_k1: float = 1.2
    bm25_b: float = 0.5
    judge: Literal["local", "api"] = "local"
    llm: str = DEFAULT_SKILLS_LLM
    api_model: str = DEFAULT_SKILLS_API_MODEL
    api_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    candidates: int = Field(default=8, ge=1)
    min_incidents: int = Field(default=3, ge=1)
    min_episodes: int = Field(default=2, ge=1)
    episode_gap: int = Field(default=3, ge=0)
    min_participant_words: int = Field(default=40, ge=0)
    min_similarity: float = Field(default=0.3, ge=-1.0, le=1.0)
    max_new_tokens: int = Field(default=700, ge=16)
    use_cache: bool = True
    cache_dir: str = "data/skills_cache"
    progress: bool = True
    lab_dir: str = "experiments/skills"
    device: str = "auto"
    torch_threads: Optional[int] = Field(default=None, ge=1)
    cpu_cores: Literal["auto", "all", "performance"] = "auto"
    runtime_plan_path: str = "data/runtime_plan.json"
    min_tokens_per_second: float = Field(default=4.0, gt=0)
    model_ladder: Tuple[Tuple[str, float], ...] = SKILLS_MODEL_LADDER
