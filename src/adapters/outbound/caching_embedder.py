"""Embedding cache: never embed the same text twice with one model."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.application.ports import Embedder, FloatArray


class CachingEmbedder:
    """Decorate an :class:`Embedder` with a persistent vector cache.

    Vectors are keyed by the hash of the text. Experiments that change
    lexical settings or re-chunk part of the corpus then only embed the
    texts they have not seen, instead of the whole corpus.
    """

    def __init__(self, inner: Embedder, directory: str, namespace: str):
        """Cache the vectors of ``inner`` under ``directory/namespace``."""
        self.inner = inner
        slug = hashlib.sha256(namespace.encode()).hexdigest()[:12]
        self.directory = Path(directory) / slug
        self.hits = 0
        self.misses = 0
        self._rows: Dict[str, int] = {}
        self._vectors: List[FloatArray] = []
        self._loaded = False

    @property
    def model_name(self) -> str:
        """Return the identifier of the wrapped model."""
        return self.inner.model_name

    def _load(self) -> None:
        """Read the cache from disk once; a broken cache is ignored."""
        if self._loaded:
            return
        self._loaded = True
        keys = self.directory / "keys.json"
        vectors = self.directory / "vectors.npy"
        try:
            names = json.loads(keys.read_text(encoding="utf-8"))
            matrix = np.load(vectors)
        except (OSError, ValueError, EOFError):
            return
        if isinstance(names, list) and len(names) == len(matrix):
            self._rows = {str(k): i for i, k in enumerate(names)}
            self._vectors = [matrix.astype(np.float32)]

    def _save(self) -> None:
        """Persist the cache; failures only cost a later re-embedding."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            matrix = np.vstack(self._vectors)
            self._vectors = [matrix]
            ordered = sorted(self._rows, key=self._rows.__getitem__)
            tmp_vectors = self.directory / "vectors.tmp.npy"
            tmp_keys = self.directory / "keys.tmp.json"
            np.save(tmp_vectors, matrix)
            tmp_keys.write_text(json.dumps(ordered), encoding="utf-8")
            os.replace(tmp_vectors, self.directory / "vectors.npy")
            os.replace(tmp_keys, self.directory / "keys.json")
        except OSError:
            pass

    def encode(
        self, texts: Sequence[str], progress: bool = False
    ) -> FloatArray:
        """Return one embedding row per text, embedding only new texts."""
        self._load()
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._rows]
        fresh_keys = list(dict.fromkeys(keys[i] for i in missing))
        first = {keys[i]: i for i in reversed(missing)}
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        if fresh_keys:
            new = self.inner.encode(
                [texts[first[k]] for k in fresh_keys], progress
            )
            base = len(self._rows)
            for offset, key in enumerate(fresh_keys):
                self._rows[key] = base + offset
            self._vectors.append(new.astype(np.float32))
            self._save()
        if not self._vectors:
            return self.inner.encode(texts, progress)
        matrix = np.vstack(self._vectors)
        self._vectors = [matrix]
        return matrix[[self._rows[k] for k in keys]]
