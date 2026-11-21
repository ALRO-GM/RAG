"""Embedding adapter: one embedder shared by threads, one call at a time.

A fast tokenizer used by two threads at once can fail ("Already
borrowed"). The knowledge queries and the mandatory-item checks of a
live session (LV-16) share the knowledge model, so its calls go through
this lock; each call is a single short sentence, so waiting is short.
"""

import threading
from typing import Sequence

from src.application.ports import Embedder, FloatArray


class SerialEmbedder:
    """Wrap an embedder so that only one thread encodes at a time."""

    def __init__(self, inner: Embedder) -> None:
        """Keep the embedder and make the lock."""
        self.inner = inner
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Return the wrapped model's identifier."""
        return self.inner.model_name

    def encode(
        self, texts: Sequence[str], progress: bool = False
    ) -> FloatArray:
        """Encode ``texts`` while holding the lock."""
        with self._lock:
            return self.inner.encode(texts, progress)
