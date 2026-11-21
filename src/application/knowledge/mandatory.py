"""Use case: a domain's mandatory items and whether a sentence says one.

The live page lists the items of the product's domain
(:meth:`MandatoryItems.items`) and the orchestrator asks, for each final
sentence of the employee, which items it says
(:meth:`MandatoryItems.check`): the sentence is embedded with the
knowledge embedding model and compared with every ``must_say``
example; an item is said when its best cosine reaches the list's
threshold (``src/domain/knowledge/mandatory.py``). The examples'
vectors are computed once per list version.
"""

import threading
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np

from src.application.ports import Embedder, FloatArray
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.mandatory import (
    MandatoryList,
    describe,
    item_scores,
)

MAX_CHECK_TEXT = 2000
"""Longest sentence checked (an employee's final is far shorter)."""


class MandatoryItems:
    """Read the mandatory items of a domain and check sentences on them.

    Args:
        manifest: Returns a domain's manifest (``NotFoundError`` when
            there is no such domain).
        embedder: The knowledge embedding model (L2-normalised vectors).
    """

    def __init__(
        self,
        manifest: Callable[[str], DomainManifest],
        embedder: Embedder,
    ) -> None:
        """Keep the ports; nothing is embedded yet."""
        self.manifest = manifest
        self.embedder = embedder
        self._vectors: Dict[Tuple[str, str], FloatArray] = {}
        self._lock = threading.Lock()

    def items(self, domain_id: str) -> Mapping[str, object]:
        """Return the domain's list (version 0 and no items if none)."""
        return describe(domain_id, self.manifest(domain_id).mandatory)

    def check(self, domain_id: str, text: str) -> Mapping[str, object]:
        """Return how close ``text`` comes to each item of the domain.

        ``InvalidInputError`` for an empty or too long text,
        ``NotFoundError`` when the domain has no mandatory items.
        """
        sentence = " ".join(str(text or "").split())
        if not sentence or len(sentence) > MAX_CHECK_TEXT:
            raise InvalidInputError(
                f"The text to check must have 1 to {MAX_CHECK_TEXT} "
                "characters."
            )
        listed = self.manifest(domain_id).mandatory
        if listed is None:
            raise NotFoundError(
                f"The domain {domain_id!r} has no mandatory items."
            )
        examples = self._examples(domain_id, listed)
        query = np.asarray(self.embedder.encode([sentence]),
                           dtype=np.float32)[0]
        similarities: List[float] = (examples @ query).tolist()
        scores = item_scores(listed, similarities)
        return {
            "domain": domain_id, "version": listed.version,
            "threshold": listed.threshold,
            "items": [s.model_dump() for s in scores],
        }

    def _examples(self, domain_id: str, listed: MandatoryList) -> FloatArray:
        """Return the examples' vectors, computed once per list."""
        key = (domain_id, listed.model_dump_json())
        with self._lock:
            found = self._vectors.get(key)
        if found is None:
            found = np.asarray(self.embedder.encode(listed.examples()),
                               dtype=np.float32)
            with self._lock:
                self._vectors = {
                    k: v for k, v in self._vectors.items()
                    if k[0] != domain_id
                }
                self._vectors[key] = found
        return found
