"""Use case: a domain's mandatory items and whether a sentence says one.

The live page lists the items of the product's domain
(:meth:`MandatoryItems.items`) and the orchestrator asks, for each final
sentence of the employee, which items it says
(:meth:`MandatoryItems.check`): the sentence is embedded with the
knowledge embedding model and compared with every ``must_say``
example; an item is said when its best cosine reaches the list's
threshold (``src/domain/knowledge/mandatory.py``). The examples'
vectors are computed once per list version.

Since #294 a list can also come with the request
(:meth:`MandatoryItems.check_list`: an owner's list of the
orchestrator, kept outside the manifests), and
:meth:`MandatoryItems.review` tells which examples come close to an
example of another item of the same list (a warning for the editor).
"""

import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Tuple

import numpy as np

from src.application.ports import Embedder, FloatArray
from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.mandatory import (
    MandatoryList,
    close_examples,
    describe,
    item_scores,
)

MAX_CHECK_TEXT = 2000
"""Longest sentence checked (an employee's final is far shorter)."""
KEEP_LISTS = 16
"""Lists sent with the request whose examples' vectors are kept."""


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
        self._sent: "OrderedDict[str, FloatArray]" = OrderedDict()
        self._lock = threading.Lock()

    def items(self, domain_id: str) -> Mapping[str, object]:
        """Return the domain's list (version 0 and no items if none)."""
        return describe(domain_id, self.manifest(domain_id).mandatory)

    def check(self, domain_id: str, text: str) -> Mapping[str, object]:
        """Return how close ``text`` comes to each item of the domain.

        ``InvalidInputError`` for an empty or too long text,
        ``NotFoundError`` when the domain has no mandatory items.
        """
        sentence = _sentence(text)
        listed = self.manifest(domain_id).mandatory
        if listed is None:
            raise NotFoundError(
                f"The domain {domain_id!r} has no mandatory items."
            )
        examples = self._examples(domain_id, listed)
        return {"domain": domain_id,
                **self._scores(listed, examples, sentence)}

    def check_list(
        self, listed: Mapping[str, Any], text: str
    ) -> Mapping[str, object]:
        """Return how close ``text`` comes to each item of ``listed``.

        ``listed`` is a whole list (``version``, ``threshold``,
        ``items``…) sent with the request (#294); the check is the
        same as :meth:`check`. ``InvalidInputError`` for an invalid
        list or text.
        """
        sentence = _sentence(text)
        owned = _valid(listed)
        return dict(self._scores(owned, self._sent_examples(owned),
                                 sentence))

    def review(self, listed: Mapping[str, Any]) -> Mapping[str, object]:
        """Return the examples close to an example of another item.

        A warning for the list's editor (#294; a heuristic: the list's
        threshold applied to example against example, not measured).
        """
        owned = _valid(listed)
        vectors = self._sent_examples(owned)
        matrix: List[List[float]] = (vectors @ vectors.T).tolist()
        return {
            "threshold": owned.threshold,
            "close": [c.model_dump() for c in close_examples(owned, matrix)],
        }

    def _scores(
        self, listed: MandatoryList, examples: FloatArray, sentence: str
    ) -> Mapping[str, object]:
        """Score ``sentence`` against the examples' vectors."""
        query = np.asarray(self.embedder.encode([sentence]),
                           dtype=np.float32)[0]
        similarities: List[float] = (examples @ query).tolist()
        scores = item_scores(listed, similarities)
        return {
            "version": listed.version, "threshold": listed.threshold,
            "items": [s.model_dump() for s in scores],
        }

    def _sent_examples(self, listed: MandatoryList) -> FloatArray:
        """Return a sent list's vectors (the last few lists are kept)."""
        key = listed.model_dump_json()
        with self._lock:
            found = self._sent.get(key)
            if found is not None:
                self._sent.move_to_end(key)
        if found is None:
            found = np.asarray(self.embedder.encode(listed.examples()),
                               dtype=np.float32)
            with self._lock:
                self._sent[key] = found
                while len(self._sent) > KEEP_LISTS:
                    self._sent.popitem(last=False)
        return found

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


def _sentence(text: str) -> str:
    """Return ``text`` on one line, or refuse an empty or long one."""
    sentence = " ".join(str(text or "").split())
    if not sentence or len(sentence) > MAX_CHECK_TEXT:
        raise InvalidInputError(
            f"The text to check must have 1 to {MAX_CHECK_TEXT} "
            "characters."
        )
    return sentence


def _valid(listed: Mapping[str, Any]) -> MandatoryList:
    """Return ``listed`` as a list, or refuse it (``InvalidInputError``)."""
    if not isinstance(listed, Mapping):
        raise InvalidInputError("The list must be an object.")
    fields = {k: v for k, v in listed.items()
              if k in ("version", "threshold", "remind_seconds", "items")}
    try:
        return MandatoryList.model_validate(fields)
    except ValueError as exc:
        raise InvalidInputError(f"Invalid list: {exc}"[:300]) from exc
