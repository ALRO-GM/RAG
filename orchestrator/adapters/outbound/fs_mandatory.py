"""Mandatory items read from the domain manifests, checked by word overlap.

For ``serve --knowledge fake`` (demos and tests without the RAG): the
list comes from ``knowledge/domains/<id>/manifest.yaml`` and a sentence
is compared with each ``must_say`` example by the cosine of their word
counts instead of the RAG's embedding model, so the measured threshold
of ``docs/live/MANDATORY_ITEMS.md`` does not hold here: this adapter
uses its own (:data:`WORD_THRESHOLD`). An owner's list sent with the
sentence (#294) is checked the same way.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

from orchestrator.application.ports import Document, MandatoryUnavailable

WORD_THRESHOLD = 0.5
"""Word-count cosine at which a sentence says an item (no measurement:
the fake only has to tell a copied example from an unrelated line)."""
DOMAIN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def _words(text: str) -> Counter[str]:
    """Return the counts of the lower-cased words of ``text``."""
    return Counter(w for w in re.findall(r"\w+", text.lower()) if len(w) > 2)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    """Return the cosine of two word counts."""
    dot = sum(a[w] * b[w] for w in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values())
    )
    return dot / norm if norm else 0.0


class FileMandatory:
    """The mandatory items of the authored domains, without a model.

    Args:
        root: The folder of the authored domains (``knowledge/domains``).
    """

    def __init__(self, root: Path):
        """Keep the folder."""
        self.root = root

    def items(self, domain: str) -> Document:
        """Return the domain's list (no items when it has none)."""
        listed = self._list(domain)
        return {"domain": domain, **listed} if listed else {
            "domain": domain, "version": 0, "items": [],
        }

    def check(self, domain: str, text: str) -> Document:
        """Return each item's word cosine to ``text`` (closest one said)."""
        return {"domain": domain, **self.check_list(self._list(domain), text)}

    def check_list(self, listed: Document, text: str) -> Document:
        """Return each item's word cosine to ``text``, for a sent list."""
        said = _words(text)
        scores: List[Dict[str, Any]] = [
            {"id": item["id"], "score": round(max(
                _cosine(said, _words(e)) for e in item["must_say"]
            ), 4)}
            for item in listed.get("items") or []
        ]
        top = max((s["score"] for s in scores), default=0.0)
        for s in scores:
            s["said"] = s["score"] >= WORD_THRESHOLD and s["score"] == top
        return {"version": listed.get("version", 0),
                "threshold": WORD_THRESHOLD, "items": scores}

    def review(self, listed: Document) -> Document:
        """Return the examples whose word cosine to another item's
        example reaches :data:`WORD_THRESHOLD` (closest other item)."""
        owners = [(str(item["id"]), str(e))
                  for item in listed.get("items") or []
                  for e in item.get("must_say") or []]
        close: List[Dict[str, Any]] = []
        for item, example in owners:
            others = [(_cosine(_words(example), _words(e)), other, e)
                      for other, e in owners if other != item]
            if not others:
                continue
            score, other, other_example = max(others, key=lambda o: o[0])
            if score >= WORD_THRESHOLD:
                close.append({"item": item, "example": example,
                              "other": other, "other_example": other_example,
                              "score": round(score, 4)})
        return {"threshold": WORD_THRESHOLD, "close": close}

    def _list(self, domain: str) -> Dict[str, Any]:
        """Return the manifest's ``mandatory`` block ({} when none)."""
        if not DOMAIN.fullmatch(domain):
            raise MandatoryUnavailable("invalid domain id")
        path = self.root / domain / "manifest.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise MandatoryUnavailable(type(exc).__name__) from exc
        listed = data.get("mandatory") if isinstance(data, dict) else None
        return listed if isinstance(listed, dict) else {}
