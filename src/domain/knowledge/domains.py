"""Knowledge domains: the manifest, its id and the index version.

A domain is a folder with a ``manifest.yaml`` (authored ones under
``knowledge/domains/<id>/``, created ones under ``data/knowledge/<id>/``)
and its own index under ``data/processed/domains/<id>``. The manifest
declares the vocabularies the chunks' metadata may use
(``docs/platform/ARCHITECTURE.md`` §10, ``CONTRACTS.md`` §2).
"""

import hashlib
import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.errors import InvalidInputError

DOMAIN_ID_PATTERN = r"[a-z0-9][a-z0-9._-]{0,63}"
"""Domain ids: the ``domain`` pattern of ``knowledge.query.v1``."""

MAX_VOCABULARY = 500
"""Most values one vocabulary may declare."""


def check_domain_id(value: str) -> str:
    """Return ``value`` if it is a valid domain id, else raise."""
    if not isinstance(value, str) or not re.fullmatch(
        DOMAIN_ID_PATTERN, value
    ):
        raise InvalidInputError(
            f"Invalid domain id {value!r}: use 1-64 lowercase letters, "
            "digits, '.', '_' or '-', starting with a letter or digit."
        )
    return value


class Vocabularies(BaseModel):
    """Values the chunks' metadata may take; an empty list allows any."""

    model_config = ConfigDict(extra="forbid")

    intent: List[str] = Field(default_factory=list, max_length=MAX_VOCABULARY)
    script_phase: List[str] = Field(
        default_factory=list, max_length=MAX_VOCABULARY
    )
    tags: List[str] = Field(default_factory=list, max_length=MAX_VOCABULARY)


class DomainManifest(BaseModel):
    """What ``manifest.yaml`` declares about a domain."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1, max_length=200)
    language: str = Field(default="es", min_length=2, max_length=16)
    description: str = Field(default="", max_length=4000)
    vocabularies: Vocabularies = Field(default_factory=Vocabularies)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        """Check the id against :data:`DOMAIN_ID_PATTERN`."""
        if not re.fullmatch(DOMAIN_ID_PATTERN, value):
            raise ValueError(f"invalid domain id {value!r}")
        return value


class DomainSummary(BaseModel):
    """One row of ``GET /knowledge/domains`` and ``domain list``."""

    id: str
    title: str
    language: str
    documents: int
    chunks: int
    index_version: Optional[str]
    updated_at: Optional[str]


def index_version(manifest: DomainManifest, fingerprint: str) -> str:
    """Return the version of a domain index.

    It hashes the manifest and the index fingerprint (itself a hash of
    the indexing parameters and of every indexed file), so it changes
    whenever the vocabularies or any indexed file change.
    """
    payload = manifest.model_dump_json() + "\n" + fingerprint
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
