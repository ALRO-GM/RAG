"""Use case: load every knowledge domain's index and the embedder at start.

The first query to a domain reads its index and loads the embedding
model (about 2 s), which a live caller with a short budget cannot
afford (issue #176). :func:`warm_domains` runs one throw-away query per
domain so that this happens when the service starts, off the request
path. A domain that cannot be searched (no index yet) is skipped.
"""

import logging
from typing import List

from src.application.knowledge.query import KnowledgeQueryService
from src.application.knowledge.registry import DomainIndexRegistry
from src.domain.errors import RagError
from src.domain.knowledge.contract import KnowledgeQuery

LOG = logging.getLogger(__name__)


def warm_domains(
    registry: DomainIndexRegistry, queries: KnowledgeQueryService
) -> List[str]:
    """Query each indexed domain once; return the ids that were warmed."""
    warmed: List[str] = []
    for row in registry.summaries():
        if row.index_version is None:
            continue
        try:
            queries.run(KnowledgeQuery(
                schema="knowledge.query.v1", request_id="warm-up",
                domain=row.id, text="warm up", top_k=1,
            ))
        except RagError as exc:
            LOG.warning("Could not warm knowledge domain %s: %s", row.id,
                        exc)
            continue
        warmed.append(row.id)
    return warmed
