"""Document processing adapter for knowledge domains.

Reads a domain's source files and chunks them with
:mod:`src.domain.knowledge.sources`, in this process (domains are
small, unlike the 42 corpus that :mod:`parallel_processor` handles).
"""

import hashlib
from typing import Iterator, Optional, Sequence

from src.adapters.outbound.parallel_processor import field_term_counts
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.sources import chunk_text, knowledge_terms
from src.domain.models import ProcessedDocument


class KnowledgeProcessor:
    """Chunk and analyse the source files of one domain.

    Args:
        manifest: The domain's manifest.
        path_weight: How much the file path counts in the lexical index.
        context_weight: How much the heading path counts.
    """

    def __init__(
        self,
        manifest: DomainManifest,
        path_weight: int = 1,
        context_weight: int = 3,
    ) -> None:
        """Keep the settings."""
        self.manifest = manifest
        self.path_weight = path_weight
        self.context_weight = context_weight

    def process(
        self, paths: Sequence[str], max_size: int, target: Optional[int]
    ) -> Iterator[ProcessedDocument]:
        """Yield one processed document per path, in order."""
        for path in paths:
            with open(path, "rb") as handle:
                raw = handle.read()
            text = raw.decode("utf-8", errors="replace")
            chunks = chunk_text(path, text, max_size, target)
            yield ProcessedDocument(
                file_path=path,
                sha256=hashlib.sha256(raw).hexdigest(),
                chunks=chunks,
                term_counts=[
                    field_term_counts(
                        c, knowledge_terms, self.path_weight,
                        self.context_weight,
                    )
                    for c in chunks
                ],
            )
