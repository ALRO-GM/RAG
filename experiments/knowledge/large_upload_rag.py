"""The RAG's real ``/knowledge`` routes for the large-upload run (#282).

The library, upload area, document store, Docling extractors (on the
CPU, captions off) and MiniLM embedder of :class:`~src.bootstrap.
Container`, with the HTTP client of the transcriber; only the 42
corpus' search and answer are stand-ins (not measured). Run it in its
own folder ``WORK`` (relative data dirs, as in production) holding
``token``:

    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=<repo> \\
    python experiments/knowledge/large_upload_rag.py WORK PORT TR_PORT on

How the whole run goes: ``large_upload_client.py``.
"""

import os
import sys
from pathlib import Path

import uvicorn

from src.adapters.inbound.http_api import (
    KnowledgeRoutes,
    TranscriptRoutes,
    create_app,
)
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.application.answering import AnswerService
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.bootstrap import Container
from src.config import KnowledgeSettings, Settings
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator


def main() -> None:
    """Serve the routes until stopped."""
    work, port, tr_port = Path(sys.argv[1]), int(sys.argv[2]), int(
        sys.argv[3]
    )
    token = (work / "token").read_text().strip()
    knowledge = KnowledgeSettings(
        domains_dir="rag/domains", data_dir="rag/knowledge",
        index_dir="rag/indexes",
        transcriber_url=f"http://127.0.0.1:{tr_port}",
        ocr=sys.argv[4] == "on", caption_backend="off",
        media_poll_seconds=2.0,
    )
    container = Container(Settings(), knowledge=knowledge)
    container.media_token = token
    store = FileSystemTranscriptStore("rag/transcripts")
    search = SearchService(
        FileSystemIndexRepository("rag/none"), tokenize, FakeEmbedder()
    )
    routes = TranscriptRoutes(
        IngestTranscript(store), ShowIngestedTranscript(store), token,
        knowledge=KnowledgeRoutes(
            container.knowledge_query, container.knowledge_library,
            uploads=container.knowledge_uploads,
        ),
    )
    container.knowledge_uploads.sweep()
    container.knowledge_library.recover()
    print("pid", os.getpid(), flush=True)
    uvicorn.run(
        create_app(search, AnswerService(search, FakeGenerator()), routes),
        host="127.0.0.1", port=port, log_level="warning",
    )


if __name__ == "__main__":
    main()
