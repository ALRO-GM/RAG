"""End to end: the transcriber publishes, the RAG receives, over HTTP only.

The two packages never import each other; this test is the one place
that holds both, and it joins them only the way production does: the
transcriber's HTTP publisher posts ``transcript.v1`` to the RAG's
FastAPI app served by a real uvicorn on a free loopback port.
"""

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import httpx
import pytest
import uvicorn

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
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
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator
from transcriber.adapters.outbound.http_transcript_publisher import (
    HttpTranscriptPublisher,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "contracts" / "examples" / "transcript.v1.json"
TOKEN = "handover-token-for-tests"
WAIT = 30.0


def free_port() -> int:
    """Return a loopback port nothing listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def rag(tmp_path: Path) -> Iterator[Tuple[str, Path]]:
    """The RAG's API on uvicorn; yields its URL and transcript folder."""
    search = SearchService(
        FileSystemIndexRepository(str(tmp_path / "index")), tokenize,
        FakeEmbedder(),
    )
    folder = tmp_path / "transcripts"
    store = FileSystemTranscriptStore(str(folder))
    app = create_app(
        search, AnswerService(search, FakeGenerator()),
        TranscriptRoutes(
            IngestTranscript(store), ShowIngestedTranscript(store), TOKEN
        ),
    )
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + WAIT
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", folder
    server.should_exit = True
    thread.join(WAIT)


def test_publish_twice_read_back_once(rag: Tuple[str, Path]) -> None:
    url, folder = rag
    document: Dict[str, Any] = json.loads(EXAMPLE.read_text("utf-8"))
    publisher = HttpTranscriptPublisher(
        url, "HANDOVER_TOKEN", {"HANDOVER_TOKEN": TOKEN}, timeout=10.0,
        retries=0,
    )
    publisher.publish(document)
    publisher.publish(document)
    answer = httpx.get(
        f"{url}/transcripts/demo-es",
        headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10.0,
    )
    assert answer.status_code == 200
    assert answer.json() == document
    assert [p.name for p in folder.iterdir()] == [
        "demo-es.transcript.v1.json"
    ]
