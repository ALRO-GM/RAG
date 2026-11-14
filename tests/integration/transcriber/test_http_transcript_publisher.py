"""The HTTP publisher against a local stand-in of the RAG.

The stand-in is :mod:`http.server` in a thread on a free loopback
port; each test scripts its answers (status, delay) and reads back the
requests it received. Nothing leaves the machine.
"""

import http.server
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from tests.integration.transcriber.test_pipeline import (
    enrol_ana,
    record,
    transcribe,
)
from tests.integration.transcriber.test_transcript_publisher import example
from transcriber.adapters.outbound.fs_transcript_publisher import (
    DirectoryTranscriptPublisher,
)
from transcriber.adapters.outbound.http_transcript_publisher import (
    HttpTranscriptPublisher,
)
from transcriber.bootstrap import build_container, build_publisher
from transcriber.config import HttpPublishSettings, TranscriberSettings
from transcriber.domain.errors import ContractError, PublishError

TOKEN = "s3cret-token-for-tests"
ENV = {"RAG_TOKEN": TOKEN}
CONTRACT_KEYS = {"schema", "session", "speakers", "utterances", "pipeline"}
UTTERANCE_KEYS = {
    "id", "turn", "index", "speaker", "start", "end", "text",
    "previous_speaker", "previous_utterance", "confidence",
    "speaker_confidence",
}

Answer = Tuple[int, float]
"""A scripted answer: HTTP status and seconds to wait before it."""


class Received:
    """What the stand-in RAG saw and what it will answer next."""

    def __init__(self, answers: List[Answer]) -> None:
        """Answer with ``answers`` in order, then always 201."""
        self.answers = list(answers)
        self.requests: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def next_answer(self) -> Answer:
        """Return the next scripted answer."""
        with self.lock:
            return self.answers.pop(0) if self.answers else (201, 0.0)


def handler_for(received: Received) -> type:
    """A request handler class that records into ``received``."""

    class Handler(http.server.BaseHTTPRequestHandler):
        """Record every POST and answer as scripted."""

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            """Record the request and send the scripted answer."""
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            with received.lock:
                received.requests.append({
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": body,
                })
            status, delay = received.next_answer()
            if delay:
                time.sleep(delay)
            reply = json.dumps({"detail": f"answer {status}"}).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
            except OSError:  # the client gave up waiting
                pass

        def log_message(self, *args: Any) -> None:
            """Keep the test output clean."""

    return Handler


class StandIn:
    """A running stand-in RAG and what it received."""

    def __init__(self, url: str, received: Received) -> None:
        """Keep the base URL and the record."""
        self.url = url
        self.received = received

    def answer(self, *answers: Answer) -> None:
        """Script the next answers."""
        self.received.answers.extend(answers)

    def bodies(self) -> List[Dict[str, Any]]:
        """Return the JSON bodies received, in order."""
        return [json.loads(r["body"]) for r in self.received.requests]


@pytest.fixture()
def rag() -> Iterator[StandIn]:
    """Serve the stand-in; a test scripts it with :meth:`StandIn.answer`."""
    received = Received([])
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), handler_for(received)
    )
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield StandIn(f"http://{host!s}:{port}/rag", received)
    finally:
        server.shutdown()
        server.server_close()


def publisher(
    url: str, environ: Optional[Dict[str, str]] = None, **options: Any
) -> Tuple[HttpTranscriptPublisher, List[float]]:
    """A publisher to ``url`` that records its back-off instead of sleeping."""
    waits: List[float] = []
    built = HttpTranscriptPublisher(
        url, "RAG_TOKEN", ENV if environ is None else environ,
        sleep=waits.append, **{"timeout": 5.0, **options},
    )
    return built, waits


def test_success_sends_exactly_the_document_with_the_token(
    rag: StandIn,
) -> None:
    sender, waits = publisher(rag.url)
    document = example()
    sender.publish(document)
    [sent] = rag.received.requests
    assert sent["path"] == "/rag/transcripts"
    assert sent["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert sent["headers"]["Content-Type"] == "application/json"
    assert json.loads(sent["body"]) == document
    assert waits == []


def test_a_wrong_token_is_final_and_named_without_the_token(
    rag: StandIn,
) -> None:
    rag.answer((401, 0.0))
    sender, waits = publisher(rag.url, retries=3)
    with pytest.raises(PublishError, match=r"\$RAG_TOKEN \(401\)") as error:
        sender.publish(example())
    assert TOKEN not in str(error.value)
    assert len(rag.received.requests) == 1  # not retried
    assert waits == []


def test_a_refused_document_is_not_retried(rag: StandIn) -> None:
    rag.answer((422, 0.0))
    sender, _ = publisher(rag.url)
    with pytest.raises(PublishError, match=r"refused .*\(422\): answer 422"):
        sender.publish(example())
    assert len(rag.received.requests) == 1


def test_server_errors_are_retried_with_back_off(rag: StandIn) -> None:
    rag.answer((503, 0.0), (500, 0.0))
    sender, waits = publisher(
        rag.url, retries=3, backoff_initial=0.5, backoff_factor=3.0
    )
    sender.publish(example())
    assert len(rag.received.requests) == 3
    assert waits == [0.5, 1.5]
    first, *rest = rag.bodies()
    assert all(body == first for body in rest)  # the same document


def test_retries_are_bounded(rag: StandIn) -> None:
    rag.answer((502, 0.0), (502, 0.0))
    sender, waits = publisher(rag.url, retries=1)
    with pytest.raises(PublishError, match="after 2 attempt.*answered 502"):
        sender.publish(example())
    assert len(rag.received.requests) == 2
    assert waits == [1.0]


def test_a_timeout_is_retried(rag: StandIn) -> None:
    rag.answer((201, 1.5), (201, 0.0))
    sender, waits = publisher(rag.url, timeout=0.3, retries=1)
    sender.publish(example())
    assert len(rag.received.requests) == 2
    assert waits == [1.0]


def test_a_timeout_on_the_last_attempt_fails(rag: StandIn) -> None:
    rag.answer((201, 1.5))
    sender, _ = publisher(rag.url, timeout=0.3, retries=0)
    with pytest.raises(PublishError, match="no answer within 0.3 s"):
        sender.publish(example())


def free_port() -> int:
    """A loopback port nobody listens on (right now)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


def test_an_unreachable_rag_is_retried_then_reported() -> None:
    sender, waits = publisher(f"http://127.0.0.1:{free_port()}", retries=2)
    with pytest.raises(PublishError, match="after 3 attempt.*cannot reach"):
        sender.publish(example())
    assert waits == [1.0, 2.0]


def test_without_a_token_nothing_is_sent(rag: StandIn) -> None:
    sender, _ = publisher(rag.url, environ={})
    with pytest.raises(PublishError, match=r"set \$RAG_TOKEN"):
        sender.publish(example())
    assert rag.received.requests == []


def test_an_invalid_document_is_not_sent(rag: StandIn) -> None:
    sender, _ = publisher(rag.url)
    broken = example()
    broken["utterances"][0]["words"] = []
    with pytest.raises(ContractError):
        sender.publish(broken)
    assert rag.received.requests == []


def test_the_token_is_not_in_the_repr() -> None:
    sender, _ = publisher("https://rag.example/")
    assert TOKEN not in repr(sender)
    assert "$RAG_TOKEN" in repr(sender)
    assert sender.destination() == "https://rag.example/transcripts"
    with pytest.raises(ValueError, match="http"):
        HttpTranscriptPublisher("ftp://x", "RAG_TOKEN", ENV)


def test_a_real_session_sends_no_audio_no_voiceprint_no_words(
    rag: StandIn, tmp_path: Path,
) -> None:
    settings = TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake", publisher="http",
        publish_http=HttpPublishSettings(url=rag.url, token_env="RAG_TOKEN"),
    )
    container = build_container(settings, environ=ENV)
    try:
        enrol_ana(container)
        record(container, "equipo-a")
        transcribe(container, "equipo-a")
        assert container.sessions.words("equipo-a")  # kept, not sent
        returned = container.publish("equipo-a")
    finally:
        container.runner.close(5.0)
    [body] = rag.bodies()
    assert body == returned
    assert set(body) == CONTRACT_KEYS
    assert all(set(u) == UTTERANCE_KEYS for u in body["utterances"])
    raw = rag.received.requests[0]["body"]
    for forbidden in (b'"words"', b'"audio"', b'"voiceprint"', b'"vector"',
                      b'"embedding"', b"RIFF", b"fLaC"):
        assert forbidden not in raw
    audio = tmp_path / "data" / "sessions" / "equipo-a" / "audio.flac"
    assert len(raw) < audio.stat().st_size / 10


def test_settings_choose_the_publisher(tmp_path: Path) -> None:
    default = TranscriberSettings(data_dir=tmp_path)
    assert isinstance(build_publisher(default), DirectoryTranscriptPublisher)
    assert default.publish_target() == str(tmp_path / "published")
    http = TranscriberSettings(
        data_dir=tmp_path, publisher="http",
        publish_http=HttpPublishSettings(
            url="https://rag.example", retries=5
        ),
    )
    built = build_publisher(http, {})
    assert isinstance(built, HttpTranscriptPublisher)
    assert built.retries == 5
    assert built.token_env == "TRANSCRIBER_PUBLISH_TOKEN"
    assert http.publish_target() == "https://rag.example/transcripts"
    with pytest.raises(ValueError, match="publish_http.url"):
        TranscriberSettings(publisher="http")
    with pytest.raises(ValueError):
        HttpPublishSettings(url="rag.example")
    with pytest.raises(ValueError):
        HttpPublishSettings(token_env="A B")
