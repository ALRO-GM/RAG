"""Receiving transcripts: the file-system store and the HTTP routes."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import (
    TranscriptRoutes,
    authorised,
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
from src.config import Settings
from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
    RagError,
)
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "contracts" / "examples" / "transcript.v1.json"
TOKEN = "a-token-only-tests-know"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def example() -> Dict[str, Any]:
    """The contract's example document."""
    document: Dict[str, Any] = json.loads(EXAMPLE.read_text("utf-8"))
    return document


# -- the store ----------------------------------------------------------

def test_store_writes_reads_and_replaces(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path / "t"))
    document = example()
    assert store.put("demo-es", document) is False
    assert store.put("demo-es", document) is True
    assert store.get("demo-es") == document
    assert [p.name for p in (tmp_path / "t").iterdir()] == [
        "demo-es.transcript.v1.json"
    ]


def test_store_refuses_unsafe_ids(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path))
    for bad in ("../x", "", ".hidden", "a/b"):
        with pytest.raises(InvalidInputError, match="Invalid session id"):
            store.path_for(bad)
    assert list(tmp_path.iterdir()) == []


def test_store_reports_missing_and_damaged_files(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path))
    with pytest.raises(NotFoundError):
        store.get("demo-es")
    store.path_for("x").write_text("{not json", encoding="utf-8")
    with pytest.raises(CorruptedDataError, match="not valid JSON"):
        store.get("x")
    store.path_for("y").write_text("[]", encoding="utf-8")
    with pytest.raises(CorruptedDataError, match="not a JSON object"):
        store.get("y")
    store.path_for("z").mkdir()
    with pytest.raises(RagError, match="Cannot read"):
        store.get("z")


def test_a_failed_write_leaves_the_old_copy(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path))
    store.put("demo-es", example())
    with pytest.raises(ValueError):
        store.put("demo-es", {"duration": float("nan")})
    assert store.get("demo-es") == example()
    tmp_path.chmod(0o500)
    try:
        if os.access(tmp_path, os.W_OK):  # running as root
            pytest.skip("the directory stays writable")
        with pytest.raises(RagError, match="Cannot write"):
            store.put("demo-es", {"other": True})
    finally:
        tmp_path.chmod(0o700)
    assert store.get("demo-es") == example()
    assert [p.name for p in tmp_path.iterdir()] == [
        "demo-es.transcript.v1.json"
    ]


# -- the routes ---------------------------------------------------------

def app_for(
    folder: Path, token: str = TOKEN, mounted: bool = True,
    max_bytes: int = 1024 * 1024,
) -> TestClient:
    """A client of the API with transcripts stored under ``folder``."""
    search = SearchService(
        FileSystemIndexRepository(str(folder / "index")), tokenize,
        FakeEmbedder(),
    )
    store = FileSystemTranscriptStore(str(folder / "transcripts"))
    routes = TranscriptRoutes(
        IngestTranscript(store), ShowIngestedTranscript(store), token,
        max_bytes,
    )
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()),
        routes if mounted else None,
    ))


@pytest.mark.parametrize("token, mounted", [("", True), (TOKEN, False)])
def test_without_a_token_the_routes_do_not_exist(
    tmp_path: Path, token: str, mounted: bool
) -> None:
    client = app_for(tmp_path, token, mounted)
    assert client.post("/transcripts", json=example()).status_code == 404
    assert client.post(
        "/transcripts", json=example(), headers={"Authorization": "Bearer "}
    ).status_code == 404
    assert client.get("/transcripts/demo-es").status_code == 404
    assert not (tmp_path / "transcripts").exists()


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN},
    {"Authorization": f"Basic {TOKEN}"}, {"Authorization": "Bearer "},
    {"Authorization": f"Bearer {TOKEN}x"},
])
def test_a_wrong_token_is_refused(
    tmp_path: Path, headers: Dict[str, str]
) -> None:
    client = app_for(tmp_path)
    posted = client.post("/transcripts", json=example(), headers=headers)
    assert posted.status_code == 401
    assert posted.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in posted.text
    assert client.get(
        "/transcripts/demo-es", headers=headers
    ).status_code == 401
    assert not (tmp_path / "transcripts").exists()


def test_the_token_check(tmp_path: Path) -> None:
    assert authorised(f"Bearer {TOKEN}", TOKEN)
    assert authorised(f"bearer  {TOKEN} ", TOKEN)
    assert not authorised(f"Token {TOKEN}", TOKEN)
    assert not authorised("", TOKEN)


def test_post_twice_then_read_back(tmp_path: Path) -> None:
    client = app_for(tmp_path)
    document = example()
    first = client.post("/transcripts", json=document, headers=AUTH)
    assert first.status_code == 201
    assert first.json() == {"id": "demo-es", "replaced": False,
                            "utterances": 4}
    again = client.post("/transcripts", json=document, headers=AUTH)
    assert again.status_code == 200 and again.json()["replaced"] is True
    read = client.get("/transcripts/demo-es", headers=AUTH)
    assert read.status_code == 200 and read.json() == document
    assert [p.name for p in (tmp_path / "transcripts").iterdir()] == [
        "demo-es.transcript.v1.json"
    ]


def test_an_invalid_document_is_refused(tmp_path: Path) -> None:
    client = app_for(tmp_path)
    broken = example()
    broken["utterances"][0]["start"] = "0.4"
    answer = client.post("/transcripts", json=broken, headers=AUTH)
    assert answer.status_code == 422
    assert "utterances.0.start" in answer.json()["detail"]
    not_json = client.post(
        "/transcripts", content=b"{nope", headers=AUTH
    )
    assert not_json.status_code == 400
    assert not (tmp_path / "transcripts").exists()


def test_a_body_too_large_is_refused(tmp_path: Path) -> None:
    client = app_for(tmp_path, max_bytes=100)
    answer = client.post("/transcripts", json=example(), headers=AUTH)
    assert answer.status_code == 413

    def chunks() -> Any:
        """A body without a length, sent in pieces."""
        yield b"{" + b" " * 80
        yield b" " * 80 + b"}"

    streamed = client.post("/transcripts", content=chunks(), headers=AUTH)
    assert streamed.status_code == 413
    assert not (tmp_path / "transcripts").exists()


def test_reading_unknown_or_invalid_sessions(tmp_path: Path) -> None:
    client = app_for(tmp_path)
    assert client.get(
        "/transcripts/demo-es", headers=AUTH
    ).status_code == 404
    assert client.get(
        "/transcripts/-bad", headers=AUTH
    ).status_code == 400


def test_storage_failures_are_server_errors(tmp_path: Path) -> None:
    client = app_for(tmp_path)
    (tmp_path / "transcripts").write_text("a file, not a directory")
    posted = client.post("/transcripts", json=example(), headers=AUTH)
    assert posted.status_code == 500
    assert "Cannot write" in posted.json()["detail"]
    read = client.get("/transcripts/demo-es", headers=AUTH)
    assert read.status_code == 500


def test_the_container_wires_the_store(tmp_path: Path) -> None:
    app = Container(Settings(transcripts_dir=str(tmp_path / "in")))
    assert app.transcript_store.directory == tmp_path / "in"
    assert app.ingest_transcript(example()).session_id == "demo-es"
    assert app.show_transcript("demo-es") == example()
    assert app.transcripts_token({"RAG_TRANSCRIPTS_TOKEN": " t "}) == "t"
    assert app.transcripts_token({}) == ""


@pytest.mark.parametrize("token", ["", TOKEN])
def test_serve_mounts_the_routes_only_with_a_token(
    indexed_project: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], token: str,
) -> None:
    import uvicorn

    from src.adapters.inbound.cli import RagCLI

    served: Dict[str, Any] = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kw: served.update(app=app, **kw)
    )
    monkeypatch.setenv("RAG_TRANSCRIPTS_TOKEN", token)
    RagCLI(mode="bm25").serve(port=8123)
    out = capsys.readouterr().out
    assert served["port"] == 8123
    client = TestClient(served["app"])
    answer = client.post("/transcripts", json=example(), headers=AUTH)
    if token:
        assert "POST /transcripts into data/transcripts" in out
        assert answer.status_code == 201
        assert Path("data/transcripts/demo-es.transcript.v1.json").is_file()
    else:
        assert "routes off (set $RAG_TRANSCRIPTS_TOKEN" in out
        assert answer.status_code == 404
    assert TOKEN not in out
