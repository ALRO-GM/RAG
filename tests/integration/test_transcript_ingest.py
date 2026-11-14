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
from src.adapters.outbound.token_file import (
    main,
    make_token_file,
    read_token_file,
)
from src.application.skills.ingest import (
    ForgetIngestedTranscript,
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
        max_bytes, ForgetIngestedTranscript(store),
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
    assert client.delete(
        "/transcripts/demo-es", headers=AUTH
    ).status_code == 404
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
    assert client.delete(
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


def test_the_store_reads_a_document_as_a_transcript(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path))
    store.put("demo-es", example())
    transcript = store.find("demo-es")
    assert [t.speaker for t in transcript.turns] == ["Marta", "Luis", "S1"]
    assert [s.id for s in transcript.turns[0].sentences] == ["t0.s0", "t0.s1"]
    broken = example()
    broken["schema"] = "transcript.v0"
    store.put("old", broken)
    with pytest.raises(CorruptedDataError, match="old.transcript.v1.json"):
        store.find("old")
    with pytest.raises(InvalidInputError):
        store.find("../x")


# -- deleting -----------------------------------------------------------

def test_store_deletes_once(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path / "in"))
    assert store.delete("demo-es") is False
    store.put("demo-es", example())
    assert store.delete("demo-es") is True
    assert list((tmp_path / "in").iterdir()) == []
    assert store.delete("demo-es") is False
    with pytest.raises(InvalidInputError):
        store.delete("../demo-es")


def test_store_reports_a_failed_delete(tmp_path: Path) -> None:
    store = FileSystemTranscriptStore(str(tmp_path))
    (tmp_path / "demo-es.transcript.v1.json").mkdir()
    with pytest.raises(RagError, match="Cannot delete"):
        store.delete("demo-es")


def test_delete_removes_the_copy(tmp_path: Path) -> None:
    client = app_for(tmp_path)
    client.post("/transcripts", json=example(), headers=AUTH)
    deleted = client.delete("/transcripts/demo-es", headers=AUTH)
    assert deleted.status_code == 204 and deleted.content == b""
    assert list((tmp_path / "transcripts").iterdir()) == []
    assert client.get(
        "/transcripts/demo-es", headers=AUTH
    ).status_code == 404
    again = client.delete("/transcripts/demo-es", headers=AUTH)
    assert again.status_code == 404
    assert client.delete(
        "/transcripts/-bad", headers=AUTH
    ).status_code == 400


def test_a_failed_delete_is_a_server_error(tmp_path: Path) -> None:
    client = app_for(tmp_path)
    (tmp_path / "transcripts" / "demo-es.transcript.v1.json").mkdir(
        parents=True
    )
    answer = client.delete("/transcripts/demo-es", headers=AUTH)
    assert answer.status_code == 500
    assert "Cannot delete" in answer.json()["detail"]


def test_without_forget_there_is_no_delete_route(tmp_path: Path) -> None:
    search = SearchService(
        FileSystemIndexRepository(str(tmp_path / "index")), tokenize,
        FakeEmbedder(),
    )
    store = FileSystemTranscriptStore(str(tmp_path / "transcripts"))
    client = TestClient(create_app(
        search, AnswerService(search, FakeGenerator()),
        TranscriptRoutes(
            IngestTranscript(store), ShowIngestedTranscript(store), TOKEN
        ),
    ))
    client.post("/transcripts", json=example(), headers=AUTH)
    assert client.delete(
        "/transcripts/demo-es", headers=AUTH
    ).status_code == 405
    assert store.get("demo-es") == example()


# -- the token file and the workstation's serve ---------------------------

FILE_TOKEN = "f" * 43


def private_token(folder: Path, text: str = FILE_TOKEN) -> Path:
    """Write ``text`` to a file only its owner may read."""
    path = folder / "rag-token"
    path.write_text(text + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_the_token_file_is_read_when_private(tmp_path: Path) -> None:
    assert read_token_file(private_token(tmp_path)) == FILE_TOKEN


def test_the_token_file_is_refused_unless_safe(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="make rag-token"):
        read_token_file(tmp_path / "missing")
    with pytest.raises(InvalidInputError, match="not a regular file"):
        read_token_file(tmp_path)
    shared = private_token(tmp_path)
    shared.chmod(0o640)
    with pytest.raises(InvalidInputError, match="chmod 600") as caught:
        read_token_file(shared)
    assert FILE_TOKEN not in str(caught.value)
    short = private_token(tmp_path, "short")
    with pytest.raises(InvalidInputError, match="does not hold a token"):
        read_token_file(short)
    link = tmp_path / "link"
    link.symlink_to(short)
    with pytest.raises(InvalidInputError, match="not a regular file"):
        read_token_file(link)


@pytest.mark.parametrize("host, loopback", [
    ("127.0.0.1", True), ("127.0.0.2", True), ("::1", True),
    ("localhost", True), ("0.0.0.0", False), ("::", False),
    ("192.168.1.10", False), ("rag.example", False),
])
def test_is_loopback(host: str, loopback: bool) -> None:
    from src.adapters.inbound.cli import is_loopback

    assert is_loopback(host) is loopback


def test_serve_reads_the_token_file(
    indexed_project: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import uvicorn

    from src.adapters.inbound.cli import RagCLI

    served: Dict[str, Any] = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kw: served.update(app=app, **kw)
    )
    monkeypatch.delenv("RAG_TRANSCRIPTS_TOKEN", raising=False)
    token_file = private_token(indexed_project)
    RagCLI(mode="bm25").serve(token_file=str(token_file))
    out = capsys.readouterr().out
    assert (served["host"], served["port"]) == ("127.0.0.1", 8000)
    assert f"token from file {token_file}" in out
    assert FILE_TOKEN not in out
    client = TestClient(served["app"])
    auth = {"Authorization": f"Bearer {FILE_TOKEN}"}
    assert client.post(
        "/transcripts", json=example(), headers=auth
    ).status_code == 201
    assert client.delete(
        "/transcripts/demo-es", headers=auth
    ).status_code == 204
    assert not Path("data/transcripts/demo-es.transcript.v1.json").exists()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_serve_with_a_token_file_stays_on_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], host: str,
) -> None:
    import uvicorn

    from src.adapters.inbound.cli import RagCLI

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    with pytest.raises(SystemExit) as stopped:
        RagCLI(mode="bm25").serve(
            host=host, token_file=str(private_token(tmp_path))
        )
    assert stopped.value.code == 1
    assert "--host 127.0.0.1" in capsys.readouterr().err


def test_serve_refuses_an_unsafe_token_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from src.adapters.inbound.cli import RagCLI

    shared = private_token(tmp_path)
    shared.chmod(0o644)
    with pytest.raises(SystemExit):
        RagCLI(mode="bm25").serve(token_file=str(shared))
    err = capsys.readouterr().err
    assert "chmod 600" in err and FILE_TOKEN not in err


def test_the_token_file_is_made_once_private(tmp_path: Path) -> None:
    path = tmp_path / "config" / "rag-token"
    assert make_token_file(path) is True
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    token = read_token_file(path)
    assert len(token) == 43
    assert make_token_file(path) is False
    assert read_token_file(path) == token
    path.chmod(0o644)
    with pytest.raises(InvalidInputError, match="chmod 600"):
        make_token_file(path)


def test_the_token_command_never_prints_the_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "rag-token"
    assert main([str(path)]) == 0
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        f"Made the RAG token in {path} (mode 600).",
        f"Kept the RAG token in {path} (mode 600).",
    ]
    assert read_token_file(path) not in out
    path.chmod(0o604)
    assert main([str(path)]) == 1
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "chmod 600" in err and "usage:" in err
