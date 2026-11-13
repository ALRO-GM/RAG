"""FastAPI inbound adapter through TestClient."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import create_app
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.application.answering import AnswerService
from src.application.search import SearchService
from src.domain.text_analysis import tokenize
from tests.fakes import FakeEmbedder, FakeGenerator
from tests.integration.conftest import INDEX_DIR
from tests.samples import PY_PATH


def client_for(index_dir: str, mode: str = "auto") -> TestClient:
    """Return a client over a real index with a fake generator."""
    search = SearchService(FileSystemIndexRepository(index_dir), tokenize,
                           FakeEmbedder(), mode=mode)
    return TestClient(create_app(search, AnswerService(search,
                                                       FakeGenerator())))


@pytest.fixture()
def client(indexed_project: Path) -> TestClient:
    """Return a client over the indexed tiny corpus."""
    return client_for(INDEX_DIR)


def test_health(client: TestClient) -> None:
    """Health reports the index statistics."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["mode"] == "bm25"
    assert len(body["fingerprint"]) == 16


def test_search_get(client: TestClient) -> None:
    """GET /search returns sources, scores and text."""
    response = client.get("/search", params={"q": "load_lora_adapter",
                                             "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "load_lora_adapter"
    assert body["mode"] == "bm25"
    assert 0 < len(body["results"]) <= 2
    first = body["results"][0]
    assert first["source"]["file_path"] == PY_PATH
    assert "load_lora_adapter" in first["text"]
    assert first["score"] > 0


def test_search_post(client: TestClient) -> None:
    """POST /search accepts a JSON body and defaults k."""
    response = client.post("/search", json={"query": "FP8 quantization"})
    assert response.status_code == 200
    assert response.json()["results"][0]["source"]["file_path"].endswith(
        "serving.md"
    )


def test_search_no_match(client: TestClient) -> None:
    """A query without matches returns an empty list."""
    response = client.get("/search", params={"q": "zzzqqq"})
    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize("params", [
    {}, {"q": ""}, {"q": "x", "k": 0}, {"q": "x", "k": 101},
    {"q": "x", "k": "abc"},
])
def test_search_get_validation(
    client: TestClient, params: dict[str, str | int]
) -> None:
    """Invalid query parameters are rejected with 422."""
    assert client.get("/search", params=params).status_code == 422


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": "x", "k": -1}, {"query": 3, "k": "z"},
])
def test_search_post_validation(
    client: TestClient, body: dict[str, object]
) -> None:
    """Invalid JSON bodies are rejected with 422."""
    assert client.post("/search", json=body).status_code == 422


def test_blank_query_is_400(client: TestClient) -> None:
    """A whitespace-only query passes validation but is refused."""
    assert client.get("/search", params={"q": "   "}).status_code == 400
    assert client.post("/answer", json={"query": "  "}).status_code == 400


def test_answer(client: TestClient) -> None:
    """POST /answer retrieves context and calls the generator."""
    response = client.post("/answer", json={"query": "kv cache", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "kv cache"
    assert body["answer"].startswith("answer to 'kv cache'")
    assert 0 < len(body["retrieved_sources"]) <= 2


def test_no_index_is_503(project: Path) -> None:
    """Without an index every endpoint answers 503, not 500."""
    client = client_for("missing_index")
    assert client.get("/health").status_code == 503
    response = client.get("/search", params={"q": "lora"})
    assert response.status_code == 503
    assert "No index found" in response.json()["detail"]
    assert client.post("/search", json={"query": "x"}).status_code == 503
    assert client.post("/answer", json={"query": "x"}).status_code == 503


def test_semantic_mode_on_lexical_index_is_503(
    indexed_project: Path
) -> None:
    """Dense retrieval on a lexical index is reported as unavailable."""
    client = client_for(INDEX_DIR, mode="semantic")
    response = client.get("/search", params={"q": "lora"})
    assert response.status_code == 503
    assert "no embeddings" in response.json()["detail"]
