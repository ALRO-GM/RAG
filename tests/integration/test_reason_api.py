"""``POST /reason``: bearer, contract validation and the three roles."""

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from src.adapters.inbound.http_api import TranscriptRoutes, create_app
from src.adapters.outbound.fs_index_repository import (
    FileSystemIndexRepository,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.application.answering import AnswerService
from src.application.reasoning import Reasoner
from src.application.search import SearchService
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.domain.text_analysis import tokenize
from tests.fakes import FakeChatModel, FakeEmbedder, FakeGenerator
from tests.integration.test_transcript_ingest import AUTH, TOKEN

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
EXAMPLE = json.loads(
    (CONTRACTS / "examples/reasoner.request.v1.json").read_text()
)
RESPONSE = Draft202012Validator(json.loads(
    (CONTRACTS / "reasoner.response.v1.schema.json").read_text()
))
REPLY = json.dumps({"suggestions": [{
    "kind": "follow_up", "text": "¿Cómo gestionasteis las transacciones?",
    "cites": ["interview.backend:q-arch-07"], "about": ["t7.s1"],
}]})


def client(tmp: Path, live: FakeChatModel, cold: FakeChatModel,
           token: str = TOKEN) -> TestClient:
    """Return a client whose ``/reason`` uses the two fake models."""
    store = FileSystemTranscriptStore(str(tmp / "transcripts"))
    search = SearchService(FileSystemIndexRepository(str(tmp / "none")),
                           tokenize, FakeEmbedder())
    reasoner = Reasoner(lambda alias: cold, live_models=lambda alias: live)
    return TestClient(create_app(
        search, AnswerService(search, FakeGenerator()),
        TranscriptRoutes(
            IngestTranscript(store), ShowIngestedTranscript(store), token,
            reasoner=reasoner,
        ),
    ))


def test_suggester_request_returns_a_valid_response(tmp_path: Path) -> None:
    live, cold = FakeChatModel([REPLY]), FakeChatModel(["x"])
    response = client(tmp_path, live, cold).post(
        "/reason", json=EXAMPLE, headers=AUTH
    )
    assert response.status_code == 200
    body = response.json()
    assert list(RESPONSE.iter_errors(body)) == []
    assert body["status"] == "ok" and body["request_id"] == "r-91"
    assert body["suggestions"][0]["about"] == ["t7.s1"]
    assert live.calls and not cold.calls


def test_judge_role_uses_the_cold_model(tmp_path: Path) -> None:
    live, cold = FakeChatModel(["x"]), FakeChatModel(["confirmado"])
    judge: Dict[str, Any] = {
        "schema": "reasoner.request.v1", "request_id": "j1",
        "role": "judge", "template": "judge.raw@1",
        "context": {"window": []},
        "payload": {"system": "s", "user": "u"},
    }
    body = client(tmp_path, live, cold).post(
        "/reason", json=judge, headers=AUTH
    ).json()
    assert body["verdict"] == {"reply": "confirmado"}
    assert cold.calls and not live.calls


def test_requires_the_bearer_token(tmp_path: Path) -> None:
    app = client(tmp_path, FakeChatModel(), FakeChatModel())
    assert app.post("/reason", json=EXAMPLE).status_code == 401
    assert app.post("/reason", json=EXAMPLE, headers={
        "Authorization": "Bearer nope"}).status_code == 401


def test_not_mounted_without_a_token(tmp_path: Path) -> None:
    app = client(tmp_path, FakeChatModel(), FakeChatModel(), token="")
    assert app.post("/reason", json=EXAMPLE, headers=AUTH).status_code == 404


def test_contract_violations_are_422(tmp_path: Path) -> None:
    app = client(tmp_path, FakeChatModel(), FakeChatModel())
    for bad in (
        {**EXAMPLE, "role": "oracle"},
        {**EXAMPLE, "surprise": 1},
        {**EXAMPLE, "template": "suggester.nope@9"},
        {"request_id": "x"},
    ):
        out = app.post("/reason", json=bad, headers=AUTH)
        assert out.status_code == 422, bad
    assert app.post("/reason", content=b"not json",
                    headers=AUTH).status_code == 422


def test_oversized_body_is_413(tmp_path: Path) -> None:
    app = client(tmp_path, FakeChatModel(), FakeChatModel())
    huge: List[str] = ["x" * 1000] * 300
    assert app.post("/reason", json={**EXAMPLE, "payload": {"a": huge}},
                    headers=AUTH).status_code == 413


def stream_lines(response: Any) -> List[Dict[str, Any]]:
    """Return the NDJSON lines of a streamed answer."""
    return [json.loads(line) for line in response.text.splitlines() if line]


class Pieces(FakeChatModel):
    """A live backend that streams its reply in pieces."""

    def complete_streaming(self, system: str, user: str, max_tokens: int,
                           deadline_s: Any, on_text: Any) -> str:
        """Answer in four-character pieces."""
        text = self.complete(system, user, max_tokens)
        for at in range(0, len(text), 4):
            on_text(text[at:at + 4])
        return text


def bot_request() -> Dict[str, Any]:
    """A conversational request."""
    return {
        "schema": "reasoner.request.v1", "request_id": "b1",
        "role": "conversational", "template": "conversational.voicebot@1",
        "context": {"window": [{"id": "u1", "speaker": "S1",
                                "text": "¿Cuándo abrís?"}]},
        "retrieved": [{"id": "kb:1", "text": "Abrimos de 9 a 18."}],
    }


def test_stream_sends_text_pieces_then_the_response(tmp_path: Path) -> None:
    reply = '{"text": "Abrimos de 9 a 18. Te espero.", "cites": ["kb:1"]}'
    live, cold = Pieces([reply]), FakeChatModel(["x"])
    response = client(tmp_path, live, cold).post(
        "/reason/stream", json=bot_request(), headers=AUTH
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/x-ndjson")
    lines = stream_lines(response)
    texts = [x["text"] for x in lines if x["type"] == "text"]
    assert len(texts) > 3 and "".join(texts) == reply
    last = lines[-1]
    assert last["type"] == "response"
    assert list(RESPONSE.iter_errors(last["response"])) == []
    assert last["response"]["reply"]["cites"] == ["kb:1"]


def test_stream_refuses_bad_requests_before_any_line(tmp_path: Path) -> None:
    api = client(tmp_path, FakeChatModel(["x"]), FakeChatModel(["x"]))
    bad = bot_request()
    bad["template"] = "conversational.nope@1"
    assert api.post("/reason/stream", json=bad,
                    headers=AUTH).status_code == 422
    assert api.post("/reason/stream", json={"x": 1},
                    headers=AUTH).status_code == 422
    assert api.post("/reason/stream", json=bot_request()).status_code == 401


def test_stream_reports_an_error_line(tmp_path: Path) -> None:
    from src.domain.errors import RagError

    class Broken(FakeChatModel):
        def complete(self, system: str, user: str, max_tokens: int) -> str:
            raise RagError("boom")

    lines = stream_lines(client(tmp_path, Broken(), FakeChatModel(["x"]))
                         .post("/reason/stream", json=bot_request(),
                               headers=AUTH))
    assert lines == [{"type": "error", "message": "boom"}]
