"""HTTP adapters against an in-process server (``httpx.MockTransport``)."""

import json
from typing import Any, Dict, List

import httpx
import jsonschema
import pytest

from orchestrator.adapters.outbound.http_knowledge import HttpKnowledge
from orchestrator.adapters.outbound.http_reasoner import HttpReasoner
from orchestrator.application.ports import (
    CancelToken,
    KnowledgeUnavailable,
    ReasonerRejected,
    ReasonerUnavailable,
)
from tests.unit.orchestrator.fixtures import REPO

QUERY = {"schema": "knowledge.query.v1", "request_id": "r", "domain": "d",
         "text": "hola"}
REQUEST = json.loads((REPO / "contracts/examples/reasoner.request.v1.json")
                     .read_text("utf-8"))
RESPONSE = json.loads((REPO / "contracts/examples/reasoner.response.v1.json")
                      .read_text("utf-8"))


def client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_knowledge_posts_with_the_token() -> None:
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"chunks": []})

    out = HttpKnowledge("http://rag:8000/", "tok", client=client(handler)
                        ).query(QUERY)
    assert out == {"chunks": []}
    assert str(seen[0].url) == "http://rag:8000/knowledge/query"
    assert seen[0].headers["authorization"] == "Bearer tok"
    assert json.loads(seen[0].content) == QUERY


@pytest.mark.parametrize("response", [
    httpx.Response(500), httpx.Response(200, text="not json"),
    httpx.Response(200, json=[1]),
])
def test_knowledge_failures(response: httpx.Response) -> None:
    http = HttpKnowledge("http://x", client=client(lambda r: response))
    with pytest.raises(KnowledgeUnavailable):
        http.query(QUERY)


def test_knowledge_unreachable() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(KnowledgeUnavailable):
        HttpKnowledge("http://x", client=client(boom)).query(QUERY)


def test_reasoner_round_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/reason"
        assert "authorization" not in request.headers
        return httpx.Response(200, json=RESPONSE)

    out = HttpReasoner("http://x", client=client(handler)).reason(
        REQUEST, CancelToken())
    schema = json.loads((REPO / "contracts/reasoner.response.v1.schema.json")
                        .read_text("utf-8"))
    jsonschema.validate(out, schema)


def test_cancelled_before_sending_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    token = CancelToken()
    token.cancel()
    out: Dict[str, Any] = HttpReasoner(
        "http://x", client=client(handler)).reason(REQUEST, token)
    assert out["status"] == "cancelled"


@pytest.mark.parametrize("response", [
    httpx.Response(503), httpx.Response(200, text="<html>"),
    httpx.Response(200, json="x"),
])
def test_reasoner_failures(response: httpx.Response) -> None:
    http = HttpReasoner("http://x", client=client(lambda r: response))
    with pytest.raises(ReasonerUnavailable):
        http.reason(REQUEST, CancelToken())


def test_a_refused_request_is_told_apart() -> None:
    """#286 review: an unknown template revision (HTTP 422)."""
    http = HttpReasoner("http://x", client=client(
        lambda r: httpx.Response(422, json={"detail": "Unknown template"})))
    with pytest.raises(ReasonerRejected):
        http.reason(REQUEST, CancelToken())


def ndjson(*items: Dict[str, Any]) -> bytes:
    return "".join(json.dumps(i) + "\n" for i in items).encode()


def test_reason_stream_passes_pieces_then_returns_the_response() -> None:
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, content=ndjson(
            {"type": "text", "text": '{"text": "Ho'}, {},
            {"type": "text", "text": 'la."}'},
            {"type": "response", "response": RESPONSE},
        ) + b"\n")

    pieces: List[str] = []
    out = HttpReasoner("http://rag", client=client(handler)).reason_stream(
        REQUEST, CancelToken(), pieces.append)
    assert seen == ["/reason/stream"]
    assert pieces == ['{"text": "Ho', 'la."}'] and out == RESPONSE


def test_reason_stream_falls_back_to_reason_on_an_old_service() -> None:
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/stream"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json=RESPONSE)

    pieces: List[str] = []
    out = HttpReasoner("http://rag", client=client(handler)).reason_stream(
        REQUEST, CancelToken(), pieces.append)
    assert seen == ["/reason/stream", "/reason"]
    assert out == RESPONSE and pieces == []


def test_reason_stream_cancel_stops_reading() -> None:
    cancel = CancelToken()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ndjson(
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"},
            {"type": "response", "response": RESPONSE}))

    def piece(text: str) -> None:
        cancel.cancel()

    out = HttpReasoner("http://rag", client=client(handler)).reason_stream(
        REQUEST, cancel, piece)
    assert out["status"] == "cancelled"
    already = CancelToken()
    already.cancel()
    out = HttpReasoner("http://rag", client=client(handler)).reason_stream(
        REQUEST, already, piece)
    assert out["status"] == "cancelled"


@pytest.mark.parametrize("body", [
    ndjson({"type": "error", "message": "boom"}),
    ndjson({"type": "text", "text": "a"}),
    b"not json\n",
])
def test_reason_stream_broken_streams_are_unavailable(body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(ReasonerUnavailable):
        HttpReasoner("http://rag", client=client(handler)).reason_stream(
            REQUEST, CancelToken(), lambda text: None)


def test_reason_stream_http_errors_are_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(ReasonerUnavailable):
        HttpReasoner("http://rag", client=client(handler)).reason_stream(
            REQUEST, CancelToken(), lambda text: None)
