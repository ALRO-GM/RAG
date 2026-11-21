"""The pipeline settings on the transcriber's side: ``/api/settings``
forwarded to the RAG, and the choices read when a job starts."""

import dataclasses
import http.server
import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.test_analysis_judge import StandInRag
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.outbound.http_pipeline_settings import (
    HttpPipelineSettings,
)
from transcriber.application.pipeline_settings import PipelineSettingsAccess
from transcriber.bootstrap import (
    NameModelChoice,
    build_container,
    with_recogniser,
)
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
    PublishError,
    StateError,
)

CODE = "a-long-enough-access-code"
TOKEN = "r" * 40
CHOICES = {"transcription.diarizer": "nemotron", "live.asr": "cpu"}
SAVED: Dict[str, Any] = {"stored": True, "revision": 2, "choices": CHOICES}


@pytest.fixture()
def rag() -> Iterator[Tuple[str, StandInRag]]:
    """Serve a stand-in RAG on loopback."""
    stand_in = StandInRag()
    stand_in.body = json.dumps(SAVED).encode()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), stand_in.handler()
    )
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host!s}:{port}/rag", stand_in
    finally:
        server.shutdown()
        server.server_close()


def test_the_adapter_forwards_with_the_token(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    port = HttpPipelineSettings(url, lambda: TOKEN)
    assert port.show() == SAVED
    assert port.change({"preset": "cost"}) == SAVED
    assert port.current() == SAVED["choices"]
    get, put, again = stand_in.requests
    assert (get["method"], get["path"]) == ("GET", "/rag/settings")
    assert json.loads(put["body"]) == {"preset": "cost"}
    assert put["auth"] == f"Bearer {TOKEN}"
    assert again["method"] == "GET"


def test_the_rag_s_refusals_keep_their_reason(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    port = HttpPipelineSettings(url, lambda: TOKEN)
    stand_in.status = 422
    stand_in.body = json.dumps({"detail": "Invalid option 'x'"}).encode()
    with pytest.raises(InvalidInputError, match="Invalid option"):
        port.change({"choices": {"live.asr": "x"}})
    stand_in.status, stand_in.body = 409, b"{}"
    with pytest.raises(StateError, match="refused"):
        port.change({"revision": 0})
    stand_in.status = 500
    with pytest.raises(AnalysisUnavailableError, match="500"):
        port.show()
    assert port.current() is None, "a job keeps its configuration"


def test_current_is_none_when_nothing_is_saved_or_nobody_answers(
    rag: Tuple[str, StandInRag],
) -> None:
    url, stand_in = rag
    stand_in.body = json.dumps({"stored": False, "choices": {}}).encode()
    assert HttpPipelineSettings(url, lambda: TOKEN).current() is None
    stand_in.body = b"[]"
    assert HttpPipelineSettings(url, lambda: TOKEN).current() is None
    stand_in.body = json.dumps({"stored": True, "choices": [1]}).encode()
    assert HttpPipelineSettings(url, lambda: TOKEN).current() is None
    assert HttpPipelineSettings(
        "http://127.0.0.1:9", lambda: TOKEN, read_timeout=1
    ).current() is None

    def missing() -> str:
        """No token file."""
        raise PublishError("no token")

    with pytest.raises(AnalysisUnavailableError, match="no token"):
        HttpPipelineSettings(url, missing).show()


class FakePort:
    """The RAG's settings in memory."""

    def __init__(self) -> None:
        """Nothing sent yet."""
        self.sent: Optional[Dict[str, Any]] = None

    def show(self) -> Dict[str, Any]:
        """Return the saved settings."""
        return SAVED

    def change(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the body."""
        self.sent = body
        return SAVED

    def current(self) -> Optional[Dict[str, str]]:
        """Return the saved choices."""
        return dict(CHOICES)


def client(tmp_path: Path, port: Optional[FakePort]) -> TestClient:
    """The transcriber API with ``port`` behind ``/api/settings``."""
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    container = dataclasses.replace(
        container, pipeline_settings=PipelineSettingsAccess(port)
    )
    return TestClient(create_app(container, CODE))


def test_the_routes_need_the_code_and_check_the_body(tmp_path: Path) -> None:
    port = FakePort()
    api = client(tmp_path, port)
    assert api.get("/api/settings").status_code == 401
    assert api.put("/api/settings", json={}).status_code == 401
    auth = {"X-Access-Code": CODE}
    assert api.get("/api/settings", headers=auth).json() == SAVED
    saved = api.put("/api/settings", headers=auth,
                    json={"preset": "cost", "revision": 2})
    assert saved.status_code == 200
    assert port.sent == {"preset": "cost", "revision": 2}
    for bad in (b"[1]", b"{x", json.dumps({"choices": {"a": "b c"}}).encode()):
        answer = api.put("/api/settings", headers=auth, content=bad)
        assert answer.status_code == 400, bad
    off = client(tmp_path, None).get("/api/settings", headers=auth)
    assert off.status_code == 503
    assert off.json()["code"] == "analysis_unavailable"


def test_the_container_reads_the_choices_from_the_rag(
    rag: Tuple[str, StandInRag], tmp_path: Path,
) -> None:
    url, _ = rag
    token = tmp_path / "token"
    token.write_text(TOKEN, encoding="utf-8")
    token.chmod(0o600)
    linked = build_container(TranscriberSettings.model_validate({
        "data_dir": str(tmp_path / "data"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": url, "token_file": str(token)},
    }), environ={})
    assert linked.pipeline_settings.current() == SAVED["choices"]
    assert linked.transcribe._choices is not None
    plain = build_container(TranscriberSettings(
        data_dir=tmp_path / "plain", runtime="fake",
    ))
    assert plain.pipeline_settings.current() is None


def test_the_name_model_is_built_once_per_choice(tmp_path: Path) -> None:
    saved: Dict[str, str] = {}
    settings = TranscriberSettings(data_dir=tmp_path, runtime="real",
                                   device="cpu")
    names = NameModelChoice(settings, lambda: saved)
    configured = names.chosen()
    assert configured is names.configured is not None
    saved["transcription.names"] = "qwen3-4b"
    small = names.chosen()
    assert small is not None and small is not configured
    assert small.models == ("Qwen/Qwen3-4B",)
    assert names.chosen() is small
    saved["transcription.names"] = "rules"
    assert names.chosen() is None
    names.release()


def test_the_recogniser_is_chosen_once_at_start_up(tmp_path: Path) -> None:
    real = TranscriberSettings(data_dir=tmp_path, runtime="real")
    port = FakePort()
    assert with_recogniser(real, PipelineSettingsAccess(port)) is real
    port.current = lambda: {  # type: ignore[method-assign]
        "transcription.recogniser": "whisper-large-v3",
    }
    chosen = with_recogniser(real, PipelineSettingsAccess(port))
    assert chosen.recogniser == "openai/whisper-large-v3"
    fake = TranscriberSettings(data_dir=tmp_path, runtime="fake")
    assert with_recogniser(fake, PipelineSettingsAccess(port)) is fake
