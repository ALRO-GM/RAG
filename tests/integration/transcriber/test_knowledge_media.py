"""``/internal/knowledge-media``: the knowledge library's recordings (KN-3a).

Fake models (``runtime="fake"``): the whole pipeline runs, in the
service's job runner, without torch or a download.
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.audio import (
    mp4_aac,
    scene,
    sound_file,
    tone,
)
from transcriber.adapters.inbound.http_api import create_app
from transcriber.application.knowledge_media import KNOWLEDGE_MEDIA_DIR
from transcriber.bootstrap import (
    TranscriberContainer,
    build_container,
    start_container,
)
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import StateError
from transcriber.domain.models import Session, SessionState

CODE = "a-long-enough-access-code"
TOKEN = "k" * 40
BEARER = {"Authorization": f"Bearer {TOKEN}"}
JOB = "kn-" + "0" * 32
OTHER = "kn-" + "1" * 32
WAIT = 30.0
ROUTE = "/internal/knowledge-media"


def settings(tmp_path: Path) -> TranscriberSettings:
    """Fake models, transcripts posted to a RAG over HTTP."""
    return TranscriberSettings.model_validate({
        "data_dir": str(tmp_path / "data"), "runtime": "fake",
        "publisher": "http",
        "publish_http": {"url": "http://127.0.0.1:9/rag"},
    })


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A container linked to a RAG; its runner is closed afterwards."""
    built = build_container(
        settings(tmp_path), environ={"TRANSCRIBER_PUBLISH_TOKEN": TOKEN}
    )
    yield built
    built.runner.close(WAIT)


@pytest.fixture()
def client(container: TranscriberContainer) -> TestClient:
    """A client of the API (no access code: these routes want the token)."""
    return TestClient(create_app(container, CODE))


def two_voices() -> bytes:
    """Two voices, 14 seconds, as a WAV file."""
    return sound_file(scene([
        ("220", 5.0), ("-", 1.6), ("880", 5.0), ("-", 1.6), ("220", 1.6),
    ]), 16000, "wav")


def wait(client: TestClient, job: str = JOB) -> Dict[str, Any]:
    """Poll the job until it ends; return its last view."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        view: Dict[str, Any] = client.get(
            f"{ROUTE}/{job}", headers=BEARER
        ).json()
        if view.get("state") in ("done", "failed"):
            return view
        time.sleep(0.02)
    raise AssertionError("the job did not end")


def submit(
    client: TestClient, data: bytes, kind: str = "wav", job: str = JOB,
    **params: Any,
) -> Any:
    """Send ``data`` as job ``job``."""
    return client.put(
        f"{ROUTE}/{job}", headers=BEARER, content=data,
        params={"format": kind, "language": "es", **params},
    )


def test_a_recording_is_transcribed_and_never_a_session(
    container: TranscriberContainer, client: TestClient,
) -> None:
    """The transcript comes back; «Sesiones» and publishing never see it."""
    started = submit(client, two_voices())
    assert started.status_code == 202, started.text
    assert started.json()["state"] in ("queued", "running", "done")
    view = wait(client)
    assert view["state"] == "done", view
    transcript = view["transcript"]
    assert transcript["schema"] == "transcript.v1"
    assert transcript["session"]["id"] == JOB
    assert len(transcript["speakers"]) == 2
    assert not any(s["enrolled"] for s in transcript["speakers"])
    assert transcript["utterances"]
    assert view["duration"] == pytest.approx(14.8, abs=0.1)
    # Sent again: left as it is, not transcribed twice.
    assert submit(client, two_voices()).json()["state"] == "done"
    # Not a session: the list, the API's list and the published folder.
    assert container.list_sessions().sessions == []
    listed = client.get("/api/sessions", headers={"X-Access-Code": CODE})
    assert listed.json()["sessions"] == []
    stored = container.settings.data_dir / KNOWLEDGE_MEDIA_DIR / "sessions"
    assert (stored / JOB).is_dir()
    assert not (container.settings.data_dir / "sessions" / JOB).exists()
    assert not container.settings.published().exists()
    assert client.delete(f"{ROUTE}/{JOB}", headers=BEARER).status_code == 204
    assert not (stored / JOB).exists()
    assert client.get(f"{ROUTE}/{JOB}", headers=BEARER).status_code == 404
    assert client.delete(f"{ROUTE}/{JOB}", headers=BEARER).status_code == 204


def test_a_video_file_is_read(client: TestClient) -> None:
    """The sound of an MP4 (AAC) is transcribed like any recording."""
    video = mp4_aac(tone(4.0, rate=44100, hz=220.0))
    assert submit(client, video, "mp4").status_code == 202
    view = wait(client)
    assert view["state"] == "done", view
    assert view["transcript"]["utterances"]


@pytest.mark.parametrize("data, params, reason", [
    (sound_file(tone(0.5, level=0.0), 16000, "wav"), {},
     "No speech recognised"),
    (b"RIFF this is not a wav file at all", {}, "decode:"),
    (sound_file(tone(130.0), 16000, "wav"), {"max_seconds": 120},
     "lasts 3 min; the limit is 2 min"),
])
def test_failures_end_the_job_with_a_reason(
    client: TestClient, data: bytes, params: Dict[str, Any], reason: str,
) -> None:
    """Silence, an unreadable file and a too long recording fail."""
    assert submit(client, data, **params).status_code == 202
    view = wait(client)
    assert view["state"] == "failed"
    assert reason in view["message"]
    assert view["transcript"] is None
    # A failed job sent again starts again.
    again = submit(client, two_voices())
    assert again.status_code == 202
    assert wait(client)["state"] == "done"


def test_the_routes_want_the_rag_token_from_this_machine(
    client: TestClient,
) -> None:
    """No token, a wrong one or a request through the tunnel are refused."""
    assert client.get(f"{ROUTE}/{JOB}").status_code == 401
    wrong = client.get(f"{ROUTE}/{JOB}",
                       headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    code = client.get(f"{ROUTE}/{JOB}", headers={"X-Access-Code": CODE})
    assert code.status_code == 401
    tunnel = client.get(f"{ROUTE}/{JOB}",
                        headers={**BEARER, "X-Client-IP": "203.0.113.5"})
    assert tunnel.status_code == 403
    assert client.get(f"{ROUTE}/not-a-job", headers=BEARER).status_code \
        == 400
    refused = submit(client, b"x", "exe")
    assert refused.status_code == 415
    assert refused.json()["code"] == "unsupported_media_type"
    assert client.get(f"{ROUTE}/{OTHER}", headers=BEARER).status_code == 404


def test_without_a_rag_there_are_no_such_routes(tmp_path: Path) -> None:
    """The directory publisher has no RAG to answer: no routes."""
    plain = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    assert plain.knowledge_media is None
    api = TestClient(create_app(plain, CODE))
    assert api.get(f"{ROUTE}/{JOB}", headers=BEARER).status_code == 404


def test_a_missing_token_and_a_running_job(
    tmp_path: Path, container: TranscriberContainer,
) -> None:
    """No token configured is 503; a running job cannot be deleted."""
    linked = build_container(settings(tmp_path / "other"), environ={})
    api = TestClient(create_app(linked, CODE))
    assert api.get(f"{ROUTE}/{JOB}", headers=BEARER).status_code == 503
    jobs = container.knowledge_media
    assert jobs is not None
    repository = jobs._sessions
    repository.create(Session(id=OTHER, language="es",
                              state=SessionState.RUNNING))
    with pytest.raises(StateError):
        jobs.forget(OTHER)
    client = TestClient(create_app(container, CODE))
    answer = client.delete(f"{ROUTE}/{OTHER}", headers=BEARER)
    assert answer.status_code == 409
    assert client.get(f"{ROUTE}/{OTHER}", headers=BEARER).json()["state"] \
        == "running"


def test_the_service_start_deletes_what_a_stopped_one_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jobs found when the service starts are deleted (the RAG resends)."""
    first = build_container(
        settings(tmp_path), environ={"TRANSCRIBER_PUBLISH_TOKEN": TOKEN}
    )
    assert first.knowledge_media is not None
    first.knowledge_media._sessions.create(
        Session(id=JOB, language="es", state=SessionState.RUNNING)
    )
    first.runner.close(WAIT)
    monkeypatch.setenv("TRANSCRIBER_PUBLISH_TOKEN", TOKEN)
    again = start_container(settings(tmp_path))
    try:
        assert again.knowledge_media is not None
        assert again.knowledge_media._sessions.ids() == []
    finally:
        again.runner.close(WAIT)
