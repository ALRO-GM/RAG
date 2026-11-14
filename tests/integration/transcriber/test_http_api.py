"""The HTTP API with TestClient over the real adapters and fake models.

``build_container`` with ``runtime="fake"`` gives real storage in a
temporary directory, the real decoder and job runner, and the
deterministic stand-in models (a 220 Hz tone is one person, an 880 Hz
tone another).
"""

import io
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.audio import (
    pcm_bytes,
    scene,
    sound_file,
    tone,
)
from transcriber.adapters.inbound.access import (
    PLAYBACK_COOKIE,
    FailureLimiter,
    check_code,
    client_key,
)
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.playback import (
    UNSATISFIABLE,
    RenderCache,
    parse_range,
)
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import ApiSettings, TranscriberSettings

CODE = "a-long-enough-access-code"
AUTH = {"X-Access-Code": CODE}
CHUNK = 32000  # bytes: one second of 16 kHz 16-bit PCM
WAIT = 30.0
WAV_HEADER = 44


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_container(tmp_path: Path, **api: Any) -> TranscriberContainer:
    """A fake-runtime container over ``tmp_path / "data"``."""
    return build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
        api=ApiSettings(**api),
    ))


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """The container; its job runner is closed afterwards."""
    built = make_container(tmp_path)
    yield built
    built.runner.close(WAIT)


@pytest.fixture()
def client(container: TranscriberContainer) -> TestClient:
    """A client of the API that sends the access code."""
    return TestClient(create_app(container, CODE), headers=AUTH)


def assert_error(response: Any, status: int, code: str) -> Dict[str, Any]:
    """Check an error response; return its body."""
    assert response.status_code == status, response.text
    body: Dict[str, Any] = response.json()
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]
    return body


def recording() -> bytes:
    """Ana, someone else, Ana: 20 seconds of 16 kHz PCM."""
    return pcm_bytes(scene([
        ("220", 6.0), ("-", 1.6), ("880", 6.0), ("-", 1.6), ("220", 4.8),
    ]))


def enrol(client: TestClient, speaker: str = "ana") -> Any:
    """Enrol ``speaker`` from 30 seconds of a 220 Hz voice (WAV)."""
    sample = sound_file(tone(30.0, hz=220.0), 16000, "wav")
    return client.post(
        f"/api/speakers/{speaker}/enrolment",
        params={"name": "Ana", "language": "es", "consent": "true",
                "format": "wav"},
        content=sample,
    )


def record(client: TestClient, session_id: str, data: bytes) -> None:
    """Open ``session_id`` and send ``data`` in one-second chunks."""
    opened = client.post("/api/sessions", json={
        "id": session_id, "language": "es", "title": "Reparto",
    })
    assert opened.status_code == 201, opened.text
    for number, start in enumerate(range(0, len(data), CHUNK)):
        sent = client.put(
            f"/api/sessions/{session_id}/chunks/{number}",
            content=data[start:start + CHUNK],
            headers={"Content-Type": "application/octet-stream"},
        )
        assert sent.json() == {"stored": True}


def wait_until_done(client: TestClient, session_id: str) -> Dict[str, Any]:
    """Poll the session until its job ends; return the last view."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        view: Dict[str, Any] = client.get(f"/api/sessions/{session_id}").json()
        if view["state"] in ("done", "failed"):
            return view
        time.sleep(0.02)
    raise AssertionError("the job did not end")


def test_the_whole_life_of_a_session(
    client: TestClient, tmp_path: Path
) -> None:
    enrolled = enrol(client)
    assert enrolled.status_code == 201, enrolled.text
    assert enrolled.json()["id"] == "ana"
    assert client.get("/api/speakers").json()["speakers"][0]["name"] == "Ana"

    data = recording()
    record(client, "equipo-a", data)
    last = (len(data) - 1) // CHUNK
    again = client.put(
        f"/api/sessions/equipo-a/chunks/{last}",
        content=data[last * CHUNK:],
    )
    assert again.json() == {"stored": False}  # a retried chunk, kept once
    view = client.get("/api/sessions/equipo-a").json()
    assert (view["state"], view["chunks"]) == ("recording", last + 1)

    closed = client.post("/api/sessions/equipo-a/close")
    assert closed.json()["state"] == "uploaded"
    queued = client.post(
        "/api/sessions/equipo-a/transcription",
        json={"num_speakers": 2, "expected_speakers": ["ana"]},
    )
    assert queued.status_code == 202, queued.text
    assert queued.json()["state"] == "queued"
    done = wait_until_done(client, "equipo-a")
    assert (done["state"], done["stage"], done["fraction"]) == (
        "done", "done", 1.0,
    )

    transcript = client.get("/api/sessions/equipo-a/transcript").json()
    assert [(s["id"], s["enrolled"]) for s in transcript["speakers"]] == [
        ("ana", True), ("S1", False),
    ]
    assert transcript["utterances"][0]["words"]

    whole = client.get("/api/sessions/equipo-a/audio")
    assert whole.status_code == 200
    assert whole.headers["content-type"] == "audio/wav"
    assert whole.headers["accept-ranges"] == "bytes"
    with wave.open(io.BytesIO(whole.content)) as heard:
        assert (heard.getframerate(), heard.getnchannels()) == (16000, 1)
        frames = heard.readframes(heard.getnframes())
    assert frames == data  # lossless from the phone to the player
    size = len(whole.content)
    first = client.get(
        "/api/sessions/equipo-a/audio", headers={"Range": "bytes=0-99"}
    )
    assert first.status_code == 206
    assert first.headers["content-range"] == f"bytes 0-99/{size}"
    assert first.content == whole.content[:100]

    renamed = client.patch(
        "/api/sessions/equipo-a/speakers/S1", json={"name": "Luis"}
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["speakers"][1]["name"] == "Luis"
    utterance = renamed.json()["utterances"][-1]["id"]
    moved = client.patch(
        f"/api/sessions/equipo-a/utterances/{utterance}",
        json={"speaker": "S1"},
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["utterances"][-1]["speaker"] == "S1"

    published = client.post("/api/sessions/equipo-a/publish")
    assert published.status_code == 200
    assert published.json()["schema"] == "transcript.v1"
    assert (tmp_path / "data" / "published"
            / "equipo-a.transcript.v1.json").is_file()

    listed = client.get("/api/sessions").json()
    assert [s["id"] for s in listed["sessions"]] == ["equipo-a"]
    assert listed["unreadable"] == 0

    assert client.delete("/api/sessions/equipo-a").status_code == 204
    assert_error(client.get("/api/sessions/equipo-a"), 404, "not_found")
    assert not (tmp_path / "data" / "sessions" / "equipo-a").exists()
    assert client.delete("/api/speakers/ana").status_code == 204
    assert client.get("/api/speakers").json() == {"speakers": []}


def test_an_uploaded_file_and_a_language_given_at_queueing(
    client: TestClient,
) -> None:
    wav = sound_file(scene([("220", 4.0), ("-", 1.6), ("880", 4.0)]),
                     16000, "wav")
    made = client.post(
        "/api/sessions/importada/upload",
        params={"filename": "reunión final.WAV", "language": "es"},
        content=wav,
    )
    assert made.status_code == 201, made.text
    assert made.json()["state"] == "uploaded"
    queued = client.post(
        "/api/sessions/importada/transcription", json={"language": "en"}
    )
    assert queued.json()["language"] == "en"
    assert wait_until_done(client, "importada")["state"] == "done"
    transcript = client.get("/api/sessions/importada/transcript").json()
    assert transcript["session"]["language"] == "en"
    no_extension = client.post(
        "/api/sessions/otra/upload",
        params={"filename": "audio", "language": "es"}, content=wav,
    )
    assert_error(no_extension, 400, "invalid_input")


class TestChunks:
    def test_a_missing_chunk_is_refused(self, client: TestClient) -> None:
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        client.put("/api/sessions/s1/chunks/0", content=b"\x01\x00")
        gap = client.put("/api/sessions/s1/chunks/2", content=b"\x02\x00")
        assert_error(gap, 400, "invalid_input")
        assert "gap" in gap.json()["message"]
        assert client.get("/api/sessions/s1").json()["chunks"] == 1

    def test_an_earlier_chunk_or_other_bytes_are_refused(
        self, client: TestClient
    ) -> None:
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        client.put("/api/sessions/s1/chunks/0", content=b"\x01\x00")
        client.put("/api/sessions/s1/chunks/1", content=b"\x02\x00")
        assert_error(
            client.put("/api/sessions/s1/chunks/0", content=b"\x01\x00"),
            400, "invalid_input",
        )
        assert_error(
            client.put("/api/sessions/s1/chunks/1", content=b"\x09\x00"),
            400, "invalid_input",
        )

    def test_bad_chunks(self, client: TestClient) -> None:
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        assert_error(client.put("/api/sessions/s1/chunks/0", content=b""),
                     400, "invalid_input")
        assert_error(client.put("/api/sessions/s1/chunks/0", content=b"\x01"),
                     400, "invalid_input")
        assert_error(client.put("/api/sessions/s1/chunks/x", content=b"ab"),
                     422, "invalid_request")
        assert_error(client.put("/api/sessions/nope/chunks/0", content=b"ab"),
                     404, "not_found")

    def test_closing_without_audio_or_twice(self, client: TestClient) -> None:
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        assert_error(client.post("/api/sessions/s1/close"),
                     400, "invalid_input")
        client.put("/api/sessions/s1/chunks/0", content=b"\x01\x00")
        assert client.post("/api/sessions/s1/close").status_code == 200
        assert_error(client.post("/api/sessions/s1/close"),
                     409, "wrong_state")
        assert_error(client.put("/api/sessions/s1/chunks/1",
                                content=b"\x01\x00"), 409, "wrong_state")


ROUTES = [
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions/s1"),
    ("PUT", "/api/sessions/s1/chunks/0"),
    ("POST", "/api/sessions/s1/close"),
    ("POST", "/api/sessions/s1/upload?filename=a.wav&language=es"),
    ("POST", "/api/sessions/s1/transcription"),
    ("GET", "/api/sessions/s1/transcript"),
    ("GET", "/api/sessions/s1/audio"),
    ("PATCH", "/api/sessions/s1/speakers/S1"),
    ("PATCH", "/api/sessions/s1/utterances/t0.s0"),
    ("DELETE", "/api/sessions/s1"),
    ("POST", "/api/sessions/s1/publish"),
    ("GET", "/api/speakers"),
    ("GET", "/api/enrolment/text?language=es"),
    ("POST", "/api/speakers/ana/enrolment?name=A&language=es"),
    ("DELETE", "/api/speakers/ana"),
    ("POST", "/api/access/cookie"),
]
"""Every route but the health check, with a request that reaches it."""

AUDIO_ROUTE = ("GET", "/api/sessions/s1/audio")


def playback_cookie(client: TestClient) -> Tuple[str, str]:
    """Ask for the playback cookie; return its value and its header."""
    answer = client.post("/api/access/cookie")
    assert answer.status_code == 204, answer.text
    header = answer.headers["set-cookie"]
    name, _, rest = header.partition("=")
    assert name == PLAYBACK_COOKIE
    return rest.split(";")[0], header


def with_cookie(value: str) -> Dict[str, str]:
    """Headers that send the playback cookie and no access code.

    Sent by hand: the client's cookie jar would hold back a ``Secure``
    cookie over the test client's plain http.
    """
    return {"Cookie": f"{PLAYBACK_COOKIE}={value}"}


class TestAccess:
    def test_health_needs_no_code_and_says_nothing_else(
        self, container: TranscriberContainer
    ) -> None:
        plain = TestClient(create_app(container, CODE))
        answer = plain.get("/api/health")
        assert answer.status_code == 200
        assert answer.json() == {"status": "ok"}

    def test_the_route_list_is_complete(
        self, container: TranscriberContainer
    ) -> None:
        app = create_app(container, CODE)
        served = {
            (method.upper(), path)
            for path, item in app.openapi()["paths"].items()
            for method in item
        }
        listed = {
            (method, path.split("?")[0]
             .replace("/s1", "/{session_id}")
             .replace("/ana", "/{speaker_id}")
             .replace("/0", "/{number}")
             .replace("/S1", "/{label}")
             .replace("/t0.s0", "/{utterance_id}"))
            for method, path in ROUTES
        }
        assert served == listed | {("GET", "/api/health")}

    @pytest.mark.parametrize("method, path", ROUTES)
    def test_every_route_needs_the_code(
        self, container: TranscriberContainer, method: str, path: str
    ) -> None:
        plain = TestClient(create_app(container, CODE))
        container.open_session("s1", "es")
        for headers in ({}, {"X-Access-Code": "wrong"},
                        {"X-Access-Code": CODE + "x"}):
            answer = plain.request(method, path, headers=headers,
                                   content=b"\x00\x00")
            assert_error(answer, 401, "unauthorized")
        assert container.sessions.get("s1") is not None  # untouched

    def test_the_playback_cookie_is_a_token_not_the_code(
        self, client: TestClient
    ) -> None:
        value, header = playback_cookie(client)
        assert value != CODE and CODE not in header
        assert len(value) >= 32
        attributes = {
            part.strip().lower() for part in header.split(";")[1:]
        }
        assert {"httponly", "secure", "samesite=strict", "path=/api"} <= (
            attributes
        )
        assert playback_cookie(client)[0] == value  # one token per start

    def test_secure_can_be_dropped_for_local_http(
        self, tmp_path: Path
    ) -> None:
        container = make_container(tmp_path, cookie_secure=False)
        client = TestClient(create_app(container, CODE), headers=AUTH)
        _, header = playback_cookie(client)
        attributes = [p.strip().lower() for p in header.split(";")[1:]]
        assert "secure" not in attributes and "httponly" in attributes
        container.runner.close(WAIT)

    def test_a_new_start_makes_a_new_token(
        self, container: TranscriberContainer
    ) -> None:
        first = TestClient(create_app(container, CODE), headers=AUTH)
        second = TestClient(create_app(container, CODE), headers=AUTH)
        old, _ = playback_cookie(first)
        assert playback_cookie(second)[0] != old
        record(second, "s1", pcm_bytes(tone(1.0)))
        plain = TestClient(second.app)
        assert_error(plain.get("/api/sessions/s1/audio",
                               headers=with_cookie(old)),
                     401, "unauthorized")

    def test_the_audio_plays_with_the_cookie_alone(
        self, client: TestClient
    ) -> None:
        record(client, "s1", pcm_bytes(tone(1.0)))
        client.post("/api/sessions/s1/close")
        value, _ = playback_cookie(client)
        plain = TestClient(client.app)
        whole = plain.get("/api/sessions/s1/audio",
                          headers=with_cookie(value))
        assert whole.status_code == 200
        assert whole.content == client.get("/api/sessions/s1/audio").content
        part = plain.get("/api/sessions/s1/audio", headers={
            **with_cookie(value), "Range": "bytes=0-99",
        })
        assert part.status_code == 206 and len(part.content) == 100
        for wrong in ("", "nope", value + "x", CODE):
            assert_error(plain.get("/api/sessions/s1/audio",
                                   headers=with_cookie(wrong)),
                         401, "unauthorized")

    @pytest.mark.parametrize(
        "method, path", [r for r in ROUTES if r != AUDIO_ROUTE]
    )
    def test_every_other_route_ignores_the_cookie(
        self, client: TestClient, method: str, path: str
    ) -> None:
        value, _ = playback_cookie(client)
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        plain = TestClient(client.app)
        answer = plain.request(method, path, headers=with_cookie(value),
                               content=b"\x00\x00")
        assert_error(answer, 401, "unauthorized")
        assert client.get("/api/sessions/s1").status_code == 200

    def test_wrong_cookies_are_throttled(self, tmp_path: Path) -> None:
        container = make_container(
            tmp_path, failed_code_limit=3, failed_code_cooldown=120.0,
        )
        app = create_app(container, CODE, now=Clock())
        owner = TestClient(app, headers=AUTH)
        value, _ = playback_cookie(owner)
        record(owner, "s1", pcm_bytes(tone(1.0)))
        attacker = TestClient(app, client=("203.0.113.9", 4000))
        for _ in range(3):
            assert_error(attacker.get("/api/sessions/s1/audio",
                                      headers=with_cookie("guess")),
                         401, "unauthorized")
        assert_error(attacker.get("/api/sessions/s1/audio",
                                  headers=with_cookie(value)),
                     429, "too_many_attempts")
        container.runner.close(WAIT)

    def test_the_code_is_never_taken_from_the_url(
        self, container: TranscriberContainer
    ) -> None:
        plain = TestClient(create_app(container, CODE))
        for query in ({"code": CODE}, {"access_code": CODE},
                      {"X-Access-Code": CODE}, {"k": CODE}):
            assert_error(plain.get("/api/sessions", params=query),
                         401, "unauthorized")
        assert_error(plain.get(f"/api/sessions/{CODE}"), 401, "unauthorized")

    def test_a_short_code_is_refused(
        self, container: TranscriberContainer
    ) -> None:
        with pytest.raises(ValueError, match="at least"):
            create_app(container, "short")

    def test_failed_codes_are_throttled(self, tmp_path: Path) -> None:
        container = make_container(
            tmp_path, failed_code_limit=3, failed_code_window=60.0,
            failed_code_cooldown=120.0,
        )
        clock = Clock()
        app = create_app(container, CODE, now=clock)
        attacker = TestClient(app, client=("203.0.113.9", 4000))
        other = TestClient(app, client=("203.0.113.10", 4000), headers=AUTH)
        wrong = {"X-Access-Code": "nope"}
        for _ in range(3):
            assert_error(attacker.get("/api/sessions", headers=wrong),
                         401, "unauthorized")
        blocked = attacker.get("/api/sessions", headers=wrong)
        assert_error(blocked, 429, "too_many_attempts")
        assert blocked.headers["retry-after"] == "120"
        # Even the right code waits for the cool-down...
        assert_error(attacker.get("/api/sessions", headers=AUTH),
                     429, "too_many_attempts")
        # ...while another client is not affected.
        assert other.get("/api/sessions").status_code == 200
        clock.now += 121
        assert attacker.get("/api/sessions", headers=AUTH).status_code == 200
        container.runner.close(WAIT)

    def test_failures_outside_the_window_are_forgotten(
        self, tmp_path: Path
    ) -> None:
        container = make_container(
            tmp_path, failed_code_limit=2, failed_code_window=10.0,
        )
        clock = Clock()
        client = TestClient(create_app(container, CODE, now=clock))
        wrong = {"X-Access-Code": "nope"}
        client.get("/api/sessions", headers=wrong)
        clock.now += 11
        assert_error(client.get("/api/sessions", headers=wrong),
                     401, "unauthorized")
        assert client.get("/api/sessions", headers=AUTH).status_code == 200
        container.runner.close(WAIT)

    def test_behind_the_tunnel_clients_are_told_apart_by_the_proxy(
        self, tmp_path: Path
    ) -> None:
        container = make_container(tmp_path, failed_code_limit=2)
        app = create_app(container, CODE, now=Clock())
        tunnel = TestClient(app, client=("127.0.0.1", 4000))
        wrong = {"X-Access-Code": "nope"}
        attacker = {**wrong, "X-Forwarded-For": "198.51.100.7, 10.0.0.2"}
        for _ in range(2):
            tunnel.get("/api/sessions", headers=attacker)
        assert_error(tunnel.get("/api/sessions", headers=attacker),
                     429, "too_many_attempts")
        phone = {**AUTH, "X-Forwarded-For": "198.51.100.8"}
        assert tunnel.get("/api/sessions", headers=phone).status_code == 200
        container.runner.close(WAIT)


class TestErrors:
    def test_unknown_session(self, client: TestClient) -> None:
        for method, path in (
            ("GET", "/api/sessions/nope"),
            ("GET", "/api/sessions/nope/transcript"),
            ("GET", "/api/sessions/nope/audio"),
            ("POST", "/api/sessions/nope/close"),
            ("POST", "/api/sessions/nope/transcription"),
            ("POST", "/api/sessions/nope/publish"),
            ("DELETE", "/api/sessions/nope"),
            ("DELETE", "/api/speakers/nope"),
        ):
            assert_error(client.request(method, path), 404, "not_found")

    @pytest.mark.parametrize("bad", ["..", "%2E%2E", ".hidden", "a%5Cb"])
    def test_ids_that_could_escape_are_refused(
        self, client: TestClient, tmp_path: Path, bad: str
    ) -> None:
        for method, path in (
            ("POST", f"/api/sessions/{bad}/upload?filename=a.wav&language=es"),
            ("GET", f"/api/sessions/{bad}"),
            ("DELETE", f"/api/sessions/{bad}"),
            ("PUT", f"/api/sessions/{bad}/chunks/0"),
            ("DELETE", f"/api/speakers/{bad}"),
        ):
            answer = client.request(method, path, content=b"\x00\x00")
            assert answer.status_code in (400, 404), (path, answer.text)
            assert answer.json()["code"] in ("invalid_input", "not_found")
        created = client.post("/api/sessions",
                              json={"id": "../x", "language": "es"})
        assert_error(created, 400, "invalid_input")
        assert [p.name for p in tmp_path.iterdir() if p.name != "data"] == []
        sessions = tmp_path / "data" / "sessions"
        assert not sessions.exists() or list(sessions.iterdir()) == []

    def test_a_slash_in_an_id_matches_no_route(
        self, client: TestClient
    ) -> None:
        assert_error(client.get("/api/sessions/..%2F..%2Fetc"),
                     404, "not_found")

    def test_oversized_bodies_are_refused_while_streaming(
        self, tmp_path: Path
    ) -> None:
        container = make_container(
            tmp_path, chunk_max_bytes=64, upload_max_bytes=128,
            enrolment_max_bytes=128, json_max_bytes=64,
        )
        client = TestClient(create_app(container, CODE), headers=AUTH)
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        assert_error(client.put("/api/sessions/s1/chunks/0",
                                content=b"\x00" * 66),
                     413, "payload_too_large")

        def stream() -> Iterator[bytes]:
            for _ in range(10):
                yield b"\x00" * 32

        streamed = client.put("/api/sessions/s1/chunks/0", content=stream())
        assert_error(streamed, 413, "payload_too_large")
        assert client.get("/api/sessions/s1").json()["chunks"] == 0
        assert_error(client.post(
            "/api/sessions/s2/upload",
            params={"filename": "a.wav", "language": "es"},
            content=b"\x00" * 200), 413, "payload_too_large")
        assert_error(client.post(
            "/api/speakers/ana/enrolment",
            params={"name": "Ana", "language": "es", "consent": "true"},
            content=b"\x00" * 200), 413, "payload_too_large")
        assert_error(client.post(
            "/api/sessions",
            json={"id": "s3", "language": "es", "title": "x" * 100}),
            413, "payload_too_large")
        assert client.get("/api/sessions").json()["sessions"][0]["id"] == "s1"
        container.runner.close(WAIT)

    def test_invalid_json_bodies(self, client: TestClient) -> None:
        assert_error(client.post("/api/sessions", content=b"{"),
                     422, "invalid_request")
        assert_error(client.post("/api/sessions", json={"id": "s1"}),
                     422, "invalid_request")
        assert_error(client.post("/api/sessions", json={
            "id": "s1", "language": "es", "extra": 1,
        }), 422, "invalid_request")

    def test_enrolment_needs_explicit_consent(
        self, client: TestClient, container: TranscriberContainer
    ) -> None:
        sample = sound_file(tone(30.0), 16000, "wav")
        for consent in ("", "false", "1", "yes", "True"):
            answer = client.post(
                "/api/speakers/ana/enrolment",
                params={"name": "Ana", "language": "es",
                        "consent": consent, "format": "wav"},
                content=sample,
            )
            assert_error(answer, 400, "consent_required")
        assert container.voiceprints.get("ana") is None

    def test_a_rejected_sample_says_why(self, client: TestClient) -> None:
        short = sound_file(tone(3.0), 16000, "wav")
        answer = client.post(
            "/api/speakers/ana/enrolment",
            params={"name": "Ana", "language": "es", "consent": "true",
                    "format": "wav"},
            content=short,
        )
        body = assert_error(answer, 422, "sample_rejected")
        assert body["reason"]

    def test_enrolment_text(self, client: TestClient) -> None:
        text = client.get("/api/enrolment/text",
                          params={"language": "es", "name": "Ana"}).json()
        assert text["version"] == "es-1"
        assert text["text"].startswith("Me llamo Ana.")
        assert text["consent_version"] == "consent-1"
        assert_error(client.get("/api/enrolment/text",
                                params={"language": "xx"}),
                     400, "invalid_input")

    def test_server_errors_hide_their_details(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        (tmp_path / "data" / "sessions" / "s1" / "session.json").write_text(
            "{", encoding="utf-8"
        )
        answer = client.get("/api/sessions/s1")
        body = assert_error(answer, 500, "storage_error")
        assert "session.json" not in body["message"]
        assert str(tmp_path) not in answer.text
        listed = client.get("/api/sessions").json()
        assert (listed["sessions"], listed["unreadable"]) == ([], 1)

    def test_unexpected_errors_are_generic(
        self, container: TranscriberContainer
    ) -> None:
        def broken() -> None:
            raise RuntimeError("/secret/path exploded")

        app = create_app(container, CODE)
        # The container is frozen; this test alone swaps a use case.
        object.__setattr__(container, "list_sessions", broken)
        client = TestClient(app, headers=AUTH,
                            raise_server_exceptions=False)
        answer = client.get("/api/sessions")
        assert_error(answer, 500, "internal_error")
        assert "secret" not in answer.text

    def test_unknown_routes_and_methods(self, client: TestClient) -> None:
        assert_error(client.get("/api/nothing"), 404, "not_found")
        assert_error(client.put("/api/sessions"), 405, "method_not_allowed")

    def test_api_answers_are_not_cached_or_sniffed(
        self, client: TestClient
    ) -> None:
        answer = client.get("/api/sessions")
        assert answer.headers["cache-control"] == "no-store"
        assert answer.headers["x-content-type-options"] == "nosniff"


class TestAudioRanges:
    @pytest.fixture()
    def audio(self, client: TestClient) -> bytes:
        """The WAV of a recorded session ``s1``."""
        record(client, "s1", pcm_bytes(tone(1.0)))
        client.post("/api/sessions/s1/close")
        whole = client.get("/api/sessions/s1/audio")
        assert whole.status_code == 200
        return whole.content

    @pytest.mark.parametrize("header, first, last", [
        ("bytes=0-0", 0, 0),
        ("bytes=0-99", 0, 99),
        ("bytes=1000-1999", 1000, 1999),
        ("bytes=1000-", 1000, None),
        ("bytes=-500", -500, None),
        ("bytes=31000-999999", 31000, None),
        ("bytes=-999999", 0, None),
    ])
    def test_satisfiable_ranges(
        self, client: TestClient, audio: bytes, header: str, first: int,
        last: Any,
    ) -> None:
        size = len(audio)
        start = first if first >= 0 else size + first
        end = size - 1 if last is None else last
        answer = client.get("/api/sessions/s1/audio",
                            headers={"Range": header})
        assert answer.status_code == 206
        assert answer.headers["content-range"] == f"bytes {start}-{end}/{size}"
        assert answer.headers["content-type"] == "audio/wav"
        assert answer.headers["accept-ranges"] == "bytes"
        assert int(answer.headers["content-length"]) == end - start + 1
        assert answer.content == audio[start:end + 1]

    @pytest.mark.parametrize("header", ["bytes=999999-", "bytes=-0"])
    def test_unsatisfiable_ranges(
        self, client: TestClient, audio: bytes, header: str
    ) -> None:
        answer = client.get("/api/sessions/s1/audio",
                            headers={"Range": header})
        assert_error(answer, 416, "range_not_satisfiable")
        assert answer.headers["content-range"] == f"bytes */{len(audio)}"

    @pytest.mark.parametrize("header", [
        "bytes=5-2", "items=0-1", "bytes=0-1,4-5", "bytes=x-y", "bytes=-",
    ])
    def test_ranges_not_understood_get_the_whole_audio(
        self, client: TestClient, audio: bytes, header: str
    ) -> None:
        answer = client.get("/api/sessions/s1/audio",
                            headers={"Range": header})
        assert answer.status_code == 200
        assert answer.content == audio

    def test_a_time_is_a_byte_offset(
        self, client: TestClient, audio: bytes
    ) -> None:
        """What makes seeking exact: second ``t`` starts at a known byte."""
        offset = WAV_HEADER + 2 * 16000 // 2  # 0.5 s
        part = client.get("/api/sessions/s1/audio",
                          headers={"Range": f"bytes={offset}-{offset + 1}"})
        expected = pcm_bytes(tone(1.0))[16000:16002]
        assert part.content == expected

    def test_a_session_still_recording_plays_what_arrived(
        self, client: TestClient
    ) -> None:
        client.post("/api/sessions", json={"id": "live", "language": "es"})
        client.put("/api/sessions/live/chunks/0", content=b"\x01\x00" * 8)
        first = client.get("/api/sessions/live/audio").content
        client.put("/api/sessions/live/chunks/1", content=b"\x02\x00" * 8)
        second = client.get("/api/sessions/live/audio").content
        assert len(second) - len(first) == 16


class TestPlaybackParts:
    def test_parse_range_edges(self) -> None:
        assert parse_range(None, 10) is None
        assert parse_range("bytes=0-", 0) is UNSATISFIABLE
        assert parse_range("bytes=-5", 0) is UNSATISFIABLE
        assert parse_range(" BYTES = 2 - 4 ", 10) == (2, 4)

    def test_the_cache_keeps_at_most_its_limit(self) -> None:
        cache = RenderCache(limit=10)
        made: List[str] = []

        def render(value: bytes) -> Any:
            def run() -> bytes:
                made.append(value.decode())
                return value
            return run

        assert cache.get(("a", "1"), render(b"aaaaaa")) == b"aaaaaa"
        assert cache.get(("a", "1"), render(b"zzzzzz")) == b"aaaaaa"
        cache.get(("b", "1"), render(b"bbbbbb"))  # evicts a
        cache.get(("a", "1"), render(b"aaaaaa"))
        assert made == ["aaaaaa", "bbbbbb", "aaaaaa"]
        cache.get(("c", "1"), render(b"c" * 11))  # too big to keep
        cache.forget("a")
        cache.get(("a", "1"), render(b"aaaaaa"))
        assert made[-2:] == ["c" * 11, "aaaaaa"]


class TestAccessParts:
    def test_check_code(self) -> None:
        assert check_code(CODE, CODE)
        assert not check_code(CODE, None)
        assert not check_code(CODE, "")
        assert not check_code(CODE, "ñ" * 20)

    @pytest.mark.parametrize("peer, forwarded, key", [
        ("203.0.113.9", "198.51.100.7", "203.0.113.9"),
        ("127.0.0.1", "198.51.100.7, 10.0.0.1", "198.51.100.7"),
        ("::1", " 2001:db8::1 ", "2001:db8::1"),
        (None, "198.51.100.7", "198.51.100.7"),
        (None, None, "unknown"),
        ("127.0.0.1", None, "127.0.0.1"),
        ("testclient", "198.51.100.7", "testclient"),
    ])
    def test_client_key(
        self, peer: Any, forwarded: Any, key: str
    ) -> None:
        headers = {} if forwarded is None else {"X-Forwarded-For": forwarded}
        assert client_key(peer, headers) == key

    def test_the_tables_stay_bounded(self) -> None:
        clock = Clock()
        limiter = FailureLimiter(2, 10.0, 10.0, now=clock, max_clients=4)
        for n in range(20):
            limiter.failed(f"c{n}")
            limiter.failed(f"c{n}")
            clock.now += 1
        assert len(limiter._failures) + len(limiter._blocked) <= 4
        assert limiter.retry_after("c19") > 0


def test_static_pages_in_development(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "index.html").write_text("<p>hola</p>", encoding="utf-8")
    container = make_container(tmp_path, static_dir=pages)
    client = TestClient(create_app(container, CODE))
    assert client.get("/").text == "<p>hola</p>"
    assert client.get("/api/health").json() == {"status": "ok"}
    assert_error(client.get("/api/sessions"), 401, "unauthorized")
    container.runner.close(WAIT)


def test_wav_rendering_is_exact_for_decoded_audio() -> None:
    """Samples decoded from 16-bit PCM come back as the same integers."""
    from transcriber.adapters.inbound.playback import wav_bytes
    from transcriber.application.ports import Audio

    ints = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
    audio = Audio(ints.astype(np.float32) / 32768.0, 16000)
    with wave.open(io.BytesIO(wav_bytes(audio))) as heard:
        assert heard.readframes(5) == ints.tobytes()
