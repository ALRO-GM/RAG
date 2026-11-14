"""The HTTP API with TestClient over the real adapters and fake models.

``build_container`` with ``runtime="fake"`` gives real storage in a
temporary directory, the real decoder and job runner, and the
deterministic stand-in models (a 220 Hz tone is one person, an 880 Hz
tone another).
"""

import hmac
import io
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

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
    parse_range,
)
from transcriber.application.ports import Progress, RecordedAudio
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


def ana_alone() -> bytes:
    """Only Ana, with two pauses: 20 seconds of 16 kHz PCM.

    For a session declared as one person: forcing the two voices of
    :func:`recording` into one group gives a mean voice whose cosine
    with Ana's voiceprint (about 0.78 with the fake encoder) is below
    the measured match threshold, so it rightly stays anonymous.
    """
    return pcm_bytes(scene([
        ("220", 6.0), ("-", 1.6), ("220", 6.0), ("-", 1.6), ("220", 4.8),
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
    assert_error(no_extension, 400, "unsupported_format")


class TestConsentAndOptions:
    def test_the_notices_come_from_the_api(self, client: TestClient) -> None:
        text = client.get("/api/enrolment/text",
                          params={"language": "en", "name": "Ana"}).json()
        assert text["consent"]["kind"] == "enrolment"
        assert text["consent"]["version"] == text["consent_version"]
        assert text["consent"]["points"] and text["consent"]["accept"]
        notice = client.get("/api/consent",
                            params={"kind": "recording", "language": "es"})
        assert notice.status_code == 200
        body = notice.json()
        assert (body["kind"], body["language"], body["version"]) == (
            "recording", "es", "consent-1",
        )
        assert body["title"] == "Antes de empezar"
        for wrong in ({"kind": "other", "language": "es"},
                      {"kind": "recording", "language": "xx"}):
            assert_error(client.get("/api/consent", params=wrong),
                         400, "invalid_input")

    def test_options_chosen_at_the_start_are_used_from_any_device(
        self, client: TestClient, container: TranscriberContainer
    ) -> None:
        assert enrol(client).status_code == 201
        opened = client.post("/api/sessions", json={
            "id": "s1", "language": "es", "consent_version": "consent-1",
            "num_speakers": 1, "expected_speakers": ["ana"],
        })
        assert opened.status_code == 201, opened.text
        assert (opened.json()["consent_version"],
                opened.json()["num_speakers"],
                opened.json()["expected_speakers"]) == (
            "consent-1", 1, ["ana"])
        data = ana_alone()
        for number, start in enumerate(range(0, len(data), CHUNK)):
            client.put(f"/api/sessions/s1/chunks/{number}",
                       content=data[start:start + CHUNK])
        client.post("/api/sessions/s1/close")
        other_device = TestClient(client.app, headers=AUTH)
        queued = other_device.post("/api/sessions/s1/transcription")
        assert queued.status_code == 202, queued.text
        assert wait_until_done(other_device, "s1")["state"] == "done"
        transcript = client.get("/api/sessions/s1/transcript").json()
        assert [s["id"] for s in transcript["speakers"]] == ["ana"]

    def test_the_queueing_body_overrides_them(
        self, client: TestClient
    ) -> None:
        client.post("/api/sessions", json={
            "id": "s1", "language": "es", "num_speakers": 1,
        })
        data = recording()
        for number, start in enumerate(range(0, len(data), CHUNK)):
            client.put(f"/api/sessions/s1/chunks/{number}",
                       content=data[start:start + CHUNK])
        client.post("/api/sessions/s1/close")
        client.post("/api/sessions/s1/transcription",
                    json={"num_speakers": 2})
        assert wait_until_done(client, "s1")["state"] == "done"
        transcript = client.get("/api/sessions/s1/transcript").json()
        assert len(transcript["speakers"]) == 2

    def test_a_language_the_recogniser_lacks_is_refused(
        self, client: TestClient
    ) -> None:
        refused = client.post("/api/sessions", json={
            "id": "s1", "language": "xx"})
        assert_error(refused, 400, "invalid_input")
        assert "Unknown language" in refused.json()["message"]
        wav = sound_file(tone(1.0), 16000, "wav")
        assert_error(client.post(
            "/api/sessions/s2/upload",
            params={"filename": "a.wav", "language": "xx"}, content=wav,
        ), 400, "invalid_input")
        made = client.post("/api/sessions", json={
            "id": "s3", "language": "es-ES"})
        assert made.status_code == 201, made.text

    def test_an_unknown_consent_version_is_refused(
        self, client: TestClient
    ) -> None:
        assert_error(client.post("/api/sessions", json={
            "id": "s1", "language": "es", "consent_version": "v9",
        }), 400, "invalid_input")
        wav = sound_file(tone(1.0), 16000, "wav")
        assert_error(client.post(
            "/api/sessions/s2/upload",
            params={"filename": "a.wav", "language": "es",
                    "consent_version": "v9"}, content=wav,
        ), 400, "invalid_input")
        made = client.post(
            "/api/sessions/s2/upload",
            params={"filename": "a.wav", "language": "es",
                    "consent_version": "consent-1"}, content=wav,
        )
        assert made.json()["consent_version"] == "consent-1"
        assert client.get("/api/sessions").json()["sessions"][0]["id"] == "s2"

    def test_the_listing_carries_the_job_notes(
        self, client: TestClient, container: TranscriberContainer
    ) -> None:
        client.post("/api/sessions/s1/upload",
                    params={"filename": "a.wav", "language": "es"},
                    content=sound_file(tone(2.0), 16000, "wav"))
        listed = client.get("/api/sessions").json()["sessions"][0]
        assert {"stage", "fraction", "message"} <= set(listed)
        assert listed["message"] is None


class TestUploads:
    @pytest.mark.parametrize("filename", [
        "a.wav\n", "a.pcm", "rec.PCM", "notes.txt", "data.json", "audio",
        "a.flac.exe", ".wav",
    ])
    def test_only_audio_and_video_files_are_accepted(
        self, client: TestClient, tmp_path: Path, filename: str
    ) -> None:
        wav = sound_file(tone(1.0), 16000, "wav")
        answer = client.post(
            "/api/sessions/s1/upload",
            params={"filename": filename, "language": "es"}, content=wav,
        )
        body = assert_error(answer, 400, "unsupported_format")
        assert "m4a" in body["message"]
        assert_error(client.get("/api/sessions/s1"), 404, "not_found")
        assert not (tmp_path / "data" / "sessions" / "s1").exists()

    def test_a_refused_upload_can_be_sent_again(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        wav = sound_file(tone(1.0), 16000, "wav")
        refused = client.post(
            "/api/sessions/s1/upload",
            params={"filename": "a.wav\n", "language": "es"}, content=wav,
        )
        assert_error(refused, 400, "unsupported_format")
        bad_language = client.post(
            "/api/sessions/s1/upload",
            params={"filename": "a.wav", "language": "x"}, content=wav,
        )
        assert bad_language.status_code in (400, 422), bad_language.text
        made = client.post(
            "/api/sessions/s1/upload",
            params={"filename": "a.wav", "language": "es"}, content=wav,
        )
        assert made.status_code == 201, made.text
        folder = tmp_path / "data" / "sessions" / "s1"
        assert (folder / "imported.wav").is_file()

    def test_an_imported_pcm_never_takes_the_recorder_files(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        flac = sound_file(tone(1.0), 16000, "flac")
        made = client.post(
            "/api/sessions/s1/upload",
            params={"filename": "rec.flac", "language": "es"}, content=flac,
        )
        assert made.status_code == 201, made.text
        folder = tmp_path / "data" / "sessions" / "s1"
        assert sorted(p.name for p in folder.iterdir()) == [
            "imported.flac", "session.json",
        ]


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

    def test_closing_without_audio_or_again(self, client: TestClient) -> None:
        client.post("/api/sessions", json={"id": "s1", "language": "es"})
        assert_error(client.post("/api/sessions/s1/close"),
                     400, "invalid_input")
        client.put("/api/sessions/s1/chunks/0", content=b"\x01\x00")
        first = client.post("/api/sessions/s1/close")
        assert first.status_code == 200
        # A retry after a lost answer: the session as it is, no error.
        again = client.post("/api/sessions/s1/close")
        assert again.status_code == 200
        assert again.json()["state"] == "uploaded" == first.json()["state"]
        assert_error(client.put("/api/sessions/s1/chunks/1",
                                content=b"\x01\x00"), 409, "wrong_state")


SAMPLES = {
    "session_id": "s1", "number": "0", "label": "S1",
    "utterance_id": "t0.s0", "speaker_id": "ana",
}
"""A value for every path parameter the API has."""

QUERIES = {
    "/api/sessions/{session_id}/upload": "?filename=a.wav&language=es",
    "/api/enrolment/text": "?language=es",
    "/api/speakers/{speaker_id}/enrolment": "?name=A&language=es",
}
"""Required query parameters, so that the request reaches the route."""

OPEN_ROUTES = {("GET", "/api/health")}


def walk_routes(routes: Any) -> Iterator[Tuple[str, str]]:
    """Yield (method, path) of every endpoint in ``routes``, recursively.

    Newer FastAPI versions keep an included router as one entry that
    points at the router (``original_router``); its routes carry the
    prefix already.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            yield from walk_routes(included.routes)
            continue
        methods = getattr(route, "methods", None) or set()
        for method in sorted(set(methods) - {"HEAD"}):
            yield method, str(getattr(route, "path"))


def served_routes() -> List[Tuple[str, str]]:
    """Every (method, path template) the API serves, from ``app.routes``.

    Built from the app itself, so a route added later is covered by
    the tests below without being listed by hand; the OpenAPI listing
    of the same app must agree.
    """
    app = create_app(build_container(TranscriberSettings(
        data_dir=Path("/nonexistent/transcriber"), runtime="fake",
    )), CODE)
    found = sorted(set(walk_routes(app.routes)))
    documented = sorted(
        (method.upper(), path)
        for path, item in app.openapi()["paths"].items() for method in item
    )
    assert found == documented
    return found


def sample_path(template: str) -> str:
    """Fill ``template``'s parameters with :data:`SAMPLES` values."""
    path = template
    for name, value in SAMPLES.items():
        path = path.replace("{" + name + "}", value)
    assert "{" not in path, f"no sample value for a parameter of {template}"
    return path + QUERIES.get(template, "")


ROUTES = [
    (method, sample_path(path)) for method, path in served_routes()
    if (method, path) not in OPEN_ROUTES
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
        served = set(served_routes())
        assert all(path.startswith("/api/") for _, path in served)
        assert OPEN_ROUTES <= served
        assert len(ROUTES) == len(served) - len(OPEN_ROUTES) >= 17
        assert AUDIO_ROUTE in ROUTES
        assert ("POST", "/api/access/cookie") in ROUTES

    @pytest.mark.parametrize("method, path", ROUTES)
    def test_every_route_needs_the_code(
        self, container: TranscriberContainer, method: str, path: str
    ) -> None:
        app = create_app(container, CODE)
        plain = TestClient(app)
        container.open_session("s1", "es")
        for headers in ({}, {"X-Access-Code": "wrong"},
                        {"X-Access-Code": CODE + "x"}):
            answer = plain.request(method, path, headers=headers,
                                   content=b"\x00\x00")
            assert_error(answer, 401, "unauthorized")
        assert container.sessions.get("s1") is not None  # untouched
        # With the code the same request reaches the route.
        reached = TestClient(app, headers=AUTH).request(
            method, path, content=b"\x00\x00"
        )
        body = reached.json() if reached.status_code != 204 else {}
        assert reached.status_code != 405, reached.text
        assert body.get("code") != "unauthorized", reached.text
        assert body.get("message") != "Not Found", reached.text  # routed

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
                                  headers=with_cookie("guess")),
                     429, "too_many_attempts")
        # The right cookie still plays, even from the blocked client.
        assert attacker.get("/api/sessions/s1/audio",
                            headers=with_cookie(value)).status_code == 200
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
        # The right code is never refused, even from the blocked client;
        # it does not lift the block for wrong codes either...
        assert attacker.get("/api/sessions", headers=AUTH).status_code == 200
        assert_error(attacker.get("/api/sessions", headers=wrong),
                     429, "too_many_attempts")
        assert_error(attacker.get("/api/sessions"),
                     429, "too_many_attempts")
        # ...and another client is not affected.
        assert other.get("/api/sessions").status_code == 200
        clock.now += 121
        assert_error(attacker.get("/api/sessions", headers=wrong),
                     401, "unauthorized")
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
        for hop in range(2):
            tunnel.get("/api/sessions", headers={
                **wrong, "X-Client-IP": "198.51.100.7",
                "X-Forwarded-For": f"203.0.113.{hop}",
            })
        assert_error(tunnel.get("/api/sessions", headers={
            **wrong, "X-Client-IP": "198.51.100.7",
            "X-Forwarded-For": "203.0.113.99",
        }), 429, "too_many_attempts")
        assert_error(tunnel.get("/api/sessions", headers={
            **wrong, "X-Client-IP": "198.51.100.8",
        }), 401, "unauthorized")
        container.runner.close(WAIT)

    def test_the_client_header_is_trusted_only_from_the_tunnel(
        self, tmp_path: Path
    ) -> None:
        container = make_container(tmp_path, failed_code_limit=2)
        app = create_app(container, CODE, now=Clock())
        direct = TestClient(app, client=("203.0.113.9", 4000))
        wrong = {"X-Access-Code": "nope"}
        for n in range(2):
            direct.get("/api/sessions", headers={
                **wrong, "X-Client-IP": f"198.51.100.{n}",
            })
        assert_error(direct.get("/api/sessions", headers={
            **wrong, "X-Client-IP": "198.51.100.50",
        }), 429, "too_many_attempts")
        container.runner.close(WAIT)

    def test_the_header_name_comes_from_the_settings(
        self, tmp_path: Path
    ) -> None:
        container = make_container(
            tmp_path, failed_code_limit=1, client_ip_header="X-Real-IP",
        )
        app = create_app(container, CODE, now=Clock())
        tunnel = TestClient(app, client=("127.0.0.1", 4000))
        wrong = {"X-Access-Code": "nope"}
        tunnel.get("/api/sessions", headers={**wrong, "X-Real-IP": "a"})
        assert_error(tunnel.get("/api/sessions", headers={
            **wrong, "X-Real-IP": "b",
        }), 401, "unauthorized")
        assert_error(tunnel.get("/api/sessions", headers={
            **wrong, "X-Real-IP": "a",
        }), 429, "too_many_attempts")
        container.runner.close(WAIT)

    def test_the_code_is_compared_in_constant_time(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        compared: List[Tuple[bytes, bytes]] = []
        real = hmac.compare_digest

        def spy(a: bytes, b: bytes) -> bool:
            compared.append((a, b))
            return real(a, b)

        monkeypatch.setattr(hmac, "compare_digest", spy)
        assert client.get("/api/sessions").status_code == 200
        assert (CODE.encode(), CODE.encode()) in compared
        compared.clear()
        plain = TestClient(client.app)
        plain.get("/api/sessions", headers={"X-Access-Code": "nope"})
        assert (CODE.encode(), b"nope") in compared


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


class TestRendering:
    def test_rendered_once_on_disk_and_removed_with_the_session(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        data = pcm_bytes(tone(2.0))
        record(client, "s1", data)
        folder = tmp_path / "data" / "sessions" / "s1"
        live = client.get("/api/sessions/s1/audio")
        assert live.content[WAV_HEADER:] == data
        assert not list(folder.glob("playback-*.wav"))  # live: no file
        client.post("/api/sessions/s1/close")
        whole = client.get("/api/sessions/s1/audio")
        assert whole.content[WAV_HEADER:] == data
        renderings = list(folder.glob("playback-*.wav"))
        assert len(renderings) == 1
        assert renderings[0].read_bytes() == whole.content
        made = renderings[0].stat().st_mtime_ns
        client.get("/api/sessions/s1/audio", headers={"Range": "bytes=0-9"})
        assert renderings[0].stat().st_mtime_ns == made  # not made again
        assert client.delete("/api/sessions/s1").status_code == 204
        assert not folder.exists()

    def test_new_audio_makes_a_new_rendering(
        self, client: TestClient, container: TranscriberContainer,
        tmp_path: Path,
    ) -> None:
        first = sound_file(tone(1.0, hz=220.0), 16000, "wav")
        client.post("/api/sessions/s1/upload",
                    params={"filename": "a.wav", "language": "es"},
                    content=first)
        before = client.get("/api/sessions/s1/audio").content
        folder = tmp_path / "data" / "sessions" / "s1"
        old = list(folder.glob("playback-*.wav"))
        container.sessions.store_recording("s1", RecordedAudio(
            sound_file(tone(1.0, hz=880.0), 16000, "wav"), "wav"
        ))
        after = client.get("/api/sessions/s1/audio").content
        assert after != before
        new = list(folder.glob("playback-*.wav"))
        assert len(new) == 1 and new != old

    def test_an_undecodable_recording_is_a_client_error(
        self, client: TestClient
    ) -> None:
        client.post("/api/sessions/s1/upload",
                    params={"filename": "a.mp3", "language": "es"},
                    content=b"not audio at all" * 10)
        assert_error(client.get("/api/sessions/s1/audio"),
                     400, "invalid_input")


class TestEnrolmentLimit:
    def test_waiting_enrolments_are_bounded_and_other_routes_answer(
        self, tmp_path: Path
    ) -> None:
        container = make_container(tmp_path, max_pending_enrolments=2,
                                   busy_retry_after=7)
        client = TestClient(create_app(container, CODE), headers=AUTH)
        release = threading.Event()
        started = threading.Event()

        def hold(progress: Progress) -> None:
            started.set()
            release.wait(WAIT)

        container.runner.submit("blocker", hold)  # the GPU is busy
        assert started.wait(WAIT)
        answers: List[Any] = []
        waiting = [
            threading.Thread(target=lambda: answers.append(enrol(client)))
            for _ in range(2)
        ]
        for thread in waiting:
            thread.start()
        deadline = time.monotonic() + WAIT
        while container.runner._queue.qsize() < 2:
            assert time.monotonic() < deadline, "the enrolments never queued"
            time.sleep(0.01)
        busy = enrol(client)
        body = assert_error(busy, 409, "busy")
        assert busy.headers["retry-after"] == "7"
        assert "busy" in body["message"]
        # Nothing else waits for the runner.
        assert client.get("/api/sessions").status_code == 200
        assert client.get("/api/speakers").status_code == 200
        record(client, "s1", pcm_bytes(tone(1.0)))
        assert client.post("/api/sessions/s1/close").status_code == 200
        queued = client.post("/api/sessions/s1/transcription")
        assert queued.status_code == 202
        assert client.get("/api/sessions/s1").json()["state"] == "queued"
        release.set()
        for thread in waiting:
            thread.join(WAIT)
        assert [a.status_code for a in answers] == [201, 201]
        assert enrol(client, "luis").status_code == 201  # room again
        container.runner.close(WAIT)

    def test_a_refused_consent_does_not_take_a_place(
        self, tmp_path: Path
    ) -> None:
        container = make_container(tmp_path, max_pending_enrolments=1)
        client = TestClient(create_app(container, CODE), headers=AUTH)
        for _ in range(3):
            refused = client.post(
                "/api/speakers/ana/enrolment",
                params={"name": "Ana", "language": "es"}, content=b"x",
            )
            assert_error(refused, 400, "consent_required")
        too_big = client.post(
            "/api/speakers/ana/enrolment",
            params={"name": "Ana", "language": "es", "consent": "true"},
            content=b"\x00" * (21 * 1024 * 1024),
        )
        assert_error(too_big, 413, "payload_too_large")
        assert enrol(client).status_code == 201
        container.runner.close(WAIT)


class TestPlaybackParts:
    def test_parse_range_edges(self) -> None:
        assert parse_range(None, 10) is None
        assert parse_range("bytes=0-", 0) is UNSATISFIABLE
        assert parse_range("bytes=-5", 0) is UNSATISFIABLE
        assert parse_range(" BYTES = 2 - 4 ", 10) == (2, 4)
        assert parse_range("bytes=1-2-3", 10) is None
        assert parse_range("bytes=١-2", 10) is None  # non-ASCII digit
        assert parse_range("bytes=" + "9" * 19 + "-", 10) is None

    def test_parsing_is_linear_and_long_headers_are_ignored(self) -> None:
        """The old regular expression took quadratic time on spaces."""
        hostile = "bytes=" + " " * 50000 + "x"
        started = time.perf_counter()
        for _ in range(100):
            assert parse_range(hostile, 10) is None
            assert parse_range("bytes=0" + " " * 90 + "-1", 10) == (0, 1)
        assert time.perf_counter() - started < 1.0
        assert parse_range("bytes=0-1" + " " * 200, 10) is None


class TestAccessParts:
    def test_check_code(self) -> None:
        assert check_code(CODE, CODE)
        assert not check_code(CODE, None)
        assert not check_code(CODE, "")
        assert not check_code(CODE, "ñ" * 20)

    @pytest.mark.parametrize("peer, header, key", [
        ("203.0.113.9", "198.51.100.7", "203.0.113.9"),
        ("127.0.0.1", "198.51.100.7", "198.51.100.7"),
        ("::1", " 2001:db8::1 ", "2001:db8::1"),
        (None, "198.51.100.7", "198.51.100.7"),
        (None, None, "unknown"),
        ("127.0.0.1", None, "127.0.0.1"),
        ("127.0.0.1", "  ", "127.0.0.1"),
        ("127.0.0.1", "x" * 100, "x" * 64),
        ("testclient", "198.51.100.7", "testclient"),
    ])
    def test_client_key(self, peer: Any, header: Any, key: str) -> None:
        headers = {"X-Forwarded-For": "192.0.2.1"}
        if header is not None:
            headers["X-Client-IP"] = header
        assert client_key(peer, headers) == key

    def test_x_forwarded_for_is_never_read(self) -> None:
        for peer in ("127.0.0.1", None, "203.0.113.9"):
            assert client_key(peer, {"X-Forwarded-For": "192.0.2.1"}) in (
                "127.0.0.1", "unknown", "203.0.113.9",
            )

    def test_the_tables_stay_bounded(self) -> None:
        clock = Clock()
        limiter = FailureLimiter(2, 10.0, 10.0, now=clock, max_clients=4)
        for n in range(20):
            limiter.failed(f"c{n}")
            limiter.failed(f"c{n}")
            clock.now += 1
        assert len(limiter._failures) + len(limiter._blocked) <= 4
        assert limiter.retry_after("c19") > 0

    def test_blocked_clients_are_kept_before_unblocked_ones(self) -> None:
        clock = Clock()
        limiter = FailureLimiter(2, 60.0, 600.0, now=clock, max_clients=4)
        limiter.failed("attacker")
        limiter.failed("attacker")  # blocked
        for n in range(10):  # a flood of single failures
            limiter.failed(f"noise{n}")
            clock.now += 1
        assert len(limiter._failures) + len(limiter._blocked) <= 4
        assert limiter.retry_after("attacker") > 0
        assert "noise9" in limiter._failures  # the newest are kept

    def test_blocked_clients_alone_are_dropped_soonest_ending_first(
        self,
    ) -> None:
        clock = Clock()
        limiter = FailureLimiter(1, 60.0, 600.0, now=clock, max_clients=2)
        for n in range(3):
            limiter.failed(f"b{n}")
            clock.now += 1
        assert sorted(limiter._blocked) == ["b1", "b2"]


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
