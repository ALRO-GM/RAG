"""The API behind a real uvicorn server, configured as ``serve`` runs it.

TestClient skips the HTTP server, so what uvicorn does to a request
before the app sees it (the client address, proxy headers, a body
announced but never sent) is checked here over a real socket, with the
same :func:`server_config` ``transcriber serve`` uses.
"""

import socket
import threading
import time
from pathlib import Path
from typing import Iterator, Tuple

import httpx
import pytest
import uvicorn

from tests.integration.transcriber.audio import pcm_bytes, tone
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.serve import server_config
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import ApiSettings, TranscriberSettings

CODE = "a-long-enough-access-code"
WAIT = 30.0
LIMIT = 3


def free_port() -> int:
    """Return a port nothing listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture()
def server(tmp_path: Path) -> Iterator[Tuple[str, TranscriberContainer]]:
    """The API on a real uvicorn server; yields its URL and container."""
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
        api=ApiSettings(failed_code_limit=LIMIT, cookie_secure=False,
                        chunk_max_bytes=64 * 1024),
    ))
    port = free_port()
    running = uvicorn.Server(server_config(create_app(container, CODE), port))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + WAIT
    while not running.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", container
    running.should_exit = True
    thread.join(WAIT)
    container.runner.close(WAIT)


def test_uvicorn_does_not_rewrite_the_client_from_proxy_headers() -> None:
    config = server_config(create_app(build_container(TranscriberSettings(
        data_dir=Path("/nonexistent/transcriber"), runtime="fake",
    )), CODE), free_port())
    assert config.proxy_headers is False
    assert config.host == "127.0.0.1"


def test_throttling_follows_the_proxy_header_and_never_locks_the_owner_out(
    server: Tuple[str, TranscriberContainer],
) -> None:
    url, _ = server
    attacker = "198.51.100.7"
    with httpx.Client(base_url=url, timeout=10.0) as http:
        for hop in range(LIMIT):
            wrong = http.get("/api/sessions", headers={
                "X-Access-Code": "nope", "X-Client-IP": attacker,
                "X-Forwarded-For": f"203.0.113.{hop}",
            })
            assert wrong.status_code == 401, wrong.text
        for hop in range(LIMIT, LIMIT + 5):  # rotating the header is no use
            blocked = http.get("/api/sessions", headers={
                "X-Access-Code": "nope", "X-Client-IP": attacker,
                "X-Forwarded-For": f"203.0.113.{hop}",
            })
            assert blocked.status_code == 429, blocked.text
            assert int(blocked.headers["retry-after"]) > 0
        # The owner, even behind the same address, is never refused...
        for headers in (
            {"X-Client-IP": attacker},
            {"X-Client-IP": "192.0.2.10", "X-Forwarded-For": attacker},
            {},
        ):
            owner = http.get(
                "/api/sessions", headers={"X-Access-Code": CODE, **headers}
            )
            assert owner.status_code == 200, owner.text
        # ...and someone else behind the proxy is counted apart.
        other = http.get("/api/sessions", headers={
            "X-Access-Code": "nope", "X-Client-IP": "192.0.2.11",
        })
        assert other.status_code == 401


def test_a_body_announced_too_large_is_refused_before_it_is_sent(
    server: Tuple[str, TranscriberContainer],
) -> None:
    url, container = server
    container.open_session("s1", "es")
    port = int(url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=10.0) as raw:
        raw.sendall(
            b"PUT /api/sessions/s1/chunks/0 HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"X-Access-Code: {CODE}\r\n".encode()
            + b"Content-Type: application/octet-stream\r\n"
            b"Content-Length: 1000000000\r\n\r\n"
        )
        answer = raw.recv(65536).decode("latin-1")
    assert answer.startswith("HTTP/1.1 413"), answer
    assert '"code":"payload_too_large"' in answer
    session = container.sessions.get("s1")
    assert session is not None and session.chunks == 0


def test_the_audio_streams_ranges_with_the_cookie(
    server: Tuple[str, TranscriberContainer],
) -> None:
    url, _ = server
    data = pcm_bytes(tone(1.5))
    with httpx.Client(base_url=url, timeout=10.0) as http:
        auth = {"X-Access-Code": CODE}
        opened = http.post("/api/sessions", headers=auth,
                           json={"id": "s1", "language": "es"})
        assert opened.status_code == 201
        http.put("/api/sessions/s1/chunks/0", headers=auth, content=data)
        http.post("/api/sessions/s1/close", headers=auth)
        cookie = http.post("/api/access/cookie", headers=auth)
        assert cookie.status_code == 204
        whole = http.get("/api/sessions/s1/audio")  # the cookie alone
        assert whole.status_code == 200
        assert whole.headers["content-length"] == str(44 + len(data))
        assert whole.content[44:] == data
        part = http.get("/api/sessions/s1/audio",
                        headers={"Range": "bytes=44-1043"})
        assert part.status_code == 206
        assert part.content == data[:1000]
