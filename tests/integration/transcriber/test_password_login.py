"""The password login: the hash on disk, the route and the command."""

import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.test_cli import environment
from tests.integration.transcriber.test_http_api import (
    CODE,
    Clock,
    make_container,
)
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.login_throttle import LoginThrottle
from transcriber.adapters.outbound.fs_password import (
    PasswordStore,
    password_file,
    set_password,
)
from transcriber.bootstrap import TranscriberContainer
from transcriber.domain.errors import PasswordError

SECRET = "tango-9271"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A fake-runtime container."""
    built = make_container(tmp_path)
    yield built
    built.runner.close(30.0)


def test_the_hash_is_salted_private_and_not_the_password(
    tmp_path: Path,
) -> None:
    path = password_file(tmp_path / "d")
    set_password(path, SECRET)
    first = path.read_text()
    assert SECRET not in first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    set_password(path, SECRET)
    assert path.read_text() != first  # a new salt every time
    store = PasswordStore(path)
    assert store.is_set()
    assert store.verify(SECRET)
    assert not store.verify("tango-9272")
    assert not store.verify("")
    assert not store.verify("x" * 5000)


def test_no_file_or_a_bad_one_never_verifies(tmp_path: Path) -> None:
    path = tmp_path / "password.json"
    store = PasswordStore(path)
    assert not store.is_set() and not store.verify(SECRET)
    path.write_text("{not json")
    path.chmod(0o600)
    assert not store.verify(SECRET)
    set_password(path, SECRET)
    path.chmod(0o644)
    assert not store.verify(SECRET)
    path.chmod(0o600)
    assert store.verify(SECRET)
    record = json.loads(path.read_text())
    record["scheme"] = "md5"
    path.write_text(json.dumps(record))
    assert not store.verify(SECRET)


def test_a_too_short_password_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PasswordError):
        set_password(tmp_path / "p.json", "abc")
    assert not (tmp_path / "p.json").exists()


def make_client(
    container: TranscriberContainer, tmp_path: Path, clock: Clock
) -> TestClient:
    """An API with the password set and a small throttle; no code header."""
    path = tmp_path / "pw.json"
    set_password(path, SECRET)
    throttle = LoginThrottle(
        per_client_limit=3, window=100.0, base_cooldown=10.0,
        global_limit=1000, now=clock,
    )
    app = create_app(
        container, CODE, clock, PasswordStore(path).verify, throttle
    )
    return TestClient(app, client=("127.0.0.1", 5000))


def login(
    client: TestClient, password: str, ip: str = "1.1.1.1"
) -> Any:
    """POST the login as ``ip`` (from the tunnel's loopback)."""
    return client.post(
        "/api/login", json={"password": password},
        headers={"X-Client-IP": ip},
    )


def test_the_right_password_gives_the_code_that_opens_the_api(
    container: TranscriberContainer, tmp_path: Path
) -> None:
    client = make_client(container, tmp_path, Clock())
    response = login(client, SECRET)
    assert response.status_code == 200
    code = response.json()["code"]
    assert code == CODE
    assert client.get(
        "/api/sessions", headers={"X-Access-Code": code}
    ).status_code == 200


def test_a_wrong_password_is_a_generic_401_then_a_lock(
    container: TranscriberContainer, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock()
    client = make_client(container, tmp_path, clock)
    with caplog.at_level("INFO"):
        for _ in range(3):
            wrong = login(client, "guess-0000")
            assert wrong.status_code == 401
            assert CODE not in wrong.text
        locked = login(client, SECRET)  # even the right one waits
        assert locked.status_code == 429
        assert locked.headers["Retry-After"] == "10"
        other = login(client, SECRET, "2.2.2.2")
        assert other.status_code == 200
    assert "guess-0000" not in caplog.text and SECRET not in caplog.text
    assert "login failed for 1.1.1.1" in caplog.text
    clock.now += 11
    assert login(client, SECRET).status_code == 200


def test_without_a_password_set_login_is_refused(
    container: TranscriberContainer, tmp_path: Path
) -> None:
    client = TestClient(create_app(container, CODE))
    assert login(client, SECRET).status_code == 401
    bad = client.post("/api/login", json={"nope": 1})
    assert bad.status_code == 422


def test_set_password_reads_stdin_and_never_argv(tmp_path: Path) -> None:
    data = tmp_path / "data"
    done = subprocess.run(
        [sys.executable, "-m", "transcriber", "set-password"],
        input=SECRET + "\n", capture_output=True, text=True, cwd=ROOT,
        env=environment(data), timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert SECRET not in done.stdout + done.stderr
    assert PasswordStore(password_file(data)).verify(SECRET)
    short = subprocess.run(
        [sys.executable, "-m", "transcriber", "set-password"],
        input="ab\n", capture_output=True, text=True, cwd=ROOT,
        env=environment(data), timeout=120,
    )
    assert short.returncode == 1
    assert PasswordStore(password_file(data)).verify(SECRET)
