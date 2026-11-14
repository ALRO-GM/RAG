"""Every command of ``python -m transcriber`` in a subprocess.

The commands run with ``TRANSCRIBER_RUNTIME=fake`` (no torch, no model)
against a temporary data directory. ``serve`` is started for real on a
free local port, used over HTTP and stopped with SIGINT; once with
``--no-tunnel`` and once with a fake ``chisel`` client (the real server
is never contacted).
"""

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List

import httpx
import pytest

from tests.integration.transcriber.audio import scene, sound_file, tone
from tests.integration.transcriber.fake_chisel import (
    install_fake_chisel,
    write_credential,
)

ROOT = Path(__file__).resolve().parents[3]
WAIT = 60.0


def environment(data: Path, **extra: str) -> Dict[str, str]:
    """The environment of a command: fake models, ``data`` as data dir."""
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "TRANSCRIBER_RUNTIME": "fake",
        "TRANSCRIBER_DATA_DIR": str(data),
        "HF_HUB_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    env.update(extra)
    return env


def run(
    data: Path, *args: str, **env: str
) -> "subprocess.CompletedProcess[str]":
    """Run ``python -m transcriber *args`` against ``data``."""
    return subprocess.run(
        [sys.executable, "-m", "transcriber", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=WAIT,
        env=environment(data, **env),
    )


def ok(data: Path, *args: str) -> str:
    """Run a command that must succeed; return its output."""
    result = run(data, *args)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """An enrolment sample (a 220 Hz voice) and a two-voice recording."""
    folder = tmp_path_factory.mktemp("audio")
    sample = folder / "ana.wav"
    sample.write_bytes(sound_file(tone(30.0, hz=220.0), 16000, "wav"))
    talk = folder / "Reparto de tareas.wav"
    talk.write_bytes(sound_file(scene([
        ("220", 6.0), ("-", 1.6), ("880", 6.0), ("-", 1.6), ("220", 4.8),
    ]), 16000, "wav"))
    return {"sample": sample, "talk": talk}


def test_a_session_from_enrolment_to_publication(
    tmp_path: Path, files: Dict[str, Path]
) -> None:
    data = tmp_path / "data"
    assert ok(data, "speakers") == "Nobody is enrolled.\n"
    assert ok(data, "sessions") == "No sessions.\n"

    refused = run(data, "enrol", "Ana María", str(files["sample"]))
    assert refused.returncode == 2
    assert "--consent" in refused.stderr
    assert not (data / "voiceprints").exists()

    enrolled = ok(data, "enrol", "Ana María", str(files["sample"]),
                  "--consent")
    assert enrolled.startswith("Enrolled Ana María as ana-maria")
    listed = json.loads(ok(data, "speakers", "--json"))["speakers"]
    assert [(s["id"], s["name"]) for s in listed] == [
        ("ana-maria", "Ana María")
    ]

    out = ok(data, "import", str(files["talk"]), "--id", "team",
             "--title", "Reparto", "--speakers", "2",
             "--expect", "ana-maria", "--wait")
    lines = out.splitlines()
    assert lines[0] == "Imported team; transcribing."
    assert lines[-2] == "done 100%"
    assert lines[-1].startswith("Done: ") and "2 speakers" in lines[-1]

    assert ok(data, "sessions").splitlines() == [
        "team  done  00:00:20  es  Reparto"
    ]
    shown = ok(data, "show", "team").splitlines()
    assert shown[0] == "team - Reparto (done, 00:00:20)"
    assert shown[1].startswith("[00:00:00] Ana María: ")
    assert any(" (after Ana María): " in line for line in shown[2:])
    led = ok(data, "show", "team", "--lead", "100").splitlines()
    assert led[1] == (
        "Times are shifted 100 s earlier than spoken (--lead 100)."
    )
    assert sum("[00:00:00]" in line for line in led) == len(shown) - 1
    assert "shifted" not in "\n".join(shown)
    detail = json.loads(ok(data, "show", "team", "--json"))
    assert detail["session"]["state"] == "done"
    assert "words" not in detail["transcript"]["utterances"][0]

    document = json.loads(ok(data, "export", "team"))
    assert document["schema"] == "transcript.v1"
    assert document["session"]["id"] == "team"
    target = tmp_path / "team.json"
    written = run(data, "export", "team", "--output", str(target))
    assert written.returncode == 0 and written.stdout == ""
    assert json.loads(target.read_text()) == document

    assert ok(data, "publish", "team").startswith("Published team to ")
    assert (data / "published").is_dir()
    assert any((data / "published").iterdir())

    assert ok(data, "sessions", "delete", "team") == "Deleted team.\n"
    assert ok(data, "speakers", "forget", "ana-maria") == (
        "Forgot ana-maria.\n"
    )
    assert json.loads(ok(data, "sessions", "--json")) == {
        "sessions": [], "unreadable": 0,
    }


def test_import_without_wait_leaves_the_session_ready(
    tmp_path: Path, files: Dict[str, Path]
) -> None:
    data = tmp_path / "data"
    imported = json.loads(ok(data, "import", str(files["talk"]), "--json"))
    assert imported["id"].startswith("reparto-de-tareas-")
    assert imported["state"] == "uploaded"
    shown = ok(data, "show", imported["id"])
    assert "(uploaded, 00:00:00)" in shown
    usage = run(data, "import", str(files["talk"]), "--speakers", "2")
    assert usage.returncode == 2 and "--wait" in usage.stderr


def test_import_wait_as_json(tmp_path: Path, files: Dict[str, Path]) -> None:
    data = tmp_path / "data"
    done = json.loads(ok(data, "import", str(files["talk"]), "--id", "j",
                         "--wait", "--json"))
    assert done == {"id": "j", "state": "done", "failure": None}


def test_errors_have_messages_and_exit_codes(
    tmp_path: Path, files: Dict[str, Path]
) -> None:
    data = tmp_path / "data"
    for args in (["show", "nope"], ["export", "nope"], ["publish", "nope"],
                 ["sessions", "delete", "nope"],
                 ["speakers", "forget", "nope"],
                 ["import", str(tmp_path / "missing.wav")]):
        result = run(data, *args)
        assert result.returncode == 1, args
        assert "error:" in result.stderr, args
    no_extension = tmp_path / "recording"
    no_extension.write_bytes(b"x")
    assert run(data, "import", str(no_extension)).returncode == 1
    assert run(data, "show", "x", "--lead", "-1").returncode == 2
    assert run(data, "enrol", "日本", str(files["sample"]),
               "--consent").returncode == 2
    assert run(data, "import", str(files["talk"]), "--expect", ",",
               "--wait").returncode == 2
    assert run(data, "frobnicate").returncode == 2


def test_enrolment_text(tmp_path: Path) -> None:
    data = tmp_path / "data"
    assert ok(data, "enrolment-text", "--name", "Ana").startswith(
        "Me llamo Ana."
    )
    shown = json.loads(ok(data, "enrolment-text", "--json"))
    assert shown["language"] == "es" and shown["text"] is None
    assert "{name}" in shown["template"]
    assert shown["consent_version"] == "consent-1"
    assert "{name}" in ok(data, "enrolment-text")
    assert json.loads(ok(data, "settings", "--json"))["runtime"] == "fake"


def free_port() -> int:
    """Return a local port nobody listens on (now)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Service:
    """``transcriber serve`` in a subprocess, its output read by a thread."""

    def __init__(self, data: Path, args: List[str],
                 **env: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "transcriber", "serve", *args],
            cwd=ROOT, env=environment(data, **env), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.seen: List[str] = []
        self.ended = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.ended.set()

    def wait_for(self, text: str) -> str:
        """Return the first output line containing ``text``."""
        deadline = time.monotonic() + WAIT
        for line in self.seen:
            if text in line:
                return line
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.1)
            except queue.Empty:
                if self.ended.is_set() and self.lines.empty():
                    raise AssertionError(
                        f"{text!r} not printed: {self.stderr()}"
                    )
                continue
            self.seen.append(line)
            if text in line:
                return line
        raise AssertionError(f"{text!r} never printed")

    def stop(self) -> int:
        """Send SIGINT and return the exit code."""
        self.process.send_signal(signal.SIGINT)
        return self.process.wait(WAIT)

    def stderr(self) -> str:
        """Return the error output (once the process ended)."""
        if self.process.poll() is None:
            return ""
        assert self.process.stderr is not None
        return self.process.stderr.read()


@pytest.fixture()
def services() -> Iterator[List[Service]]:
    """Services started by a test; killed if the test left them running."""
    started: List[Service] = []
    yield started
    for service in started:
        if service.process.poll() is None:
            service.process.kill()
            service.process.wait()


def wait_healthy(port: int) -> None:
    """Poll ``/api/health`` until the API answers."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health",
                         timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    raise AssertionError("the API never answered")


def port_is_free(port: int) -> bool:
    """Tell whether ``port`` can be listened on again.

    ``SO_REUSEADDR`` as servers set it: connections the API closed
    first (left in TIME_WAIT) do not count, a listening socket does.
    """
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def code_of(link: str) -> str:
    """Return the access code in a join link."""
    return link.split("#k=", 1)[1]


def test_serve_without_tunnel(
    tmp_path: Path, files: Dict[str, Path], services: List[Service]
) -> None:
    data = tmp_path / "data"
    ok(data, "import", str(files["talk"]), "--id", "s1")
    port = free_port()
    service = Service(data, ["--no-tunnel", "--port", str(port),
                             "--keep-code"])
    services.append(service)
    wait_healthy(port)
    link = service.wait_for("#k=")
    assert link.startswith(f"http://127.0.0.1:{port}/#k=")
    code = code_of(link)
    assert len(code) >= 32
    kept = data / "access_code"
    assert kept.read_text().strip() == code
    assert kept.stat().st_mode & 0o777 == 0o600
    service.wait_for("Ctrl-C to stop.")

    api = f"http://127.0.0.1:{port}/api"
    assert httpx.get(f"{api}/sessions").status_code == 401
    listed = httpx.get(f"{api}/sessions", headers={"X-Access-Code": code})
    assert listed.status_code == 200
    assert [s["id"] for s in listed.json()["sessions"]] == ["s1"]
    queued = httpx.post(f"{api}/sessions/s1/transcription",
                        headers={"X-Access-Code": code})
    assert queued.status_code == 202, queued.text
    assert service.wait_for("  s1  done").endswith("s1  done")

    assert service.stop() == 0
    service.wait_for("Stopped.")
    assert code not in service.stderr()  # only the terminal shows it
    assert port_is_free(port)

    again = Service(data, ["--no-tunnel", "--port", str(port),
                           "--keep-code"])
    services.append(again)
    assert code_of(again.wait_for("#k=")) == code  # the QR still works
    assert again.stop() == 0


def test_serve_with_a_fake_tunnel(
    tmp_path: Path, services: List[Service]
) -> None:
    data = tmp_path / "data"
    log = tmp_path / "chisel.log"
    chisel = install_fake_chisel(tmp_path / "bin", log)
    credential = write_credential(tmp_path / "credential")
    port = free_port()
    config = tmp_path / "settings.json"
    # The probe goes to the host of server_url: here the API itself.
    config.write_text(json.dumps({"tunnel": {
        "public_host": "app.example", "chisel": str(chisel),
        "credential_file": str(tmp_path / "credential"),
        "server_url": f"http://127.0.0.1:{port}/_tunnel",
        "backoff_initial": 0.05, "probe_interval": 0.1,
    }}))
    service = Service(data, ["--port", str(port)],
                      TRANSCRIBER_CONFIG=str(config))
    services.append(service)
    link = service.wait_for("#k=")
    assert link.startswith("https://app.example/#k=")
    service.wait_for("tunnel  connected")
    clients = log.read_text().splitlines()
    assert len(clients) == 2  # the first one was refused and retried
    assert clients[-1] == (
        "client --keepalive 25s --max-retry-count 0"
        f" http://127.0.0.1:{port}/_tunnel R:0.0.0.0:9000:127.0.0.1:{port}"
    )
    assert credential not in log.read_text()
    assert (data / "tunnel.pid").exists()
    assert service.stop() == 0
    assert Path(f"{log}.term").exists()  # the client was ended
    assert not (data / "tunnel.pid").exists()
    assert credential not in "\n".join(service.seen) + service.stderr()
    assert port_is_free(port)


def test_serve_without_a_tunnel_client_says_how_to_get_one(
    tmp_path: Path,
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"tunnel": {
        "chisel": str(tmp_path / "missing"),
        "credential_file": str(tmp_path / "credential"),
    }}))
    result = run(tmp_path / "data", "serve", "--port", str(free_port()),
                 TRANSCRIBER_CONFIG=str(config))
    assert result.returncode != 0
    assert "make tunnel-client" in result.stderr
    assert "--no-tunnel" in result.stderr


def test_serve_refuses_a_taken_port(
    tmp_path: Path, services: List[Service]
) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        result = subprocess.run(
            [sys.executable, "-m", "transcriber", "serve", "--no-tunnel",
             "--port", str(port)],
            cwd=ROOT, env=environment(tmp_path / "data"), text=True,
            capture_output=True, timeout=WAIT,
        )
    assert result.returncode == 1
    assert "could not listen" in result.stdout
    assert "#k=" not in result.stdout


def running_session(data: Path, files: Dict[str, Path]) -> Path:
    """Import a session and mark it running, as a live service's job."""
    ok(data, "import", str(files["talk"]), "--id", "live")
    record = next(data.rglob("live/session.json"))
    state = json.loads(record.read_text())
    state["state"] = "running"
    record.write_text(json.dumps(state))
    return record


def test_a_second_serve_leaves_the_live_one_alone(
    tmp_path: Path, files: Dict[str, Path]
) -> None:
    import fcntl

    data = tmp_path / "data"
    record = running_session(data, files)
    with open(data / "serve.lock", "a") as held:  # the live service
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run(data, "serve", "--no-tunnel", "--port",
                     str(free_port()))
    assert result.returncode == 3
    assert "another transcriber serve is using" in result.stderr
    assert "#k=" not in result.stdout
    assert json.loads(record.read_text())["state"] == "running"


def test_a_taken_port_stops_serve_before_recovery(
    tmp_path: Path, files: Dict[str, Path]
) -> None:
    data = tmp_path / "data"
    record = running_session(data, files)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        result = run(data, "serve", "--no-tunnel", "--port",
                     str(taken.getsockname()[1]))
    assert result.returncode == 1
    assert "could not listen" in result.stdout
    assert json.loads(record.read_text())["state"] == "running"


def test_serve_new_code_replaces_the_kept_one(
    tmp_path: Path, services: List[Service]
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    kept = data / "access_code"
    kept.write_text("o" * 43 + "\n")
    kept.chmod(0o600)
    port = free_port()
    service = Service(data, ["--no-tunnel", "--port", str(port),
                             "--new-code"])
    services.append(service)
    code = code_of(service.wait_for("#k="))
    assert code != "o" * 43
    assert kept.read_text().strip() == code
    assert kept.stat().st_mode & 0o777 == 0o600
    assert service.stop() == 0


def test_serve_refuses_a_kept_code_others_can_read(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    kept = data / "access_code"
    kept.write_text("x" * 43)
    kept.chmod(0o644)
    result = run(data, "serve", "--no-tunnel", "--keep-code")
    assert result.returncode == 1
    assert "readable by others" in result.stderr
