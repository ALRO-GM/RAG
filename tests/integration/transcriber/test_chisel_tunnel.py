"""The tunnel supervisor running real processes.

With a fake ``chisel`` script: start, a refused first client, the
restart, the environment and the stop. With the real chisel (the binary
of the image pinned in ``infra/compose.yml``, copied out with docker;
skipped without docker): a local chisel server, the tunnel up end to end
to a local HTTP server, and a client left by a crashed ``serve`` ended
before the new one connects. Nothing outside this machine is contacted.
"""

import http.server
import json
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, List

import httpx
import pytest

from tests.integration.transcriber.fake_chisel import (
    install_fake_chisel,
    write_credential,
)
from transcriber.adapters.outbound.chisel_tunnel import (
    PID_FILE,
    ChiselTunnel,
    client_command,
    command_line,
    http_probe,
)
from transcriber.config import TunnelSettings

ROOT = Path(__file__).resolve().parents[3]
WAIT = 20.0


def until(condition: Callable[[], bool], wait: float = WAIT) -> None:
    """Wait until ``condition`` holds, or fail."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def free_port() -> int:
    """Return a local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def logged(log: Path) -> List[str]:
    """Return the argument lines the fake chisel recorded."""
    return log.read_text().splitlines() if log.exists() else []


def test_start_refusal_restart_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "chisel.log"
    chisel = install_fake_chisel(tmp_path / "bin", log)
    monkeypatch.setenv("AUTH", "not:this-one")
    credential = write_credential(tmp_path / "credential")
    settings = TunnelSettings(
        chisel=chisel, credential_file=tmp_path / "credential",
        public_host="app.example", backoff_initial=0.05, backoff_max=0.1,
    )
    seen: List[str] = []

    def pause(seconds: float) -> bool:
        seen.append(tunnel.status().describe())
        time.sleep(seconds)
        return False

    tunnel = ChiselTunnel(settings, 8799, tmp_path / "data", pause=pause)
    tunnel.start()
    until(lambda: tunnel.status().state == "connected")
    lines = logged(log)
    assert lines == [" ".join(client_command(settings, 8799)[1:])] * 2
    assert lines[0].endswith(
        "--max-retry-count 0 https://app.example/_tunnel"
        " R:0.0.0.0:9000:127.0.0.1:8799"
    )
    assert credential not in log.read_text()
    assert Path(f"{log}.env").read_text().splitlines() == ["AUTH=set"] * 2
    assert seen == [
        "reconnecting (chisel exited with code 1: client: Connection"
        " error: ssh: handshake failed: ssh: unable to authenticate)"
    ]
    pid = int((tmp_path / "data" / PID_FILE).read_text())
    assert command_line(pid) is not None
    assert tunnel.connections == 2

    tunnel.stop()
    until(lambda: Path(f"{log}.term").exists())
    assert tunnel.status().state == "stopped"
    assert not (tmp_path / "data" / PID_FILE).exists()
    time.sleep(0.2)
    assert len(logged(log)) == 2  # nothing started after the stop


def docker_can_run() -> bool:
    """Tell whether docker is installed and its daemon answers."""
    if shutil.which("docker") is None:
        return False
    done = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    return done.returncode == 0


def pinned_image() -> str:
    """Return the chisel image pinned in ``infra/compose.yml``."""
    text = (ROOT / "infra" / "compose.yml").read_text(encoding="utf-8")
    found = re.search(r"image: (jpillora/chisel:\S+)", text)
    assert found is not None
    return found.group(1)


@pytest.fixture(scope="module")
def real_chisel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The chisel binary of the pinned image, copied out with docker."""
    if not docker_can_run():
        pytest.skip("docker is not available to copy the pinned chisel out")
    target = tmp_path_factory.mktemp("chisel") / "chisel"
    name = f"ragputin-chisel-copy-{time.monotonic_ns()}"
    subprocess.run(["docker", "create", "--name", name, pinned_image()],
                   capture_output=True, check=True, timeout=300)
    try:
        subprocess.run(["docker", "cp", f"{name}:/ko-app/chisel",
                        str(target)], capture_output=True, check=True,
                       timeout=60)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                       check=False, timeout=60)
    target.chmod(0o755)
    return target


class Health(http.server.BaseHTTPRequestHandler):
    """The stand-in API: ``/api/health`` only."""

    def log_message(self, format: str, *args: object) -> None:
        """Stay quiet."""

    def do_GET(self) -> None:
        """Answer the health check."""
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200 if self.path == "/api/health" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def api() -> Iterator[int]:
    """A local HTTP server; yields its port."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield int(server.server_address[1])
    server.shutdown()
    server.server_close()


def test_real_chisel_end_to_end_and_a_leftover_client(
    tmp_path: Path, real_chisel: Path, api: int
) -> None:
    server_port, remote_port = free_port(), free_port()
    credential = write_credential(tmp_path / "credential")
    users = tmp_path / "users.json"
    users.write_text(json.dumps({
        credential: [rf"^R:0\.0\.0\.0:{remote_port}$"],
    }))
    server = subprocess.Popen(
        [str(real_chisel), "server", "--host", "127.0.0.1", "--port",
         str(server_port), "--reverse", "--authfile", str(users)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    settings = TunnelSettings(
        chisel=real_chisel, credential_file=tmp_path / "credential",
        server_url=f"http://127.0.0.1:{server_port}/_tunnel",
        remote_port=remote_port, backoff_initial=0.2, backoff_max=0.5,
        probe_interval=0.2,
    )
    through = f"http://127.0.0.1:{remote_port}/api/health"
    until(lambda: _listens(server_port))
    data = tmp_path / "data"
    data.mkdir()
    leftover = subprocess.Popen(
        client_command(settings, api), env={"AUTH": credential},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    tunnel = None
    try:
        until(lambda: _answers(through))  # the "crashed" client works
        (data / PID_FILE).write_text(f"{leftover.pid}\n")
        tunnel = ChiselTunnel(settings, api, data,
                              probe=http_probe(through, 2.0))
        tunnel.start()
        assert leftover.wait(WAIT) is not None  # ended by the new start
        until(lambda: tunnel is not None
              and tunnel.status().state == "connected")
        assert httpx.get(through, timeout=5).json() == {"status": "ok"}
        assert tunnel.connections >= 1
        tunnel.stop()
        until(lambda: not _answers(through))
    finally:
        if tunnel is not None:
            tunnel.stop()
        for process in (leftover, server):
            if process.poll() is None:
                process.kill()
            process.wait()


def test_real_chisel_with_a_wrong_credential_reconnects(
    tmp_path: Path, real_chisel: Path, api: int
) -> None:
    server_port, remote_port = free_port(), free_port()
    write_credential(tmp_path / "credential")
    users = tmp_path / "users.json"
    users.write_text(json.dumps({
        "workstation:" + "other-" * 8: [rf"^R:0\.0\.0\.0:{remote_port}$"],
    }))
    server = subprocess.Popen(
        [str(real_chisel), "server", "--host", "127.0.0.1", "--port",
         str(server_port), "--reverse", "--authfile", str(users)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    settings = TunnelSettings(
        chisel=real_chisel, credential_file=tmp_path / "credential",
        server_url=f"http://127.0.0.1:{server_port}/_tunnel",
        remote_port=remote_port, backoff_initial=5.0,
    )
    until(lambda: _listens(server_port))
    tunnel = ChiselTunnel(settings, api, tmp_path / "data")
    try:
        tunnel.start()
        until(lambda: tunnel.status().state == "reconnecting")
        assert "unable to authenticate" in (tunnel.status().reason or "")
    finally:
        tunnel.stop()
        server.kill()
        server.wait()


def _listens(port: int) -> bool:
    """Tell whether something accepts connections on ``port``."""
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _answers(url: str) -> bool:
    """Tell whether ``url`` answers 200."""
    try:
        return httpx.get(url, timeout=1.0).status_code == 200
    except httpx.TransportError:
        return False
