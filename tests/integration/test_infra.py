"""The server files in ``infra/``: their rules, the clean-up script and a
docker proof of the web container.

Nothing here contacts the server. The docker proof
(``infra/tests/docker_proof.py``) runs local containers of the image the
server uses; it is skipped when docker cannot run them.
"""

import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"
CLEAN = INFRA / "tunnel-clean.sh"


def read(name: str) -> str:
    """Return the text of a file of ``infra/``."""
    return (INFRA / name).read_text(encoding="utf-8")


class TestCaddyfile:
    def test_the_api_goes_through_the_tunnel_socket(self) -> None:
        text = read("Caddyfile")
        assert "reverse_proxy unix//run/tunnel/app.sock" in text
        assert "header_up X-Client-IP {client_ip}" in text
        for dropped in ("X-Forwarded-For", "Forwarded"):
            assert f"header_up -{dropped}" in text
        assert "flush_interval -1" in text
        size = re.search(r"max_size (\d+)MiB", text)
        assert size is not None and int(size.group(1)) > 256

    def test_security_headers(self) -> None:
        text = read("Caddyfile")
        assert 'Permissions-Policy "microphone=(self), camera=(),' in text
        csp = re.search(r'Content-Security-Policy "([^"]+)"', text)
        assert csp is not None
        policy = dict(
            (part.split()[0], part.split()[1:])
            for part in csp.group(1).split("; ")
        )
        assert policy["default-src"] == ["'self'"]
        assert policy["script-src"] == ["'self'"]
        assert policy["frame-ancestors"] == ["'none'"]
        assert "'unsafe-inline'" not in csp.group(1)
        assert 'Referrer-Policy "no-referrer"' in text
        assert "Strict-Transport-Security" not in text  # the outer Caddy's

    def test_errors_are_small_json_and_health_needs_no_tunnel(self) -> None:
        text = read("Caddyfile")
        assert "handle_errors 502 503 504" in text
        assert '"code":"workstation_unavailable"' in text
        assert "handle /healthz" in text


class TestCompose:
    def test_the_rules_of_the_shared_server(self) -> None:
        text = read("compose.yml")
        assert "image: caddy:2.10-alpine" in text
        assert 'user: "1001:1001"' in text
        assert "../run:/run/tunnel:ro" in text
        assert "../pruebas:/srv:ro" in text
        for rule in ("read_only: true", "cap_drop: [ALL]", "mem_limit: 128m",
                     "cpus: 0.5", "pids_limit: 64",
                     "no-new-privileges:true", "name: sanatorio_default",
                     "aliases: [ragputin-web]", "http://127.0.0.1/healthz"):
            assert rule in text, rule

    def test_the_tunnel_key_is_forwarding_only(self) -> None:
        line = read("authorized_keys.tunnel.example").strip()
        assert line.startswith(
            'restrict,port-forwarding,'
            'command="/opt/ragputin/infra/tunnel-clean.sh" ssh-ed25519 '
        )


@pytest.fixture()
def run_dir() -> Iterator[Path]:
    """A short directory for a unix socket (paths are limited to 108)."""
    path = Path(tempfile.mkdtemp(prefix="rp", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def clean(socket_path: Path) -> "subprocess.CompletedProcess[str]":
    """Run the clean-up script on ``socket_path``."""
    return subprocess.run(
        ["sh", str(CLEAN)], capture_output=True, text=True, check=False,
        env={**os.environ, "RAGPUTIN_SOCKET": str(socket_path)},
    )


@pytest.mark.skipif(shutil.which("ss") is None, reason="needs ss (iproute2)")
class TestTunnelClean:
    def test_nothing_to_clean(self, run_dir: Path) -> None:
        done = clean(run_dir / "app.sock")
        assert done.returncode == 0 and "no socket" in done.stderr

    def test_a_live_forward_keeps_its_socket(self, run_dir: Path) -> None:
        path = run_dir / "app.sock"
        with socket.socket(socket.AF_UNIX) as listening:
            listening.bind(str(path))
            listening.listen(1)
            done = clean(path)
            assert done.returncode == 0 and "kept" in done.stderr
            assert path.exists()

    def test_a_stale_socket_is_removed(self, run_dir: Path) -> None:
        path = run_dir / "app.sock"
        with socket.socket(socket.AF_UNIX) as stale:
            stale.bind(str(path))
        done = clean(path)
        assert done.returncode == 0 and "removed" in done.stderr
        assert not path.exists()

    def test_anything_else_is_left_alone(self, run_dir: Path) -> None:
        path = run_dir / "app.sock"
        path.write_text("not a socket")
        done = clean(path)
        assert done.returncode == 1 and path.exists()

    def test_the_script_is_executable(self) -> None:
        assert os.access(CLEAN, os.X_OK)


def docker_can_run() -> bool:
    """Tell whether docker is installed and its daemon answers."""
    if shutil.which("docker") is None:
        return False
    done = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    return done.returncode == 0


def test_the_web_container_in_docker() -> None:
    """``infra/tests/docker_proof.py``: every check passes."""
    if not docker_can_run():
        reason = "docker is not available"
        if os.environ.get("CI"):
            reason += " on this CI runner"
        pytest.skip(reason)
    done = subprocess.run(
        [sys.executable, str(INFRA / "tests" / "docker_proof.py")],
        capture_output=True, text=True, check=False, timeout=600,
    )
    lines: Dict[str, str] = {
        line[5:].split(":")[0]: line[:4]
        for line in done.stdout.splitlines()
        if line[:4] in ("ok  ", "FAIL")
    }
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
    assert "FAIL" not in lines.values()
    assert len(lines) >= 12, done.stdout
