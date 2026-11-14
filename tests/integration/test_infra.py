"""The server files in ``infra/``: their rules, the workstation set-up
(``infra/tunnel_setup.py``) and a docker proof of the two containers.

Nothing here contacts the server or GitHub. The docker proof
(``infra/tests/docker_proof.py``) runs local containers of the images the
server uses; it is skipped when docker cannot run them.
"""

import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"


def read(name: str) -> str:
    """Return the text of a file of ``infra/``."""
    return (INFRA / name).read_text(encoding="utf-8")


def load_setup() -> ModuleType:
    """Import ``infra/tunnel_setup.py`` (``infra`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "tunnel_setup", INFRA / "tunnel_setup.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_setup()


class TestCaddyfile:
    def test_the_api_goes_through_the_tunnel_port(self) -> None:
        text = read("Caddyfile")
        assert "reverse_proxy tunnel:9000 {" in text
        assert "unix/" not in text
        assert "header_up X-Client-IP {client_ip}" in text
        for dropped in ("X-Forwarded-For", "Forwarded"):
            assert f"header_up -{dropped}" in text
        assert "flush_interval -1" in text
        size = re.search(r"max_size (\d+)MiB", text)
        assert size is not None and int(size.group(1)) > 256

    def test_only_websocket_upgrades_reach_chisel(self) -> None:
        text = read("Caddyfile")
        tunnel = re.search(r"@tunnel \{(.*?)\n\t\}", text, re.DOTALL)
        assert tunnel is not None
        assert "path /_tunnel\n" in tunnel.group(1)
        assert "header Upgrade websocket" in tunnel.group(1)
        assert "handle @tunnel {\n\t\treverse_proxy tunnel:8080\n" in text
        assert "handle /_tunnel* {\n\t\trespond 404" in text

    def test_trusted_proxies_come_from_the_environment(self) -> None:
        assert ("trusted_proxies static {$TRUSTED_PROXIES:private_ranges}"
                in read("Caddyfile"))
        assert ("TRUSTED_PROXIES: ${TRUSTED_PROXIES:-private_ranges}"
                in read("compose.yml"))

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


def service(name: str) -> str:
    """Return the block of ``name`` in ``compose.yml``."""
    text = read("compose.yml")
    block = text.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-z]+:\n|\nnetworks:\n", block, maxsplit=1)[0]


class TestCompose:
    def test_the_rules_of_the_shared_server(self) -> None:
        for name, memory in (("web", "128m"), ("tunnel", "64m")):
            block = service(name)
            for rule in ("read_only: true", "cap_drop: [ALL]",
                         f"mem_limit: {memory}", "cpus: ", "pids_limit: ",
                         "no-new-privileges:true", 'user: "65534:65534"',
                         "restart: unless-stopped"):
                assert rule in block, (name, rule)
            assert "ports:" not in block, name
            assert re.search(r"image: \S+:\S+@sha256:[0-9a-f]{64}", block)

    def test_web(self) -> None:
        block = service("web")
        assert "image: caddy:2.10-alpine@" in block
        assert "../pruebas:/srv:ro" in block
        assert "aliases: [ragputin-web]" in block
        assert "tunnel: {}" in block
        assert "http://127.0.0.1/healthz" in block
        assert "/run/tunnel" not in read("compose.yml")
        assert "1001" not in read("compose.yml")

    def test_tunnel(self) -> None:
        block = service("tunnel")
        assert "image: jpillora/chisel:1.12.0@sha256:" in block
        assert "networks: [tunnel]" in block  # not on sanatorio_default
        assert ("../secrets/tunnel-users.json:/etc/chisel/users.json:ro"
                in block)
        command = block.split("command:\n", 1)[1]
        command = command.split("\n    environment:")[0]
        flags = re.findall(r"^      - (\S+)$", command, re.MULTILINE)
        assert flags == [
            "server", "--host=0.0.0.0", "--port=8080", "--reverse",
            "--authfile=/etc/chisel/users.json", "--keepalive=25s",
        ]

    def test_networks(self) -> None:
        tail = read("compose.yml").split("\nnetworks:\n")[-1]
        assert "name: sanatorio_default" in tail
        assert "  tunnel:\n    internal: true" in tail

    def test_the_ssh_tunnel_is_gone(self) -> None:
        for name in ("tunnel-clean.sh", "authorized_keys.tunnel.example"):
            assert not (INFRA / name).exists()
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        assert "tunnel-key" not in makefile
        assert "ssh-keygen" not in makefile


class TestInstallScripts:
    def test_the_server_side_is_streamed(self) -> None:
        text = read("install-infra.sh")
        assert ('ssh "$target" "sh -s -- $R \'$image\'" <'
                ' "$here/install-remote.sh"') in text
        assert "REMOTE" not in text

    def test_validated_with_the_proxies_web_gets(self) -> None:
        text = read("install-remote.sh")
        assert 'cp "$R/infra/.env" .env' in text
        assert "config --format json" in text
        assert '-e "TRUSTED_PROXIES=$proxies"' in text
        validate = text.index("caddy validate --config")
        assert validate < text.index('cp Caddyfile compose.yml "$R/infra/"')

    def test_web_is_recreated_and_checked_before_success(self) -> None:
        text = read("install-remote.sh")
        up = text.index("docker compose up -d --remove-orphans")
        recreate = text.index(
            "docker compose up -d --no-deps --force-recreate web")
        health = text.index("http://127.0.0.1/healthz")
        chisel = text.index("http://tunnel:8080/health")
        done = text.index('echo "RAGPutin containers up')
        assert up < recreate < health < chisel < done
        assert text.startswith("#!/bin/sh\n")
        assert "pipefail" not in text  # POSIX sh: the CLI image has no bash


class TestClientInstall:
    def test_the_pinned_checksums(self) -> None:
        sums = setup.pinned_checksums()
        assert set(sums) == {
            f"chisel_{setup.VERSION}_{name}.gz" for name in (
                "linux_amd64", "linux_arm64", "darwin_amd64",
                "darwin_arm64",
            )
        }
        assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in sums.values())

    @pytest.mark.parametrize("system, machine, name", [
        ("Linux", "x86_64", "linux_amd64"),
        ("Linux", "aarch64", "linux_arm64"),
        ("Darwin", "arm64", "darwin_arm64"),
        ("Darwin", "x86_64", "darwin_amd64"),
    ])
    def test_the_platform(self, system: str, machine: str, name: str) -> None:
        assert setup.platform_name(system, machine) == name

    def test_an_unknown_platform_is_refused(self) -> None:
        with pytest.raises(setup.SetupError):
            setup.platform_name("Windows", "x86_64")

    def test_a_matching_download_is_installed(self, tmp_path: Path) -> None:
        archive = gzip.compress(b"#!/bin/sh\necho chisel\n")
        asked: List[str] = []

        def download(url: str) -> bytes:
            asked.append(url)
            return archive

        target = tmp_path / "bin" / "chisel"
        name = f"chisel_{setup.VERSION}_linux_amd64.gz"
        digest = hashlib.sha256(archive).hexdigest()
        assert setup.install_client(
            target, "linux_amd64", {name: digest}, download
        ) == digest
        assert asked == [
            "https://github.com/jpillora/chisel/releases/download/"
            f"v{setup.VERSION}/{name}"
        ]
        assert target.read_bytes() == b"#!/bin/sh\necho chisel\n"
        assert target.stat().st_mode & 0o777 == 0o755

    def test_a_mismatch_installs_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "bin" / "chisel"
        name = f"chisel_{setup.VERSION}_linux_amd64.gz"
        with pytest.raises(setup.SetupError, match="nothing installed"):
            setup.install_client(target, "linux_amd64", {name: "0" * 64},
                                 lambda url: gzip.compress(b"evil"))
        assert not (tmp_path / "bin").exists()

    def test_only_https_is_fetched(self) -> None:
        with pytest.raises(setup.SetupError, match="not https"):
            setup.fetch("http://example.invalid/chisel.gz")


class TestSecret:
    def run(self, tmp_path: Path, *extra: str) -> List[str]:
        lines: List[str] = []
        code = setup.main(
            ["secret", "--credential", str(tmp_path / "cfg" / "cred"),
             "--users", str(tmp_path / "out" / "users.json"), *extra],
            out=lines.append,
        )
        assert code == 0
        return lines

    def test_made_once_and_matched_by_the_users_file(
        self, tmp_path: Path
    ) -> None:
        lines = self.run(tmp_path)
        cred = tmp_path / "cfg" / "cred"
        made = cred.read_text().strip()
        user, password = made.split(":")
        assert user == "workstation" and len(password) >= 32
        assert cred.stat().st_mode & 0o777 == 0o600
        assert cred.parent.stat().st_mode & 0o777 == 0o700
        users = tmp_path / "out" / "users.json"
        assert json.loads(users.read_text()) == {
            made: [r"^R:0\.0\.0\.0:9000$"],
        }
        assert users.stat().st_mode & 0o777 == 0o600
        assert password not in "\n".join(lines)
        assert "made" in lines[0] and str(users) in lines[1]
        self.run(tmp_path, "--remote-port", "9100")
        assert cred.read_text().strip() == made  # kept
        assert json.loads(users.read_text()) == {
            made: [r"^R:0\.0\.0\.0:9100$"],
        }

    def test_the_regex_matches_only_the_one_remote(self) -> None:
        allowed = list(json.loads(setup.users_file("u:p", 9000)).values())
        pattern = re.compile(allowed[0][0])
        assert pattern.search("R:0.0.0.0:9000")
        for other in ("R:0.0.0.0:90001", "R:0.0.0.0:9001", "socks",
                      "R:127.0.0.1:9000", "xR:0.0.0.0:9000",
                      "R:0a0a0a0:9000", "web:80"):
            assert not pattern.search(other), other

    def test_a_credential_others_can_read_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cred = tmp_path / "cred"
        cred.write_text("workstation:" + "x" * 40)
        cred.chmod(0o640)
        assert setup.main(["secret", "--credential", str(cred), "--users",
                           str(tmp_path / "u.json")]) == 1
        assert "read by others" in capsys.readouterr().err
        assert not (tmp_path / "u.json").exists()


def docker_can_run() -> bool:
    """Tell whether docker is installed and its daemon answers."""
    if shutil.which("docker") is None:
        return False
    done = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True, text=True, check=False, timeout=30,
    )
    return done.returncode == 0


def test_the_containers_in_docker() -> None:
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
    assert len(lines) >= 35, done.stdout
