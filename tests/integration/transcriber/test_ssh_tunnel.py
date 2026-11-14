"""The tunnel supervisor running a fake ``ssh`` found first on ``PATH``."""

import os
import time
from pathlib import Path
from typing import Callable, List

import pytest

from tests.integration.transcriber.fake_ssh import install_fake_ssh
from transcriber.adapters.outbound.ssh_tunnel import SshTunnel
from transcriber.config import TunnelSettings

WAIT = 20.0


def until(condition: Callable[[], bool]) -> None:
    """Wait until ``condition`` holds, or fail."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def logged(log: Path) -> List[str]:
    """Return the argument lines the fake ssh recorded."""
    return log.read_text().splitlines() if log.exists() else []


def test_start_exit_reconnect_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    install_fake_ssh(bin_dir)
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    settings = TunnelSettings(
        target="tester@example.invalid", identity_file=tmp_path / "key",
        known_hosts=tmp_path / "known_hosts", backoff_initial=0.05,
        backoff_max=0.1, connect_grace=0.3,
    )
    tunnel = SshTunnel(settings, 8799)
    tunnel.start()
    until(lambda: tunnel.status().state == "connected")
    lines = logged(log)
    assert [line.split()[-1] for line in lines] == [
        "true", "tester@example.invalid", "true", "tester@example.invalid",
    ]
    assert lines[1].startswith(
        "-N -R /opt/ragputin/run/app.sock:127.0.0.1:8799 "
        "-o ExitOnForwardFailure=yes"
    )
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in lines[1]
    assert tunnel.connections == 2

    tunnel.stop()
    until(lambda: Path(f"{log}.term").exists())
    assert tunnel.status().state == "stopped"
    time.sleep(0.2)
    assert len(logged(log)) == 4  # nothing started after the stop


def test_the_reason_of_an_exit_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    install_fake_ssh(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SSH_LOG", str(tmp_path / "ssh.log"))
    seen: List[str] = []
    settings = TunnelSettings(identity_file=tmp_path / "key",
                              backoff_initial=0.5, connect_grace=0.3)

    def pause(seconds: float) -> bool:
        seen.append(tunnel.status().describe())
        return True  # stop after the first failure

    tunnel = SshTunnel(settings, 8799, pause=pause)
    tunnel.start()
    until(lambda: bool(seen))
    tunnel.stop()
    assert seen == [
        "reconnecting (ssh exited with code 255: Error: remote port"
        " forwarding failed for listen path)"
    ]
