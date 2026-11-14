"""The SSH tunnel supervisor with fake processes (no ssh is run)."""

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pytest

from transcriber.adapters.outbound.ssh_tunnel import (
    SshTunnel,
    backoff,
    cleanup_command,
    forward_command,
    ssh_options,
)
from transcriber.application.ports import TunnelStatus
from transcriber.config import TunnelSettings

WAIT = 10.0


def settings(**changes: object) -> TunnelSettings:
    """Fast settings: tiny delays, a known key path."""
    values: dict[str, object] = dict(
        identity_file=Path("/keys/tunnel"), backoff_initial=0.01,
        backoff_max=0.04, connect_grace=0.02, cleanup_timeout=0.2,
    )
    values.update(changes)
    return TunnelSettings.model_validate(values)


class FakeProcess:
    """A child that exits with ``code`` at once, or runs until stopped."""

    def __init__(self, code: Optional[int], output: str = "") -> None:
        self.code = code
        self._output = output
        self.terminated = threading.Event()

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if self.code is not None:
            return self.code
        if self.terminated.wait(timeout):
            return -15
        return None

    def terminate(self) -> None:
        self.terminated.set()

    def output(self) -> str:
        return self._output


class FakeRunner:
    """Start fake processes: clean-ups and forwards from scripted lists."""

    def __init__(
        self,
        forwards: List[FakeProcess],
        cleanups: Optional[List[FakeProcess]] = None,
    ) -> None:
        self.forwards = forwards
        self.cleanups = cleanups or []
        self.started: List[List[str]] = []
        self.children: List[FakeProcess] = []
        self.fail = False

    def __call__(self, args: Sequence[str]) -> FakeProcess:
        if self.fail:
            raise FileNotFoundError("no ssh here")
        self.started.append(list(args))
        is_cleanup = args[-1] == "true"
        queue = self.cleanups if is_cleanup else self.forwards
        child = queue.pop(0) if queue else FakeProcess(
            0 if is_cleanup else None
        )
        self.children.append(child)
        return child

    def kinds(self) -> List[str]:
        return ["cleanup" if a[-1] == "true" else "forward"
                for a in self.started]


def until(condition: Callable[[], bool]) -> None:
    """Wait until ``condition`` holds, or fail."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached")


class TestCommands:
    def test_options_are_strict_and_use_the_key(self) -> None:
        assert ssh_options(settings()) == [
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-i", "/keys/tunnel", "-o", "StrictHostKeyChecking=yes",
        ]

    def test_known_hosts_and_home_are_expanded(self) -> None:
        chosen = settings(identity_file=Path("~/k"),
                          known_hosts=Path("/etc/kh"))
        options = ssh_options(chosen)
        assert options[5] == str(Path("~/k").expanduser())
        assert options[-2:] == ["-o", "UserKnownHostsFile=/etc/kh"]

    def test_cleanup_then_forward(self) -> None:
        chosen = settings(target="me@host", remote_socket="/run/a.sock")
        assert cleanup_command(chosen) == [
            "ssh", *ssh_options(chosen), "me@host", "true",
        ]
        assert forward_command(chosen, 8770) == [
            "ssh", "-N", "-R", "/run/a.sock:127.0.0.1:8770",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            *ssh_options(chosen), "me@host",
        ]

    def test_defaults_point_at_the_server(self) -> None:
        default = TunnelSettings()
        assert default.target == "ubuntu@rag.peer4u.app"
        assert default.remote_socket == "/opt/ragputin/run/app.sock"
        assert default.public_host == "rag.peer4u.app"
        assert default.known_hosts is None

    def test_backoff_doubles_up_to_the_cap(self) -> None:
        chosen = settings(backoff_initial=1.0, backoff_factor=2.0,
                          backoff_max=10.0)
        assert [backoff(chosen, n) for n in range(1, 7)] == [
            1.0, 2.0, 4.0, 8.0, 10.0, 10.0,
        ]


class TestStatus:
    def test_describe(self) -> None:
        assert TunnelStatus().describe() == "stopped"
        assert TunnelStatus(state="connected").describe() == "connected"
        down = TunnelStatus(state="reconnecting", reason="ssh exited")
        assert down.describe() == "reconnecting (ssh exited)"


class TestSupervisor:
    def test_cleans_up_forwards_and_stops(self) -> None:
        runner = FakeRunner([])
        tunnel = SshTunnel(settings(), 9000, runner)
        assert tunnel.status().state == "stopped"
        tunnel.start()
        tunnel.start()  # once only
        until(lambda: tunnel.status().state == "connected")
        assert runner.kinds() == ["cleanup", "forward"]
        assert "/opt/ragputin/run/app.sock:127.0.0.1:9000" in (
            runner.started[1]
        )
        tunnel.stop()
        assert runner.children[1].terminated.is_set()
        assert tunnel.status().state == "stopped"
        assert runner.kinds() == ["cleanup", "forward"]

    def test_an_exit_reconnects_after_a_new_cleanup(self) -> None:
        dropped = FakeProcess(255, "debug\nremote port forwarding failed")
        runner = FakeRunner([dropped])
        delays: List[float] = []

        def pause(seconds: float) -> bool:
            delays.append(seconds)
            return False

        tunnel = SshTunnel(settings(), 9000, runner, pause=pause)
        tunnel.start()
        until(lambda: len(runner.started) == 4)
        until(lambda: tunnel.status().state == "connected")
        assert runner.kinds() == ["cleanup", "forward", "cleanup", "forward"]
        assert delays == [0.01]
        assert tunnel.connections == 2
        tunnel.stop()

    def test_the_reason_is_shown_while_reconnecting(self) -> None:
        runner = FakeRunner([FakeProcess(255, "Permission denied")])
        tunnel = SshTunnel(settings(), 9000, runner, pause=lambda s: True)
        tunnel.start()
        until(lambda: tunnel.status().state == "reconnecting")
        assert tunnel.status().describe() == (
            "reconnecting (ssh exited with code 255: Permission denied)"
        )
        tunnel.stop()
        assert tunnel.status().state == "stopped"

    def test_failures_back_off_and_a_stable_connection_resets(self) -> None:
        times = iter([0.0, 1.0, 1.0, 2.0, 2.0, 500.0, 500.0, 501.0])
        runner = FakeRunner([FakeProcess(1) for _ in range(4)])
        stopped = threading.Event()

        def pause(seconds: float) -> bool:
            delays.append(seconds)
            if len(delays) == 4:
                stopped.set()
            return stopped.is_set()

        delays: List[float] = []
        tunnel = SshTunnel(
            settings(backoff_initial=1.0, backoff_max=100.0,
                     stable_after=60.0),
            9000, runner, clock=lambda: next(times), pause=pause,
        )
        tunnel.start()
        until(lambda: len(delays) == 4)
        # 1 s, 2 s, then a connection that lasted 498 s resets to 1 s.
        assert delays == [1.0, 2.0, 1.0, 2.0]
        tunnel.stop()

    def test_a_failed_cleanup_is_not_fatal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        runner = FakeRunner([], [FakeProcess(1, "no such file")])
        tunnel = SshTunnel(settings(), 9000, runner)
        tunnel.start()
        until(lambda: tunnel.status().state == "connected")
        tunnel.stop()
        assert "clean-up failed" in caplog.text
        assert "no such file" in caplog.text

    def test_a_cleanup_that_hangs_is_ended(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hung = FakeProcess(None)
        runner = FakeRunner([], [hung])
        tunnel = SshTunnel(settings(), 9000, runner)
        tunnel.start()
        until(lambda: tunnel.status().state == "connected")
        assert hung.terminated.is_set()
        assert "timed out" in caplog.text
        tunnel.stop()

    def test_a_missing_client_keeps_retrying(self) -> None:
        runner = FakeRunner([])
        runner.fail = True
        delays: List[float] = []

        def pause(seconds: float) -> bool:
            delays.append(seconds)
            time.sleep(0.001)
            return False

        tunnel = SshTunnel(settings(), 9000, runner, pause=pause)
        tunnel.start()
        until(lambda: len(delays) >= 2)
        assert tunnel.status().state == "reconnecting"
        assert "cannot start ssh" in (tunnel.status().reason or "")
        tunnel.stop()

    def test_stopping_during_a_cleanup_starts_no_forward(self) -> None:
        hung = FakeProcess(None)
        runner = FakeRunner([], [hung])
        tunnel = SshTunnel(settings(cleanup_timeout=30.0), 9000, runner)
        tunnel.start()
        until(lambda: len(runner.started) == 1)
        tunnel.stop()
        assert hung.terminated.is_set()
        assert runner.kinds() == ["cleanup"]

    def test_stop_without_start(self) -> None:
        tunnel = SshTunnel(settings(), 9000, FakeRunner([]))
        tunnel.stop()
        assert tunnel.status().state == "stopped"
