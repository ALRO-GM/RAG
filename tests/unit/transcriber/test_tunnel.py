"""The chisel tunnel supervisor with fake processes (no chisel is run)."""

import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import pytest
from pydantic import ValidationError

from transcriber.adapters.outbound.chisel_tunnel import (
    PID_FILE,
    ChiselTunnel,
    LineHandler,
    TunnelSetupError,
    backoff,
    check_client,
    client_command,
    client_environment,
    end_leftover,
    exit_reason,
    is_our_client,
    read_credential,
)
from transcriber.application.ports import TunnelStatus
from transcriber.config import TunnelSettings

WAIT = 10.0
CREDENTIAL = "workstation:" + "p" * 40
CONNECTED = "2026/11/14 08:22:15 client: Connected (Latency 171µs)"


def settings(tmp_path: Path, **changes: object) -> TunnelSettings:
    """Fast settings with a credential file of mode 600 in ``tmp_path``."""
    credential = tmp_path / "credential"
    if not credential.exists():
        credential.write_text(CREDENTIAL + "\n")
        credential.chmod(0o600)
    values: Dict[str, object] = dict(
        chisel="/opt/bin/chisel", credential_file=str(credential),
        backoff_initial=0.01, backoff_max=0.04, probe_interval=0.01,
    )
    values.update(changes)
    return TunnelSettings.model_validate(values)


class FakeProcess:
    """A child that writes ``lines`` and exits with ``code``, or runs
    until it is stopped."""

    def __init__(
        self, code: Optional[int] = None, lines: Sequence[str] = (CONNECTED,)
    ) -> None:
        self.code = code
        self.lines = list(lines)
        self.pid = 4242
        self.terminated = threading.Event()

    def begin(self, on_line: LineHandler) -> None:
        for line in self.lines:
            on_line(line)

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if self.code is not None:
            return self.code
        if self.terminated.wait(timeout):
            return 0
        return None

    def terminate(self) -> None:
        self.terminated.set()

    def output(self) -> str:
        return "\n".join(self.lines)


class FakeRunner:
    """Start fake processes from a scripted list (then ones that run)."""

    def __init__(self, children: Optional[List[FakeProcess]] = None) -> None:
        self.queue = children or []
        self.started: List[List[str]] = []
        self.environments: List[Dict[str, str]] = []
        self.children: List[FakeProcess] = []
        self.fail = False

    def __call__(
        self, args: Sequence[str], env: Mapping[str, str],
        on_line: LineHandler,
    ) -> FakeProcess:
        if self.fail:
            raise FileNotFoundError("no chisel here")
        self.started.append(list(args))
        self.environments.append(dict(env))
        child = self.queue.pop(0) if self.queue else FakeProcess()
        self.children.append(child)
        child.begin(on_line)
        return child


def until(condition: Callable[[], bool]) -> None:
    """Wait until ``condition`` holds, or fail."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached")


def no_process(pid: int) -> Optional[List[str]]:
    """``describe`` for a machine where no leftover runs."""
    return None


class TestSettings:
    def test_defaults_point_at_the_server(self) -> None:
        default = TunnelSettings()
        assert default.url() == "https://rag.peer4u.app/_tunnel"
        assert default.health_url() == "https://rag.peer4u.app/api/health"
        assert default.remote_port == 9000
        assert default.chisel == Path("~/.local/share/ragputin/bin/chisel")
        assert default.credential_file == Path(
            "~/.config/ragputin/tunnel-credential"
        )

    def test_a_local_server_url_for_tests(self) -> None:
        chosen = TunnelSettings(server_url="http://127.0.0.1:8123/_tunnel")
        assert chosen.url() == "http://127.0.0.1:8123/_tunnel"
        assert chosen.health_url() == "http://127.0.0.1:8123/api/health"

    @pytest.mark.parametrize("field, value", [
        ("public_host", "-oProxyCommand=x"),
        ("public_host", "rag.peer4u.app:443"),
        ("public_host", "rag.peer4u.app/x"),
        ("public_host", "a b"),
        ("server_url", "--auth=x:y"),
        ("server_url", "http://rag.peer4u.app/_tunnel"),
        ("server_url", "https://host/_tunnel R:1:2"),
        ("server_url", "https://-host/_tunnel"),
        ("remote_port", 80),
        ("remote_port", 70000),
        ("chisel", "-v"),
        ("chisel", "chisel"),
        ("chisel", "/bin/chisel\n--auth"),
        ("credential_file", "--auth"),
        ("keepalive", 0),
    ])
    def test_nothing_can_reach_the_arguments_as_an_option(
        self, field: str, value: object
    ) -> None:
        with pytest.raises(ValidationError):
            TunnelSettings.model_validate({field: value})

    def test_the_command(self, tmp_path: Path) -> None:
        chosen = settings(tmp_path, public_host="app.example",
                          remote_port=9100, keepalive=30)
        assert client_command(chosen, 8770) == [
            "/opt/bin/chisel", "client", "--keepalive", "30s",
            "--max-retry-count", "0", "https://app.example/_tunnel",
            "R:0.0.0.0:9100:127.0.0.1:8770",
        ]

    def test_home_is_expanded(self, tmp_path: Path) -> None:
        chosen = settings(tmp_path, chisel="~/bin/chisel")
        assert client_command(chosen, 1)[0] == str(
            Path("~/bin/chisel").expanduser()
        )

    @pytest.mark.parametrize("port", [0, 70000])
    def test_a_wrong_port_is_refused(self, tmp_path: Path, port: int) -> None:
        with pytest.raises(ValueError):
            client_command(settings(tmp_path), port)

    def test_backoff_doubles_up_to_the_cap(self, tmp_path: Path) -> None:
        chosen = settings(tmp_path, backoff_initial=1.0, backoff_factor=2.0,
                          backoff_max=10.0)
        assert [backoff(chosen, n) for n in range(1, 7)] == [
            1.0, 2.0, 4.0, 8.0, 10.0, 10.0,
        ]


class TestCredential:
    def test_read_from_a_private_file(self, tmp_path: Path) -> None:
        path = tmp_path / "credential"
        path.write_text(CREDENTIAL + "\n")
        path.chmod(0o600)
        assert read_credential(path) == CREDENTIAL

    @pytest.mark.parametrize("mode", [0o640, 0o604, 0o660])
    def test_a_file_others_may_read_is_refused(
        self, tmp_path: Path, mode: int
    ) -> None:
        path = tmp_path / "credential"
        path.write_text(CREDENTIAL)
        path.chmod(mode)
        with pytest.raises(TunnelSetupError, match="read by others") as err:
            read_credential(path)
        assert CREDENTIAL not in str(err.value)

    def test_a_missing_file_says_how_to_make_it(self, tmp_path: Path) -> None:
        with pytest.raises(TunnelSetupError, match="make tunnel-secret"):
            read_credential(tmp_path / "missing")

    def test_a_link_is_refused(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.write_text(CREDENTIAL)
        real.chmod(0o600)
        link = tmp_path / "link"
        link.symlink_to(real)
        with pytest.raises(TunnelSetupError, match="not a regular file"):
            read_credential(link)

    @pytest.mark.parametrize("content", [
        "no-colon" + "x" * 40, "user:short", "us er:" + "x" * 40, "",
    ])
    def test_a_file_without_a_credential_is_refused(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / "credential"
        path.write_text(content)
        path.chmod(0o600)
        with pytest.raises(TunnelSetupError, match="does not hold") as err:
            read_credential(path)
        assert content not in str(err.value) or not content

    def test_check_client_wants_the_executable(self, tmp_path: Path) -> None:
        executable = tmp_path / "chisel"
        chosen = settings(tmp_path, chisel=str(executable))
        with pytest.raises(TunnelSetupError, match="make tunnel-client"):
            check_client(chosen)
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        check_client(chosen)

    def test_the_environment_carries_the_credential_and_little_else(
        self,
    ) -> None:
        env = client_environment(CREDENTIAL, {
            "PATH": "/bin", "HOME": "/home/me", "HTTPS_PROXY": "http://p",
            "TRANSCRIBER_PUBLISH_TOKEN": "secret", "AUTH": "old:value",
        })
        assert env == {"PATH": "/bin", "HOME": "/home/me",
                       "HTTPS_PROXY": "http://p", "AUTH": CREDENTIAL}


class TestExitReason:
    def test_the_error_line_without_its_time(self) -> None:
        output = "\n".join([
            "2026/11/14 08:22:00 client: Connecting to wss://x/_tunnel",
            "2026/11/14 08:22:00 client: Connection error: server: access"
            " to 'R:0.0.0.0:9001' denied",
            "2026/11/14 08:22:00 client: Give up",
            "2026/11/14 08:22:00 connection attempts exhausted",
        ])
        assert exit_reason(1, output) == (
            "chisel exited with code 1: client: Connection error: server:"
            " access to 'R:0.0.0.0:9001' denied"
        )

    def test_without_output(self) -> None:
        assert exit_reason(2, "") == "chisel exited with code 2"


class TestStatus:
    def test_describe(self) -> None:
        assert TunnelStatus().describe() == "stopped"
        assert TunnelStatus(state="connected").describe() == "connected"
        down = TunnelStatus(state="reconnecting", reason="chisel exited")
        assert down.describe() == "reconnecting (chisel exited)"
        waiting = TunnelStatus(state="connecting", reason="no answer")
        assert waiting.describe() == "connecting (no answer)"


class TestSupervisor:
    def test_connects_and_stops(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path, runner,
                              describe=no_process, environ={"PATH": "/bin"})
        assert tunnel.status().state == "stopped"
        tunnel.start()
        tunnel.start()  # once only
        until(lambda: tunnel.status().state == "connected")
        assert runner.started == [client_command(settings(tmp_path), 8770)]
        assert runner.environments == [{"PATH": "/bin", "AUTH": CREDENTIAL}]
        assert not any(CREDENTIAL in arg for arg in runner.started[0])
        assert (tmp_path / PID_FILE).read_text() == "4242\n"
        assert (tmp_path / PID_FILE).stat().st_mode & 0o777 == 0o600
        tunnel.stop()
        assert runner.children[0].terminated.is_set()
        assert tunnel.status().state == "stopped"
        assert not (tmp_path / PID_FILE).exists()
        assert len(runner.started) == 1

    def test_not_connected_before_chisel_says_so(
        self, tmp_path: Path
    ) -> None:
        runner = FakeRunner([FakeProcess(lines=["client: Connecting"])])
        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path, runner,
                              describe=no_process)
        tunnel.start()
        until(lambda: len(runner.started) == 1)
        time.sleep(0.3)
        assert tunnel.status().state == "connecting"
        tunnel.stop()

    def test_connected_only_once_the_probe_answers(
        self, tmp_path: Path
    ) -> None:
        calls: List[int] = []
        gate = threading.Event()

        def slow_probe() -> Optional[str]:
            calls.append(1)
            if len(calls) == 1:
                return "https://x/api/health answered 502"
            gate.wait(WAIT)
            return None

        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path,
                              FakeRunner(), probe=slow_probe,
                              describe=no_process)
        tunnel.start()
        until(lambda: tunnel.status().state == "connecting"
              and tunnel.status().reason is not None)
        assert tunnel.status().describe() == (
            "connecting (chisel connected; https://x/api/health"
            " answered 502)"
        )
        gate.set()
        until(lambda: tunnel.status().state == "connected")
        tunnel.stop()

    def test_an_exit_reconnects(self, tmp_path: Path) -> None:
        dropped = FakeProcess(1, [CONNECTED, "client: Connection error:"
                                  " websocket: close 1006"])
        runner = FakeRunner([dropped])
        delays: List[float] = []

        def pause(seconds: float) -> bool:
            delays.append(seconds)
            return False

        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path, runner,
                              pause=pause, describe=no_process)
        tunnel.start()
        until(lambda: len(runner.started) == 2)
        until(lambda: tunnel.status().state == "connected")
        assert delays == [0.01]
        assert tunnel.connections == 2
        tunnel.stop()

    def test_the_reason_is_shown_while_reconnecting(
        self, tmp_path: Path
    ) -> None:
        refused = FakeProcess(1, [
            "client: Connection error: ssh: handshake failed: ssh: unable"
            " to authenticate", "connection attempts exhausted",
        ])
        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path,
                              FakeRunner([refused]), pause=lambda s: True,
                              describe=no_process)
        tunnel.start()
        until(lambda: tunnel.status().state == "reconnecting")
        assert tunnel.status().describe() == (
            "reconnecting (chisel exited with code 1: client: Connection"
            " error: ssh: handshake failed: ssh: unable to authenticate)"
        )
        tunnel.stop()
        assert tunnel.status().state == "stopped"

    def test_failures_back_off_and_a_stable_connection_resets(
        self, tmp_path: Path
    ) -> None:
        times = iter([0.0, 1.0, 1.0, 2.0, 2.0, 500.0, 500.0, 501.0])
        runner = FakeRunner([FakeProcess(1) for _ in range(4)])
        stopped = threading.Event()
        delays: List[float] = []

        def pause(seconds: float) -> bool:
            delays.append(seconds)
            if len(delays) == 4:
                stopped.set()
            return stopped.is_set()

        tunnel = ChiselTunnel(
            settings(tmp_path, backoff_initial=1.0, backoff_max=100.0,
                     stable_after=60.0),
            8770, tmp_path, runner, clock=lambda: next(times), pause=pause,
            describe=no_process,
        )
        tunnel.start()
        until(lambda: len(delays) == 4)
        # 1 s, 2 s, then a connection that lasted 498 s resets to 1 s.
        assert delays == [1.0, 2.0, 1.0, 2.0]
        tunnel.stop()

    def test_a_missing_client_keeps_retrying(self, tmp_path: Path) -> None:
        runner = FakeRunner()
        runner.fail = True
        delays: List[float] = []

        def pause(seconds: float) -> bool:
            delays.append(seconds)
            time.sleep(0.001)
            return False

        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path, runner,
                              pause=pause, describe=no_process)
        tunnel.start()
        until(lambda: len(delays) >= 2)
        assert tunnel.status().state == "reconnecting"
        assert "cannot start /opt/bin/chisel" in (
            tunnel.status().reason or ""
        )
        tunnel.stop()

    def test_an_unsafe_credential_is_reported_not_used(
        self, tmp_path: Path
    ) -> None:
        chosen = settings(tmp_path)
        chosen.credential_file.chmod(0o644)
        runner = FakeRunner()
        tunnel = ChiselTunnel(chosen, 8770, tmp_path, runner,
                              pause=lambda s: True, describe=no_process)
        tunnel.start()
        until(lambda: tunnel.status().state == "reconnecting")
        assert "read by others" in (tunnel.status().reason or "")
        assert CREDENTIAL not in tunnel.status().describe()
        assert runner.started == []
        tunnel.stop()

    def test_stop_without_start(self, tmp_path: Path) -> None:
        tunnel = ChiselTunnel(settings(tmp_path), 8770, tmp_path,
                              FakeRunner(), describe=no_process)
        tunnel.stop()
        assert tunnel.status().state == "stopped"

    def test_a_leftover_client_is_ended_first(self, tmp_path: Path) -> None:
        (tmp_path / PID_FILE).write_text("777\n")
        running = {777}
        signals: List[tuple[int, int]] = []

        def describe(pid: int) -> Optional[List[str]]:
            if pid in running:
                return ["/opt/bin/chisel", "client", "--keepalive", "25s"]
            return None

        def send(pid: int, number: int) -> None:
            signals.append((pid, number))
            running.discard(pid)

        runner = FakeRunner()
        tunnel = ChiselTunnel(
            settings(tmp_path), 8770, tmp_path, runner, describe=describe,
            signal_process=send,
        )
        tunnel.start()
        until(lambda: tunnel.status().state == "connected")
        assert signals == [(777, signal.SIGTERM)]
        assert (tmp_path / PID_FILE).read_text() == "4242\n"
        tunnel.stop()


class TestLeftover:
    def test_a_reused_pid_is_left_alone(self, tmp_path: Path) -> None:
        path = tmp_path / PID_FILE
        path.write_text(f"{os.getpid()}\n")
        signals: List[int] = []
        ended = end_leftover(
            path, Path("/opt/bin/chisel"),
            lambda pid: ["/usr/bin/python3", "-m", "pytest"],
            lambda pid, number: signals.append(number),
        )
        assert ended is None and signals == []
        assert not path.exists()

    def test_one_that_ignores_sigterm_is_killed(self, tmp_path: Path) -> None:
        path = tmp_path / PID_FILE
        path.write_text("777")
        signals: List[int] = []
        ended = end_leftover(
            path, Path("/opt/bin/chisel"),
            lambda pid: ["/opt/bin/chisel", "client"],
            lambda pid, number: signals.append(number),
            pause=lambda s: None,
        )
        assert ended == 777
        assert signals == [signal.SIGTERM, signal.SIGKILL]

    @pytest.mark.parametrize("content", ["", "abc", "1", "0", "-5"])
    def test_a_pid_file_without_a_pid_is_removed(
        self, tmp_path: Path, content: str
    ) -> None:
        path = tmp_path / PID_FILE
        path.write_text(content)
        assert end_leftover(path, Path("/x"), lambda pid: ["/x", "client"],
                            lambda pid, number: None) is None
        assert not path.exists()

    def test_nothing_to_end(self, tmp_path: Path) -> None:
        assert end_leftover(tmp_path / PID_FILE, Path("/x")) is None

    def test_one_already_gone(self, tmp_path: Path) -> None:
        path = tmp_path / PID_FILE
        path.write_text("777")

        def send(pid: int, number: int) -> None:
            raise ProcessLookupError()

        assert end_leftover(path, Path("/x"), lambda pid: ["/x", "client"],
                            send) == 777

    def test_is_our_client(self) -> None:
        chisel = Path("/opt/bin/chisel")
        assert is_our_client(["/opt/bin/chisel", "client", "x"], chisel)
        assert not is_our_client(["/opt/bin/chisel", "server"], chisel)
        assert not is_our_client(["/other/chisel", "client"], chisel)
        assert not is_our_client(None, chisel)
        assert not is_our_client([], chisel)
