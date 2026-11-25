"""The services guard (#264): ``infra/live_guard.py`` and the copy of
``ragputin-servicios`` in the repository.

The guard reads the GPU lease and the transcriber's live marker under
a temporary ``XDG_RUNTIME_DIR``. The script is run with stand-ins for
``systemctl``, ``pkill``, ``pgrep``, ``systemd-run`` and ``curl`` first
on its ``PATH``, so nothing real is ever stopped or started here.
"""

import importlib.util
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "live_guard", ROOT / "infra" / "live_guard.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_guard()


def write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str)
                    else json.dumps(document))


def lease(holder: str = "ragputin-live", minutes: float = 0.25,
          pid: int = 4242) -> Dict[str, Any]:
    return {"schema": "gpu-lease.v1", "holder": holder, "priority": "live",
            "pid": pid, "acquired_at": NOW.isoformat(),
            "expires_at": (NOW + timedelta(minutes=minutes)).isoformat(),
            "need_mib": 2200}


def alive(pid: Any) -> bool:
    return bool(pid == 4242)


def env(tmp: Path) -> Dict[str, str]:
    return {"XDG_RUNTIME_DIR": str(tmp)}


def test_nothing_running_is_free(tmp_path: Path) -> None:
    assert guard.check(env(tmp_path), NOW, alive) == (
        0, "No hay ninguna entrevista en directo en marcha.")


def test_a_valid_live_lease_is_an_interview(tmp_path: Path) -> None:
    write(tmp_path / "gpu-lease" / "lease.json", lease())
    status, line = guard.check(env(tmp_path), NOW, alive)
    assert status == 3 and "reservada para el directo" in line


@pytest.mark.parametrize("found", [
    lease(minutes=-1), lease(holder="transcriber-batch"), lease(pid=7)])
def test_an_expired_foreign_or_dead_lease_is_not(
    tmp_path: Path, found: Dict[str, Any],
) -> None:
    write(tmp_path / "gpu-lease" / "lease.json", found)
    assert guard.check(env(tmp_path), NOW, alive)[0] == 0


def test_a_session_waiting_for_its_audio_is_an_interview_too(
    tmp_path: Path,
) -> None:
    """It may hold no lease (#263 grace, or the CPU model)."""
    write(tmp_path / "ragputin" / "live-active.json", {
        "pid": 4242, "sessions": [
            {"id": "live-20261125-1000", "state": "running",
             "source": "lost"}]})
    status, line = guard.check(env(tmp_path), NOW, alive)
    assert status == 3
    assert line == ("Entrevista en directo en marcha: live-20261125-1000"
                    " (esperando a que vuelva el audio).")


def test_a_stale_marker_or_an_empty_one_is_not(tmp_path: Path) -> None:
    marker = tmp_path / "ragputin" / "live-active.json"
    write(marker, {"pid": 7, "sessions": [{"id": "x"}]})
    assert guard.check(env(tmp_path), NOW, alive)[0] == 0
    write(marker, {"pid": 4242, "sessions": []})
    assert guard.check(env(tmp_path), NOW, alive)[0] == 0


def test_files_that_cannot_be_read_refuse_too(tmp_path: Path) -> None:
    write(tmp_path / "gpu-lease" / "lease.json", "{broken")
    status, line = guard.check(env(tmp_path), NOW, alive)
    assert status == 2 and line.startswith("No se pudo comprobar")
    write(tmp_path / "gpu-lease" / "lease.json",
          {**lease(), "expires_at": "2026-11-25T12:00:00"})
    assert guard.check(env(tmp_path), NOW, alive)[0] == 2


def test_without_a_runtime_directory_the_temporary_one_is_read() -> None:
    lease_dir, marker_dir = guard.runtime_dir({})
    assert lease_dir.name == f"gpu-lease-{os.getuid()}"
    assert marker_dir.name == f"ragputin-{os.getuid()}"
    assert guard.pid_alive(os.getpid()) and not guard.pid_alive("x")


# ---- the services script, with stand-ins for the system's commands ------

STUBS = ("systemctl", "pkill", "pgrep", "systemd-run", "curl")


def run_script(
    tmp_path: Path, *args: str,
) -> "subprocess.CompletedProcess[str]":
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "calls.log"
    for name in STUBS:
        stub = bin_dir / name
        code = 1 if name == "pgrep" else 0
        stub.write_text(f"#!/bin/sh\necho \"{name} $*\" >> '{calls}'\n"
                        f"exit {code}\n")
        stub.chmod(0o755)
    environ = {
        "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home"),
        "XDG_RUNTIME_DIR": str(tmp_path / "run"), "RAGPUTIN_DIR": str(ROOT),
    }
    return subprocess.run(
        ["bash", str(ROOT / "infra" / "ragputin-servicios"), *args],
        env=environ, capture_output=True, text=True, timeout=60)


def busy(tmp_path: Path) -> None:
    write(tmp_path / "run" / "ragputin" / "live-active.json", {
        "pid": os.getpid(), "sessions": [
            {"id": "live-1", "state": "running", "source": "connected"}]})


def calls(tmp_path: Path) -> str:
    path = tmp_path / "calls.log"
    return path.read_text() if path.exists() else ""


def test_stop_and_restart_refuse_under_an_interview(tmp_path: Path) -> None:
    busy(tmp_path)
    for action in ("stop", "restart"):
        done = run_script(tmp_path, action)
        assert done.returncode == 3, done.stderr
        assert "Entrevista en directo en marcha: live-1." in done.stdout
        assert "--force" in done.stdout
    assert "systemctl" not in calls(tmp_path)
    assert "pkill" not in calls(tmp_path), "nothing was stopped"


def test_force_stops_anyway_and_a_free_machine_stops(tmp_path: Path) -> None:
    busy(tmp_path)
    forced = run_script(tmp_path, "stop", "--force")
    assert forced.returncode == 0, forced.stderr
    assert "systemctl --user stop ragputin-orchestrator" in calls(tmp_path)
    (tmp_path / "run" / "ragputin" / "live-active.json").unlink()
    (tmp_path / "calls.log").unlink()
    free = run_script(tmp_path, "stop")
    assert free.returncode == 0, free.stderr
    assert "pkill -f" in calls(tmp_path)


def test_status_says_whether_an_interview_is_running(tmp_path: Path) -> None:
    done = run_script(tmp_path, "status")
    assert "Entrevista en directo:  No hay ninguna entrevista" in done.stdout
    busy(tmp_path)
    done = run_script(tmp_path, "status")
    assert "Entrevista en directo:  Entrevista en directo en marcha" \
        in done.stdout


def test_units_restart_on_failure() -> None:
    text = (ROOT / "infra" / "ragputin-servicios").read_text()
    assert "-p Restart=on-failure -p RestartSec=2" in text
    assert "/home/" not in text, "no one's home folder in the repository"
