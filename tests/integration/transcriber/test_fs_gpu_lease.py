"""The file lease: atomic files, heartbeat, expiry after a crash, CLI."""

import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from transcriber.adapters.inbound.cli import main
from transcriber.adapters.outbound.fs_gpu_lease import (
    ACK_FILE, FileSystemGpuLease, lease_directory, pid_alive,
)
from transcriber.adapters.outbound.thread_job_runner import ThreadJobRunner
from transcriber.application.ports import Progress
from transcriber.domain.errors import StateError
from transcriber.domain.gpu_lease import AckFile, LeaseFile, new_lease
from transcriber.domain.models import SessionState

WAIT = 10.0


class Clock:
    """A clock the test moves."""

    def __init__(self) -> None:
        self.at = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.at


def test_acquire_writes_a_valid_file_and_release_deletes_it(
    tmp_path: Path,
) -> None:
    lease = FileSystemGpuLease(tmp_path / "gpu-lease", heartbeat_seconds=60)
    held = lease.acquire(6000)
    data = json.loads(lease.path.read_text())
    assert data["schema"] == "gpu-lease.v1" and data["need_mib"] == 6000
    assert data["pid"] == os.getpid() and data["priority"] == "live"
    assert LeaseFile.model_validate(data) == held
    assert lease.current() == held
    assert [p.name for p in lease.path.parent.iterdir()] == ["lease.json"]
    lease.release()
    assert not lease.path.exists() and lease.current() is None


def test_heartbeat_pushes_the_expiry(tmp_path: Path) -> None:
    clock = Clock()
    lease = FileSystemGpuLease(
        tmp_path, clock=clock, heartbeat_seconds=3600
    )
    first = lease.acquire(1)
    clock.at += timedelta(seconds=5)
    lease.heartbeat()
    current = lease.current()
    assert current is not None
    assert current.expires_at == first.expires_at + timedelta(seconds=5)
    lease.release()


def test_the_heartbeat_thread_renews_by_itself(tmp_path: Path) -> None:
    lease = FileSystemGpuLease(tmp_path, heartbeat_seconds=0.02)
    first = lease.acquire(1)
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        current = lease.current()
        if current is not None and current.expires_at > first.expires_at:
            break
        time.sleep(0.02)
    else:
        pytest.fail("the lease was never renewed")
    lease.release()
    assert not lease.path.exists()
    time.sleep(0.1)
    assert not lease.path.exists()  # no write after release


def test_a_lease_expires_when_not_renewed(tmp_path: Path) -> None:
    clock = Clock()
    lease = FileSystemGpuLease(tmp_path, clock=clock, heartbeat_seconds=3600)
    lease.acquire(1)
    clock.at += timedelta(seconds=16)
    assert lease.current() is None
    lease.release()


def test_a_dead_pid_makes_the_lease_invalid(tmp_path: Path) -> None:
    other = FileSystemGpuLease(
        tmp_path, holder="x", alive=lambda pid: False
    )
    lease = FileSystemGpuLease(tmp_path)
    lease.acquire(1)
    assert other.current() is None
    lease.release()


def test_acquire_refuses_while_another_process_holds_one(
    tmp_path: Path,
) -> None:
    mine = FileSystemGpuLease(tmp_path, pid=os.getpid() + 100000,
                              alive=lambda pid: True)
    mine.acquire(1)
    with pytest.raises(StateError, match="held by"):
        FileSystemGpuLease(tmp_path, alive=lambda pid: True).acquire(1)
    mine.release()


def test_garbage_files_are_ignored(tmp_path: Path) -> None:
    lease = FileSystemGpuLease(tmp_path)
    (tmp_path / "lease.json").write_text("{not json")
    (tmp_path / ACK_FILE).write_text("[]")
    assert lease.current() is None and lease.ack() is None
    assert not lease.freed()


def test_ack_is_read_and_stale_ones_dropped_at_acquire(
    tmp_path: Path,
) -> None:
    clock = Clock()
    lease = FileSystemGpuLease(tmp_path, clock=clock, heartbeat_seconds=3600)
    old = AckFile(holder="img-worker", freed=True, at=clock.at)
    (tmp_path / ACK_FILE).write_text(old.model_dump_json(by_alias=True))
    clock.at += timedelta(seconds=1)
    lease.acquire(1)
    assert lease.ack() is None and not lease.freed()
    fresh = AckFile(
        holder="img-worker", freed=True, at=clock.at + timedelta(seconds=1)
    )
    (tmp_path / ACK_FILE).write_text(fresh.model_dump_json(by_alias=True))
    assert lease.ack() == fresh and lease.freed()
    lease.release()


def test_lease_directory_uses_the_runtime_dir(tmp_path: Path) -> None:
    assert lease_directory({"XDG_RUNTIME_DIR": str(tmp_path)}) == (
        tmp_path / "gpu-lease"
    )
    assert lease_directory({}).name.startswith("gpu-lease-")


def test_pid_alive_reads_proc() -> None:
    assert pid_alive(os.getpid()) and not pid_alive(0)


HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from transcriber.adapters.outbound.fs_gpu_lease import FileSystemGpuLease
    lease = FileSystemGpuLease(Path(sys.argv[1]), heartbeat_seconds=0.2)
    lease.acquire(4000)
    print("held", flush=True)
    time.sleep(60)
    """
)


def test_a_killed_holder_lets_the_lease_expire(tmp_path: Path) -> None:
    """A holder killed without release: dead pid, then expiry (SIGKILL)."""
    child = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(tmp_path)],
        stdout=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        reader = FileSystemGpuLease(tmp_path)
        held = reader.current()
        assert held is not None and held.pid == child.pid
        child.kill()  # the test's own child: no release runs
        child.wait(WAIT)
        assert (tmp_path / "lease.json").exists()  # nobody deleted it
        assert reader.current() is None  # pid gone: not valid
        # and by time alone, whatever the pid check says
        late = FileSystemGpuLease(
            tmp_path, alive=lambda pid: True,
            clock=_At(held.expires_at + timedelta(seconds=1)),
        )
        assert late.current() is None
    finally:
        child.kill()
        child.wait(WAIT)
        if child.stdout is not None:
            child.stdout.close()


class _At:
    """A clock fixed at one time."""

    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


def test_job_runner_defers_until_the_live_lease_is_gone(
    tmp_path: Path,
) -> None:
    live = FileSystemGpuLease(tmp_path, heartbeat_seconds=0.05)
    live.acquire(1)
    reader = FileSystemGpuLease(tmp_path, holder="b", priority="batch")
    runner = ThreadJobRunner(lease=reader, lease_poll=0.02)
    ran: List[str] = []

    def job(progress: Progress) -> None:
        ran.append("job")

    runner.submit("s", job)
    time.sleep(0.3)
    status = runner.status("s")
    assert ran == [] and status is not None
    assert status.state is SessionState.QUEUED
    assert status.message is not None and "live" in status.message
    live.release()
    assert runner.wait(WAIT)
    assert ran == ["job"]
    runner.close(WAIT)


def test_job_runner_ignores_a_batch_lease(tmp_path: Path) -> None:
    batch = FileSystemGpuLease(tmp_path, holder="b", priority="batch",
                               heartbeat_seconds=60)
    batch.acquire(1)
    runner = ThreadJobRunner(lease=batch, lease_poll=0.02)
    done = threading.Event()
    runner.submit("s", lambda progress: done.set())
    assert done.wait(WAIT)
    runner.close(WAIT)
    batch.release()


def test_close_stops_the_waiting(tmp_path: Path) -> None:
    live = FileSystemGpuLease(tmp_path, heartbeat_seconds=60)
    live.acquire(1)
    runner = ThreadJobRunner(
        lease=FileSystemGpuLease(tmp_path, priority="batch"),
        lease_poll=60,
    )
    runner.submit("s", lambda progress: None)
    time.sleep(0.1)
    runner.close(WAIT)
    assert runner.wait(WAIT)
    live.release()


def _cli(tmp_path: Path, *argv: str) -> "tuple[int, str]":
    out = io.StringIO()
    code = main(
        ["gpu-lease", *argv],
        environ={"XDG_RUNTIME_DIR": str(tmp_path),
                 "TRANSCRIBER_RUNTIME": "fake",
                 "TRANSCRIBER_DATA_DIR": str(tmp_path / "data")},
        out=out, err=io.StringIO(),
    )
    return code, out.getvalue()


def test_cli_status_release_and_acquire(tmp_path: Path) -> None:
    assert _cli(tmp_path, "status") == (0, "GPU free: no valid lease.\n")
    assert _cli(tmp_path, "status", "--json") == (0, "null\n")
    assert _cli(tmp_path, "release")[1] == "No lease file.\n"
    code, text = _cli(
        tmp_path, "acquire", "--need-mib", "3000", "--duration", "0.6",
    )
    assert code == 0 and "Lease acquired" in text
    assert text.rstrip().endswith("Lease released.")
    assert not (tmp_path / "gpu-lease" / "lease.json").exists()
    held = new_lease("ragputin-live", "live", os.getpid(),
                     datetime.now(timezone.utc), 10)
    (tmp_path / "gpu-lease").mkdir(exist_ok=True)
    (tmp_path / "gpu-lease" / "lease.json").write_text(
        held.model_dump_json(by_alias=True)
    )
    code, text = _cli(tmp_path, "status")
    assert code == 0 and "held by ragputin-live" in text
    code, text = _cli(tmp_path, "status", "--json")
    assert json.loads(text)["schema"] == "gpu-lease.v1"
    assert _cli(tmp_path, "release")[1] == "Lease deleted.\n"


def test_cli_acquire_waits_for_the_ack_and_refuses_a_held_card(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "gpu-lease"

    def answer() -> None:
        for _ in range(200):
            if (directory / "lease.json").exists():
                break
            time.sleep(0.01)
        time.sleep(0.1)
        ack = AckFile(holder="img-worker", freed=True,
                      at=datetime.now(timezone.utc) + timedelta(seconds=1))
        (directory / ACK_FILE).write_text(ack.model_dump_json(by_alias=True))

    thread = threading.Thread(target=answer)
    thread.start()
    code, text = _cli(
        tmp_path, "acquire", "--need-mib", "1", "--wait-ack",
        "--duration", "0.5",
    )
    thread.join()
    assert code == 0 and "Freed by the other side." in text


def test_cli_acquire_fails_when_another_process_holds_the_lease(
    tmp_path: Path,
) -> None:
    foreign = FileSystemGpuLease(
        tmp_path / "gpu-lease", pid=os.getppid(), heartbeat_seconds=60
    )
    foreign.acquire(1)
    try:
        code, _ = _cli(tmp_path, "acquire", "--need-mib", "1")
        assert code == 1
    finally:
        foreign.release()
