"""Reading the live mode's GPU lease from the file system."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from src.adapters.outbound.gpu_lease_file import (
    FileGpuGate,
    lease_directory,
    process_alive,
    utc_now,
)

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def gate(tmp_path: Path, alive: bool = True) -> FileGpuGate:
    """Return a gate at ``NOW`` whose holder is (not) running."""
    return FileGpuGate(tmp_path, lambda: NOW, lambda pid: alive)


def write(tmp_path: Path, expires: Optional[object], pid: object = 7) -> None:
    """Write a ``lease.json`` expiring at ``expires``."""
    (tmp_path / "lease.json").write_text(json.dumps({
        "schema": "gpu-lease.v1", "holder": "ragputin-live",
        "priority": "live", "pid": pid, "expires_at": expires,
    }), encoding="utf-8")


def test_free_without_a_lease(tmp_path: Path) -> None:
    """No file, no lease."""
    assert gate(tmp_path).free()


def test_held_by_a_live_holder(tmp_path: Path) -> None:
    """An unexpired lease of a running process holds the card."""
    write(tmp_path, (NOW + timedelta(seconds=10)).isoformat())
    assert not gate(tmp_path).free()
    write(tmp_path, "2026-11-21T12:00:10.123456Z")
    assert not gate(tmp_path).free()


def test_expired_lease_or_dead_holder_frees_the_card(tmp_path: Path) -> None:
    """A lease not renewed, or whose process is gone, does not count."""
    write(tmp_path, (NOW - timedelta(seconds=1)).isoformat())
    assert gate(tmp_path).free()
    write(tmp_path, (NOW + timedelta(seconds=10)).isoformat())
    assert gate(tmp_path, alive=False).free()


@pytest.mark.parametrize("content", [
    "not json", "[]", '{"pid": 1}', '{"pid": "x", "expires_at": "2030-01-01"}',
    '{"pid": 1, "expires_at": 5}', '{"pid": 1, "expires_at": "garbage"}',
    '{"pid": 1, "expires_at": "2030-01-01T00:00:00"}',
])
def test_unreadable_lease_is_ignored(tmp_path: Path, content: str) -> None:
    """A malformed file never blocks ingestion."""
    (tmp_path / "lease.json").write_text(content, encoding="utf-8")
    assert gate(tmp_path).free()


def test_directory_and_clock_helpers(tmp_path: Path) -> None:
    """The directory follows ``XDG_RUNTIME_DIR``; the clock is aware."""
    assert lease_directory({"XDG_RUNTIME_DIR": "/run/u"}) == Path(
        "/run/u/gpu-lease")
    assert lease_directory({}).name.startswith("gpu-lease-")
    assert utc_now().tzinfo is not None
    assert process_alive(1) and not process_alive(0)
    assert FileGpuGate(tmp_path).free()
