"""Writes are atomic: a killed writer leaves the old file or the new one."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from transcriber.adapters.outbound.fs_storage import (
    atomic_write,
    read_json,
    write_json,
)
from transcriber.domain.errors import StorageError

ROOT = Path(__file__).resolve().parents[3]

# Writes version 1, then stops in the middle of writing version 2 (its
# bytes are on disk, the rename has not happened) and waits there to be
# killed.
STOPPED_WRITER = """
import sys, time
from pathlib import Path
from transcriber.adapters.outbound import fs_storage
target = Path(sys.argv[1])
fs_storage.write_json(target, {"version": 1})
def stop(descriptor):
    print("mid-write", flush=True)
    time.sleep(60)
fs_storage.os.fsync = stop
fs_storage.write_json(target, {"version": 2, "payload": "x" * 100000})
"""


def start(script: str, target: Path) -> "subprocess.Popen[str]":
    """Run ``script`` on ``target`` in a fresh interpreter."""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(target)],
        cwd=ROOT, stdout=subprocess.PIPE, text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
    )


def test_killed_mid_write_the_old_version_stays(tmp_path: Path) -> None:
    target = tmp_path / "session.json"
    process = start(STOPPED_WRITER, target)
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "mid-write"
    process.send_signal(signal.SIGKILL)
    process.wait(timeout=10)
    assert read_json(target) == {"version": 1}
    # Only the hidden temporary file of the interrupted write remains.
    others = [p.name for p in tmp_path.iterdir() if p != target]
    assert len(others) == 1 and others[0].startswith(".session.json.")


def test_a_failed_write_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.json"
    write_json(target, {"version": 1})

    def crash(*args: object) -> None:
        raise KeyboardInterrupt  # stands for the process being stopped

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        write_json(target, {"version": 2})
    assert read_json(target) == {"version": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_bytes_and_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "er" / "audio.flac"
    atomic_write(target, b"abc")
    atomic_write(target, b"defg")
    assert target.read_bytes() == b"defg"


def test_reading_damage(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    target.write_bytes(b"\xff\xfe")
    with pytest.raises(StorageError, match="Cannot read x.json"):
        read_json(target)
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")
