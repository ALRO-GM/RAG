"""The transcriber command line, run as ``python -m transcriber``."""

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from transcriber import __version__
from transcriber.adapters.inbound.cli import main

ROOT = Path(__file__).resolve().parents[3]

COMMANDS = (
    "version", "settings", "serve", "enrol", "import", "sessions", "show",
    "export", "publish", "speakers", "enrolment-text",
)


def run(args: List[str]) -> "subprocess.CompletedProcess[str]":
    """Run ``python -m transcriber *args`` from the repository root."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PAGER="cat")
    return subprocess.run(
        [sys.executable, "-m", "transcriber", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=60, env=env,
    )


def test_help_lists_the_commands_and_the_models_warning() -> None:
    result = run(["--help"])
    assert result.returncode == 0, result.stderr
    for command in COMMANDS:
        assert command in result.stdout
    shown = " ".join(result.stdout.split())
    assert "do not use them while serve is working" in shown


def test_commands_that_run_models_warn_in_their_help() -> None:
    for command in ("enrol", "import", "serve"):
        shown = " ".join(run([command, "--help"]).stdout.split())
        assert "while serve is working on a job" in shown, command


def test_version() -> None:
    result = run(["version"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__


def test_settings_in_process() -> None:
    out = io.StringIO()
    assert main(["settings"], environ={"TRANSCRIBER_DATA_DIR": "elsewhere"},
                out=out, err=io.StringIO()) == 0
    assert "data_dir: elsewhere" in out.getvalue()
    assert "runtime: real" in out.getvalue()
    assert "public_host: rag.peer4u.app" in out.getvalue()


def test_invalid_settings_are_a_usage_error() -> None:
    err = io.StringIO()
    code = main(["version"], environ={"TRANSCRIBER_RUNTIME": "quantum"},
                out=io.StringIO(), err=err)
    assert code == 2 and "invalid settings" in err.getvalue()
