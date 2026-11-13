"""The transcriber command line, run as ``python -m transcriber``."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from transcriber import __version__
from transcriber.adapters.inbound.cli import TranscriberCLI
from transcriber.bootstrap import build_container
from transcriber.config import TranscriberSettings

ROOT = Path(__file__).resolve().parents[3]


def run(args: List[str]) -> "subprocess.CompletedProcess[str]":
    """Run ``python -m transcriber *args`` from the repository root."""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PAGER="cat")
    return subprocess.run(
        [sys.executable, "-m", "transcriber", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=60, env=env,
    )


def test_help_lists_the_commands() -> None:
    result = run(["--help"])
    assert result.returncode == 0, result.stderr
    shown = result.stdout + result.stderr  # Fire prints help to stderr
    assert "version" in shown and "settings" in shown


def test_version() -> None:
    result = run(["version"])
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__


def test_settings_come_from_the_container() -> None:
    settings = TranscriberSettings(data_dir=Path("elsewhere"))
    cli = TranscriberCLI(build_container(settings))
    assert cli.settings()["data_dir"] == "elsewhere"
    assert TranscriberCLI().settings() == {
        "data_dir": "data/transcriber",
        "recogniser": "openai/whisper-large-v3-turbo",
        "voice_model": "microsoft/wavlm-base-plus-sv",
    }
