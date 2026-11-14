"""The static bundle CI deploys: the pages plus ``version.json``."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from transcriber.adapters.inbound.bundle import WEB_DIR, build_bundle, main

ROOT = Path(__file__).resolve().parents[3]
WHEN = datetime(2026, 11, 14, 9, 30, tzinfo=timezone.utc)


def test_the_pages_and_the_version(tmp_path: Path) -> None:
    out = tmp_path / "site"
    files = build_bundle(out, " abc1234 ", now=lambda: WHEN)
    assert "index.html" in files and "js/main.js" in files
    assert "js/worklet/recorder-processor.js" in files
    assert "package.json" not in files
    sources = {
        p.relative_to(WEB_DIR).as_posix() for p in WEB_DIR.rglob("*")
        if p.is_file() and p.name != "package.json"
        and not p.name.startswith(".")
    }
    assert set(files) == sources | {"version.json"}
    version = json.loads((out / "version.json").read_text())
    assert version == {"commit": "abc1234",
                       "built_at": "2026-11-14T09:30:00+00:00"}
    assert (out / "index.html").read_bytes() == (
        WEB_DIR / "index.html"
    ).read_bytes()


def test_an_old_bundle_is_replaced(tmp_path: Path) -> None:
    out = tmp_path / "site"
    out.mkdir()
    (out / "data").mkdir()
    (out / "data" / "manifest.json").write_text("{}")
    build_bundle(out, "x")
    assert not (out / "data").exists()


@pytest.mark.parametrize("where", ["same", "inside", "around"])
def test_it_never_writes_over_its_source(tmp_path: Path, where: str) -> None:
    source = tmp_path / "web"
    source.mkdir()
    (source / "index.html").write_text("<p>x</p>")
    out = {"same": source, "inside": source / "dist",
           "around": tmp_path}[where]
    with pytest.raises(ValueError, match="overwrite"):
        build_bundle(out, "x", source=source)
    assert (source / "index.html").is_file()


def test_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]
                          ) -> None:
    assert main(["--out", str(tmp_path / "s"), "--commit", "1234567"]) == 0
    assert "commit 1234567" in capsys.readouterr().out
    assert main(["--out", str(tmp_path / "s"), "--commit", " "]) == 1
    assert "commit is required" in capsys.readouterr().err


def test_it_runs_without_the_project_installed(tmp_path: Path) -> None:
    """CI builds with ``python -m`` and the standard library only."""
    done = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]);"
         " import runpy; sys.argv = ['bundle', '--out', sys.argv[2],"
         " '--commit', 'abc']; runpy.run_module("
         "'transcriber.adapters.inbound.bundle', run_name='__main__')",
         str(ROOT), str(tmp_path / "site")],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "site" / "version.json").read_text())[
        "commit"] == "abc"
