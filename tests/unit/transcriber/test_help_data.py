"""The help page's rubric module: built from the rubric, never stale."""

from pathlib import Path

import pytest

from transcriber.adapters.inbound.help_data import (
    OUT, build_rubric, main, render,
)


def test_every_competency_comes_with_its_help_texts() -> None:
    """Both languages list the same competencies with practice texts."""
    rubric = build_rubric()
    assert [c["id"] for c in rubric["es"]] == [c["id"] for c in rubric["en"]]
    for competencies in rubric.values():
        for c in competencies:
            assert c["positive"] and c["negative"] and c["limits"]
            assert c["helps"] and c["hurts"]


def test_the_committed_module_is_up_to_date() -> None:
    """Changing a rubric without rebuilding the module fails here."""
    assert OUT.read_text(encoding="utf-8") == render(build_rubric())
    assert main(["--check"]) == 0


def test_write_and_check_on_a_stale_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing file is stale; writing makes the check pass."""
    out = tmp_path / "help-rubric.js"
    assert main(["--check", "--out", str(out)]) == 1
    assert "out of date" in capsys.readouterr().out
    assert main(["--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("// The rubric")
    assert main(["--check", "--out", str(out)]) == 0
