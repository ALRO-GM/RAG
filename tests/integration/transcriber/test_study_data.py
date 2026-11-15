"""The study page's data, built from the committed results and documents."""

import json
from datetime import date
from pathlib import Path

import pytest

from transcriber.adapters.inbound.study_data import (
    OUT, ROOT, StudyDataError, build_study, load_inputs, main,
)


def test_the_committed_data_file_is_up_to_date() -> None:
    """Rebuilding from the repository gives the committed file.

    Only the date may differ: run ``python -m
    transcriber.adapters.inbound.study_data`` after changing a result.
    """
    committed = json.loads(OUT.read_text(encoding="utf-8"))
    built = build_study(
        load_inputs(ROOT), date.fromisoformat(committed["updated"])
    )
    assert json.loads(json.dumps(built)) == committed


def test_the_figures_match_lab_section_13_8() -> None:
    """The headline figures the page shows, as LAB.md §13.8 states them.

    The document rounds the DER to three decimals; the summaries keep
    four.
    """
    data = build_study(load_inputs(ROOT), date(2026, 11, 15))
    corpora = {c["id"]: c for c in data["separation"]["corpora"]}
    ami, callhome = corpora["ami"], corpora["callhome"]
    assert ami["headline"]["best"] == "none+diarizen-wavlm-large-s80-md-v2"
    assert abs(1 - ami["headline"]["success"] - 0.089) <= 0.0006
    assert ami["rows"][0]["non_commercial"] is True
    assert ami["headline"]["best_commercial"]["id"] == "none+pyannote-3.1"
    assert callhome["headline"]["best"] == "none+nemotron-3-diarization"
    assert abs(1 - callhome["headline"]["success"] - 0.106) <= 0.0006
    assert callhome["non_commercial"] is True
    assert (callhome["excluded_no_speech"], callhome["excluded_invalid"]) \
        == (3, 1)
    assert [c["combinations"] for c in (ami, callhome)] == [24, 24]
    builtin = {c: corpora[c]["headline"]["builtin"] for c in corpora}
    assert abs(1 - builtin["ami"]["success"] - 0.498) <= 0.0006
    assert abs(1 - builtin["callhome"]["success"] - 0.487) <= 0.0006
    assert data["separation"]["rule"]["status"] in ("pending", "done")
    assert data["labelling"]["events"]["given_links"]["expected"] == 70


def test_the_command_writes_the_file(tmp_path: Path) -> None:
    out = tmp_path / "study" / "data.json"
    assert main(["--out", str(out), "--date", "2026-11-15"]) == 0
    assert json.loads(out.read_text())["updated"] == "2026-11-15"


def test_the_command_fails_cleanly_without_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--root", str(tmp_path), "--out",
                 str(tmp_path / "x.json")]) == 1
    assert "study_data:" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_optional_results_appear_when_their_files_exist(
    tmp_path: Path,
) -> None:
    """#68 and #82 switch from pending to measured with their files."""
    for rel in ("experiments", "lab", "docs", "examples"):
        (tmp_path / rel).symlink_to(ROOT / rel)
    inputs = load_inputs(tmp_path)
    switch = {"corpora": {"ami": {"der": 0.09, "n": 4}}}
    tagger = [{"approach": "rules", "items": 1534, "top1_accuracy": 0.4}]
    data = build_study(
        type(inputs)(**{**inputs.__dict__, "switch": switch,
                        "tagger": tagger}),
        date(2026, 11, 15),
    )
    assert data["separation"]["rule"]["status"] == "done"
    assert data["labelling"]["tagger"]["status"] == "done"


def test_a_figure_that_left_its_document_stops_the_build() -> None:
    inputs = load_inputs(ROOT)
    documents = dict(inputs.documents)
    path = "docs/skills/DESIGN.md"
    documents[path] = documents[path].replace("0.365 (3)", "0.366 (3)")
    with pytest.raises(StudyDataError, match="qwen3-1.7b"):
        build_study(
            type(inputs)(**{**inputs.__dict__, "documents": documents}),
            date(2026, 11, 15),
        )
