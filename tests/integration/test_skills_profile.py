"""Profiles across sessions on disk: store, command and committed example.

The example under ``examples/skills/profile/`` is synthetic (invented
people, hand-written levels); these tests check that it is up to date
with its sources and that ``skills profile`` reads it end to end.
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType
from typing import List

import pytest

from src.adapters.outbound.fs_assessment_store import FileSystemAssessmentStore
from src.adapters.outbound.json_catalogue_repository import (
    JsonCatalogueRepository,
)
from src.domain.errors import CorruptedDataError, NotFoundError
from src.domain.skills.transcript import from_contract
from src.domain.skills.transcript_contract import read_contract
from tests.integration.test_cli import run_cli

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "skills" / "profile"
SCRIPT = ROOT / "experiments" / "skills" / "build_profile_example.py"
RUBRICS = ROOT / "knowledge" / "rubrics"


def load_builder() -> ModuleType:
    """Import the example's build script, which is not in a package."""
    spec = importlib.util.spec_from_file_location(
        "build_profile_example", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example(tmp_path: Path) -> Path:
    """Copy the example's transcripts and assessments into ``tmp_path``."""
    for folder in ("transcripts", "output"):
        shutil.copytree(EXAMPLE / folder, tmp_path / folder)
    return tmp_path


def profile(
    args: List[str], cwd: Path
) -> "subprocess.CompletedProcess[str]":
    """Run ``skills profile`` over the copied example."""
    return run_cli(
        ["skills", "--rubric_dir", str(RUBRICS), "--output_dir", "pages",
         "--transcripts_dir", "transcripts", "profile",
         "--assessments_dir", "output", *args],
        cwd,
    )


# -------------------------------------------------------------- store


def test_store_lists_and_finds_assessments(example: Path) -> None:
    """Every ``*.assessment.json``; by session id or by path."""
    store = FileSystemAssessmentStore(str(example / "output"))
    assert [a.session_id for a in store.all()] == [
        "perfil_s1", "perfil_s2", "perfil_s3", "perfil_s4",
    ]
    assert store.get("perfil_s2").conversation_type == (
        "resolucion_problemas"
    )
    path = example / "output" / "perfil_s3.assessment.json"
    assert store.get(str(path)).session_id == "perfil_s3"
    with pytest.raises(NotFoundError, match="No assessment 'nope'"):
        store.get("nope")
    assert FileSystemAssessmentStore(str(example / "none")).all() == []


def test_store_rejects_a_broken_file(tmp_path: Path) -> None:
    """A file that is not an assessment is a clean error."""
    (tmp_path / "bad.assessment.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CorruptedDataError):
        FileSystemAssessmentStore(str(tmp_path)).all()


# ------------------------------------------------------------ example


def test_committed_example_matches_its_sources() -> None:
    """Rebuilding from the hand-written files changes nothing."""
    assert load_builder().check() == 0


def test_example_is_marked_synthetic_and_joins_by_person_id() -> None:
    """The people are enrolled ids; the judge says it is hand-written."""
    store = FileSystemAssessmentStore(str(EXAMPLE / "output"))
    for item in store.all():
        assert "ejemplo inventado" in item.judge_model
        assert all(card.person_id for card in item.scorecards)
        assert item.started_at and item.conversation_type
    note = json.loads((EXAMPLE / "example.json").read_text("utf-8"))["note"]
    assert note.startswith("Synthetic example")


def test_builder_refuses_a_level_the_rules_would_not_keep() -> None:
    """Two quotes are below the minimum: a hand-written 4 is an error."""
    builder = load_builder()
    spec = json.loads((EXAMPLE / "example.json").read_text("utf-8"))
    first = spec["sessions"][0]
    first["ratings"] = {"marta": {"LISTEN": [4, [[2, "LISTEN.P1"],
                                                 [9, "LISTEN.P4"]]]}}
    text = (EXAMPLE / "sessions" / "perfil_s1.txt").read_text("utf-8")
    document = builder.contract(first, text, spec["people"], "es")
    rubric = JsonCatalogueRepository(str(RUBRICS)).load("core", "es")
    transcript = from_contract(read_contract(document))
    with pytest.raises(ValueError, match="the rules turn 4 into None"):
        builder.assessment(first, transcript, rubric, "hand")
    first["ratings"] = {"marta": {"LISTEN": [4, [[0, "LISTEN.P1"]]]}}
    with pytest.raises(ValueError, match="is Pablo's, not Marta's"):
        builder.assessment(first, transcript, rubric, "hand")


# ------------------------------------------------------------ command


def test_profile_command_writes_the_page(example: Path) -> None:
    """Marta: two overall levels, one under construction, quotes timed."""
    result = profile(["--participant", "marta", "--synthetic"], example)
    assert result.returncode == 0, result.stderr
    assert "LISTEN   [4 4 - 4] 3/4 sessions; overall 4" in result.stdout
    assert "CONFL    [2 - - 2] 2/4 sessions; building, 1 more" in (
        result.stdout
    )
    page = (example / "pages" / "marta.profile.html").read_text("utf-8")
    assert page.startswith('<!doctype html>\n<html lang="es">')
    assert "Perfil de Marta" in page
    assert "Datos de ejemplo inventados" in page
    assert "Conversación 2 · Marta, minuto 00:36" in page
    assert "Escucha activa" in page and "En construcción" in page
    assert "http" not in page


def test_profile_command_json_and_a_chosen_list(example: Path) -> None:
    """``--format json`` is the profile; a session without Pablo is out."""
    result = profile(
        ["--participant", "pablo", "--sessions",
         "perfil_s1,perfil_s4,output/perfil_s3.assessment.json",
         "--format", "json", "--output", "pablo.json"],
        example,
    )
    assert result.returncode == 0, result.stderr
    assert "Left out perfil_s4: absent" in result.stdout
    data = json.loads((example / "pablo.json").read_text("utf-8"))
    assert [s["session_id"] for s in data["sessions"]] == [
        "perfil_s1", "perfil_s3",
    ]
    incl = next(c for c in data["competencies"]
                if c["competency_id"] == "INCL")
    assert (incl["rated"], incl["aggregate"], incl["missing"]) == (2, None, 1)


def test_profile_command_in_english(example: Path) -> None:
    """The page in English, written where asked."""
    result = profile(
        ["--participant", "luis", "--language", "en", "--output",
         "luis.html"],
        example,
    )
    assert result.returncode == 0, result.stderr
    page = (example / "luis.html").read_text("utf-8")
    assert "Profile of Luis" in page and "Initiative" in page


@pytest.mark.parametrize("args, message", [
    (["--participant", "marta", "--format", "pdf"], "Unknown format 'pdf'"),
    (["--participant", "nobody"], "No session has a scorecard"),
    (["--participant", "marta", "--sessions", "nope"], "No assessment"),
])
def test_profile_command_errors(
    example: Path, args: List[str], message: str
) -> None:
    """Bad format, unknown person and unknown session are clean errors."""
    result = profile(args, example)
    assert result.returncode == 1, result.stdout
    assert result.stderr.startswith("Error: ") and message in result.stderr
    assert "Traceback" not in result.stderr
