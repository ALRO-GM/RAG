"""The comparison of two analyses (#96), on small hand-made runs."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "skills" / "compare_analyses.py"


def load_comparer() -> ModuleType:
    """Import the comparison script, which is not in a package."""
    spec = importlib.util.spec_from_file_location("compare_analyses", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparer = load_comparer()


def run(ratings: Dict[str, Any], estimates: Dict[str, Any],
        quotes: List[str], confirmed_text: str,
        events: List[str]) -> Dict[str, Any]:
    """Return one run of a single person, Ana, as :func:`read_run` would."""
    return {
        "summary": {"people": [{
            "name": "Ana", "words": 100, "rated": 1, "provisional": 1,
            "observations": 0, "ratings": ratings, "estimates": estimates,
        }]},
        "assessment": {"scorecards": [{
            "participant": "Ana",
            "assessments": [{"competency_id": "COMM", "evidence": [
                {"indicator_id": "COMM.P1", "quote": q} for q in quotes]}],
        }]},
        "confirmation": {"sentences": [
            {"sentence": {"speaker": "Ana", "text": confirmed_text},
             "tags": [],
             "decision": {"status": "confirmed",
                          "indicator_id": "COMM.P1"}},
            {"sentence": {"speaker": "Ana", "text": "Otra."}, "tags": [],
             "decision": {"status": "rejected", "indicator_id": "X"}},
            {"sentence": {"speaker": "Ana", "text": "Sin."}, "tags": []},
        ]},
        "events": {"events": [{"type": t} for t in events]},
    }


def test_normalise_drops_case_and_punctuation() -> None:
    """Two quotes that differ only in punctuation and case are equal."""
    assert comparer.normalise("¿Vale, Luis?") == comparer.normalise(
        "vale luis")


def test_levels_tell_firm_provisional_and_none() -> None:
    """A firm level, an estimate and a competency with neither."""
    person = {"ratings": {"A": 4, "B": None, "C": None},
              "estimates": {"B": {"rating": 3}}}
    assert comparer.levels(person) == {
        "A": ("firm", 4), "B": ("provisional", 3), "C": ("none", None)}


def test_compare_reports_changes_shared_evidence_and_events() -> None:
    """Every figure of one person whose level and evidence changed."""
    first = run({"COMM": 4, "LISTEN": None}, {"LISTEN": {"rating": 3}},
                ["Lo cerramos el jueves.", "Te escucho."],
                "Lo cerramos el jueves.", ["decision", "action_item"])
    second = run({"COMM": 3, "LISTEN": None}, {"LISTEN": {"rating": 3}},
                 ["lo cerramos el jueves", "Otra cita."],
                 "Lo cerramos el viernes.", ["decision", "decision"])
    result = comparer.compare(first, second)
    ana = result["people"][0]
    assert ana["changes"] == [{"competency": "COMM",
                               "first": ["firm", 4],
                               "second": ["firm", 3]}]
    assert ana["evidence"] == [2, 2, 1]
    assert ana["evidence_indicators"] == [1, 1, 1]
    assert ana["confirmed"] == [1, 1, 0]
    assert ana["counts"]["words"] == [100, 100]
    assert result["events"] == {"action_item": [1, 0], "decision": [1, 2]}


def test_main_reads_two_directories(tmp_path: Path,
                                    capsys: pytest.CaptureFixture[str]
                                    ) -> None:
    """The command line reads both runs; missing files count as empty."""
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        data = run({"COMM": 4}, {}, ["Hola."], "Hola.", ["decision"])
        for key in ("summary", "assessment", "confirmation"):
            (folder / f"{key}.json").write_text(json.dumps(data[key]),
                                                encoding="utf-8")
    out = tmp_path / "out.json"
    assert comparer.main([str(tmp_path / "a"), str(tmp_path / "b"),
                          "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Ana: words 100→100" in printed
    assert "evidence 1→1 (1 the same)" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["events"] == {}
