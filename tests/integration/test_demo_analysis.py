"""The live demo's prepared analysis result (#232).

It is installed in the analysis store when the RAG starts and read back
through the usual result use case; it is marked demo everywhere and
never words a hiring decision.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.adapters.outbound.fs_analysis_store import FileSystemAnalysisStore
from src.adapters.outbound.fs_demo_analyses import read_demo_analyses
from src.application.skills.demo_analysis import install_demo, install_demos
from src.application.skills.session_analysis import AnalysisResults
from src.domain.skills.demo_analysis import DEMO_NOTE, DemoAnalysis
from src.domain.skills.synthesis import _VERDICT

DEMOS = Path(__file__).resolve().parents[2] / "examples" / "demo"


def bundle() -> DemoAnalysis:
    found = read_demo_analyses(DEMOS)
    assert [b.session_id for b in found] == ["demo-entrevista-lucia-marin"]
    return found[0]


def test_the_demo_result_says_what_the_owner_asked() -> None:
    person = bundle().people()[0]
    assert person.participant == "Lucía Marín"
    assert person.overall is not None and person.overall.percent == 88.7
    assert person.top_strengths == [
        "resolucion_datos", "comunicacion", "responsabilidad"]
    assert person.top_improvements == ["pruebas_automaticas", "delegar"]
    raw: Dict[str, Any] = bundle().synthesis["people"][0]
    testing = next(s for s in raw["skills"]
                   if s["competency_id"] == "pruebas_automaticas")
    assert testing["score"] < testing["required"]


def test_no_text_words_a_hiring_decision() -> None:
    person = bundle().people()[0]
    assert person.overall is not None
    texts = [person.overall.line] + [s.conclusion for s in person.skills]
    texts += [p.text for p in person.strengths + person.fit]
    texts += person.questions
    for text in texts:
        assert not _VERDICT.search(text), text
        assert "siguiente fase" not in text.lower()


def test_installing_is_done_once_and_read_back(tmp_path: Path) -> None:
    store = FileSystemAnalysisStore(str(tmp_path))
    demo = bundle()
    assert install_demos(store, [demo]) == [demo.session_id]
    assert not install_demo(store, demo)
    run = store.load(demo.session_id)
    assert run is not None and run.synthetic and run.status == "done"
    assert run.step("synthesis").status == "done"
    assert {s.note for s in run.steps} == {DEMO_NOTE}
    results = AnalysisResults(store)
    stored = results.result(demo.session_id, "synthesis")
    assert stored["demo"] is True
    assert stored["people"][0]["overall"]["percent"] == 88.7
    newer = demo.model_copy(update={
        "started_at": demo.started_at.replace(year=2027)})
    assert install_demo(store, newer)


def test_bad_files_are_skipped(tmp_path: Path) -> None:
    good: Dict[str, Any] = json.loads(
        (DEMOS / "entrevista-lucia.analysis.json").read_text("utf-8"))
    (tmp_path / "a.analysis.json").write_text("{", "utf-8")
    unmarked = {**good, "synthesis": {**good["synthesis"], "demo": False}}
    (tmp_path / "b.analysis.json").write_text(json.dumps(unmarked), "utf-8")
    assert read_demo_analyses(tmp_path) == []
    assert read_demo_analyses(tmp_path / "none") == []
    for broken in (
        {**good["synthesis"], "session_id": "demo-other"},
        {**good["synthesis"], "people": []},
    ):
        with pytest.raises(ValueError):
            DemoAnalysis.model_validate({**good, "synthesis": broken})


def test_the_container_installs_the_demos(tmp_path: Path) -> None:
    from src.config import SkillsSettings
    from src.skills_bootstrap import SkillsContainer

    skills = SkillsContainer(SkillsSettings(analysis_dir=str(tmp_path)))
    assert skills.install_demo_analyses() == ["demo-entrevista-lucia-marin"]
    assert skills.install_demo_analyses() == []
