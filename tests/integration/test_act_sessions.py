"""Reading labelled sessions for the second layer from files."""

import json
from pathlib import Path

import pytest

from src.adapters.outbound.act_sessions import read_act_sessions
from src.domain.errors import InvalidInputError, NotFoundError

SET = """\
=== dec-01 | reunion_decision | Prueba
Marta: ¿Lo habéis mirado? | pregunta #q1
Luis: Sí, ayer. | respuesta >q1
"""


def test_set_files_directories_and_json(tmp_path: Path) -> None:
    """A ``.txt`` file, a directory of them and both JSON shapes."""
    (tmp_path / "a.txt").write_text(SET, encoding="utf-8")
    (tmp_path / "b.txt").write_text(
        SET.replace("dec-01", "dec-02"), encoding="utf-8")
    assert [s.id for s in read_act_sessions(str(tmp_path / "a.txt"))] == [
        "dec-01"]
    assert [s.id for s in read_act_sessions(str(tmp_path))] == [
        "dec-01", "dec-02"]
    one = {"session": {"id": "x"}, "utterances": [
        {"speaker": "A", "text": "¿Y?", "act": "pregunta"}]}
    (tmp_path / "x.acts.json").write_text(json.dumps(one), "utf-8")
    (tmp_path / "all.json").write_text(json.dumps([one, one]), "utf-8")
    assert read_act_sessions(str(tmp_path / "x.acts.json"))[0].id == "x"
    assert len(read_act_sessions(str(tmp_path / "all.json"))) == 2


def test_unreadable_inputs(tmp_path: Path) -> None:
    """Missing, empty, wrong suffix and broken JSON are clean errors."""
    with pytest.raises(NotFoundError, match="No such file"):
        read_act_sessions(str(tmp_path / "missing.json"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NotFoundError, match="No \\*.txt"):
        read_act_sessions(str(empty))
    (tmp_path / "notes.md").write_text("x", "utf-8")
    with pytest.raises(InvalidInputError, match="expected a .txt"):
        read_act_sessions(str(tmp_path / "notes.md"))
    (tmp_path / "bad.json").write_text("{nope", "utf-8")
    with pytest.raises(InvalidInputError, match="not valid JSON"):
        read_act_sessions(str(tmp_path / "bad.json"))
