"""JSON Lines trial store on a real file."""

from pathlib import Path

import pytest

from src.adapters.outbound.jsonl_trial_store import JsonlTrialStore
from src.domain.errors import CorruptedDataError
from src.domain.experiments import TrialResult
from tests.fakes import make_trial


def result(trial_id: str, name: str = "") -> TrialResult:
    """Return a small trial result."""
    return make_trial(trial_id, {"docs": [[1], [0]]}, params={"x": 1},
                      name=name)


def store(tmp_path: Path) -> JsonlTrialStore:
    """Return a store writing under a directory that does not exist."""
    return JsonlTrialStore(str(tmp_path / "results" / "trials.jsonl"))


def test_empty_store_reads_nothing_and_creates_no_file(
    tmp_path: Path
) -> None:
    """Without a file there is no result, and reading writes nothing."""
    trials = store(tmp_path)
    assert trials.all() == []
    assert trials.get("missing") is None
    assert not trials.path.exists()


def test_append_writes_one_json_line_per_result(tmp_path: Path) -> None:
    """Results are appended to the file, creating its directory."""
    trials = store(tmp_path)
    first, second = result("a"), result("b")
    trials.append(first)
    trials.append(second)
    lines = trials.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert TrialResult.model_validate_json(lines[0]) == first
    assert trials.get("a") == first
    assert trials.get("b") == second
    assert trials.get("c") is None
    assert trials.all() == [first, second]


def test_latest_result_of_a_trial_wins(tmp_path: Path) -> None:
    """A trial run again keeps both lines but only the last result."""
    trials = store(tmp_path)
    trials.append(result("a", "first run"))
    trials.append(result("b"))
    trials.append(result("a", "second run"))
    assert len(trials.path.read_text().splitlines()) == 3
    found = trials.get("a")
    assert found is not None and found.name == "second run"
    assert [r.trial_id for r in trials.all()] == ["b", "a"]


def test_results_survive_a_reload_from_disk(tmp_path: Path) -> None:
    """A new store over the same file sees the same latest results."""
    trials = store(tmp_path)
    trials.append(result("a", "first run"))
    trials.append(result("b"))
    trials.append(result("a", "second run"))
    reloaded = store(tmp_path)
    assert reloaded.all() == trials.all()
    found = reloaded.get("a")
    assert found is not None and found.name == "second run"
    reloaded.append(result("c"))
    assert [r.trial_id for r in store(tmp_path).all()] == ["b", "a", "c"]


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    """Empty lines between results are not an error."""
    trials = store(tmp_path)
    trials.append(result("a"))
    with open(trials.path, "a", encoding="utf-8") as handle:
        handle.write("\n   \n")
    store(tmp_path).append(result("b"))
    assert [r.trial_id for r in store(tmp_path).all()] == ["a", "b"]


@pytest.mark.parametrize("line", [
    "{not json", '{"trial_id": "a"}', "[1, 2]", '"text"',
])
def test_corrupted_line_is_reported_with_its_number(
    tmp_path: Path, line: str
) -> None:
    """A line that is not a trial result names the file and the line."""
    trials = store(tmp_path)
    trials.append(result("a"))
    with open(trials.path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    broken = store(tmp_path)
    with pytest.raises(CorruptedDataError, match="line 2: not a trial"):
        broken.all()
    with pytest.raises(CorruptedDataError, match="trials.jsonl"):
        broken.get("a")
    with pytest.raises(CorruptedDataError):
        broken.append(result("b"))
    assert len(trials.path.read_text().splitlines()) == 2


def test_unwritable_path_is_a_clean_error(tmp_path: Path) -> None:
    """A path that is a directory cannot be appended to."""
    trials = JsonlTrialStore(str(tmp_path))
    assert trials.all() == []
    with pytest.raises(CorruptedDataError, match="Cannot write"):
        trials.append(result("a"))
