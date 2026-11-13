"""Trial store adapter: one JSON object per line, append-only."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.domain.errors import CorruptedDataError
from src.domain.experiments import TrialResult


class JsonlTrialStore:
    """Keep every trial result in a JSON Lines file.

    The file is only appended to, so an interrupted suite loses nothing.
    There is one row per (suite, trial): when it was written several
    times, the latest wins.
    """

    def __init__(self, path: str):
        """Store results in ``path``."""
        self.path = Path(path)
        self._results: Optional[Dict[Tuple[str, str], TrialResult]] = None

    def _load(self) -> Dict[Tuple[str, str], TrialResult]:
        """Read the file once and index the results by trial id."""
        if self._results is None:
            results: Dict[Tuple[str, str], TrialResult] = {}
            if self.path.is_file():
                try:
                    text = self.path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise CorruptedDataError(
                        f"Cannot read {self.path}: {exc}"
                    )
                for number, line in enumerate(text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        result = TrialResult.model_validate_json(line)
                    except (ValidationError, ValueError):
                        raise CorruptedDataError(
                            f"{self.path}, line {number}: not a trial result."
                        )
                    key = (result.suite, result.trial_id)
                    results.pop(key, None)
                    results[key] = result
            self._results = results
        return self._results

    def append(self, result: TrialResult) -> None:
        """Append one result to the file."""
        results = self._load()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(result.model_dump_json() + "\n")
        except OSError as exc:
            raise CorruptedDataError(f"Cannot write {self.path}: {exc}")
        key = (result.suite, result.trial_id)
        results.pop(key, None)
        results[key] = result

    def get(self, trial_id: str) -> Optional[TrialResult]:
        """Return the latest result of ``trial_id`` in any suite."""
        for result in reversed(list(self._load().values())):
            if result.trial_id == trial_id:
                return result
        return None

    def all(self) -> List[TrialResult]:
        """Return the latest result of every (suite, trial)."""
        return list(self._load().values())
