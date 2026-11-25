"""Interview plans and sheets on disk (JSON files).

``<dir>/plans/<plan id>.json`` holds the current revision of each
``interview.plan.v1``; ``<dir>/results/<plan id>/<session id>.json``
the ``interview.sheet.v1`` of every interview held from it (the
comparison view reads them): the AI interviewer's sheets and the
live-page reports (#262). Ids are checked against fixed patterns
before a path is built; writes go through a temporary file created
with mode 600, so every file is readable by its owner only.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from orchestrator.domain.interview import InterviewPlan

PLAN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _write(path: Path, document: Dict[str, Any]) -> None:
    """Write ``document`` as JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(document, out, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


class JsonPlanStore:
    """Plans and sheets as JSON files under ``root``."""

    def __init__(self, root: Path):
        """Keep the folder (created on the first write)."""
        self.root = Path(root)

    def _plan_path(self, plan_id: str) -> Path:
        if not PLAN_ID.fullmatch(plan_id):
            raise KeyError(plan_id)
        return self.root / "plans" / f"{plan_id}.json"

    def save(self, plan: InterviewPlan) -> None:
        """Keep ``plan`` (replacing its older revision)."""
        _write(self._plan_path(plan.id), plan.to_contract())

    def load(self, plan_id: str) -> InterviewPlan:
        """Return a plan (``KeyError`` when there is none)."""
        path = self._plan_path(plan_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise KeyError(plan_id) from exc
        return InterviewPlan.model_validate_json(text)

    def list(self) -> List[InterviewPlan]:
        """Every readable plan, newest first."""
        plans = []
        for path in sorted((self.root / "plans").glob("*.json")):
            try:
                plans.append(InterviewPlan.model_validate_json(
                    path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        plans.sort(key=lambda p: (p.updated_at.isoformat()
                                  if p.updated_at else ""), reverse=True)
        return plans

    def _results(self, plan_id: str) -> Path:
        if not PLAN_ID.fullmatch(plan_id):
            raise KeyError(plan_id)
        return self.root / "results" / plan_id

    def save_result(self, plan_id: str, sheet: Dict[str, Any]) -> None:
        """Keep the sheet of one interview of a plan."""
        session = str(sheet.get("session_id") or "")
        if not SESSION_ID.fullmatch(session):
            raise KeyError(session)
        _write(self._results(plan_id) / f"{session}.json", sheet)

    def results(self, plan_id: str) -> List[Dict[str, Any]]:
        """The sheets of a plan's interviews, newest first."""
        found: List[Dict[str, Any]] = []
        for path in self._results(plan_id).glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                found.append(data)
        found.sort(key=lambda d: str(d.get("finished_at") or ""),
                   reverse=True)
        return found

    def _result_path(self, plan_id: str, session_id: str) -> Path:
        if not SESSION_ID.fullmatch(session_id):
            raise KeyError(session_id)
        return self._results(plan_id) / f"{session_id}.json"

    def result(self, plan_id: str, session_id: str) -> Dict[str, Any]:
        """One stored report (``KeyError`` when there is none)."""
        path = self._result_path(plan_id, session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(session_id) from exc
        if not isinstance(data, dict):
            raise ValueError(f"report {session_id} is not an object")
        return data

    def delete_result(self, plan_id: str, session_id: str) -> bool:
        """Delete one stored report (``False`` when it was not there)."""
        try:
            self._result_path(plan_id, session_id).unlink()
        except FileNotFoundError:
            return False
        return True
