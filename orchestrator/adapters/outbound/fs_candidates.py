"""Candidate profiles on disk (``interview.candidate.v1``, #235).

``<dir>/candidates/<plan id>/<candidate id>.json``, beside the plans and
sheets of :mod:`.fs_interview_plans`. The folders are created with mode
700 and the files written with mode 600 through a temporary file; ids
are checked against fixed patterns before a path is built. An
unreadable file reads as no profile. Deleting a profile removes its
file (and the role's folder when it is left empty).
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import List

from orchestrator.domain.candidate import CandidateProfile

ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class JsonCandidateStore:
    """The :class:`~orchestrator.application.ports.CandidateStore` on disk.

    Args:
        root: The interview folder (``data/interview``).
    """

    def __init__(self, root: Path):
        """Keep the folder (created on the first write)."""
        self.root = Path(root) / "candidates"

    def _folder(self, plan_id: str) -> Path:
        if not ID.fullmatch(plan_id):
            raise KeyError(plan_id)
        return self.root / plan_id

    def _path(self, plan_id: str, candidate_id: str) -> Path:
        if not ID.fullmatch(candidate_id):
            raise KeyError(candidate_id)
        return self._folder(plan_id) / f"{candidate_id}.json"

    def load(self, plan_id: str, candidate_id: str) -> CandidateProfile:
        """Return one profile (``KeyError`` when there is none)."""
        path = self._path(plan_id, candidate_id)
        try:
            return CandidateProfile.model_validate_json(
                path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KeyError(candidate_id) from exc

    def save(self, profile: CandidateProfile) -> None:
        """Keep ``profile`` atomically, readable by its owner only."""
        path = self._path(profile.plan_id, profile.id)
        for folder in (self.root, path.parent):
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(folder, 0o700)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(profile.to_contract(), out, ensure_ascii=False,
                      indent=1)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def list(self, plan_id: str) -> List[CandidateProfile]:
        """Every readable profile of a role."""
        found = []
        for path in sorted(self._folder(plan_id).glob("*.json")):
            try:
                found.append(CandidateProfile.model_validate_json(
                    path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return found

    def plans(self) -> List[str]:
        """The roles with a folder of profiles."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and ID.fullmatch(p.name))

    def delete(self, plan_id: str, candidate_id: str) -> bool:
        """Delete one profile; ``False`` when there was none."""
        path = self._path(plan_id, candidate_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True
