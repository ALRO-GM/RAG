"""The owner's lists of «Lo que hay que decir» on disk (#294).

Owner data, like the interview plans, outside the repository's seed
manifests: ``<dir>/lists/<id>.json`` is the current version of a list
and ``<dir>/lists/<id>/v<version>.json`` every version saved (a session
records the version it used, so an old one can be read again);
``<dir>/offers/<plan id>.json`` says which list a job offer uses. Ids
are checked against fixed patterns before a path is built; writes go
through a temporary file created with mode 600 (readable by its owner
only) and replace the file at once.
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.domain.mandatory_lists import LIST_ID, OwnerList

LIST = re.compile(rf"^{LIST_ID}$")
PLAN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _write(path: Path, document: Dict[str, Any]) -> None:
    """Write ``document`` as JSON atomically (mode 600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(document, out, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


class JsonMandatoryLists:
    """The owner's lists as JSON files under ``root``."""

    def __init__(self, root: Path):
        """Keep the folder (created on the first write)."""
        self.root = Path(root)

    def _path(self, list_id: str, version: Optional[int] = None) -> Path:
        if not LIST.fullmatch(list_id):
            raise KeyError(list_id)
        if version is None:
            return self.root / "lists" / f"{list_id}.json"
        return self.root / "lists" / list_id / f"v{int(version)}.json"

    def list(self) -> List[OwnerList]:
        """Every readable list, by name."""
        found: List[OwnerList] = []
        for path in sorted((self.root / "lists").glob("*.json")):
            try:
                found.append(OwnerList.model_validate_json(
                    path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        found.sort(key=lambda o: (o.name.lower(), o.id))
        return found

    def load(self, list_id: str, version: Optional[int] = None) -> OwnerList:
        """Return a list (``KeyError`` when there is none)."""
        path = self._path(list_id, version)
        try:
            return OwnerList.model_validate_json(
                path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(list_id) from exc

    def save(self, owned: OwnerList) -> None:
        """Keep ``owned``: its version first, then the current file."""
        document = owned.to_json()
        _write(self._path(owned.id, owned.version), document)
        _write(self._path(owned.id), document)

    def delete(self, list_id: str) -> bool:
        """Delete a list, its versions and the offers that used it."""
        path = self._path(list_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        shutil.rmtree(self.root / "lists" / list_id, ignore_errors=True)
        for plan, used in self.offers().items():
            if used == list_id:
                self.assign(plan, None)
        return existed

    def _offer(self, plan_id: str) -> Path:
        if not PLAN.fullmatch(plan_id):
            raise KeyError(plan_id)
        return self.root / "offers" / f"{plan_id}.json"

    def offer_list(self, plan_id: str) -> Optional[str]:
        """The list a job offer uses (``None``: the general list)."""
        try:
            data = json.loads(self._offer(plan_id).read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        found = data.get("list") if isinstance(data, dict) else None
        return found if isinstance(found, str) and LIST.fullmatch(
            found) else None

    def assign(self, plan_id: str, list_id: Optional[str]) -> None:
        """Make a job offer use ``list_id`` (``None``: the general list)."""
        path = self._offer(plan_id)
        if list_id is None:
            path.unlink(missing_ok=True)
            return
        if not LIST.fullmatch(list_id):
            raise KeyError(list_id)
        _write(path, {"plan": plan_id, "list": list_id})

    def offers(self) -> Dict[str, str]:
        """Every job offer with a list of its own."""
        found: Dict[str, str] = {}
        for path in sorted((self.root / "offers").glob("*.json")):
            used = self.offer_list(path.stem)
            if used is not None:
                found[path.stem] = used
        return found
