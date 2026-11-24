"""Scripted demos of the live page: ``*.demo.json`` files (``live.demo.v1``).

The demo id is the file name without ``.demo.json``; it must match the
``id`` inside. Only safe ids are looked up, so an id never reaches
outside the folder.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from pydantic import ValidationError

from orchestrator.application.ports import Document
from orchestrator.domain.demo import DEMO_ID, DemoScript

SUFFIX = ".demo.json"


class DirDemos:
    """The demos kept as files in ``folder``."""

    def __init__(self, folder: Path):
        """Keep the folder (a missing one has no demos)."""
        self.folder = Path(folder)

    def _files(self) -> Dict[str, Path]:
        if not self.folder.is_dir():
            return {}
        return {p.name[:-len(SUFFIX)]: p
                for p in sorted(self.folder.glob(f"*{SUFFIX}"))}

    def list(self) -> List[Document]:
        """Every readable demo, as a page lists it."""
        out: List[Document] = []
        for demo_id in sorted(self._files()):
            try:
                out.append(self.load(demo_id).summary())
            except KeyError:
                continue
        return out

    def load(self, demo_id: str) -> DemoScript:
        """Return one demo; ``KeyError`` when there is none (or unreadable)."""
        path = self._files().get(demo_id) if re.fullmatch(
            DEMO_ID, demo_id or "") else None
        if path is None:
            raise KeyError(demo_id)
        try:
            script = DemoScript.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise KeyError(demo_id) from exc
        if script.id != demo_id:
            raise KeyError(demo_id)
        return script
