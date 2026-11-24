"""The scripted examples' clips on disk (#254), outside the repository.

``<root>/<demo id>/manifest.json`` (``demo.voices.v1``) and one WAV per
line, ``<line key>-<hash>.wav``. Folders are mode 700 and files 600,
written to a temporary file and renamed, so a reader never sees half a
clip. The default root is ``data/transcriber/demo-voices`` (git-ignored,
beside the voiceprints): the transcriber deletes a person's clips there
when their clone consent is withdrawn or they are forgotten.

The repository is public, so :func:`check_outside_repository` refuses a
root inside the checkout unless it is under its git-ignored ``data/``.
Ids and names are checked before they become paths.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from orchestrator.domain.demo import DEMO_ID
from orchestrator.domain.demo_voices import DemoVoiceManifest

MANIFEST = "manifest.json"
CLIP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.wav")


def check_outside_repository(root: Path, repository: Path) -> None:
    """Refuse a clip folder the repository could commit.

    Raises:
        ValueError: ``root`` is inside ``repository`` but not under its
            ignored ``data/`` folder.
    """
    root = root.expanduser().resolve()
    repository = repository.resolve()
    if root == repository or repository in root.parents:
        data = repository / "data"
        if not (root == data or data in root.parents):
            raise ValueError(
                f"{root} is inside the repository: demo voices live under"
                " data/ or outside the checkout, never where git sees them"
            )


class FsDemoVoices:
    """The :class:`~orchestrator.application.ports.DemoVoiceStore` on disk.

    Args:
        root: The folder of every demo's clips (made on first write).
    """

    def __init__(self, root: Path):
        """Keep the folder; nothing is written yet."""
        self.root = Path(root).expanduser()

    def _folder(self, demo_id: str) -> Path:
        """The demo's folder after checking the id."""
        if not re.fullmatch(DEMO_ID, demo_id or ""):
            raise KeyError(demo_id)
        return self.root / demo_id

    def _file(self, demo_id: str, name: str) -> Path:
        """A clip's file after checking the name."""
        if not CLIP_NAME.fullmatch(name or "") or ".." in name:
            raise KeyError(name)
        return self._folder(demo_id) / name

    def _write(self, path: Path, data: bytes) -> None:
        """Write ``data`` to ``path`` atomically, mode 600 (folders 700)."""
        for folder in (self.root, path.parent):
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(folder, 0o700)
        handle, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=".part-", suffix=path.suffix)
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def manifest(self, demo_id: str) -> Optional[DemoVoiceManifest]:
        """The demo's manifest; ``None`` when missing or unreadable."""
        try:
            path = self._folder(demo_id) / MANIFEST
            found = DemoVoiceManifest.model_validate(
                json.loads(path.read_text(encoding="utf-8")))
        except (KeyError, OSError, ValueError, ValidationError):
            return None
        return found if found.demo == demo_id else None

    def save_manifest(self, manifest: DemoVoiceManifest) -> None:
        """Write the manifest (atomically, mode 600)."""
        text = json.dumps(manifest.model_dump(mode="json", by_alias=True),
                          ensure_ascii=False, indent=1) + "\n"
        self._write(self._folder(manifest.demo) / MANIFEST,
                    text.encode("utf-8"))

    def write_clip(self, demo_id: str, name: str, wav: bytes) -> None:
        """Write one clip (atomically, mode 600)."""
        self._write(self._file(demo_id, name), wav)

    def read_clip(self, demo_id: str, name: str) -> bytes:
        """Return one clip; ``KeyError`` when there is none."""
        try:
            return self._file(demo_id, name).read_bytes()
        except OSError as exc:
            raise KeyError(name) from exc

    def has_clip(self, demo_id: str, name: str) -> bool:
        """Whether the clip's file is there."""
        try:
            return self._file(demo_id, name).is_file()
        except KeyError:
            return False

    def delete_clip(self, demo_id: str, name: str) -> None:
        """Delete one clip (nothing when it is gone already)."""
        try:
            self._file(demo_id, name).unlink(missing_ok=True)
        except KeyError:
            return

    def clear(self, demo_id: str) -> int:
        """Delete every clip and the manifest of a demo; how many clips."""
        folder = self._folder(demo_id)
        if not folder.is_dir():
            return 0
        count = 0
        for path in folder.iterdir():
            if path.is_file() and (CLIP_NAME.fullmatch(path.name)
                                   or path.name == MANIFEST
                                   or path.name.startswith(".part-")):
                if path.suffix == ".wav" and not path.name.startswith("."):
                    count += 1
                path.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
        return count
