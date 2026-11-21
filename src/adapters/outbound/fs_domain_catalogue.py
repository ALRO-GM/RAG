"""Domain catalogue adapter: ``manifest.yaml`` files on disk.

Authored domains live in the repository (``knowledge/domains/<id>/``);
domains created through the API live with their uploads under
``data/knowledge/<id>/``. An authored domain cannot be deleted here.
"""

import shutil
from pathlib import Path
from typing import List, Set

import yaml
from pydantic import ValidationError

from src.domain.errors import (
    BusyError,
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
    short_reason,
)
from src.domain.knowledge.domains import DomainManifest, check_domain_id

MANIFEST = "manifest.yaml"


class FileSystemDomainCatalogue:
    """Read, create and delete domain manifests.

    Args:
        authored_root: Folder of the domains kept in the repository.
        data_root: Folder of the domains created through the API.
    """

    def __init__(self, authored_root: str, data_root: str) -> None:
        """Keep the two roots; nothing is read yet."""
        self.authored_root = Path(authored_root)
        self.data_root = Path(data_root)

    def ids(self) -> List[str]:
        """Return the ids of every folder holding a manifest, sorted."""
        found: Set[str] = set()
        for root in (self.authored_root, self.data_root):
            if root.is_dir():
                found.update(
                    p.parent.name for p in root.glob(f"*/{MANIFEST}")
                )
        return sorted(found)

    def _path(self, domain_id: str) -> Path:
        """Return the manifest of ``domain_id`` (authored first)."""
        check_domain_id(domain_id)
        for root in (self.authored_root, self.data_root):
            path = root / domain_id / MANIFEST
            if path.is_file():
                return path
        raise NotFoundError(f"No knowledge domain {domain_id!r}.")

    def manifest(self, domain_id: str) -> DomainManifest:
        """Return the validated manifest of a domain."""
        path = self._path(domain_id)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            manifest = DomainManifest.model_validate(data)
        except (OSError, yaml.YAMLError) as exc:
            raise CorruptedDataError(f"{path}: unreadable ({exc}).")
        except ValidationError as exc:
            raise CorruptedDataError(f"{path}: {short_reason(exc)}")
        if manifest.id != domain_id:
            raise CorruptedDataError(
                f"{path}: id {manifest.id!r} differs from its folder."
            )
        return manifest

    def authored(self, domain_id: str) -> bool:
        """Tell whether the domain's manifest is in the repository."""
        return self._path(domain_id).is_relative_to(self.authored_root)

    def create(self, manifest: DomainManifest) -> None:
        """Write the manifest of a new domain under the data root."""
        if manifest.id in self.ids():
            raise BusyError(f"The domain {manifest.id!r} already exists.")
        folder = self.data_root / manifest.id
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".{MANIFEST}.tmp"
        tmp.write_text(
            yaml.safe_dump(
                manifest.model_dump(mode="json"), allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(folder / MANIFEST)

    def remove(self, domain_id: str) -> None:
        """Delete a created domain's folder and everything uploaded."""
        if self.authored(domain_id):
            raise InvalidInputError(
                f"The domain {domain_id!r} is authored in the repository "
                f"({self.authored_root}); delete it there."
            )
        shutil.rmtree(self.data_root / domain_id)
