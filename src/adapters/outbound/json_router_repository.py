"""Persistence of the query router model as a small JSON file."""

from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.domain.errors import CorruptedDataError
from src.domain.priors import FEATURE_NAMES, RouterModel


class JsonRouterRepository:
    """Load and save a :class:`RouterModel`."""

    def __init__(self, path: str):
        """Use ``path`` as the model file."""
        self.path = Path(path)

    def load(self) -> Optional[RouterModel]:
        """Return the saved model, or None when there is no file."""
        if not self.path.is_file():
            return None
        try:
            model = RouterModel.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise CorruptedDataError(
                f"Invalid router model {self.path}: {exc}"
            )
        if len(model.weights) != len(FEATURE_NAMES) + len(model.vocabulary):
            raise CorruptedDataError(
                f"Router model {self.path} does not match the current "
                "features; run 'train_router' again."
            )
        return model

    def save(self, model: RouterModel) -> str:
        """Write ``model`` and return its path."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                model.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:
            raise CorruptedDataError(f"Cannot write {self.path}: {exc}")
        return str(self.path)
