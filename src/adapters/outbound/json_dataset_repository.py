"""Dataset repository adapter: pydantic models as JSON files."""

import json
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.domain.errors import CorruptedDataError, NotFoundError, RagError

ModelT = TypeVar("ModelT", bound=BaseModel)


class JsonDatasetRepository:
    """Read and write evaluator documents as UTF-8 JSON files."""

    def read(self, path: str, model: Type[ModelT]) -> ModelT:
        """Load and validate a JSON file as ``model``."""
        return read_model(path, model)

    def write(self, directory: str, filename: str, data: BaseModel) -> str:
        """Write ``data`` as indented JSON and return the output path."""
        return str(write_model(directory, filename, data))


def read_model(path: str, model: Type[ModelT]) -> ModelT:
    """Load and validate a JSON file as ``model``."""
    file = Path(str(path))
    if not file.is_file():
        raise NotFoundError(f"File not found: {file}")
    try:
        with open(file, encoding="utf-8") as handle:
            return model.model_validate_json(handle.read())
    except UnicodeDecodeError:
        raise CorruptedDataError(f"{file} is not a UTF-8 text file.")
    except ValidationError as exc:
        first = exc.errors()[0]
        if first.get("type") == "json_invalid":
            raise CorruptedDataError(
                f"Malformed JSON in {file}: {first.get('msg')}"
            )
        raise CorruptedDataError(
            f"{file} does not match the {model.__name__} format "
            f"({exc.error_count()} error(s), first: "
            f"{'.'.join(str(p) for p in first.get('loc', ()))}: "
            f"{first.get('msg')})"
        )
    except OSError as exc:
        raise RagError(f"Cannot read {file}: {exc}")


def write_model(directory: str, filename: str, data: BaseModel) -> Path:
    """Write ``data`` as indented JSON to ``directory/filename``."""
    out_dir = Path(str(directory))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / filename
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(data.model_dump(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise RagError(f"Cannot write to {out_dir}: {exc}")
    return target
