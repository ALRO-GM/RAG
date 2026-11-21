"""File readers: the product YAML and the transcript JSON."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pydantic import ValidationError

from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import Transcript

PRODUCT_DIR = Path("config") / "products"


class _Loader(yaml.SafeLoader):
    """A YAML loader where ``on`` and ``off`` stay plain strings.

    YAML 1.1 reads a bare ``on`` key as the boolean true, which would
    turn ``trigger: {on: final}`` into ``{True: final}``. Only ``true``
    and ``false`` are booleans here.
    """


_Loader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers
          if tag != "tag:yaml.org,2002:bool"]
    + [("tag:yaml.org,2002:bool",
        re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"))]
    if key in "tTfF" else
    [(tag, regexp) for tag, regexp in resolvers
     if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


class ConfigError(Exception):
    """A product file is unreadable or invalid."""


def load_product(path: Path) -> ProductConfig:
    """Read and validate one product YAML.

    Raises:
        ConfigError: The file is missing, not YAML, or breaks the schema.
    """
    try:
        data: Any = yaml.load(
            path.read_text(encoding="utf-8"), Loader=_Loader
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        return ProductConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def check_dir(directory: Path) -> Tuple[List[str], List[str]]:
    """Validate every ``*.yaml`` of ``directory``.

    Returns:
        ``(valid product names, error messages)``.
    """
    names: List[str] = []
    errors: List[str] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            config = load_product(path)
        except ConfigError as exc:
            errors.append(str(exc))
            continue
        if path.stem != config.product:
            errors.append(f"{path}: file name differs from "
                          f"product {config.product!r}")
        else:
            names.append(config.product)
    if not names and not errors:
        errors.append(f"{directory}: no product files")
    return names, errors


def load_transcript(path: Path) -> Transcript:
    """Read a ``transcript.v1`` file.

    Raises:
        ConfigError: The file is missing or is not a transcript.
    """
    try:
        data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return Transcript.model_validate(data)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
