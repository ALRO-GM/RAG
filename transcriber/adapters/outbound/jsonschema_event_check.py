"""Contract check of live events against ``contracts/live.event.v1``.

Implements :class:`~transcriber.application.ports.EventCheck` with the
JSON Schema of the contract (draft 2020-12, formats checked). The
``jsonschema`` package is loaded when the check is built; without it
:func:`build_event_check` gives no check and says why.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

CONTRACT = "live.event.v1.schema.json"


class JsonSchemaEventCheck:
    """Validates events with the contract's schema."""

    def __init__(self, schema_path: Path) -> None:
        """Load the schema; raise ``ImportError`` without ``jsonschema``."""
        from jsonschema import Draft202012Validator, FormatChecker

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._validator = Draft202012Validator(
            schema, format_checker=FormatChecker()
        )

    def problem(self, event: Mapping[str, Any]) -> Optional[str]:
        """Return the first rule ``event`` breaks, or ``None``."""
        for error in self._validator.iter_errors(dict(event)):
            where = "/".join(str(p) for p in error.absolute_path) or "event"
            return f"{where}: {error.message}"
        return None


def build_event_check(
    contracts: Path,
) -> Tuple[Optional[JsonSchemaEventCheck], Optional[str]]:
    """Return the check of ``contracts``, or why there is none."""
    path = contracts / CONTRACT
    try:
        return JsonSchemaEventCheck(path), None
    except ImportError:
        return None, "jsonschema is not installed"
    except OSError as exc:
        return None, f"{path}: {exc.strerror}"
