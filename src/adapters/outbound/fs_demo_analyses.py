"""The prepared demo results: ``*.analysis.json`` files (``analysis.demo.v1``).

They live beside the live page's demo scripts (``examples/demo``). A
file that cannot be read or does not follow the schema is skipped.
"""

import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.domain.skills.demo_analysis import DemoAnalysis

SUFFIX = ".analysis.json"


def read_demo_analyses(folder: Path) -> List[DemoAnalysis]:
    """Return the readable demo results of ``folder`` (none if missing)."""
    found: List[DemoAnalysis] = []
    if not Path(folder).is_dir():
        return found
    for path in sorted(Path(folder).glob(f"*{SUFFIX}")):
        try:
            found.append(DemoAnalysis.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            ))
        except (OSError, ValueError, ValidationError):
            continue
    return found
