"""Outbound adapter: read labelled sessions for the second layer.

Accepted inputs:

- a ``.txt`` file of the Spanish set (``examples/skills/dialogue_acts``),
  or a directory of them (every ``*.txt``, in name order);
- a ``.json`` file: one session in the ``acts.v1`` shape (a tagger's
  ``<session>.acts.json``) or a list of them (the set's
  ``dialogues.json``); see :mod:`src.domain.skills.act_sequence`.
"""

import json
from pathlib import Path
from typing import List

from src.domain.errors import InvalidInputError, NotFoundError
from src.domain.skills.act_sequence import (
    ActSession,
    session_from_set,
    sessions_from_json,
)
from src.domain.skills.dialogue_act_set import parse_set


def read_act_sessions(path: str) -> List[ActSession]:
    """Return the sessions stored at ``path`` (file or set directory)."""
    target = Path(path)
    if target.is_dir():
        files = sorted(target.glob("*.txt"))
        if not files:
            raise NotFoundError(f"No *.txt dialogue files in {path}.")
        return [s for f in files for s in _read_set(f)]
    if not target.is_file():
        raise NotFoundError(f"No such file: {path}.")
    if target.suffix == ".txt":
        return _read_set(target)
    if target.suffix != ".json":
        raise InvalidInputError(
            f"{path}: expected a .txt dialogue file, a .json acts file or "
            "a directory.")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InvalidInputError(f"{path}: not valid JSON ({error}).")
    return sessions_from_json(document, target.name)


def _read_set(path: Path) -> List[ActSession]:
    """Parse one text file of the Spanish set."""
    return [
        session_from_set(d)
        for d in parse_set(path.read_text(encoding="utf-8"), path.name)
    ]
