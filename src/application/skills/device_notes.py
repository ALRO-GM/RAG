"""Notes of models that changed device during an analysis (#110).

An adapter whose model ran out of GPU memory and moved to the CPU (or
that runs on the CPU when a GPU was planned) keeps one line per event
in a ``device_notes`` list. The use cases read it without knowing the
adapter, the way they call ``release``: a component without the list
has no notes.
"""

from collections import Counter
from typing import Any, Callable, Iterable, List


def device_notes(component: Any) -> List[str]:
    """Return a component's device notes (none when it keeps none)."""
    notes = getattr(component, "device_notes", None)
    if notes is None:
        return []
    return [str(n) for n in notes if str(n)]


def collect_notes(components: Iterable[Any]) -> List[str]:
    """Return the device notes of several components, in order."""
    return [n for c in components for n in device_notes(c)]


def new_notes(before: List[str], after: List[str]) -> List[str]:
    """Return the notes of ``after`` that ``before`` did not have.

    Counted as a multiset: a note added again counts as new.
    """
    seen = Counter(before)
    fresh: List[str] = []
    for note in after:
        if seen[note]:
            seen[note] -= 1
        else:
            fresh.append(note)
    return fresh


def join_notes(*parts: str) -> str:
    """Join the non-empty notes with spaces, without repeating one."""
    unique: List[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return " ".join(unique)


NoteSource = Callable[[], List[str]]
"""Reads the device notes of the models shared by the steps."""
