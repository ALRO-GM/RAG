"""Use case: read and change the pipeline settings (``settings.v1``).

The web's «Ajustes» page reads them with ``GET /settings`` and changes
them with ``PUT /settings`` (through the transcriber, behind the access
code); every consumer reads them when a job or session starts
(:meth:`PipelineSettings.current`), so no service restarts, except for
the steps whose ``applies`` is ``restart``.

The analysis judge already had its own record (``GET/PUT
/analysis/judge``, :class:`~src.application.skills.judges.JudgeSelection`);
it stays the one source of that choice: a change here is written there
too, and a read shows what it holds.
"""

from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from src.domain.errors import BusyError, InvalidInputError, RagError
from src.domain.pipeline_settings import (
    PRESET_NAMES,
    PRESETS,
    Choices,
    SettingsDocument,
    catalogue,
    check_choices,
    preset_of,
    resolve,
)


class SettingsStore(Protocol):
    """Where the ``settings.v1`` document is kept."""

    def load(self) -> Optional[SettingsDocument]:
        """Return the stored document, or ``None`` when there is none."""
        ...

    def update(
        self,
        change: Callable[[Optional[SettingsDocument]], SettingsDocument],
    ) -> SettingsDocument:
        """Apply ``change`` to the stored document atomically."""
        ...


class JudgeChoice(Protocol):
    """The analysis judge's own record (``/analysis/judge``)."""

    def mode(self) -> str:
        """Return the judge the next analyses use."""
        ...

    def choose(self, mode: str) -> Any:
        """Store ``mode`` as the judge of the next analyses."""
        ...


class PipelineSettings:
    """Read, validate and store the pipeline settings.

    Args:
        store: Where the document is kept.
        now: The clock (replaced in tests).
        judge: The analysis judge's record, kept in step with the
            ``analysis.judge`` choice (``None``: not linked).
    """

    def __init__(
        self,
        store: SettingsStore,
        now: Callable[[], datetime],
        judge: Optional[JudgeChoice] = None,
    ) -> None:
        """Wire the store, the clock and the judge record."""
        self.store = store
        self.now = now
        self.judge = judge

    def _choices(self, document: Optional[SettingsDocument]) -> Choices:
        """Return every choice, the judge's from its own record."""
        chosen = resolve(document.choices if document else {})
        if self.judge is not None:
            try:
                mode = self.judge.mode()
            except RagError:
                mode = ""
            chosen = resolve({**chosen, "analysis.judge": mode})
        return chosen

    def show(self) -> Dict[str, Any]:
        """Return the document, whether it is stored, and the catalogue."""
        document = self.store.load()
        choices = self._choices(document)
        return {
            "schema": "settings.v1",
            "revision": document.revision if document else 0,
            "updated_at": (
                document.updated_at.isoformat()
                if document and document.updated_at else None
            ),
            "preset": preset_of(choices),
            "choices": choices,
            "stored": document is not None,
            "catalogue": catalogue(),
        }

    def change(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Store the choices of ``body``; return :meth:`show`.

        ``body`` has ``preset`` (a preset name, whose choices are taken)
        and/or ``choices`` (steps to change, over the current ones, or
        over the preset's when both are given), and optionally the
        ``revision`` it was based on: an older one is refused
        (:class:`~src.domain.errors.BusyError`, someone else saved in
        between). Unknown steps or options are refused
        (:class:`~src.domain.errors.InvalidInputError`).
        """
        preset = body.get("preset")
        given = body.get("choices") or {}
        revision = body.get("revision")
        if preset is not None and preset not in (*PRESET_NAMES, "custom"):
            raise InvalidInputError(f"Unknown preset {str(preset)[:20]!r}.")
        if not isinstance(given, Mapping):
            raise InvalidInputError("choices must be an object.")
        if revision is not None and (
            not isinstance(revision, int) or isinstance(revision, bool)
        ):
            raise InvalidInputError("revision must be a whole number.")
        checked = check_choices(given)

        def apply(
            current: Optional[SettingsDocument],
        ) -> SettingsDocument:
            """Return the new document over ``current``."""
            held = current.revision if current else 0
            if revision is not None and revision != held:
                raise BusyError(
                    "The settings were changed meanwhile (revision "
                    f"{held}); reload them and try again."
                )
            base = self._choices(current)
            if preset in PRESETS:
                base = dict(PRESETS[preset])
            chosen = resolve({**base, **checked})
            return SettingsDocument(
                revision=held + 1, updated_at=self.now(),
                preset=preset_of(chosen), choices=chosen,
            )

        document = self.store.update(apply)
        if self.judge is not None:
            self.judge.choose(document.choices["analysis.judge"])
        return self.show()

    def current(self) -> Optional[Choices]:
        """Return the stored choices for a job starting now.

        ``None`` when nothing was saved or the file cannot be read: the
        caller keeps its own configuration (today's defaults).
        """
        try:
            document = self.store.load()
        except RagError:
            return None
        if document is None:
            return None
        return self._choices(document)
