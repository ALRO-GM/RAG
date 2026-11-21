"""Use case: the web's «Ajustes» page reads and saves the pipeline settings.

The RAG keeps them (``settings.v1``); this use case checks the shape of
what the page sends before it is forwarded (short string names, and
numbers and person ids for the voice bot; nothing else), and gives the
jobs of this service the saved choices when they start
(:meth:`PipelineSettingsAccess.current`).
"""

import re
from typing import Any, Dict, Mapping, Optional

from transcriber.application.ports import PipelineSettingsPort
from transcriber.domain.errors import (
    AnalysisUnavailableError,
    InvalidInputError,
)

NAME = re.compile(r"[a-z0-9_.-]{1,48}")
"""What a step or a preset name may look like."""
OPTION = re.compile(r"[A-Za-z0-9_.:-]{1,80}")
"""What an option may look like (a name, a number, ``clone:<person>``,
a natural voice such as ``pocket:lola``)."""
MAX_CHOICES = 64


def checked_body(body: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``PUT /api/settings`` body with its shape checked.

    Accepted keys: ``preset`` (a name), ``choices`` (an object of names
    to names) and ``revision`` (a whole number). The RAG checks the
    names against its allow-lists.
    """
    unknown = sorted(set(body) - {"preset", "choices", "revision"})
    if unknown:
        raise InvalidInputError(f"Unknown field(s): {', '.join(unknown)}.")
    out: Dict[str, Any] = {}
    preset = body.get("preset")
    if preset is not None:
        if not isinstance(preset, str) or not NAME.fullmatch(preset):
            raise InvalidInputError("Invalid preset.")
        out["preset"] = preset
    choices = body.get("choices")
    if choices is not None:
        if not isinstance(choices, dict) or len(choices) > MAX_CHOICES:
            raise InvalidInputError("choices must be an object.")
        for step, option in choices.items():
            if not (isinstance(option, str) and NAME.fullmatch(step)
                    and OPTION.fullmatch(option)):
                raise InvalidInputError("Invalid setting name or option.")
        out["choices"] = dict(choices)
    revision = body.get("revision")
    if revision is not None:
        if not isinstance(revision, int) or isinstance(revision, bool) \
                or revision < 0:
            raise InvalidInputError("revision must be a whole number.")
        out["revision"] = revision
    return out


class PipelineSettingsAccess:
    """Read and save the RAG's pipeline settings.

    ``port`` is ``None`` when transcripts do not go to a RAG over HTTP:
    nothing can be saved and every job keeps the configuration.
    """

    def __init__(self, port: Optional[PipelineSettingsPort]) -> None:
        """Wire the port (or its absence)."""
        self._port = port

    def _require(self) -> PipelineSettingsPort:
        """Return the port, or say the RAG is not connected."""
        if self._port is None:
            raise AnalysisUnavailableError(
                "Transcripts are not sent to a RAG over HTTP, so there are "
                "no pipeline settings to change."
            )
        return self._port

    def show(self) -> Dict[str, Any]:
        """Return the settings, the options and when each applies."""
        return self._require().show()

    def change(self, body: Mapping[str, Any]) -> Dict[str, Any]:
        """Check and forward a change; return the saved settings."""
        return self._require().change(checked_body(body))

    def current(self) -> Optional[Dict[str, str]]:
        """Return the saved choices (``None``: keep the configuration)."""
        if self._port is None:
            return None
        return self._port.current()
