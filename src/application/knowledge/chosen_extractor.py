"""Read each upload with the OCR and captions chosen on «Ajustes».

The pipeline settings (``knowledge.ocr``, ``knowledge.captions``) are
read when an upload is checked and when it is extracted, so a change on
the page applies to the next upload without a restart. One reader is
built per combination, on first use, and kept.
"""

import threading
from typing import Callable, Dict, Mapping, Optional, Tuple

from src.application.knowledge.ports import (
    DocumentExtractor,
    ExtractedDocument,
)

Choice = Tuple[bool, str]
"""(read with layout and OCR, caption backend: florence|claude|off)."""


class ChosenExtractor:
    """A :class:`DocumentExtractor` that follows the pipeline settings.

    Args:
        choices: The saved pipeline settings, or ``None`` (none saved:
            ``default`` applies).
        build: Builds the reader of one (OCR, captions) combination.
        default: The combination the service is configured with.
    """

    def __init__(
        self,
        choices: Callable[[], Optional[Mapping[str, str]]],
        build: Callable[[bool, str], DocumentExtractor],
        default: Choice,
    ) -> None:
        """Keep the factories; nothing is built yet."""
        self._choices = choices
        self._build = build
        self._default = default
        self._built: Dict[Choice, DocumentExtractor] = {}
        self._lock = threading.Lock()

    def choice(self) -> Choice:
        """Return the combination a document read now uses."""
        chosen = self._choices()
        if not chosen:
            return self._default
        ocr = chosen.get("knowledge.ocr", "on") == "on"
        captions = chosen.get("knowledge.captions", self._default[1])
        if captions not in ("florence", "claude", "off"):
            captions = self._default[1]
        return ocr, captions

    def current(self) -> DocumentExtractor:
        """Return the reader of :meth:`choice` (built once)."""
        key = self.choice()
        with self._lock:
            if key not in self._built:
                self._built[key] = self._build(*key)
            return self._built[key]

    def verify(self, data: bytes, document_type: str) -> None:
        """Check the file with the reader chosen now."""
        self.current().verify(data, document_type)

    def extract(self, data: bytes, document_type: str) -> ExtractedDocument:
        """Extract the file with the reader chosen now."""
        return self.current().extract(data, document_type)
