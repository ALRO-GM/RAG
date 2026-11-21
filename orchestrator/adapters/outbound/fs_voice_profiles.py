"""Voice profiles read from the transcriber's voiceprint files.

The transcriber keeps one file per enrolled person,
``<data>/voiceprints/<id>.json`` (``{"voiceprint": {...,
"words_per_second"}, "consent": {..., "voice_clone"}}``), and, only
while that person agrees to their voice being cloned, a reference clip
``<id>.clone.wav`` beside it (``docs/live/VOICE_CLONING.md``). Both
services run on the same workstation; this adapter reads those files as
plain JSON and WAV, never the transcriber's code. An id is checked
before it becomes a file name, and an unreadable file reads as no
profile.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from orchestrator.domain.voices import VoiceProfile

SPEAKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CLIP_SUFFIX = ".clone.wav"


class DirVoiceProfiles:
    """The :class:`~orchestrator.application.ports.VoiceProfiles` on disk.

    Args:
        folder: The transcriber's ``voiceprints`` folder.
    """

    def __init__(self, folder: Path):
        """Keep the folder (it may not exist yet)."""
        self.folder = folder.expanduser()

    def get(self, speaker_id: str) -> Optional[VoiceProfile]:
        """Return the profile of ``speaker_id``, or ``None``."""
        if not SPEAKER_ID.fullmatch(speaker_id) or ".." in speaker_id:
            return None
        try:
            data = json.loads((self.folder / f"{speaker_id}.json")
                              .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        voiceprint = data.get("voiceprint")
        consent = data.get("consent")
        if not isinstance(voiceprint, dict) or not isinstance(consent, dict):
            return None
        clip = self.folder / f"{speaker_id}{CLIP_SUFFIX}"
        agreed = consent.get("voice_clone") is True
        return VoiceProfile(
            speaker_id=speaker_id,
            words_per_second=_rate(voiceprint.get("words_per_second")),
            clone_consent=agreed,
            reference=str(clip) if agreed and clip.is_file() else None,
        )


def _rate(value: Any) -> Optional[float]:
    """Return a stored rate as a number, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
