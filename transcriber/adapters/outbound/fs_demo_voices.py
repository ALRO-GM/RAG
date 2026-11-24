"""Delete a person's cloned demo clips when their clone consent ends.

The orchestrator renders the scripted examples' voices ahead of time
(#254) into ``<data>/demo-voices/<demo>/``: one WAV per line and a
``manifest.json`` (``demo.voices.v1``) whose ``clips`` say, for a
cloned voice, ``cloned_from`` (the enrolled person). The two services
share the file format, not code, as for the voiceprints.

:func:`forget_cloned_demo_clips` is called by the same hook that deletes
a person's reference clip (withdrawing the clone consent, enrolling
again without it, forgetting the person): every clip cloned from that
person is deleted and dropped from its manifest, so the example falls
back to text at once. Every clip file that no manifest names is deleted
too: its owner cannot be told (a render killed half-way, or one still
running whose clips are not published yet), so it must go. Clips of
other people and stock voices that a manifest names stay.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Set

from transcriber.adapters.outbound.fs_storage import atomic_write

DIRECTORY = "demo-voices"
MANIFEST = "manifest.json"

LOG = logging.getLogger(__name__)


def forget_cloned_demo_clips(folder: Path, speaker_id: str) -> int:
    """Delete the demo clips cloned from ``speaker_id``; return how many.

    ``folder`` is ``<data>/demo-voices``. A missing folder or an
    unreadable manifest deletes nothing (and is logged); the manifest
    is rewritten atomically, mode 600, without those clips.
    """
    if not folder.is_dir():
        return 0
    deleted = 0
    for demo in sorted(p for p in folder.iterdir() if p.is_dir()):
        deleted += _forget_in(demo, speaker_id)
    return deleted


def _forget_in(demo: Path, speaker_id: str) -> int:
    """Clean one demo's folder; return how many clips were deleted."""
    path = demo / MANIFEST
    keep: Set[str] = set()
    deleted = 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        clips: Dict[str, Any] = data["clips"]
        if not isinstance(clips, dict):
            raise ValueError("clips is not an object")
    except FileNotFoundError:
        clips = {}
        data = None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.warning("Unreadable demo voices %s: %s", path, exc)
        clips = {}
        data = None
    gone = {k for k, c in clips.items()
            if isinstance(c, dict) and c.get("cloned_from") == speaker_id}
    keep = {str(c.get("file")) for k, c in clips.items()
            if isinstance(c, dict) and k not in gone}
    if gone and data is not None:
        data["clips"] = {k: c for k, c in clips.items() if k not in gone}
        atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=1)
                            + "\n").encode("utf-8"))
        os.chmod(path, 0o600)
    # Every clip file no manifest names goes: the person's, and those
    # of an unfinished render, whose owner cannot be told.
    for clip in demo.iterdir():
        if (clip.suffix == ".wav" and clip.name not in keep
                and clip.is_file()):
            clip.unlink(missing_ok=True)
            deleted += 1
    return deleted
