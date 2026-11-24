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
back to text at once. Clips of other people and stock voices stay.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

from transcriber.adapters.outbound.fs_storage import atomic_write

DIRECTORY = "demo-voices"
MANIFEST = "manifest.json"
CLIP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.wav")

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
    for path in sorted(folder.glob(f"*/{MANIFEST}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            clips: Dict[str, Any] = data["clips"]
            if not isinstance(clips, dict):
                raise ValueError("clips is not an object")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Unreadable demo voices %s: %s", path, exc)
            continue
        gone = {k: c for k, c in clips.items()
                if isinstance(c, dict) and c.get("cloned_from") == speaker_id}
        if not gone:
            continue
        for clip in gone.values():
            name = str(clip.get("file") or "")
            if CLIP_NAME.fullmatch(name) and ".." not in name:
                (path.parent / name).unlink(missing_ok=True)
                deleted += 1
        data["clips"] = {k: c for k, c in clips.items() if k not in gone}
        atomic_write(path, (json.dumps(data, ensure_ascii=False, indent=1)
                            + "\n").encode("utf-8"))
        os.chmod(path, 0o600)
    return deleted
