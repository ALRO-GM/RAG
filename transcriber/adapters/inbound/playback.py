"""Audio for the browser's player: a WAV rendering and byte ranges.

The player (``docs/transcriber/ARCHITECTURE.md`` §7) sets the audio to
a time and plays from there, so the browser must be able to seek to any
second without downloading the file. Stored audio is FLAC (recorded
sessions) or whatever was imported (m4a, mp3, webm...). Browsers seek
those by guessing a byte offset and then searching for a frame, which
is slow or imprecise for FLAC without a seek table, and some formats do
not play at all in Safari (Ogg, older WebM). So the player is given a
rendering as 16-bit PCM WAV at the decoder's rate (16 kHz mono): every
browser plays it, and a time maps to a byte offset exactly, which makes
each seek one HTTP range request. It costs 32 KB per second of audio
(115 MB per hour) in transit; only the ranges played are sent.

Renderings are kept in memory (:class:`RenderCache`), bounded in bytes,
keyed by the session and a hash of its stored audio.
"""

import io
import re
import threading
import wave
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union

import numpy as np

from transcriber.application.ports import Audio

WAV_TYPE = "audio/wav"


class Unsatisfiable:
    """A range request none of whose bytes exist (answered with 416)."""


UNSATISFIABLE = Unsatisfiable()
ByteRange = Tuple[int, int]
"""First and last byte, both included."""

_RANGE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.IGNORECASE)


def wav_bytes(audio: Audio) -> bytes:
    """Return ``audio`` as a mono 16-bit PCM WAV file."""
    scaled = np.clip(
        np.round(audio.samples.astype(np.float64) * 32768.0),
        -32768, 32767,
    ).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(audio.sample_rate)
        out.writeframes(scaled.tobytes())
    return buffer.getvalue()


def parse_range(
    header: Optional[str], size: int
) -> Union[None, ByteRange, Unsatisfiable]:
    """Return the byte range ``header`` asks of a ``size``-byte body.

    ``None`` means "send the whole body": no header, a header this
    server does not understand (another unit, several ranges, a
    malformed one, first byte after last), which RFC 9110 lets a server
    ignore. A range that starts past the end, or an empty suffix, is
    :data:`UNSATISFIABLE`. The last byte is clamped to the body.
    """
    if header is None:
        return None
    match = _RANGE.match(header)
    if match is None:
        return None
    first_text, last_text = match.groups()
    if not first_text:
        if not last_text:
            return None
        suffix = int(last_text)
        if suffix == 0 or size == 0:
            return UNSATISFIABLE
        return max(size - suffix, 0), size - 1
    first = int(first_text)
    last = int(last_text) if last_text else max(size - 1, first)
    if last < first:
        return None
    if first >= size:
        return UNSATISFIABLE
    return first, min(last, size - 1)


class RenderCache:
    """The last renderings made, at most ``limit`` bytes in all."""

    def __init__(self, limit: int) -> None:
        """Keep up to ``limit`` bytes of renderings (0 keeps none)."""
        self.limit = limit
        self._items: "OrderedDict[Tuple[str, str], bytes]" = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self, key: Tuple[str, str], render: Callable[[], bytes]
    ) -> bytes:
        """Return the rendering under ``key``, making it if needed.

        A rendering larger than the limit is returned but not kept.
        Two requests for a missing rendering may both make it.
        """
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
                return found
        made = render()
        if len(made) <= self.limit:
            with self._lock:
                self._items[key] = made
                self._items.move_to_end(key)
                while sum(map(len, self._items.values())) > self.limit:
                    self._items.popitem(last=False)
        return made

    def forget(self, session_id: str) -> None:
        """Drop every rendering of ``session_id`` (it was deleted)."""
        with self._lock:
            for key in [k for k in self._items if k[0] == session_id]:
                del self._items[key]
