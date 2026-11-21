"""What the voice bot says: sentences, the spoken text, TTS requests.

Plain data and functions: no I/O. The voice bot speaks as soon as it
can (``docs/platform/ARCHITECTURE.md`` §6, P3): the reasoner's reply
arrives as text pieces, :class:`ReplyText` takes the spoken text out
of the JSON the conversational template asks for, and
:class:`SentenceChunker` cuts it into sentences, each of which becomes
one ``tts.request.v1`` (:func:`tts_request`) and one
:class:`SpeechChunk` of audio.
"""

import re
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

TTS_SCHEMA = "tts.request.v1"
MAX_TTS_TEXT = 1000
"""Longest text of one ``tts.request.v1`` (the contract's limit)."""
MAX_REQUEST_ID = 64
VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
"""A Piper voice id (``es_ES-davefx-medium``): a file name, no path."""

_END = re.compile(r"[.!?…]+[\"'»”)\]]*(?=\s)")
"""A sentence end: punctuation, closing quotes, then a space."""
_SOFT = re.compile(r"[,;:](?=\s)")
"""Where a sentence that is too long may be cut."""


def _clean(text: str) -> str:
    """Return ``text`` on one line with single spaces."""
    return " ".join(text.split())


class SentenceChunker:
    """Cut a growing text into sentences as soon as each one ends.

    A sentence ends at ``.``, ``!``, ``?`` or ``…`` followed by a
    space; a piece shorter than ``min_chars`` waits for the next one
    (so "Sí." is not spoken on its own), and one longer than
    ``max_chars`` is cut at its last comma, semicolon or colon (or
    space) so the first audio never waits for a very long sentence.

    Args:
        min_chars: Shortest sentence spoken on its own.
        max_chars: Longest sentence before a forced cut.
    """

    def __init__(self, min_chars: int = 12, max_chars: int = 220):
        """Start with nothing buffered."""
        if min_chars < 1 or max_chars <= min_chars:
            raise ValueError("need 1 <= min_chars < max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def feed(self, text: str) -> List[str]:
        """Add ``text``; return the sentences it completed."""
        self._buffer += text
        found: List[str] = []
        while True:
            cut = self._cut()
            if cut <= 0:
                return found
            sentence = _clean(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
            if sentence:
                found.append(sentence)

    def flush(self) -> List[str]:
        """Return what is left as a last sentence (the reply ended)."""
        rest = _clean(self._buffer)
        self._buffer = ""
        return [rest] if rest else []

    def _cut(self) -> int:
        """Where the first complete sentence of the buffer ends (0: none)."""
        start = len(self._buffer) - len(self._buffer.lstrip())
        for match in _END.finditer(self._buffer):
            if match.end() - start >= self.min_chars:
                return match.end()
        if len(self._buffer) - start <= self.max_chars:
            return 0
        window = self._buffer[:start + self.max_chars]
        soft = [m.end() for m in _SOFT.finditer(window)
                if m.end() - start >= self.min_chars]
        if soft:
            return soft[-1]
        space = window.rfind(" ")
        return space if space - start >= self.min_chars else len(window)


def split_sentences(text: str, min_chars: int = 12) -> List[str]:
    """Return the sentences of a whole text, in order."""
    chunker = SentenceChunker(min_chars=min_chars)
    return chunker.feed(text) + chunker.flush()


_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": " ", "r": " ", "t": " "}


class ReplyText:
    """The spoken text inside a streamed conversational reply.

    The conversational template answers ``{"text": "...", "cites":
    [...]}``; the model's pieces arrive one by one and :meth:`feed`
    returns the new characters of the ``text`` value as soon as they
    are known (JSON escapes decoded). A reply that starts with neither
    ``{`` nor a code fence is taken as plain text, whole.
    """

    def __init__(self) -> None:
        """Start before the reply's first character."""
        self._mode = "start"
        self._raw = ""
        self._at = 0
        self._escape = ""

    @property
    def done(self) -> bool:
        """Whether the ``text`` value is complete."""
        return self._mode == "done"

    def feed(self, piece: str) -> str:
        """Add one piece of the model's output; return new spoken text."""
        if self._mode == "plain":
            return piece
        if self._mode == "done":
            return ""
        self._raw += piece
        if self._mode == "start":
            head = self._raw.lstrip()
            if not head:
                return ""
            if not head.startswith(("{", "`")):
                self._mode = "plain"
                return head
            self._mode = "key"
        if self._mode == "key":
            match = re.search(r'"text"\s*:\s*"', self._raw)
            if match is None:
                return ""
            self._at = match.end()
            self._mode = "value"
        return self._value()

    def _value(self) -> str:
        """Decode the ``text`` value from where it was left."""
        out: List[str] = []
        raw = self._raw
        while self._at < len(raw):
            char = raw[self._at]
            if self._escape:
                self._escape += char
                decoded = self._decode()
                if decoded is None:
                    self._at += 1
                    continue
                out.append(decoded)
                self._escape = ""
            elif char == "\\":
                self._escape = "\\"
            elif char == '"':
                self._mode = "done"
                self._at += 1
                break
            else:
                out.append(char)
            self._at += 1
        return "".join(out)

    def _decode(self) -> Optional[str]:
        """Return a finished escape's text, ``None`` while incomplete."""
        kind = self._escape[1]
        if kind != "u":
            return _ESCAPES.get(kind, kind)
        if len(self._escape) < 6:
            return None
        try:
            return chr(int(self._escape[2:6], 16))
        except ValueError:
            return ""


@dataclass(frozen=True)
class SpeechChunk:
    """The audio of one sentence.

    ``audio`` is a WAV file or raw PCM s16le mono, as the request's
    ``format`` asked; ``sample_rate`` is the voice's own rate.
    ``synth_ms`` is the time the synthesis took.
    """

    request_id: str
    seq: int
    text: str
    audio: bytes
    format: str
    sample_rate: int
    synth_ms: float

    @property
    def duration_ms(self) -> float:
        """Length of the audio in milliseconds (16-bit mono)."""
        header = 44 if self.format == "wav" else 0
        samples = max(0, len(self.audio) - header) // 2
        return round(samples * 1000.0 / max(1, self.sample_rate), 1)


def tts_request(
    request_id: str,
    text: str,
    voice: str,
    language: Optional[str] = None,
    interrupt_id: Optional[str] = None,
    audio_format: str = "wav",
    length_scale: float = 1.0,
    stream: bool = True,
    noise_scale: Optional[float] = None,
    noise_w: Optional[float] = None,
    sentence_silence: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a ``tts.request.v1`` document.

    ``noise_scale``, ``noise_w`` and ``sentence_silence`` are left out
    when ``None`` (the voice's own).

    Raises:
        ValueError: The text is empty, or an id, the voice or a control
            breaks the contract.
    """
    text = _clean(text)
    if not text:
        raise ValueError("nothing to say")
    if not 0 < len(request_id) <= MAX_REQUEST_ID:
        raise ValueError("request_id must be 1-64 characters")
    if not VOICE_ID.match(voice):
        raise ValueError(f"invalid voice id {voice!r}")
    if audio_format not in ("wav", "pcm_s16le"):
        raise ValueError(f"unknown audio format {audio_format!r}")
    if not 0.5 <= length_scale <= 2.0:
        raise ValueError("length_scale must be between 0.5 and 2.0")
    document: Dict[str, Any] = {
        "schema": TTS_SCHEMA, "request_id": request_id,
        "text": text[:MAX_TTS_TEXT], "language": language, "voice": voice,
        "speaker": None, "length_scale": length_scale,
        "format": audio_format, "stream": stream,
        "interrupt_id": interrupt_id,
    }
    for name, value, high in (("noise_scale", noise_scale, 1.5),
                              ("noise_w", noise_w, 1.5),
                              ("sentence_silence", sentence_silence, 2.0)):
        if value is None:
            continue
        if not 0.0 <= value <= high:
            raise ValueError(f"{name} must be between 0 and {high:g}")
        document[name] = value
    return document


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Return 16-bit mono PCM as a WAV file's bytes (in memory)."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


def pcm_of(chunk: SpeechChunk) -> bytes:
    """Return the PCM frames of a chunk (its WAV header dropped)."""
    if chunk.format != "wav":
        return chunk.audio
    with wave.open(BytesIO(chunk.audio), "rb") as reader:
        return reader.readframes(reader.getnframes())


class AudioJoiner:
    """Group a streamed sentence's small PCM pieces into growing ones.

    A streaming engine yields audio every 80-333 ms; the page decodes
    each message as one WAV. The first piece goes out at once (first
    audio as early as possible); after it, pieces are held until they
    are as long as everything already sent (at most ``max_ms``), so
    while the page plays what it has, an engine faster than real time
    has the next piece ready (no gap) and a sentence takes few
    messages. 16-bit mono PCM.
    """

    def __init__(self, sample_rate: int, max_ms: float = 1000.0):
        """Start with nothing sent."""
        self.sample_rate = max(1, sample_rate)
        self.max_ms = max_ms
        self.sent_ms = 0.0
        self._held: List[bytes] = []

    def _ms(self, size: int) -> float:
        """Milliseconds of ``size`` bytes."""
        return size / 2 * 1000.0 / self.sample_rate

    def add(self, pcm: bytes) -> Optional[bytes]:
        """Take one piece; return what should be sent now, if anything."""
        self._held.append(pcm)
        held = sum(len(p) for p in self._held)
        if self.sent_ms > 0 and self._ms(held) < min(self.sent_ms,
                                                     self.max_ms):
            return None
        return self.flush()

    def flush(self) -> Optional[bytes]:
        """Return everything held (the sentence ended), or ``None``."""
        if not self._held:
            return None
        out = b"".join(self._held)
        self._held = []
        self.sent_ms += self._ms(len(out))
        return out or None


def whole_chunk(chunks: List[SpeechChunk]) -> Optional[SpeechChunk]:
    """Join a short text's chunks into one (a streamed sentence).

    Chunks at another sample rate than the first are left out; ``None``
    when there is no audio.
    """
    if not chunks:
        return None
    first = chunks[0]
    if len(chunks) == 1:
        return first
    pcm = b"".join(pcm_of(c) for c in chunks
                   if c.sample_rate == first.sample_rate)
    audio = wav_bytes(pcm, first.sample_rate) if first.format == "wav" \
        else pcm
    return SpeechChunk(
        request_id=first.request_id, seq=first.seq,
        text=" ".join(c.text for c in chunks if c.text), audio=audio,
        format=first.format, sample_rate=first.sample_rate,
        synth_ms=chunks[-1].synth_ms,
    )
