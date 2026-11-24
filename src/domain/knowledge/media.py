"""Audio and video files as knowledge sources: what is said in them (KN-3a).

A recording uploaded to a domain is transcribed by the transcriber
service (never in the RAG's process) and its ``transcript.v1`` becomes
the document's text: one paragraph per stretch of one voice, opened by
the time it covers (``[03:12–03:40]``) and, when several voices speak,
by the speaker. The extracted file says ``timed: true`` in its front
matter; its chunks then carry the start and end of what they cover
(:func:`span_times`), so a retrieved chunk can be cited by time.

The accepted formats are the transcriber's recording formats
(``transcriber/config.py``, ``UPLOAD_FORMATS``); a test keeps the two
lists equal.
"""

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

from src.domain.skills.transcript_contract import (
    ContractTranscript,
    ContractUtterance,
)

MEDIA_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "wav": (".wav",),
    "flac": (".flac",),
    "ogg": (".ogg",),
    "opus": (".opus",),
    "mp3": (".mp3",),
    "m4a": (".m4a",),
    "mp4": (".mp4",),
    "aac": (".aac",),
    "webm": (".webm",),
    "mov": (".mov",),
    "3gp": (".3gp",),
    "amr": (".amr",),
    "wma": (".wma",),
}
"""Audio and video types and their extensions (the transcriber's)."""
MEDIA_TYPES = tuple(MEDIA_EXTENSIONS)

TIMED_KEY = "timed"
"""Front-matter key of a text whose paragraphs open with their times."""

MAX_PARAGRAPH = 600
"""Longest paragraph (characters) before a voice's stretch is split, so
every chunk starts close to a time mark."""

ISO_BOXES = (b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip", b"pnot")
"""First boxes of an ISO media file (MP4, M4A, MOV, 3GP)."""
ASF_GUID = b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"
"""Start of an ASF (WMA) file."""


def looks_like_media(head: bytes) -> bool:
    """Tell whether ``head`` starts like an audio or video container.

    Any of the accepted containers will do whatever the extension says
    (a ``.m4a`` that is really an MP4, an ``.ogg`` holding Opus): the
    transcriber reads the content, not the name. Text, documents and
    images are refused.
    """
    if head.startswith(b"ID3"):
        return True  # MP3, AAC or FLAC with an ID3 tag
    if head.startswith(b"RIFF"):
        return head[8:12] == b"WAVE"
    if head.startswith((b"fLaC", b"OggS", b"\x1a\x45\xdf\xa3", b"#!AMR",
                        b"ADIF", ASF_GUID)):
        return True
    if head[4:8] in ISO_BOXES:
        return True
    # MPEG audio frames (MP3) and ADTS (AAC) start with a frame sync.
    return len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0


def clock(seconds: float) -> str:
    """Return ``seconds`` as ``MM:SS``, or ``H:MM:SS`` from one hour."""
    whole = max(int(seconds), 0)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


_CLOCK = r"(?:\d+:)?\d{2}:\d{2}"
MARK = re.compile(rf"^\[({_CLOCK})–({_CLOCK})\]", re.MULTILINE)
"""A paragraph's time mark: ``[MM:SS–MM:SS]`` at the start of a line."""


def seconds_of(text: str) -> float:
    """Return the seconds of a :func:`clock` text."""
    total = 0
    for part in text.split(":"):
        total = total * 60 + int(part)
    return float(total)


def speaker_names(
    transcript: ContractTranscript, language: str
) -> Dict[str, str]:
    """Return the label shown for each speaker of ``transcript``.

    An enrolled or named voice keeps its name; an anonymous one becomes
    ``Voz 1``, ``Voz 2``… (``Speaker 1``… when the domain is not in
    Spanish), numbered in the order they first speak.
    """
    word = "Voz" if language.lower().startswith("es") else "Speaker"
    named = {s.id: s.name for s in transcript.speakers if s.name}
    labels: Dict[str, str] = {}
    for utterance in transcript.utterances:
        who = utterance.speaker
        if who not in labels:
            labels[who] = named.get(who) or (
                f"{word} {sum(1 for s in labels if s not in named) + 1}"
            )
    return labels


def _paragraphs(
    utterances: Sequence[ContractUtterance],
) -> List[List[ContractUtterance]]:
    """Group consecutive sentences of one voice, at most MAX_PARAGRAPH long."""
    groups: List[List[ContractUtterance]] = []
    size = 0
    for utterance in utterances:
        text = utterance.text.strip()
        if not text:
            continue
        if groups and groups[-1][-1].speaker == utterance.speaker and \
                size + len(text) + 1 <= MAX_PARAGRAPH:
            groups[-1].append(utterance)
            size += len(text) + 1
        else:
            groups.append([utterance])
            size = len(text)
    return groups


def transcript_markdown(
    transcript: ContractTranscript, language: str = "es"
) -> str:
    """Return the indexed text of a transcribed recording.

    One paragraph per stretch of one voice: ``[MM:SS–MM:SS] text``, with
    ``**Voz 1:**`` after the mark when more than one voice speaks.
    Empty when nothing was said.
    """
    labels = speaker_names(transcript, language)
    several = len(labels) > 1
    lines: List[str] = []
    for group in _paragraphs(transcript.utterances):
        mark = f"[{clock(group[0].start)}–{clock(group[-1].end)}]"
        said = " ".join(u.text.strip() for u in group)
        who = f" **{labels[group[0].speaker]}:**" if several else ""
        lines.append(f"{mark}{who} {said}")
    return "\n\n".join(lines)


Mark = Tuple[int, float, float]
"""A time mark: its offset in the text, its start and its end (seconds)."""


def time_marks(text: str) -> List[Mark]:
    """Return the time marks of ``text``, in order."""
    return [
        (m.start(), seconds_of(m.group(1)), seconds_of(m.group(2)))
        for m in MARK.finditer(text)
    ]


def span_times(
    marks: Sequence[Mark], first: int, last: int
) -> Optional[Tuple[float, float]]:
    """Return the start and end of the text ``[first, last)`` covers.

    The start is that of the paragraph the span begins in (the last mark
    at or before ``first``, else the first mark inside the span); the
    end that of the last paragraph that begins before ``last``.
    ``None`` when no paragraph touches the span.
    """
    before = [m for m in marks if m[0] <= first]
    inside = [m for m in marks if first < m[0] < last]
    opening = before[-1] if before else (inside[0] if inside else None)
    if opening is None:
        return None
    closing = inside[-1] if inside else opening
    return opening[1], max(closing[2], opening[1])


def media_failure(reason: str) -> str:
    """Return the message a document shows for a failed transcription.

    ``reason`` is the transcriber's (``"<stage>: <why>"``) or the
    client's; known causes get a plain bilingual sentence.
    """
    text = reason.strip()
    if "No speech recognised" in text:
        return ("No se oye voz en el archivo. / No speech was found in "
                "the file.")
    lasts = re.search(r"lasts (.+?); the limit is (.+?)\.?$", text)
    if lasts:
        return (f"La grabación dura {lasts.group(1)}; el límite es "
                f"{lasts.group(2)}. / The recording lasts "
                f"{lasts.group(1)}; the limit is {lasts.group(2)}.")
    if text.startswith("decode:"):
        return ("No se pudo leer el sonido del archivo (¿está dañado o "
                "no tiene audio?). / The file's sound cannot be read "
                "(damaged, or no audio track?).")
    return (f"La transcripción falló. / The transcription failed: "
            f"{text[:500]}")


def media_job_id(domain_id: str, document_id: str) -> str:
    """Return the transcriber's job id of a domain's recording.

    Derived from the two ids, so a job sent again after a restart is the
    same job (the transcriber does not transcribe it twice).
    """
    digest = hashlib.sha256(f"{domain_id}/{document_id}".encode("utf-8"))
    return f"kn-{digest.hexdigest()[:32]}"
