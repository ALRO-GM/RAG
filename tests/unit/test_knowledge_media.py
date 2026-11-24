"""Audio and video as knowledge sources: the domain rules (KN-3a)."""

import re
from typing import Any, Dict

import pytest

from src.domain.errors import UnsupportedDocumentError
from src.domain.knowledge.contract import ResultMetadata
from src.domain.knowledge.documents import (
    EXTENSIONS,
    KnowledgeDocument,
    detect_type,
    document_text,
    is_media,
)
from src.domain.knowledge.domains import DomainManifest
from src.domain.knowledge.media import (
    MAX_PARAGRAPH,
    MEDIA_EXTENSIONS,
    MEDIA_TYPES,
    clock,
    looks_like_media,
    media_failure,
    media_job_id,
    span_times,
    speaker_names,
    time_marks,
    transcript_markdown,
)
from src.domain.knowledge.sources import chunk_markdown
from src.domain.models import ChunkMetadata
from src.domain.skills.transcript_contract import read_contract
from tests.knowledge_fakes import contract_transcript
from transcriber.config import UPLOAD_FORMATS

HEADS: Dict[str, bytes] = {
    "wav": b"RIFF\x24\x00\x00\x00WAVEfmt ",
    "flac": b"fLaC\x00\x00\x00\x22",
    "ogg": b"OggS\x00\x02",
    "opus": b"OggS\x00\x02",
    "mp3": b"ID3\x04\x00\x00",
    "m4a": b"\x00\x00\x00\x20ftypM4A ",
    "mp4": b"\x00\x00\x00\x18ftypisom",
    "aac": b"\xff\xf1\x50\x80",
    "webm": b"\x1a\x45\xdf\xa3\x9f\x42",
    "mov": b"\x00\x00\x00\x14ftypqt  ",
    "3gp": b"\x00\x00\x00\x14ftyp3gp4",
    "amr": b"#!AMR\n",
    "wma": b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9",
}


def test_the_formats_are_the_transcribers() -> None:
    """The library accepts exactly what the transcriber reads."""
    assert set(MEDIA_TYPES) == set(UPLOAD_FORMATS)
    for kind, extensions in MEDIA_EXTENSIONS.items():
        assert EXTENSIONS[kind] == extensions == (f".{kind}",)


@pytest.mark.parametrize("kind", sorted(HEADS))
def test_each_format_is_recognised_by_its_content(kind: str) -> None:
    """Every container's first bytes are accepted for its extension."""
    assert looks_like_media(HEADS[kind])
    assert detect_type(f"Clase 1.{kind.upper()}", HEADS[kind]) == kind


@pytest.mark.parametrize("head", [
    b"%PDF-1.7", b"PK\x03\x04", b"RIFF\x00\x00\x00\x00WEBPVP8 ",
    b"\x89PNG\r\n\x1a\n", b"hola, esto es texto", b"", b"\xff",
])
def test_other_content_is_refused(head: bytes) -> None:
    """Documents, images and text named as audio are refused."""
    assert not looks_like_media(head)
    with pytest.raises(UnsupportedDocumentError, match="audio or video"):
        detect_type("clase.mp3", head)


def test_a_mismatched_container_is_still_media() -> None:
    """An ``.mp3`` that holds MP4 audio is read by its content."""
    assert detect_type("clase.mp3", HEADS["m4a"]) == "mp3"


def test_unknown_extensions_keep_their_answer() -> None:
    """``.aiff`` is not among the formats: the usual refusal."""
    with pytest.raises(UnsupportedDocumentError, match="Unsupported"):
        detect_type("clase.aiff", b"FORM\x00\x00AIFF")


@pytest.mark.parametrize("seconds, text", [
    (0, "00:00"), (59.9, "00:59"), (192, "03:12"), (3600, "1:00:00"),
    (3725.5, "1:02:05"), (-3, "00:00"),
])
def test_clock(seconds: float, text: str) -> None:
    """Minutes and seconds, hours from one hour."""
    assert clock(seconds) == text


def transcript(*said: Any, **kw: Any) -> Any:
    """A read ``transcript.v1``."""
    return read_contract(contract_transcript(*said, **kw))


def test_the_text_has_times_and_the_speakers() -> None:
    """One paragraph per voice's stretch; names when several speak."""
    read = transcript(
        ("SPEAKER_01", 0.0, 4.0, "Hola."),
        ("marta", 4.5, 192.0, "Bienvenidos."),
        ("SPEAKER_02", 192.5, 3725.0, "Gracias."),
        names={"marta": "Marta"},
    )
    assert speaker_names(read, "es") == {
        "SPEAKER_01": "Voz 1", "marta": "Marta", "SPEAKER_02": "Voz 2",
    }
    assert speaker_names(read, "en")["SPEAKER_02"] == "Speaker 2"
    assert transcript_markdown(read, "es") == (
        "[00:00–00:04] **Voz 1:** Hola.\n\n"
        "[00:04–03:12] **Marta:** Bienvenidos.\n\n"
        "[03:12–1:02:05] **Voz 2:** Gracias."
    )


def test_one_voice_has_no_label_and_long_stretches_split() -> None:
    """A single voice gets no label; a stretch is cut into paragraphs."""
    sentence = "Una frase bastante larga sobre la política de reembolsos."
    said = [("S1", float(n * 5), float(n * 5 + 4), sentence)
            for n in range(30)]
    contract = contract_transcript(*said)
    for n, utterance in enumerate(contract["utterances"]):
        # One turn of many sentences.
        utterance.update(id=f"t0.s{n}", turn=0, index=n,
                         previous_speaker=None, previous_utterance=None)
    text = transcript_markdown(read_contract(contract))
    paragraphs = text.split("\n\n")
    assert len(paragraphs) > 1
    assert all(len(p) <= MAX_PARAGRAPH + 20 for p in paragraphs)
    assert "**" not in text
    assert paragraphs[0].startswith("[00:00–")
    assert paragraphs[-1].endswith(sentence)
    assert transcript_markdown(read_contract(contract_transcript(
        ("S1", 0.0, 1.0, "   ")))) == ""


def test_span_times_follow_the_paragraphs() -> None:
    """A span starts in the paragraph it begins in, ends in the last one."""
    text = "x\n[00:00–00:04] uno\n\n[00:05–03:12] dos\n\n[03:13–03:40] tres"
    marks = time_marks(text)
    assert [(s, e) for _, s, e in marks] == [
        (0.0, 4.0), (5.0, 192.0), (193.0, 220.0)
    ]
    second = text.index("[00:05")
    assert span_times(marks, second, len(text)) == (5.0, 220.0)
    assert span_times(marks, second + 5, second + 10) == (5.0, 192.0)
    assert span_times(marks, 0, 1) is None
    assert span_times(marks, 0, len(text)) == (0.0, 220.0)
    assert span_times([], 0, 10) is None


@pytest.mark.parametrize("reason, expected", [
    ("recognise: No speech recognised.", "No se oye voz"),
    ("decode: The recording lasts 2 h 10 min; the limit is 2 h 00 min.",
     "La grabación dura 2 h 10 min; el límite es 2 h 00 min."),
    ("decode: Invalid data found when processing input",
     "No se pudo leer el sonido"),
    ("voices: boom", "La transcripción falló. / The transcription failed: "
     "voices: boom"),
])
def test_failures_read_plainly(reason: str, expected: str) -> None:
    """Known causes get a plain sentence, the rest the reason."""
    assert expected in media_failure(reason)


def test_job_ids_are_stable_and_valid() -> None:
    """The same document always gets the same transcriber job."""
    one = media_job_id("acme", "clase-1234abcd")
    assert one == media_job_id("acme", "clase-1234abcd")
    assert one != media_job_id("acme2", "clase-1234abcd")
    assert re.fullmatch(r"kn-[0-9a-f]{32}", one)


def test_the_text_of_a_recording_is_timed_and_its_chunks_too() -> None:
    """``timed: true`` in the front matter; every chunk gets its times."""
    document = KnowledgeDocument(
        id="clase-1", name="Clase 1.mp4", type="mp4", size=10,
        uploaded_at="2026-11-24T10:00:00+00:00",
    )
    assert is_media(document)
    said = [(f"SPEAKER_0{1 + n % 2}", float(n * 60), float(n * 60 + 50),
             f"Frase número {n} sobre devoluciones y plazos. " * 4)
            for n in range(12)]
    body = transcript_markdown(read_contract(contract_transcript(*said)))
    text = document_text(document, body)
    assert "timed: true" in text.split("---")[1]
    chunks = chunk_markdown(
        "extracted/clase-1.md", text, DomainManifest(id="acme", title="A"),
        400,
    )
    assert len(chunks) > 2
    starts = [c.metadata.start_s or 0.0 for c in chunks]
    ends = [c.metadata.end_s or 0.0 for c in chunks]
    assert starts[0] == 0.0 and ends[-1] == 11 * 60 + 50
    assert all(c.metadata.start_s is not None for c in chunks)
    assert all(s <= e for s, e in zip(starts, ends))
    assert starts == sorted(starts)
    other = document.model_copy(update={"type": "md", "name": "a.md"})
    plain = chunk_markdown(
        "a.md", document_text(other, body),
        DomainManifest(id="acme", title="A"), 400,
    )
    assert all(c.metadata.start_s is None for c in plain)


def test_times_are_in_results_only_when_set() -> None:
    """``knowledge.result.v1`` gets ``start_s``/``end_s`` only when known."""
    assert "start_s" not in ResultMetadata(title="x").model_dump()
    timed = ResultMetadata(title="x", start_s=192.0, end_s=220.0)
    assert timed.model_dump(mode="json")["start_s"] == 192.0
    assert ChunkMetadata().is_empty()
    assert not ChunkMetadata(start_s=0.0).is_empty()
