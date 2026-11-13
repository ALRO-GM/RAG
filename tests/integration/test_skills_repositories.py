"""Transcript and catalogue repositories on real files."""

import json
from pathlib import Path
from typing import List

import pytest

from src.application.skills.gold import (
    AnnotationSet,
    agreement,
    consensus,
    validate_annotations,
)
from src.application.skills.lab import lab_dataset
from src.domain.languages import detect_language
from src.domain.models import RagDataset

from src.adapters.outbound.fs_transcript_repository import (
    FileSystemTranscriptRepository,
)
from src.adapters.outbound.json_catalogue_repository import (
    JsonCatalogueRepository,
)
from src.domain.errors import (
    CorruptedDataError,
    InvalidInputError,
    NotFoundError,
)
from tests.fakes import catalogue_data
from tests.skills_samples import (
    meeting_text,
    write_rubrics,
    write_transcripts,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "skills" / "transcripts"
REAL_RUBRICS = ROOT / "knowledge" / "rubrics"


# ---------------------------------------------------------- transcripts


def test_read_text_transcript_uses_the_file_stem(tmp_path: Path) -> None:
    """A ``Speaker: text`` file is named after its file."""
    directory = write_transcripts(tmp_path)
    repository = FileSystemTranscriptRepository(str(directory))
    transcript = repository.read(str(directory / "s1.txt"))
    assert transcript.session_id == "s1"
    assert transcript.speakers() == ["Maya", "Tom", "Priya"]
    assert len(transcript.turns) == 8
    assert transcript.has_timestamps()
    assert transcript.turns[1].start == 10.0


def test_read_json_transcript_may_name_its_session(tmp_path: Path) -> None:
    """A JSON document can carry its own session id and language."""
    file = tmp_path / "file.json"
    file.write_text(json.dumps({
        "session_id": "named", "language": "es",
        "turns": [{"speaker": "Ana", "text": "hola a todos"}],
    }, indent=2), encoding="utf-8")
    transcript = FileSystemTranscriptRepository().read(str(file))
    assert (transcript.session_id, transcript.language) == ("named", "es")


def test_read_strips_a_byte_order_mark(tmp_path: Path) -> None:
    """A UTF-8 BOM does not end up in the first speaker's name."""
    text_file = tmp_path / "bom.txt"
    text_file.write_bytes(b"\xef\xbb\xbfAna: hola\r\nLuis: qu\xc3\xa9 tal\r\n")
    json_file = tmp_path / "bom.json"
    json_file.write_bytes(
        b'\xef\xbb\xbf[\n{"speaker": "Ana", "text": "hola"}\n]'
    )
    repository = FileSystemTranscriptRepository()
    transcript = repository.read(str(text_file))
    assert transcript.speakers() == ["Ana", "Luis"]
    assert transcript.turns[1].text == "qué tal"
    assert repository.read(str(json_file)).speakers() == ["Ana"]


def test_read_missing_file_or_directory(tmp_path: Path) -> None:
    """A path that is not a file is 'not found'."""
    repository = FileSystemTranscriptRepository(str(tmp_path))
    with pytest.raises(NotFoundError, match="Transcript not found"):
        repository.read(str(tmp_path / "missing.txt"))
    with pytest.raises(NotFoundError, match="Transcript not found"):
        repository.read(str(tmp_path))


def test_read_empty_binary_and_invalid_files(tmp_path: Path) -> None:
    """Degenerate files give domain errors naming the file."""
    repository = FileSystemTranscriptRepository(str(tmp_path))
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "blank.txt").write_text(" \n\n")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00\x01\x80data")
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "notes.txt").write_text("no speaker here\nat all\n")
    for name in ("empty.txt", "blank.txt"):
        with pytest.raises(InvalidInputError, match="is empty"):
            repository.read(str(tmp_path / name))
    with pytest.raises(CorruptedDataError, match="not a UTF-8 text file"):
        repository.read(str(tmp_path / "binary.txt"))
    with pytest.raises(CorruptedDataError, match="not valid JSON"):
        repository.read(str(tmp_path / "bad.json"))
    with pytest.raises(InvalidInputError, match="no speaker turn"):
        repository.read(str(tmp_path / "notes.txt"))


def test_find_looks_up_a_session_by_identifier(tmp_path: Path) -> None:
    """``find`` tries .json, .txt and .md under the directory."""
    directory = write_transcripts(tmp_path)
    (directory / "notes.md").write_text("Ana: hola\n", encoding="utf-8")
    (directory / "both.txt").write_text("Ana: from text\n")
    (directory / "both.json").write_text(
        json.dumps([{"speaker": "Ana", "text": "from json"}], indent=1)
    )
    repository = FileSystemTranscriptRepository(str(directory))
    assert repository.find("s1").session_id == "s1"
    assert repository.find("notes").speakers() == ["Ana"]
    assert repository.find("both").turns[0].text == "from json"
    with pytest.raises(NotFoundError, match="No transcript for session"):
        repository.find("s9")


def test_find_names_the_session_after_the_requested_id(
    tmp_path: Path,
) -> None:
    """``find`` resolves by file name even if the JSON names another id."""
    (tmp_path / "file.json").write_text(json.dumps({
        "session_id": "declared",
        "turns": [{"speaker": "Ana", "text": "hola a todos"}],
    }))
    repository = FileSystemTranscriptRepository(str(tmp_path))
    assert repository.find("file").session_id == "file"
    assert repository.read(str(tmp_path / "file.json")).session_id == (
        "declared"
    )


def test_sessions_lists_transcript_files_only(tmp_path: Path) -> None:
    """Session ids are the sorted stems of the transcript files."""
    directory = write_transcripts(tmp_path)
    (directory / "readme.rst").write_text("not a transcript")
    (directory / "sub").mkdir()
    (directory / "sub" / "nested.txt").write_text("Ana: hola\n")
    (directory / "es1.json").write_text("[]")
    repository = FileSystemTranscriptRepository(str(directory))
    assert repository.sessions() == ["es1", "s1"]
    assert FileSystemTranscriptRepository(
        str(tmp_path / "nowhere")
    ).sessions() == []


def test_single_line_json_file_is_read_as_json(tmp_path: Path) -> None:
    """A compact JSON array (one line) is still parsed as JSON."""
    file = tmp_path / "compact.json"
    file.write_text(json.dumps([{"speaker": "Ana", "text": "hola a todos"}]))
    transcript = FileSystemTranscriptRepository().read(str(file))
    assert transcript.speakers() == ["Ana"]


def test_single_line_whisperx_file_is_read_as_json(tmp_path: Path) -> None:
    """A diarizer output written without indentation is parsed as JSON."""
    file = tmp_path / "wx.json"
    file.write_text(json.dumps({"segments": [
        {"speaker": "SPEAKER_00", "text": "hello there", "start": 0.0,
         "end": 1.0},
    ]}))
    transcript = FileSystemTranscriptRepository().read(str(file))
    assert transcript.speakers() == ["SPEAKER_00"]


@pytest.mark.parametrize("name, speakers, language_words", [
    ("demo_es.txt", {"Ana", "Luis", "Marta", "Pablo", "Irene"}, "es"),
    ("demo_en.txt", {"Maya", "Tom", "Priya", "Sam"}, "en"),
])
def test_example_transcripts_parse(
    name: str, speakers: set[str], language_words: str
) -> None:
    """The shipped demo sessions parse into timed, multi-speaker turns."""
    transcript = FileSystemTranscriptRepository().read(str(EXAMPLES / name))
    assert speakers <= set(transcript.speakers())
    assert len(transcript.speakers()) <= len(speakers) + 2
    assert len(transcript.turns) > 10
    assert transcript.has_timestamps()
    starts = [t.start or 0.0 for t in transcript.turns]
    assert starts == sorted(starts)
    assert detect_language(transcript.text()) == language_words
    assert all(
        t.speaker != transcript.turns[i - 1].speaker
        for i, t in enumerate(transcript.turns) if i
    )


def test_sample_meeting_round_trips_through_a_file(tmp_path: Path) -> None:
    """Writing and reading the untimed meeting keeps every turn."""
    file = tmp_path / "m.txt"
    file.write_text(meeting_text(timed=False), encoding="utf-8")
    transcript = FileSystemTranscriptRepository().read(str(file))
    assert not transcript.has_timestamps()
    assert transcript.turns[6].text == "Okay."


# ------------------------------------------------------------ catalogues


def test_catalogue_available_and_load(tmp_path: Path) -> None:
    """Catalogues are found by ``<name>.<language>.json``."""
    directory = write_rubrics(tmp_path)
    (directory / "notes.json").write_text("{}")
    (directory / "extra.set.en.json").write_text(
        json.dumps(catalogue_data("en", name="extra.set"))
    )
    repository = JsonCatalogueRepository(str(directory))
    assert repository.available() == [
        ("core", "en"), ("core", "es"), ("extra.set", "en"),
    ]
    assert repository.load("core", "es").language == "es"
    assert repository.load("core", "EN").language == "en"
    assert repository.load("core", "es-MX").language == "es"
    assert repository.load("extra.set", "en").name == "extra.set"


def test_catalogue_load_by_path(tmp_path: Path) -> None:
    """``name`` may be the path of a catalogue file."""
    directory = write_rubrics(tmp_path)
    repository = JsonCatalogueRepository(str(tmp_path / "elsewhere"))
    loaded = repository.load(str(directory / "core.es.json"), "en")
    assert loaded.language == "es"


def test_catalogue_not_found_lists_what_exists(tmp_path: Path) -> None:
    """A missing catalogue or language names the available ones."""
    repository = JsonCatalogueRepository(str(write_rubrics(tmp_path, ["en"])))
    with pytest.raises(NotFoundError, match=r"Available: core \(en\)\."):
        repository.load("core", "es")
    with pytest.raises(NotFoundError, match="'other'"):
        repository.load("other", "en")
    empty = JsonCatalogueRepository(str(tmp_path / "nowhere"))
    assert empty.available() == []
    with pytest.raises(NotFoundError, match="Available: none"):
        empty.load("core", "en")


@pytest.mark.parametrize("content", [
    b"{not json", b"", b"[]", b'{"name": "core"}', b"\xff\xfe\x00binary",
    json.dumps({**catalogue_data(), "competencies": []}).encode(),
    json.dumps({**catalogue_data(),
                "scale": {"minimum": 5, "maximum": 1}}).encode(),
])
def test_catalogue_invalid_file_is_corrupted_data(
    tmp_path: Path, content: bytes
) -> None:
    """Malformed, incomplete, inconsistent or binary catalogues."""
    directory = tmp_path / "rubrics"
    directory.mkdir()
    (directory / "core.en.json").write_bytes(content)
    with pytest.raises(CorruptedDataError) as error:
        JsonCatalogueRepository(str(directory)).load("core", "en")
    message = str(error.value)
    assert message.startswith("Invalid catalogue ")
    assert "\n" not in message


def test_real_catalogue_languages_have_the_same_structure() -> None:
    """The shipped catalogue loads in en and es with identical ids."""
    if not (REAL_RUBRICS / "core.en.json").is_file() or not (
        REAL_RUBRICS / "core.es.json"
    ).is_file():
        pytest.skip("the real catalogue is not written yet")
    repository = JsonCatalogueRepository(str(REAL_RUBRICS))
    english = repository.load("core", "en")
    spanish = repository.load("core", "es")
    assert (english.language, spanish.language) == ("en", "es")
    assert english.structure() == spanish.structure()
    assert english.version == spanish.version
    assert english.scale.points() == spanish.scale.points()


# -------------------------------------------------------- example data


def example_forms() -> List[Path]:
    """Return every shipped annotation and consensus file."""
    examples = ROOT / "examples" / "skills"
    return sorted(
        [*(examples / "annotations").glob("*.json"),
         *(examples / "reference").glob("*.consensus.json")]
    )


def test_example_annotation_files_exist() -> None:
    """Annotations of two annotators and a consensus ship per demo."""
    names = [p.name for p in example_forms()]
    for session in ("demo_en", "demo_es"):
        assert sum(n.startswith(session) for n in names) >= 3


@pytest.mark.parametrize("form", example_forms(), ids=lambda p: p.name)
def test_example_annotations_are_valid(form: Path) -> None:
    """Each example form validates against its transcript and rubric."""
    annotations = AnnotationSet.model_validate_json(form.read_text("utf-8"))
    transcripts = FileSystemTranscriptRepository(str(EXAMPLES))
    transcript = transcripts.find(annotations.session_id)
    language = detect_language(transcript.text())
    catalogue = JsonCatalogueRepository(str(REAL_RUBRICS)).load(
        annotations.catalogue, language
    )
    assert catalogue.language == language
    assert validate_annotations(annotations, transcript, catalogue) == []
    assert any(a.rating is not None for a in annotations.annotations)


def test_example_annotators_can_be_compared_and_merged() -> None:
    """The two example annotators yield an agreement and a consensus."""
    annotations = ROOT / "examples" / "skills" / "annotations"
    sets = [
        AnnotationSet.model_validate_json(p.read_text("utf-8"))
        for p in sorted(annotations.glob("*.json"))
    ]
    scale = JsonCatalogueRepository(str(REAL_RUBRICS)).load(
        "core", "en"
    ).scale
    report = agreement(sets, scale)
    assert report.annotators == ["A1", "A2"] and report.sessions == 2
    assert report.alpha is not None and -1.0 <= report.alpha <= 1.0
    merged = consensus(sets)
    assert sorted(m.session_id for m in merged) == ["demo_en", "demo_es"]


def test_example_lab_dataset_matches_the_example_reference() -> None:
    """The shipped lab dataset is what ``lab_dataset`` builds today."""
    reference = ROOT / "examples" / "skills" / "reference"
    sets = [
        AnnotationSet.model_validate_json(p.read_text("utf-8"))
        for p in sorted(reference.glob("*.consensus.json"))
    ]
    shipped = RagDataset.model_validate_json(
        (reference / "lab_dataset.json").read_text("utf-8")
    )
    built = lab_dataset(sets)
    assert sorted(
        (q.question, [s.model_dump() for s in getattr(q, "sources", [])])
        for q in shipped.rag_questions
    ) == sorted(
        (q.question, [s.model_dump() for s in getattr(q, "sources", [])])
        for q in built.rag_questions
    )


def test_saved_transcript_is_found_again(tmp_path: Path) -> None:
    """A transcript made from audio is stored where ``find`` looks."""
    from src.domain.skills.speech import transcript_sentences
    from src.domain.skills.transcript import build_transcript

    repository = FileSystemTranscriptRepository(str(tmp_path / "sessions"))
    transcript = build_transcript("standup", [{
        "speaker": "Ana", "text": "Hola a todos. ¿Empezamos?",
        "start": 0.0, "end": 1.8,
        "words": [
            {"text": "Hola", "start": 0.0, "end": 0.4},
            {"text": "a", "start": 0.5, "end": 0.6},
            {"text": "todos.", "start": 0.6, "end": 1.0},
            {"text": "¿Empezamos?", "start": 1.1, "end": 1.8},
        ],
    }], language="es")
    path = repository.save(transcript)
    assert path.endswith("sessions/standup.json")
    found = repository.find("standup")
    assert found.language == "es"
    assert found.turns[0].words[3].start == 1.1
    assert [s.end for s in transcript_sentences(found)] == [1.0, 1.8]
    assert repository.sessions() == ["standup"]
    elsewhere = repository.save(transcript, str(tmp_path / "x" / "t.json"))
    assert Path(elsewhere).is_file()


def test_save_reports_an_unwritable_target(tmp_path: Path) -> None:
    """A path that cannot be created is a clean error."""
    from src.domain.errors import RagError
    from src.domain.skills.transcript import build_transcript

    blocker = tmp_path / "file"
    blocker.write_text("x")
    transcript = build_transcript("s", [{"speaker": "A", "text": "Hola."}])
    with pytest.raises(RagError, match="Cannot write"):
        FileSystemTranscriptRepository(str(tmp_path)).save(
            transcript, str(blocker / "sub" / "t.json")
        )
