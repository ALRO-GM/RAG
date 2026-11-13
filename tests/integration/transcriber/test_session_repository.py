"""The file-system session repository in a temporary directory."""

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile

from tests.integration.transcriber.audio import pcm_bytes, tone
from transcriber.adapters.outbound.fs_session_repository import (
    FileSystemSessionRepository,
)
from transcriber.application.ports import PCM_FORMAT, RecordedAudio
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StorageError,
)
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
    Word,
)

STARTED = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)


def session(
    session_id: str = "s1", state: SessionState = SessionState.RECORDING
) -> Session:
    """A session in ``state``."""
    return Session(
        id=session_id, title="Equipo A", language="es", started_at=STARTED,
        state=state,
    )


def transcript(base: Session) -> Transcript:
    """A one-sentence transcript of ``base`` with its words."""
    words = [
        Word(text=" Hola", start=0.0, end=0.4, speaker="S1"),
        Word(text=" mundo.", start=0.4, end=0.9, speaker="S1"),
    ]
    return Transcript(
        session=base.model_copy(update={
            "state": SessionState.DONE, "duration": 1.0,
        }),
        speakers=[Speaker(id="S1", enrolled=False)],
        utterances=[Utterance(
            id="t0.s0", turn=0, index=0, speaker="S1", start=0.0, end=0.9,
            text="Hola mundo.", previous_speaker=None,
            previous_utterance=None, speaker_confidence=0.7, words=words,
        )],
        pipeline=Pipeline(
            recogniser="r", voice_model="v", transcriber="0.1.0"
        ),
    )


@pytest.fixture()
def repo(tmp_path: Path) -> FileSystemSessionRepository:
    """A repository rooted in a temporary directory."""
    return FileSystemSessionRepository(tmp_path)


class TestSessions:
    def test_layout_and_round_trip(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        assert repo.get("s1") is None
        repo.create(session())
        path = tmp_path / "sessions" / "s1" / "session.json"
        assert json.loads(path.read_text())["id"] == "s1"
        assert repo.get("s1") == session()
        repo.save(session().model_copy(update={"title": "Otro"}))
        stored = repo.get("s1")
        assert stored is not None and stored.title == "Otro"

    def test_create_twice_is_refused(
        self, repo: FileSystemSessionRepository
    ) -> None:
        repo.create(session())
        with pytest.raises(InvalidInputError, match="already exists"):
            repo.create(session())

    def test_save_of_an_unknown_session(
        self, repo: FileSystemSessionRepository
    ) -> None:
        with pytest.raises(NotFoundError, match="No session s1"):
            repo.save(session())

    def test_delete_removes_everything(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x00\x01")
        repo.delete("s1")
        assert repo.get("s1") is None
        assert not (tmp_path / "sessions" / "s1").exists()
        repo.delete("s1")  # deleting twice is harmless

    def test_damaged_session_file(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session())
        path = tmp_path / "sessions" / "s1" / "session.json"
        path.write_text('{"id": "s1", "lang', encoding="utf-8")
        with pytest.raises(StorageError, match="session.json"):
            repo.get("s1")
        path.write_text('{"id": "s1"}', encoding="utf-8")
        with pytest.raises(StorageError, match="Damaged session.json"):
            repo.get("s1")


class TestIds:
    @pytest.mark.parametrize("bad", [
        "../evil", "..", ".hidden", "a/b", "a\\b", "", "x" * 65, "/etc",
        "s1\n",
    ])
    def test_unsafe_ids_never_touch_the_disk(
        self, repo: FileSystemSessionRepository, tmp_path: Path, bad: str
    ) -> None:
        with pytest.raises(InvalidInputError, match="Invalid session id"):
            repo.get(bad)
        with pytest.raises(InvalidInputError, match="Invalid session id"):
            repo.delete(bad)
        with pytest.raises(InvalidInputError, match="Invalid session id"):
            repo.append_chunk(bad, 0, b"\x00\x00")
        assert not (tmp_path / "sessions").exists()

    def test_a_symlink_out_of_the_root_is_refused(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "sessions").mkdir()
        (tmp_path / "sessions" / "link").symlink_to(outside)
        with pytest.raises(InvalidInputError, match="Invalid session id"):
            repo.get("link")


class TestRecordingAudio:
    def test_chunks_are_appended_in_order(
        self, repo: FileSystemSessionRepository
    ) -> None:
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x01\x00\x02\x00")
        repo.append_chunk("s1", 1, b"\x03\x00")
        assert repo.recording("s1") == RecordedAudio(
            b"\x01\x00\x02\x00\x03\x00", PCM_FORMAT
        )

    def test_a_retried_chunk_replaces_the_bytes_after_the_previous(
        self, repo: FileSystemSessionRepository
    ) -> None:
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x01\x00")
        repo.append_chunk("s1", 1, b"\x02\x00")
        repo.append_chunk("s1", 1, b"\x09\x00")
        assert repo.recording("s1").data == b"\x01\x00\x09\x00"

    def test_bytes_left_by_a_killed_append_are_cut(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        """The PCM grew but ``chunks.json`` was never updated."""
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x01\x00")
        with (tmp_path / "sessions" / "s1" / "audio.pcm").open("ab") as f:
            f.write(b"\x07\x07\x07")
        assert repo.recording("s1").data == b"\x01\x00"
        repo.append_chunk("s1", 1, b"\x02\x00")
        assert repo.recording("s1").data == b"\x01\x00\x02\x00"

    def test_a_gap_is_refused(
        self, repo: FileSystemSessionRepository
    ) -> None:
        repo.create(session())
        with pytest.raises(InvalidInputError, match="expects chunk 0"):
            repo.append_chunk("s1", 1, b"\x00\x00")
        with pytest.raises(InvalidInputError, match="expects chunk 0"):
            repo.append_chunk("s1", -1, b"\x00\x00")

    def test_a_truncated_pcm_is_reported(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x01\x00\x02\x00")
        (tmp_path / "sessions" / "s1" / "audio.pcm").write_bytes(b"\x01")
        with pytest.raises(StorageError, match="shorter"):
            repo.append_chunk("s1", 1, b"\x00\x00")

    def test_a_damaged_chunk_index_is_reported(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x01\x00")
        (tmp_path / "sessions" / "s1" / "chunks.json").write_text('{"a": 1}')
        with pytest.raises(StorageError, match="chunks.json"):
            repo.recording("s1")

    def test_closing_encodes_flac_and_drops_the_pcm(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        wave = tone(1.0)
        data = pcm_bytes(wave)
        repo.create(session())
        repo.append_chunk("s1", 0, data[:16000])
        repo.append_chunk("s1", 1, data[16000:])
        repo.save(session(state=SessionState.UPLOADED))
        folder = tmp_path / "sessions" / "s1"
        assert sorted(p.name for p in folder.iterdir()) == [
            "audio.flac", "session.json",
        ]
        recording = repo.recording("s1")
        assert recording.format == "flac"
        samples, rate = soundfile.read(
            io.BytesIO(recording.data), dtype="int16"
        )
        assert rate == 16000
        assert samples.tobytes() == data  # lossless
        repo.save(session(state=SessionState.UPLOADED))  # again: no-op
        assert repo.recording("s1") == recording

    def test_no_audio(self, repo: FileSystemSessionRepository) -> None:
        repo.create(session())
        with pytest.raises(NotFoundError, match="has no audio"):
            repo.recording("s1")
        with pytest.raises(NotFoundError, match="No session s2"):
            repo.recording("s2")


class TestImportedAudio:
    def test_an_imported_file_is_kept_as_it_came(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        repo.store_recording("s1", RecordedAudio(b"m4a bytes", ".M4A"))
        assert (tmp_path / "sessions" / "s1" / "audio.m4a").is_file()
        assert repo.recording("s1") == RecordedAudio(b"m4a bytes", "m4a")
        repo.store_recording("s1", RecordedAudio(b"wav bytes", "wav"))
        assert repo.recording("s1") == RecordedAudio(b"wav bytes", "wav")
        assert not (tmp_path / "sessions" / "s1" / "audio.m4a").exists()

    @pytest.mark.parametrize("kind", [
        "../x", "a b", "", PCM_FORMAT, "toolongextension",
    ])
    def test_unsafe_formats(
        self, repo: FileSystemSessionRepository, kind: str
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        with pytest.raises(InvalidInputError, match="Unsupported"):
            repo.store_recording("s1", RecordedAudio(b"x", kind))


class TestWordsAndTranscripts:
    def test_round_trip(self, repo: FileSystemSessionRepository) -> None:
        repo.create(session())
        assert repo.words("s1") == []
        assert repo.transcript("s1") is None
        made = transcript(session())
        words = made.utterances[0].words
        repo.save_words("s1", words)
        repo.save_transcript(made)
        assert repo.words("s1") == words
        assert repo.transcript("s1") == made

    def test_damaged_words(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session())
        (tmp_path / "sessions" / "s1" / "words.json").write_text(
            '[{"text": "x", "start": 2, "end": 1}]'
        )
        with pytest.raises(StorageError, match="words.json"):
            repo.words("s1")

    def test_unknown_session(self, repo: FileSystemSessionRepository) -> None:
        with pytest.raises(NotFoundError):
            repo.save_words("s1", [])
        with pytest.raises(NotFoundError):
            repo.save_transcript(transcript(session()))
        with pytest.raises(NotFoundError):
            repo.words("s1")
        with pytest.raises(NotFoundError):
            repo.transcript("s1")


def test_no_temporary_files_are_left(
    repo: FileSystemSessionRepository, tmp_path: Path
) -> None:
    repo.create(session())
    repo.append_chunk("s1", 0, pcm_bytes(np.zeros(10, dtype=np.float32)))
    repo.save(session(state=SessionState.UPLOADED))
    repo.save_transcript(transcript(session()))
    leftovers = [
        p.name for p in (tmp_path / "sessions" / "s1").iterdir()
        if p.name.startswith(".")
    ]
    assert leftovers == []
