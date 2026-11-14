"""The file-system session repository in a temporary directory."""

import io
import json
import tracemalloc
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest
import soundfile

from tests.integration.transcriber.audio import pcm_bytes, sound_file, tone
from transcriber.adapters.outbound.av_audio_decoder import pcm16_blocks
from transcriber.adapters.outbound.fs_session_repository import (
    FileSystemSessionRepository,
)
from transcriber.adapters.outbound.wav_playback import (
    MAX_DATA_BYTES,
    WAV_HEADER_BYTES,
    FilePlayback,
    LivePlayback,
    wav_header,
    write_wav,
)
from transcriber.application.ports import (
    PCM_FORMAT,
    PlaybackAudio,
    RecordedAudio,
)
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

    def test_ids_lists_stored_sessions_only(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        assert repo.ids() == []
        repo.create(session("b"))
        repo.create(session("a", SessionState.UPLOADED))
        (tmp_path / "sessions" / "empty-dir").mkdir()
        (tmp_path / "sessions" / ".hidden").mkdir()
        (tmp_path / "sessions" / ".hidden" / "session.json").write_text("{}")
        (tmp_path / "sessions" / "file").write_text("x")
        assert repo.ids() == ["a", "b"]

    def test_the_run_id_is_kept(
        self, repo: FileSystemSessionRepository
    ) -> None:
        repo.create(session())
        queued = session(state=SessionState.UPLOADED).model_copy(
            update={"run_id": "c" * 32}
        )
        repo.save(queued)
        stored = repo.get("s1")
        assert stored is not None and stored.run_id == "c" * 32

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

    def test_a_directory_in_place_of_the_session_file(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        (tmp_path / "sessions" / "s1" / "session.json").mkdir(parents=True)
        with pytest.raises(StorageError, match="Cannot read session.json"
                           ) as caught:
            repo.get("s1")
        assert str(tmp_path) not in str(caught.value)
        assert repo.ids() == []


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

    def test_a_symlink_to_another_session_is_refused(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        """An alias must not reach (or delete) the session it points to."""
        repo.create(session("real"))
        (tmp_path / "sessions" / "alias").symlink_to(
            tmp_path / "sessions" / "real"
        )
        for action in (repo.get, repo.delete, repo.recording):
            with pytest.raises(InvalidInputError, match="Invalid session"):
                action("alias")
        with pytest.raises(InvalidInputError, match="Invalid session"):
            repo.append_chunk("alias", 0, b"\x00\x00")
        assert repo.get("real") == session("real")
        assert repo.ids() == ["real"]


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

    def test_only_the_last_chunk_may_be_stored_again(
        self, repo: FileSystemSessionRepository
    ) -> None:
        """The port's contract: the last chunk is idempotent; an older
        number is refused instead of dropping every later chunk."""
        repo.create(session())
        repo.append_chunk("s1", 0, b"\x01\x00")
        repo.append_chunk("s1", 1, b"\x02\x00")
        repo.append_chunk("s1", 2, b"\x03\x00")
        repo.append_chunk("s1", 2, b"\x03\x00")
        with pytest.raises(InvalidInputError, match="expects chunk 3"):
            repo.append_chunk("s1", 1, b"\x09\x00")
        with pytest.raises(InvalidInputError, match="expects chunk 3"):
            repo.append_chunk("s1", 0, b"\x09\x00")
        assert repo.recording("s1").data == b"\x01\x00\x02\x00\x03\x00"
        with pytest.raises(InvalidInputError, match="expects chunk 3"):
            repo.append_chunk("s1", 4, b"\x00\x00")

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

    def test_an_interrupted_close_never_loses_later_chunks(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        """A process killed between the FLAC and ``session.json`` of a
        close leaves a ``recording`` session with a stale FLAC; the
        chunks received after it must reach the final FLAC."""
        data = pcm_bytes(tone(0.002))  # 32 samples
        first, second = data[:20], data[20:]  # 10 and 6 samples
        repo.create(session())
        repo.append_chunk("s1", 0, first)
        folder = tmp_path / "sessions" / "s1"
        repo._close_audio(folder)  # the close dies right after this
        assert (folder / "audio.flac").is_file()
        assert repo.get("s1") == session()  # still recording
        assert repo.recording("s1") == RecordedAudio(first, PCM_FORMAT)
        repo.append_chunk("s1", 1, second)
        assert repo.recording("s1") == RecordedAudio(data, PCM_FORMAT)
        repo.save(session(state=SessionState.UPLOADED))
        samples, _ = soundfile.read(
            io.BytesIO(repo.recording("s1").data), dtype="int16"
        )
        assert samples.tobytes() == data
        assert sorted(p.name for p in folder.iterdir()) == [
            "audio.flac", "session.json",
        ]

    def test_the_pcm_stays_until_the_session_file_is_written(
        self, repo: FileSystemSessionRepository, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Killed after the FLAC but before ``session.json``: the PCM is
        still there, so nothing is lost."""
        data = pcm_bytes(tone(0.002))
        repo.create(session())
        repo.append_chunk("s1", 0, data)
        module = "transcriber.adapters.outbound.fs_session_repository"

        def killed(path: Path, value: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(f"{module}.write_json", killed)
        with pytest.raises(KeyboardInterrupt):
            repo.save(session(state=SessionState.UPLOADED))
        monkeypatch.undo()
        folder = tmp_path / "sessions" / "s1"
        assert (folder / "audio.pcm").read_bytes() == data
        assert repo.recording("s1") == RecordedAudio(data, PCM_FORMAT)

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
        assert (tmp_path / "sessions" / "s1" / "imported.m4a").is_file()
        assert repo.recording("s1") == RecordedAudio(b"m4a bytes", "m4a")
        repo.store_recording("s1", RecordedAudio(b"wav bytes", "wav"))
        assert repo.recording("s1") == RecordedAudio(b"wav bytes", "wav")
        assert not (tmp_path / "sessions" / "s1" / "imported.m4a").exists()

    def test_an_import_never_takes_a_recorder_file_name(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        repo.store_recording("s1", RecordedAudio(b"flac bytes", "flac"))
        folder = tmp_path / "sessions" / "s1"
        names = sorted(p.name for p in folder.iterdir())
        assert names == ["imported.flac", "session.json"]
        assert repo.recording("s1") == RecordedAudio(b"flac bytes", "flac")

    def test_imports_stored_by_earlier_versions_are_still_read(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        (tmp_path / "sessions" / "s1" / "audio.m4a").write_bytes(b"old")
        assert repo.recording("s1") == RecordedAudio(b"old", "m4a")
        repo.store_recording("s1", RecordedAudio(b"new", "mp3"))
        folder = tmp_path / "sessions" / "s1"
        names = sorted(p.name for p in folder.iterdir())
        assert names == ["imported.mp3", "session.json"]

    @pytest.mark.parametrize("kind", [
        "../x", "a b", "", PCM_FORMAT, "pcm", "toolongextension", "wav\n",
    ])
    def test_unsafe_formats(
        self, repo: FileSystemSessionRepository, kind: str
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        with pytest.raises(InvalidInputError, match="Unsupported"):
            repo.store_recording("s1", RecordedAudio(b"x", kind))


def read_all(audio: PlaybackAudio) -> bytes:
    """Read a whole rendering and close it."""
    try:
        return audio.read(0, audio.size)
    finally:
        audio.close()


class TestPlayback:
    def test_a_recording_in_progress_is_its_pcm_behind_a_header(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session())
        with pytest.raises(NotFoundError, match="has no audio"):
            repo.playback("s1")
        data = pcm_bytes(tone(0.5))
        repo.append_chunk("s1", 0, data)
        live = repo.playback("s1")
        assert live.size == WAV_HEADER_BYTES + len(data)
        assert live.read(10, 30) == read_all(repo.playback("s1"))[10:40]
        repo.append_chunk("s1", 1, data)  # arrives while being read
        whole = live.read(0, live.size)
        live.close()
        assert whole[WAV_HEADER_BYTES:] == data  # only what had arrived
        with wave.open(io.BytesIO(whole)) as heard:
            assert (heard.getframerate(), heard.getnchannels(),
                    heard.getsampwidth()) == (16000, 1, 2)
            assert heard.getnframes() == len(data) // 2
        assert not list((tmp_path / "sessions" / "s1").glob("playback-*"))

    def test_a_closed_recording_is_rendered_once_and_kept(
        self, tmp_path: Path
    ) -> None:
        decoded: List[str] = []

        def counting(path: Path, kind: str) -> Iterator[bytes]:
            decoded.append(kind)
            return pcm16_blocks(path, kind)

        repo = FileSystemSessionRepository(tmp_path, blocks=counting)
        repo.create(session())
        data = pcm_bytes(tone(1.0))
        repo.append_chunk("s1", 0, data)
        repo.save(session(state=SessionState.UPLOADED))
        first = read_all(repo.playback("s1"))
        assert first[WAV_HEADER_BYTES:] == data  # lossless through FLAC
        assert read_all(repo.playback("s1")) == first
        assert decoded == ["flac"]
        folder = tmp_path / "sessions" / "s1"
        assert len(list(folder.glob("playback-*.wav"))) == 1
        repo.delete("s1")
        assert not folder.exists()

    def test_new_audio_replaces_the_rendering(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        low = sound_file(tone(1.0, hz=220.0), 16000, "wav")
        repo.store_recording("s1", RecordedAudio(low, "wav"))
        before = read_all(repo.playback("s1"))
        high = sound_file(tone(1.0, hz=880.0), 16000, "wav")
        repo.store_recording("s1", RecordedAudio(high, "wav"))
        after = read_all(repo.playback("s1"))
        assert after != before and after[WAV_HEADER_BYTES:] == (
            high[WAV_HEADER_BYTES:]
        )
        assert len(list(
            (tmp_path / "sessions" / "s1").glob("playback-*.wav")
        )) == 1

    def test_other_rates_and_channels_are_rendered_at_16_khz_mono(
        self, repo: FileSystemSessionRepository
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        stereo = np.stack([tone(2.0, rate=44100), tone(2.0, rate=44100)], 1)
        buffer = io.BytesIO()
        soundfile.write(buffer, stereo, 44100, format="WAV")
        repo.store_recording("s1", RecordedAudio(buffer.getvalue(), "wav"))
        with wave.open(io.BytesIO(read_all(repo.playback("s1")))) as heard:
            assert (heard.getframerate(), heard.getnchannels()) == (16000, 1)
            assert abs(heard.getnframes() - 32000) <= 32

    def test_an_undecodable_file_leaves_no_rendering(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        repo.create(session(state=SessionState.UPLOADED))
        repo.store_recording("s1", RecordedAudio(b"noise" * 100, "mp3"))
        with pytest.raises(InvalidInputError):
            repo.playback("s1")
        assert sorted(
            p.name for p in (tmp_path / "sessions" / "s1").iterdir()
        ) == ["imported.mp3", "session.json"]

    def test_a_deleted_session_is_not_brought_back_by_a_rendering(
        self, tmp_path: Path
    ) -> None:
        holder: List[FileSystemSessionRepository] = []

        def deleting(path: Path, kind: str) -> Iterator[bytes]:
            holder[0].delete("s1")  # deleted while being rendered
            yield b"\x00\x00"

        repo = FileSystemSessionRepository(tmp_path, blocks=deleting)
        holder.append(repo)
        repo.create(session(state=SessionState.UPLOADED))
        repo.store_recording("s1", RecordedAudio(b"x", "wav"))
        with pytest.raises(NotFoundError):
            repo.playback("s1")
        assert not (tmp_path / "sessions" / "s1").exists()

    def test_thirty_minutes_render_in_bounded_memory(
        self, repo: FileSystemSessionRepository, tmp_path: Path
    ) -> None:
        """The whole recording is never in memory, decoded or rendered."""
        repo.create(session())
        repo.save(session(state=SessionState.UPLOADED))
        minutes = 30
        flac = tmp_path / "long.flac"
        second = (tone(1.0) * 0.5).astype(np.float32)
        with soundfile.SoundFile(
            str(flac), "w", 16000, 1, format="FLAC", subtype="PCM_16"
        ) as out:
            for _ in range(minutes * 60):
                out.write(second)
        stored = tmp_path / "sessions" / "s1" / "imported.flac"
        flac.replace(stored)
        tracemalloc.start()
        try:
            rendering = repo.playback("s1")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        size = rendering.size
        middle = rendering.read(size // 2, 1000)
        rendering.close()
        assert size == WAV_HEADER_BYTES + minutes * 60 * 16000 * 2
        assert len(middle) == 1000
        print(f"\n30-minute rendering: {size} bytes on disk,"
              f" peak traced memory {peak / 2**20:.1f} MiB")
        assert peak < 16 * 2**20  # the samples alone would be 55 MiB


class TestWavParts:
    def test_header_limits_and_empty_audio(self) -> None:
        with pytest.raises(InvalidInputError, match="too long"):
            wav_header(MAX_DATA_BYTES + 1)
        with pytest.raises(InvalidInputError, match="no audio"):
            write_wav(io.BytesIO(), iter([]))
        too_long = iter([b"\x00\x00"] * 3)
        with pytest.raises(InvalidInputError, match="too long"):
            write_wav(io.BytesIO(), too_long, max_bytes=4)

    def test_reads_past_the_end_and_closing_twice(
        self, tmp_path: Path
    ) -> None:
        pcm = tmp_path / "a.pcm"
        pcm.write_bytes(b"\x01\x00" * 4)
        live = LivePlayback(pcm, 8)
        assert live.read(live.size, 10) == b""
        assert live.read(0, 10) == wav_header(8)[:10]
        pcm.write_bytes(b"\x01\x00")  # cut back by a retried chunk
        assert live.read(44, 8) == b"\x01\x00" + bytes(6)
        live.close()
        live.close()
        out = tmp_path / "a.wav"
        with out.open("w+b") as stream:
            write_wav(stream, iter([b"\x01\x00"]))
        rendered = FilePlayback(out)
        assert rendered.read(rendered.size + 5, 3) == b""
        rendered.close()
        rendered.close()


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
