"""The file-system voiceprint repository in a temporary directory."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)
from transcriber.domain.errors import (
    InvalidInputError,
    StorageError,
    VoiceModelMismatchError,
)
from transcriber.domain.models import ConsentRecord, Voiceprint

NOW = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)
MODEL = "microsoft/wavlm-base-plus-sv"


def voiceprint(speaker_id: str = "ana", model: str = MODEL) -> Voiceprint:
    """A voiceprint of ``speaker_id`` made with ``model``."""
    return Voiceprint(
        speaker_id=speaker_id, model=model, vector=[0.6, 0.8],
        consent_version="consent-1", enrolled_at=NOW,
    )


def consent(speaker_id: str = "ana") -> ConsentRecord:
    """The consent record of ``speaker_id``."""
    return ConsentRecord(
        speaker_id=speaker_id, name="Ana", consent_version="consent-1",
        text_version="es-1", given_at=NOW,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> FileSystemVoiceprintRepository:
    """A repository of :data:`MODEL` voiceprints in a temporary dir."""
    return FileSystemVoiceprintRepository(tmp_path, MODEL)


def test_round_trip_one_file_per_speaker(
    repo: FileSystemVoiceprintRepository, tmp_path: Path
) -> None:
    assert repo.all() == []
    assert repo.get("ana") is None and repo.consent("ana") is None
    repo.save(voiceprint(), consent())
    repo.save(voiceprint("luis"), consent("luis"))
    stored = json.loads((tmp_path / "voiceprints" / "ana.json").read_text())
    assert set(stored) == {"voiceprint", "consent"}
    assert stored["voiceprint"]["vector"] == [0.6, 0.8]
    assert stored["voiceprint"]["model"] == MODEL
    assert repo.get("ana") == voiceprint()
    assert repo.consent("ana") == consent()
    assert [v.speaker_id for v in repo.all()] == ["ana", "luis"]


def test_saving_again_replaces(repo: FileSystemVoiceprintRepository) -> None:
    repo.save(voiceprint(), consent())
    newer = voiceprint().model_copy(update={"vector": [1.0, 0.0]})
    repo.save(newer, consent())
    assert repo.get("ana") == newer


def test_delete(repo: FileSystemVoiceprintRepository) -> None:
    repo.save(voiceprint(), consent())
    assert repo.delete("ana") is True
    assert repo.get("ana") is None and repo.consent("ana") is None
    assert repo.delete("ana") is False


class TestAnotherVoiceModel:
    def test_saving_is_refused(
        self, repo: FileSystemVoiceprintRepository
    ) -> None:
        with pytest.raises(VoiceModelMismatchError, match="Enrol ana again"):
            repo.save(voiceprint(model="other/model"), consent())

    def test_reading_is_refused_with_a_clear_error(
        self, tmp_path: Path
    ) -> None:
        old = FileSystemVoiceprintRepository(tmp_path, "other/model")
        old.save(voiceprint(model="other/model"), consent())
        repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
        with pytest.raises(VoiceModelMismatchError) as caught:
            repo.get("ana")
        message = str(caught.value)
        assert "other/model" in message and MODEL in message
        assert "cannot be compared" in message
        assert isinstance(caught.value, InvalidInputError)
        # Left out of the candidates, still visible and forgettable.
        assert repo.all() == []
        assert repo.consent("ana") == consent()
        assert repo.delete("ana") is True


class TestUnsafeInput:
    @pytest.mark.parametrize("bad", ["../ana", ".ana", "a/b", "", "ana\n"])
    def test_ids(
        self, repo: FileSystemVoiceprintRepository, tmp_path: Path, bad: str
    ) -> None:
        with pytest.raises(InvalidInputError, match="Invalid speaker id"):
            repo.get(bad)
        with pytest.raises(InvalidInputError, match="Invalid speaker id"):
            repo.delete(bad)
        assert not (tmp_path / "voiceprints").exists()

    def test_consent_of_someone_else(
        self, repo: FileSystemVoiceprintRepository
    ) -> None:
        with pytest.raises(InvalidInputError, match="another speaker"):
            repo.save(voiceprint(), consent("luis"))

    def test_damaged_files(
        self, repo: FileSystemVoiceprintRepository, tmp_path: Path
    ) -> None:
        repo.save(voiceprint(), consent())
        path = tmp_path / "voiceprints" / "ana.json"
        path.write_text('{"voiceprint": {}}')
        with pytest.raises(StorageError, match="Damaged voiceprint"):
            repo.get("ana")
        repo.save(voiceprint("luis"), consent("luis"))
        path.write_text((tmp_path / "voiceprints" / "luis.json").read_text())
        with pytest.raises(StorageError, match="holds the voiceprint of luis"):
            repo.get("ana")

    def test_files_that_are_not_voiceprints_are_ignored(
        self, repo: FileSystemVoiceprintRepository, tmp_path: Path
    ) -> None:
        repo.save(voiceprint(), consent())
        (tmp_path / "voiceprints" / "not an id.json").write_text("{}")
        (tmp_path / "voiceprints" / ".ana.json.x.tmp").write_text("{")
        assert [v.speaker_id for v in repo.all()] == ["ana"]
