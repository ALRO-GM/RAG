"""Consent to voice cloning on disk and over HTTP (#207): the clip lives
beside the voiceprint only while the consent stands, and the
orchestrator reads both files as the transcriber writes them."""

import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from orchestrator.adapters.outbound.fs_voice_profiles import DirVoiceProfiles
from tests.integration.transcriber.test_http_api import (
    AUTH,
    CODE,
    make_container,
)
from tests.integration.transcriber.test_voiceprint_repository import (
    MODEL,
    NOW,
    consent,
    voiceprint,
)
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)


def test_the_clip_is_written_read_by_the_orchestrator_and_deleted(
    tmp_path: Path,
) -> None:
    repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
    repo.save(voiceprint().model_copy(update={"words_per_second": 3.2}),
              consent().model_copy(update={"voice_clone": True}))
    samples = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
    repo.save_clone_reference("ana", samples, 16000)
    clip = tmp_path / "voiceprints" / "ana.clone.wav"
    with wave.open(str(clip), "rb") as reader:
        assert (reader.getframerate(), reader.getnframes(),
                reader.getsampwidth()) == (16000, 1600, 2)
    assert [v.speaker_id for v in repo.all()] == ["ana"], "clip not listed"
    profile = DirVoiceProfiles(tmp_path / "voiceprints").get("ana")
    assert profile is not None and profile.clonable
    assert profile.words_per_second == 3.2
    assert repo.delete_clone_reference("ana") is True
    assert repo.delete_clone_reference("ana") is False
    repo.save_clone_reference("ana", samples, 16000)
    assert repo.delete("ana") is True
    assert not clip.exists(), "forgetting a person deletes the clip too"


def test_withdrawing_over_http_deletes_the_clip(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    try:
        client = TestClient(create_app(container, CODE), headers=AUTH)
        repo = container.voiceprints
        assert isinstance(repo, FileSystemVoiceprintRepository)
        repo.save(voiceprint(model=repo.model),
                  consent().model_copy(update={"voice_clone": True}))
        repo.save_clone_reference(
            "ana", np.zeros(160, dtype=np.float32), 16000)
        listed = client.get("/api/speakers").json()["speakers"]
        assert listed[0]["voice_clone"] is True
        assert client.delete("/api/speakers/ana/voice-clone").status_code \
            == 204
        record = repo.consent("ana")
        assert record is not None and record.voice_clone is False
        assert record.voice_clone_withdrawn_at is not None
        assert record.voice_clone_withdrawn_at >= NOW
        assert client.get("/api/speakers").json()["speakers"][0][
            "voice_clone"] is False
        assert client.delete("/api/speakers/nobody/voice-clone"
                             ).status_code == 404
        profile = DirVoiceProfiles(repo.root).get("ana")
        assert profile is not None and not profile.clonable
        assert profile.reference is None
    finally:
        container.runner.close(5.0)
