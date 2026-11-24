"""Withdrawing clone consent deletes the demo clips cloned from the voice
(#254), through the same hook that deletes the reference clip, over the
files both services share."""

import json
import os
import stat
from pathlib import Path
from typing import Any, List

import pytest

import numpy as np
from fastapi.testclient import TestClient

from orchestrator.adapters.outbound.demo_speaker import ToneDemoSpeaker
from orchestrator.adapters.outbound.fs_demo_voices import FsDemoVoices
from orchestrator.adapters.outbound.fs_demos import DirDemos
from orchestrator.adapters.outbound.fs_voice_profiles import DirVoiceProfiles
from orchestrator.application.demo_voices import (
    DemoVoiceError,
    DemoVoiceLibrary,
    RenderDemoVoices,
)
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
from tests.unit.orchestrator.test_demo_voices import DEMOS, STOCK, Gate
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.outbound.fs_demo_voices import (
    forget_cloned_demo_clips,
)
from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)

CALL = "llamada-lumbre"


def enrolled(repo: FileSystemVoiceprintRepository, person: str) -> None:
    """``person`` enrolled with clone consent and a reference clip."""
    repo.save(voiceprint(speaker_id=person, model=repo.model),
              consent(speaker_id=person).model_copy(
                  update={"voice_clone": True}))
    repo.save_clone_reference(person, np.zeros(1600, dtype=np.float32),
                              16000)


def render(data: Path) -> DemoVoiceLibrary:
    """Render the call with ``ana`` cloned for the customer."""
    store = FsDemoVoices(data / "demo-voices")
    profiles = DirVoiceProfiles(data / "voiceprints")
    RenderDemoVoices(DirDemos(DEMOS), store, ToneDemoSpeaker(), profiles,
                     Gate()).render(CALL, {**STOCK, "cliente": "clone:ana"})
    return DemoVoiceLibrary(DirDemos(DEMOS), store, profiles)


def manifest(data: Path) -> "dict[str, object]":
    path = data / "demo-voices" / CALL / "manifest.json"
    found: "dict[str, object]" = json.loads(path.read_text("utf-8"))
    return found


def test_withdrawing_deletes_her_clips_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
    enrolled(repo, "ana")
    library = render(tmp_path)
    assert library.view(CALL)["kinds"] == ["clone", "stock"]
    folder = tmp_path / "demo-voices" / CALL
    assert len(list(folder.glob("*.wav"))) == 27
    assert repo.delete_clone_reference("ana") is True
    clips = manifest(tmp_path)["clips"]
    assert isinstance(clips, dict) and len(clips) == 27 - 13
    assert all(c["cloned_from"] is None for c in clips.values())
    assert len(list(folder.glob("*.wav"))) == 14
    assert stat.S_IMODE(os.stat(folder / "manifest.json").st_mode) == 0o600
    view = library.view(CALL)
    assert view["examples"] == {
        "call": {"available": False, "clips": {}},
        "voicebot": {"available": False, "clips": {}}}


def test_forgetting_the_person_deletes_them_too(tmp_path: Path) -> None:
    repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
    enrolled(repo, "ana")
    render(tmp_path)
    assert repo.delete("ana") is True
    assert len(list((tmp_path / "demo-voices" / CALL).glob("*.wav"))) == 14


def test_withdrawing_over_http_reaches_the_demo_clips(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    try:
        repo = container.voiceprints
        assert isinstance(repo, FileSystemVoiceprintRepository)
        enrolled(repo, "ana")
        render(repo.root.parent)
        client = TestClient(create_app(container, CODE), headers=AUTH)
        assert client.delete("/api/speakers/ana/voice-clone").status_code \
            == 204
        clips = manifest(repo.root.parent)["clips"]
        assert isinstance(clips, dict) and len(clips) == 14
    finally:
        container.runner.close(5.0)


def test_withdrawing_while_a_render_runs_leaves_none_of_her_clips(
    tmp_path: Path,
) -> None:
    """Real store, real hook: she withdraws while the stock lines render."""
    repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
    enrolled(repo, "ana")
    store = FsDemoVoices(tmp_path / "demo-voices")
    profiles = DirVoiceProfiles(tmp_path / "voiceprints")
    withdrawn: List[bool] = []

    class Withdraws(ToneDemoSpeaker):
        def speak(self, text: str, voice: Any, reference: Any,
                  language: str) -> Any:
            if voice.person is None and not withdrawn:
                withdrawn.append(repo.delete_clone_reference("ana"))
            return super().speak(text, voice, reference, language)

    with pytest.raises(DemoVoiceError):
        RenderDemoVoices(DirDemos(DEMOS), store, Withdraws(), profiles,
                         Gate()).render(CALL, {**STOCK,
                                               "cliente": "clone:ana"})
    assert withdrawn == [True]
    folder = tmp_path / "demo-voices" / CALL
    found = store.manifest(CALL)
    named = {c.file for c in found.clips.values()} if found else set()
    assert all(c.cloned_from != "ana"
               for c in (found.clips.values() if found else []))
    assert {p.name for p in folder.glob("*.wav")} <= named
    assert not list(folder.glob(".part-*"))


def test_files_no_manifest_names_go_on_withdrawal(tmp_path: Path) -> None:
    repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
    enrolled(repo, "ana")
    render(tmp_path)
    folder = tmp_path / "demo-voices" / CALL
    (folder / "t0.s0-killedrun.wav").write_bytes(b"RIFF")
    (folder / ".part-x.wav").write_bytes(b"RI")
    other = tmp_path / "demo-voices" / "otra"
    other.mkdir()
    (other / "x.wav").write_bytes(b"RIFF")  # no manifest at all
    repo.delete_clone_reference("ana")
    assert not (folder / "t0.s0-killedrun.wav").exists()
    assert not (folder / ".part-x.wav").exists()
    assert not (other / "x.wav").exists()
    assert len(list(folder.glob("*.wav"))) == 14


def test_the_withdrawal_completes_when_the_clean_up_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transcriber.adapters.outbound import fs_voiceprint_repository as fs
    from transcriber.application.enrol import WithdrawVoiceClone
    from transcriber.domain.errors import StorageError

    repo = FileSystemVoiceprintRepository(tmp_path, MODEL)
    enrolled(repo, "ana")

    def broken(folder: Path, speaker_id: str) -> int:
        raise PermissionError("read-only")

    monkeypatch.setattr(fs, "forget_cloned_demo_clips", broken)
    clock = type("C", (), {"now": staticmethod(lambda: NOW)})()
    with pytest.raises(StorageError):
        WithdrawVoiceClone(repo, clock)("ana")
    assert not (tmp_path / "voiceprints" / "ana.clone.wav").exists()
    record = repo.consent("ana")
    assert record is not None and record.voice_clone is False
    with pytest.raises(StorageError):
        repo.delete("ana")
    assert repo.get("ana") is None, "forgotten all the same"


def test_other_people_and_odd_files_are_left_alone(tmp_path: Path) -> None:
    folder = tmp_path / "demo-voices"
    assert forget_cloned_demo_clips(folder, "ana") == 0
    (folder / "broken").mkdir(parents=True)
    (folder / "broken" / "manifest.json").write_text("{", encoding="utf-8")
    (folder / "x").mkdir()
    (folder / "x" / "keep.wav").write_bytes(b"RIFF")
    (folder / "x" / "manifest.json").write_text(json.dumps({"clips": {
        "a": {"file": "../keep.wav", "cloned_from": "ana"},
        "b": {"file": "keep.wav", "cloned_from": "luis"},
    }}), encoding="utf-8")
    assert forget_cloned_demo_clips(folder, "ana") == 0
    assert (folder / "x" / "keep.wav").exists()
    kept = json.loads((folder / "x" / "manifest.json").read_text("utf-8"))
    assert list(kept["clips"]) == ["b"]
    (folder / "y").mkdir()
    (folder / "y" / "manifest.json").write_text('{"clips": []}', "utf-8")
    assert forget_cloned_demo_clips(folder, "ana") == 0
