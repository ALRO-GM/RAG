"""End to end: the wired use cases, fake models, real storage on disk.

``build_container`` with ``runtime="fake"`` gives the real decoder,
repositories, job runner and publisher in a temporary directory, with
the deterministic stand-in models: a 220 Hz tone is one "person", an
880 Hz tone another.
"""

import json
import threading
from pathlib import Path
from typing import Dict

import jsonschema
import pytest

from tests.integration.transcriber.audio import (
    mp4_aac,
    pcm_bytes,
    scene,
    sound_file,
    tone,
)
from transcriber.adapters.outbound.fake_models import (
    FakeSpeechRecogniser,
    FakeVoiceEncoder,
)
from transcriber.adapters.outbound.fs_session_repository import (
    FileSystemSessionRepository,
)
from transcriber.adapters.outbound.wavlm_voice_encoder import (
    WavLMVoiceEncoder,
)
from transcriber.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
)
from transcriber.application.ports import Progress, RecordedAudio
from transcriber.application.transcribe import TranscriptionOptions
from transcriber.bootstrap import (
    TranscriberContainer,
    build_container,
    start_container,
)
from transcriber.config import TranscriberSettings
from transcriber.domain.errors import NotFoundError
from transcriber.domain.models import SessionState, Utterance

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "contracts" / "transcript.v1.schema.json"
WAIT = 30.0
CHUNK = 32000  # bytes: one second of 16 kHz 16-bit PCM


@pytest.fixture()
def container(tmp_path: Path) -> TranscriberContainer:
    """The transcriber with fake models over ``tmp_path``."""
    return build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))


def enrol_ana(container: TranscriberContainer) -> None:
    """Enrol Ana from 30 seconds of her (220 Hz) voice."""
    sample = sound_file(tone(30.0, hz=220.0), 16000, "wav")
    container.enrol_speaker(
        "ana", "Ana", "es", RecordedAudio(sample, "wav"), consent=True,
        consent_version="consent-1",
    )


def record(container: TranscriberContainer, session_id: str) -> None:
    """Record Ana, then someone else, then Ana, in one-second chunks."""
    data = pcm_bytes(scene([
        ("220", 6.0), ("-", 1.6), ("880", 6.0), ("-", 1.6), ("220", 4.8),
    ]))
    container.open_session(session_id, "es", title="Reparto")
    for number, start in enumerate(range(0, len(data), CHUNK)):
        assert container.append_audio(
            session_id, number, data[start:start + CHUNK]
        )
    # The phone retries the last chunk: accepted once only.
    assert not container.append_audio(
        session_id, number, data[start:start + CHUNK]
    )
    container.close_session(session_id)


def transcribe(container: TranscriberContainer, session_id: str) -> None:
    """Queue the session and wait for the runner to finish it."""
    container.queue_transcription(session_id, TranscriptionOptions())
    assert container.runner.wait(WAIT)


def test_record_transcribe_identify_publish_review(
    container: TranscriberContainer, tmp_path: Path
) -> None:
    enrol_ana(container)
    record(container, "equipo-a")
    folder = tmp_path / "data" / "sessions" / "equipo-a"
    assert (folder / "audio.flac").is_file()
    assert not (folder / "audio.pcm").exists()

    transcribe(container, "equipo-a")
    status = container.transcription_status("equipo-a")
    assert (status.state, status.stage, status.fraction) == (
        SessionState.DONE, "done", 1.0,
    )
    transcript = container.sessions.transcript("equipo-a")
    assert transcript is not None
    assert [(s.id, s.name, s.enrolled) for s in transcript.speakers] == [
        ("ana", "Ana", True), ("S1", None, False),
    ]
    turns: Dict[int, Utterance] = {}
    for utterance in transcript.utterances:
        turns.setdefault(utterance.turn, utterance)
    assert [(u.speaker, u.previous_speaker) for u in turns.values()] == [
        ("ana", None), ("S1", "ana"), ("ana", "S1"),
    ]
    assert transcript.session.duration == pytest.approx(20.0, abs=0.01)
    starts = [u.start for u in transcript.utterances]
    assert starts == sorted(starts)
    assert transcript.utterances[-1].end <= transcript.session.duration
    words = container.sessions.words("equipo-a")
    assert words and {w.speaker for w in words} == {"ana", "S1"}

    document = container.publish("equipo-a")
    published = tmp_path / "data" / "published"
    path = published / "equipo-a.transcript.v1.json"
    assert json.loads(path.read_text("utf-8")) == document
    schema = json.loads(SCHEMA.read_text("utf-8"))
    jsonschema.validate(document, schema)
    assert document["pipeline"] == {
        "recogniser": "fake/recogniser",
        "voice_model": "fake/voice-encoder",
        "transcriber": "0.1.0",
    }

    renamed = container.rename_speaker("equipo-a", "S1", name="Luis")
    assert renamed.speakers[1].name == "Luis"
    stored = container.sessions.transcript("equipo-a")
    assert stored is not None and stored.speakers[1].name == "Luis"

    container.forget_speaker("ana")
    assert container.voiceprints.get("ana") is None
    container.delete_session("equipo-a")
    assert not folder.exists()


def test_an_imported_mp4_is_transcribed(
    container: TranscriberContainer,
) -> None:
    samples = scene([("220", 4.0), ("-", 1.6), ("880", 4.0)], rate=44100)
    container.import_recording(
        "import", "es", RecordedAudio(mp4_aac(samples), "mp4")
    )
    transcribe(container, "import")
    transcript = container.sessions.transcript("import")
    assert transcript is not None
    assert [s.id for s in transcript.speakers] == ["S1", "S2"]
    assert all(not s.enrolled for s in transcript.speakers)


def test_a_failing_job_leaves_the_session_failed(
    container: TranscriberContainer,
) -> None:
    container.import_recording(
        "broken", "es", RecordedAudio(b"not audio at all" * 20, "m4a")
    )
    transcribe(container, "broken")
    status = container.transcription_status("broken")
    assert status.state is SessionState.FAILED
    session = container.sessions.get("broken")
    assert session is not None and session.failure is not None
    assert session.failure.startswith("decode: Cannot decode the m4a")
    assert container.runner.error("broken") is not None


def test_a_session_deleted_while_queued_stops_its_job_quietly(
    container: TranscriberContainer, tmp_path: Path
) -> None:
    """A queued session may be deleted; its job then does nothing."""
    busy = threading.Event()
    release = threading.Event()

    def hold(progress: Progress) -> None:
        busy.set()
        release.wait(WAIT)

    container.runner.submit("other", hold)
    assert busy.wait(WAIT)
    wav = sound_file(tone(2.0), 16000, "wav")
    container.import_recording("gone", "es", RecordedAudio(wav, "wav"))
    container.queue_transcription("gone")
    container.delete_session("gone")
    release.set()
    assert container.runner.wait(WAIT)
    status = container.runner.status("gone")
    assert status is not None and status.message is not None
    assert "deleted or queued again" in status.message
    assert container.runner.error("gone") is None
    assert not (tmp_path / "data" / "sessions" / "gone").exists()
    with pytest.raises(NotFoundError):
        container.transcription_status("gone")


def test_build_container_wires_real_models_without_loading_them(
    tmp_path: Path,
) -> None:
    settings = TranscriberSettings(data_dir=tmp_path, device="cpu")
    container = build_container(settings)
    assert isinstance(container.recogniser, WhisperRecogniser)
    assert isinstance(container.encoder, WavLMVoiceEncoder)
    assert container.recogniser.model_name == settings.recogniser
    assert container.encoder.model_name == settings.voice_model
    assert not container.recogniser.loaded and not container.encoder.loaded
    assert isinstance(container.sessions, FileSystemSessionRepository)
    container.release_models()
    assert list(tmp_path.iterdir()) == []  # building touched no file
    assert settings.published() == tmp_path / "published"
    other = TranscriberSettings(publish_dir=tmp_path / "rag")
    assert other.published() == tmp_path / "rag"


def test_fake_runtime(container: TranscriberContainer) -> None:
    assert isinstance(container.recogniser, FakeSpeechRecogniser)
    assert isinstance(container.encoder, FakeVoiceEncoder)
    container.release_models()


def test_enrolment_runs_on_the_job_thread_and_releases_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enrolment shares the one queue with transcriptions."""
    threads: Dict[str, str] = {}
    released: Dict[str, int] = {"recogniser": 0, "encoder": 0}
    recognise = FakeSpeechRecogniser.recognise

    def watched(self: FakeSpeechRecogniser, *args: object,
                **kwargs: object) -> object:
        threads["recognise"] = threading.current_thread().name
        return recognise(self, *args, **kwargs)  # type: ignore[arg-type]

    def counted(kind: str) -> object:
        def release(self: object) -> None:
            released[kind] += 1
        return release

    monkeypatch.setattr(FakeSpeechRecogniser, "recognise", watched)
    monkeypatch.setattr(FakeSpeechRecogniser, "release",
                        counted("recogniser"))
    monkeypatch.setattr(FakeVoiceEncoder, "release", counted("encoder"))
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    enrol_ana(container)
    assert threads == {"recognise": "transcriber-jobs"}
    assert released == {"recogniser": 1, "encoder": 1}
    assert container.voiceprints.get("ana") is not None
    container.runner.close(WAIT)


def test_start_up_skips_a_damaged_session(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = TranscriberSettings(data_dir=tmp_path / "data",
                                   runtime="fake")
    first = build_container(settings)
    first.open_session("ok", "es")
    first.open_session("broken", "es")
    (tmp_path / "data" / "sessions" / "broken" / "session.json"
     ).write_text("{", encoding="utf-8")
    with caplog.at_level("WARNING"):
        container = start_container(settings)
    assert "broken" in caplog.text
    assert container.sessions.get("ok") is not None
