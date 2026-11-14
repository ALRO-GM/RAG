"""Reference corpora over real adapters: parquet shards, storage, CLI.

A tiny fake corpus (two parquet shards of 10-second WAVs) stands in for
the Hub: the download is a copy, the listing a fixed list, so nothing
touches the network.
"""

import dataclasses
import io
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import GatedRepoError

from tests.integration.transcriber.audio import scene, sound_file
from tests.integration.transcriber.test_lab_api import settings
from tests.unit.transcriber.test_speech_check import band_noise
from transcriber.adapters.inbound.cli import Output, main
from transcriber.adapters.inbound.lab_cli import LabCommands
from transcriber.adapters.outbound.fs_lab_repository import (
    FileSystemLabRepository,
)
from transcriber.adapters.outbound.hf_reference_corpus import (
    HuggingFaceCorpus,
    _sniff,
)
from transcriber.application.lab_reference import ImportReference
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.domain.errors import CorpusAccessError, StorageError
from transcriber.domain.lab import LabRequest, Segment
from transcriber.domain.lab_reference import CORPORA, make_reference

AMI = CORPORA["ami"]
CALLHOME = CORPORA["callhome"]
WAIT = 30.0


def wav(seconds: float = 10.0) -> bytes:
    """A WAV of two tones, ``seconds`` long, at 8 kHz (resampled later)."""
    half = seconds / 2
    return sound_file(scene([("220", half), ("880", half)], 8000), 8000,
                      "wav")


def noise_wav(seconds: float = 10.0) -> bytes:
    """A WAV of steady noise cut above 4 kHz, at 16 kHz (no speech)."""
    return sound_file(band_noise(seconds), 16000, "wav")


def shard(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a parquet shard with the corpus's columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "audio": [r["audio"] for r in rows],
        "timestamps_start": [r["starts"] for r in rows],
        "timestamps_end": [r["ends"] for r in rows],
        "speakers": [r["speakers"] for r in rows],
    }), path)


def row(name: Any, data: bytes) -> Dict[str, Any]:
    """One recording: A 0-4 s, B 3-8 s (1 s of overlap)."""
    return {"audio": {"bytes": data, "path": name},
            "starts": [0.0, 3.0], "ends": [4.0, 8.0],
            "speakers": ["A", "B"]}


class FakeHub:
    """A hub that lists fixed shards and copies them on download."""

    def __init__(self, root: Path, config: str, split: str,
                 noisy: Sequence[int] = ()) -> None:
        """Write two shards of ``config``/``split`` under ``root``.

        The recordings numbered in ``noisy`` hold noise, not speech.
        """
        self.root = root
        self.downloads: List[str] = []
        self.files = [f"{config}/{split}-00000-of-00002.parquet",
                      f"{config}/{split}-00001-of-00002.parquet",
                      f"{config}/other-00000-of-00001.parquet",
                      "README.md"]
        name = (lambda n: None) if config == "spa" else (
            lambda n: f"REC{n}.Array1-01.wav"
        )
        audio = [noise_wav() if n in noisy else wav() for n in range(3)]
        shard(root / self.files[0], [row(name(0), audio[0]),
                                     row(name(1), audio[1])])
        shard(root / self.files[1], [row(name(2), audio[2])])

    def list_files(self, repo: str) -> List[str]:
        """Return the fixed listing."""
        return list(self.files)

    def download(self, repo: str, filename: str, directory: Path) -> Path:
        """Copy the shard into ``directory`` as the Hub would."""
        self.downloads.append(filename)
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.root / filename, target)
        return target


def gated(repo: str) -> None:
    """The Hub refusing a gated repo whose terms were not accepted."""
    error = GatedRepoError.__new__(GatedRepoError)
    error.args = ("403 Client Error: Cannot access gated repo",)
    raise error


def corpus(tmp_path: Path, hub: FakeHub, **kw: Any) -> HuggingFaceCorpus:
    """The adapter over the fake hub."""
    return HuggingFaceCorpus(
        tmp_path / "corpora", hub.list_files, hub.download,
        kw.get("check", lambda repo: None),
    )


class TestHuggingFaceCorpus:
    def test_reads_only_the_shards_it_needs(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub", "sdm", "test")
        reader = corpus(tmp_path, hub)
        got = list(reader.recordings(AMI, "test", 2))
        assert [r.recording for r in got] == ["REC0", "REC1"]
        assert hub.downloads == ["sdm/test-00000-of-00002.parquet"]
        first = got[0]
        assert first.audio.format == "wav"
        assert first.audio.data[:4] == b"RIFF"
        assert (list(first.starts), list(first.ends),
                list(first.speakers)) == ([0.0, 3.0], [4.0, 8.0],
                                          ["A", "B"])
        assert first.words is None
        assert (tmp_path / "corpora" / "ami" / hub.files[0]).is_file()
        every = list(reader.recordings(AMI, "test", None))
        assert [r.recording for r in every] == ["REC0", "REC1", "REC2"]
        assert len(hub.downloads) == 3

    def test_nameless_rows_are_numbered(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub", "spa", "data")
        got = list(corpus(tmp_path, hub).recordings(CALLHOME, "data", 3))
        assert [r.recording for r in got] == [
            "data-0001", "data-0002", "data-0003",
        ]
        assert got[0].audio.format == "wav"

    def test_a_gated_corpus_says_where_to_accept(
        self, tmp_path: Path
    ) -> None:
        hub = FakeHub(tmp_path / "hub", "spa", "data")
        reader = corpus(tmp_path, hub, check=gated)
        with pytest.raises(CorpusAccessError) as caught:
            list(reader.recordings(CALLHOME, "data", 1))
        message = str(caught.value)
        assert "gated" in message and CALLHOME.source_url in message
        assert "huggingface-cli login" in message
        assert hub.downloads == []

        def refused(repo: str, name: str, directory: Path) -> Path:
            """A download refused because the repo is gated."""
            gated(repo)
            return directory

        late = HuggingFaceCorpus(tmp_path / "c2", hub.list_files, refused,
                                 lambda repo: None)
        with pytest.raises(CorpusAccessError, match="accept its terms"):
            list(late.recordings(CALLHOME, "data", 1))

    def test_offline_uses_downloaded_shards(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub", "sdm", "test")
        list(corpus(tmp_path, hub).recordings(AMI, "test", 1))

        def offline(repo: str) -> List[str]:
            """No network."""
            raise OSError("offline")

        reader = HuggingFaceCorpus(tmp_path / "corpora", offline,
                                   hub.download)
        assert len(list(reader.recordings(AMI, "test", None))) == 2
        empty = HuggingFaceCorpus(tmp_path / "none", offline, hub.download)
        with pytest.raises(CorpusAccessError, match="offline"):
            list(empty.recordings(AMI, "test", None))
        with pytest.raises(CorpusAccessError, match="no shard"):
            list(corpus(tmp_path, hub).recordings(AMI, "validation", 1))

    def test_a_gated_corpus_offline_reads_its_shards(
        self, tmp_path: Path
    ) -> None:
        hub = FakeHub(tmp_path / "hub", "spa", "data")
        list(corpus(tmp_path, hub).recordings(CALLHOME, "data", 1))

        def offline(*args: Any) -> Any:
            """No network for any Hub call."""
            raise OSError("offline")

        reader = HuggingFaceCorpus(tmp_path / "corpora", offline, offline,
                                   offline)
        got = list(reader.recordings(CALLHOME, "data", 5))
        assert [r.recording for r in got] == ["data-0001", "data-0002"]
        empty = HuggingFaceCorpus(tmp_path / "none", offline, offline,
                                  offline)
        with pytest.raises(CorpusAccessError, match="offline"):
            list(empty.recordings(CALLHOME, "data", 1))
        refused = HuggingFaceCorpus(tmp_path / "corpora", offline, offline,
                                    gated)
        with pytest.raises(CorpusAccessError, match="accept its terms"):
            list(refused.recordings(CALLHOME, "data", 1))

    def test_a_row_without_audio(self, tmp_path: Path) -> None:
        hub = FakeHub(tmp_path / "hub", "sdm", "test")
        shard(tmp_path / "hub" / hub.files[0], [row("x.wav", b"")])
        with pytest.raises(CorpusAccessError, match="no audio"):
            list(corpus(tmp_path, hub).recordings(AMI, "test", 1))

    def test_format_from_the_bytes(self) -> None:
        assert _sniff(b"fLaC") == "flac" and _sniff(b"OggS") == "ogg"
        assert _sniff(b"ID3\x04") == "mp3" and _sniff(b"\xff\xfb\x90\x00") \
            == "mp3"
        assert _sniff(b"RIFF") == "wav" and _sniff(b"????") == "wav"


def test_the_reference_is_stored_with_the_session(tmp_path: Path) -> None:
    repository = FileSystemLabRepository(tmp_path)
    assert repository.reference("s1") is None
    reference = make_reference(
        AMI, "test", "R", 10.0,
        [Segment(start=0.0, end=4.0, speaker="A")], None,
    )
    repository.save_reference("s1", reference)
    assert repository.reference("s1") == reference
    (tmp_path / "lab" / "s1" / "reference.json").write_text("[1]")
    with pytest.raises(StorageError):
        repository.reference("s1")
    repository.delete_session("s1")
    assert repository.reference("s1") is None


# ------------------------------------------------------------ end to end


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """The service's container with fake models and the sandbox lab."""
    built = build_container(settings(tmp_path))
    yield built
    built.runner.close(WAIT)


def with_fake_hub(
    container: TranscriberContainer, tmp_path: Path,
    noisy: Sequence[int] = (),
) -> LabCommands:
    """Return the lab commands with the import reading the fake hub."""
    lab = container.lab
    assert lab is not None
    hub = FakeHub(tmp_path / "hub", "sdm", "test", noisy)
    importer = ImportReference(
        container.sessions, lab.repository, corpus(tmp_path, hub),
        container.decoder, container.import_recording, container.locks,
        container.clock,
    )
    lab = dataclasses.replace(lab, import_reference=importer)
    return LabCommands(lab, Output(io.StringIO(), io.StringIO()))


def text_of(commands: LabCommands) -> str:
    """Return what the commands printed so far, and forget it."""
    out = commands.output.out
    assert isinstance(out, io.StringIO)
    value = out.getvalue()
    out.seek(0)
    out.truncate()
    return value


def test_import_run_and_summary(
    tmp_path: Path, container: TranscriberContainer
) -> None:
    commands = with_fake_hub(container, tmp_path)
    assert commands.import_reference("ami", None, 2, False) == 0
    printed = text_of(commands)
    assert "ami-REC0: 10.0 s, 2 speakers, 2 turns, no transcript" in printed
    assert "CC BY 4.0" in printed
    assert commands.import_reference("ami", None, 1, True) == 0
    data = json.loads(text_of(commands))
    assert data["recordings"][0]["state"] == "exists"
    session = container.sessions.get("ami-REC0")
    assert session is not None and session.duration == pytest.approx(10.0)
    lab = commands.lab
    run = lab.start("ami-REC0", LabRequest(
        diarizers=["fake-diarize"], recognisers=["fake-asr"],
    ))
    deadline = time.monotonic() + WAIT
    while lab.status("ami-REC0", run.id).state in ("queued", "running"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert commands.summary("ami-REC0", run.id, False) == 0
    printed = text_of(commands)
    assert "Against the ami reference REC0 (CC BY 4.0)" in printed
    assert "none+fake-diarize+fake-asr |" in printed
    block = lab.results("ami-REC0", run.id)["reference"]
    track = block["tracks"][0]
    # fake-diarize: A 0-5.5, B 4.5-10 against A 0-4, B 3-8
    assert track["track"] == "none+fake-diarize"
    assert 0.0 < track["der"] < 1.0
    assert commands.reference_summary(None, False) == 0
    printed = text_of(commands)
    assert "2 imported recordings (ami); 1 without a scored run." in printed
    assert "none+fake-diarize | 1 |" in printed
    assert commands.reference_summary("ami", True) == 0
    summary = json.loads(text_of(commands))
    assert summary["unscored"] == ["ami-REC1"]


def run_cli(tmp_path: Path, *argv: str) -> "tuple[int, str]":
    """Run the CLI with the sandbox settings; return code and output."""
    config = tmp_path / "config.json"
    config.write_text(settings(tmp_path).model_dump_json())
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), {"TRANSCRIBER_CONFIG": str(config)}, out, err)
    return code, out.getvalue()


def test_noise_is_skipped_checked_and_excluded(
    tmp_path: Path, container: TranscriberContainer
) -> None:
    commands = with_fake_hub(container, tmp_path, noisy=(0,))
    assert commands.import_reference("ami", None, 1, False) == 0
    printed = text_of(commands)
    assert "ami-REC0: skipped: no speech in the audio (lag-1" in printed
    assert "ami-REC1: 10.0 s, 2 speakers" in printed
    assert "1 usable recordings of AMI" in printed
    assert "1 without speech." in printed
    assert container.sessions.get("ami-REC0") is None
    assert container.sessions.get("ami-REC2") is None
    assert commands.import_reference("ami", None, 1, True) == 0
    data = json.loads(text_of(commands))["recordings"]
    assert [(d["state"], d["reason"]) for d in data] == [
        ("skipped", "no speech in the audio"), ("exists", None),
    ]
    assert data[0]["speech"]["flatness"] > 0.2
    # the check again, over the stored audio (no download)
    code, text = run_cli(tmp_path, "lab", "import-reference", "ami",
                         "--check")
    assert code == 0, text
    assert "ami-REC1: speech (lag-1 autocorrelation" in text
    assert "1 sessions of AMI (single distant microphone) checked;" \
        " 0 marked unusable" in text
    # a session found without speech is left out of the summary
    lab = commands.lab
    reference = lab.repository.reference("ami-REC1")
    assert reference is not None and reference.speech is not None
    lab.repository.save_reference("ami-REC1", reference.model_copy(update={
        "speech": reference.speech.model_copy(update={"speech": False}),
    }))
    code, text = run_cli(tmp_path, "lab", "import-reference", "ami",
                         "--check", "--json")
    checked = json.loads(text)["recordings"]
    assert [(d["session_id"], d["state"]) for d in checked] == [
        ("ami-REC1", "checked"),
    ]
    lab.repository.save_reference("ami-REC1", reference.model_copy(update={
        "speech": reference.speech.model_copy(update={"speech": False}),
    }))
    assert commands.reference_summary(None, False) == 0
    printed = text_of(commands)
    assert "0 imported recordings (none); 0 without a scored run." in printed
    assert "1 excluded: no speech in the audio." in printed
    assert commands.check_reference("ami", False) == 0
    assert "ami-REC1: speech" in text_of(commands)
    lab.repository.save_reference("ami-REC1", reference.model_copy(update={
        "speech": reference.speech.model_copy(update={"speech": False}),
    }))
    commands.import_reference("ami", None, 1, False)
    assert "ami-REC1: unusable: no speech in the audio" in text_of(commands)
