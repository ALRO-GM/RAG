"""The search for speakers' names over real adapters (#102).

The local model adapter with its memory reader and loader replaced, the
record on disk, the settings, the composition root and the HTTP routes
(fake runtime, a transcript written to the real repository, a fake chat
model put in place of the local one).
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.unit.transcriber.names_fakes import (
    NOW,
    FakeChatModel,
    conversation,
    judged,
    marked,
    reply,
)
from transcriber.adapters.inbound.http_api import create_app, names_json
from transcriber.adapters.outbound.fs_session_repository import (
    NAMES_FILE,
    FileSystemSessionRepository,
)
from transcriber.adapters.outbound.local_name_model import (
    ALLOWED_MODELS,
    LocalNameModel,
    check_models,
    memory_gb,
)
from transcriber.bootstrap import (
    TranscriberContainer,
    build_container,
    build_name_model,
)
from transcriber.config import SpeakerNameSettings, TranscriberSettings
from transcriber.domain.errors import ModelLoadError
from transcriber.domain.speaker_names import AutoName, SpeakerNames

CODE = "a-long-enough-access-code"
WAIT = 30.0


class Backend:
    """A loaded model that answers a fixed text."""

    def __init__(self, name: str, device: str, bits: int) -> None:
        """Remember how it was loaded."""
        self.loaded = (name, device, bits)
        self.released = False

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Answer with the model's name."""
        return f"{self.loaded[0]}:{max_tokens}"

    def release(self) -> None:
        """Count the release."""
        self.released = True


class Gpu:
    """Free memory that changes as the test says."""

    def __init__(self, free: List[float], total: float = 8.2) -> None:
        """Hand out ``free`` in order (the last one stays)."""
        self.free = list(free)
        self.total = total
        self.looks = 0

    def __call__(self) -> Optional[Tuple[float, float]]:
        """Return the next reading."""
        self.looks += 1
        value = self.free.pop(0) if len(self.free) > 1 else self.free[0]
        return value, self.total


def model(
    gpu: Callable[[], Optional[Tuple[float, float]]],
    device: str = "cuda",
    quantisable: bool = True,
    wait: float = 60.0,
) -> Tuple[LocalNameModel, List[Backend], List[float]]:
    """A model adapter over fakes; the loads and the sleeps it made."""
    loads: List[Backend] = []
    sleeps: List[float] = []
    clock = {"now": 0.0}

    def load(name: str, where: str, bits: int) -> Backend:
        loads.append(Backend(name, where, bits))
        return loads[-1]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    adapter = LocalNameModel(
        device="auto", wait_seconds=wait, poll_seconds=15.0, memory=gpu,
        resolve=lambda _: device, load=load,
        quantisable=lambda: quantisable, sleep=sleep,
        clock=lambda: clock["now"],
    )
    return adapter, loads, sleeps


class TestLocalNameModel:
    def test_the_8b_when_it_fits(self) -> None:
        adapter, loads, sleeps = model(Gpu([7.5]))
        assert adapter.model_name == "Qwen/Qwen3-8B"
        assert adapter.complete("s", "u", 32) == "Qwen/Qwen3-8B:32"
        adapter.complete("s", "u", 32)
        assert [b.loaded for b in loads] == [("Qwen/Qwen3-8B", "cuda", 4)]
        assert adapter.model_name == "Qwen/Qwen3-8B@4bit"
        assert sleeps == []
        adapter.release()
        assert loads[0].released and adapter.model_name == "Qwen/Qwen3-8B"
        adapter.release()

    def test_the_4b_when_only_it_fits(self) -> None:
        adapter, loads, _ = model(Gpu([4.0]))
        adapter.complete("s", "u", 8)
        assert loads[0].loaded == ("Qwen/Qwen3-4B", "cuda", 4)

    def test_waits_for_the_last_then_loads_it(self) -> None:
        gpu = Gpu([1.0, 1.5, 3.5])
        adapter, loads, sleeps = model(gpu)
        adapter.complete("s", "u", 8)
        assert sleeps == [15.0, 15.0]
        assert loads[0].loaded == ("Qwen/Qwen3-4B", "cuda", 4)

    def test_gives_up_after_waiting_and_never_takes_a_smaller_one(
        self,
    ) -> None:
        adapter, loads, sleeps = model(Gpu([1.0]), wait=30.0)
        with pytest.raises(ModelLoadError, match="No smaller model"):
            adapter.complete("s", "u", 8)
        assert loads == [] and sleeps == [15.0, 15.0]

    def test_without_4_bits_an_8_gb_card_is_too_small(self) -> None:
        adapter, loads, sleeps = model(Gpu([7.9]), quantisable=False)
        with pytest.raises(ModelLoadError, match="local-judge"):
            adapter.complete("s", "u", 8)
        assert loads == [] and sleeps == []

    def test_no_gpu_reading(self) -> None:
        adapter, _, _ = model(lambda: None)
        with pytest.raises(ModelLoadError, match="0.0 GB in all"):
            adapter.complete("s", "u", 8)

    def test_on_the_cpu_the_last_model_in_full_precision(self) -> None:
        adapter, loads, _ = model(Gpu([0.0]), device="cpu")
        adapter.complete("s", "u", 8)
        assert loads[0].loaded == ("Qwen/Qwen3-4B", "cpu", 16)
        assert adapter.model_name == "Qwen/Qwen3-4B"

    def test_only_the_allowed_models(self) -> None:
        assert ALLOWED_MODELS == ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
        assert check_models(["Qwen/Qwen3-4B"]) == ("Qwen/Qwen3-4B",)
        for wrong in (["Qwen/Qwen3-1.7B"], []):
            with pytest.raises(ValueError):
                check_models(wrong)
        with pytest.raises(ValueError):
            LocalNameModel(["Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B"])

    def test_memory_estimates(self) -> None:
        assert memory_gb("Qwen/Qwen3-8B", 4) == 6.31
        assert memory_gb("Qwen/Qwen3-4B", 16) == pytest.approx(10.4)


class TestSettings:
    def test_defaults(self) -> None:
        names = TranscriberSettings().names
        assert names.enabled and names.models == ALLOWED_MODELS
        assert names.search.auto_confidence == 0.8

    def test_a_smaller_model_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names.models"):
            SpeakerNameSettings(models=("Qwen/Qwen3-1.7B",))
        with pytest.raises(ValidationError):
            SpeakerNameSettings(models=())

    def test_the_composition_root(self, tmp_path: Path) -> None:
        fake = TranscriberSettings(data_dir=tmp_path, runtime="fake")
        assert build_name_model(fake) is None
        real = TranscriberSettings(
            data_dir=tmp_path, device="cpu",
            names=SpeakerNameSettings(models=("Qwen/Qwen3-4B",),
                                      wait_seconds=5),
        )
        built = build_name_model(real)
        assert built is not None
        assert (built.models, built.device, built.wait_seconds) == (
            ("Qwen/Qwen3-4B",), "cpu", 5,
        )


class TestRecordOnDisk:
    def test_round_trip_and_removal_with_the_session(
        self, tmp_path: Path
    ) -> None:
        repo = FileSystemSessionRepository(tmp_path)
        transcript = conversation([("S1", "Soy Ana.")])
        repo.create(transcript.session)
        assert repo.names("s1") is None
        record = SpeakerNames(
            session="s1", searched_at=NOW, model="m", auto=[AutoName(
                speaker="S1", name="Ana", quote="Soy Ana.",
                utterance="t0.s0", start=0.0, confidence=0.9,
            )],
        )
        repo.save_names(record)
        assert repo.names("s1") == record
        assert (repo.root / "s1" / NAMES_FILE).is_file()
        repo.delete("s1")
        assert not (repo.root / "s1").exists()

    def test_a_damaged_record_is_a_storage_error(self, tmp_path: Path) -> None:
        from transcriber.domain.errors import StorageError

        repo = FileSystemSessionRepository(tmp_path)
        repo.create(conversation([("S1", "Hola.")]).session)
        (repo.root / "s1" / NAMES_FILE).write_text('{"session": 1}')
        with pytest.raises(StorageError):
            repo.names("s1")


def understands(user: str) -> str:
    """A model that gets "soy <name>" right."""
    text, word = marked(user), judged(user)
    if f"soy {word.lower()}" in text.lower():
        return reply(word, "self", 0.95, f"soy {word}")
    return reply(word, "addressed", 0.6) if word in text else reply(None)


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A fake-runtime container holding a finished session ``s1``."""
    built = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    transcript = conversation([
        ("S1", "Buenos días."), ("S2", "Hola, soy Ana."),
        ("S1", "Gracias, Ana."), ("S3", "Soy Pedro."),
    ])
    built.sessions.create(transcript.session)
    built.sessions.save_transcript(transcript)
    yield built
    built.runner.close(WAIT)


@pytest.fixture()
def client(container: TranscriberContainer) -> TestClient:
    """A client of the API that sends the access code."""
    return TestClient(
        create_app(container, CODE), headers={"X-Access-Code": CODE}
    )


def settled(client: TestClient) -> Dict[str, Any]:
    """Poll the names until no search runs; return the last view."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        view: Dict[str, Any] = client.get("/api/sessions/s1/names").json()
        if not view["running"]:
            return view
        time.sleep(0.02)
    raise AssertionError("the search did not end")


class TestHttpRoutes:
    def test_never_searched(self, client: TestClient) -> None:
        view = client.get("/api/sessions/s1/names")
        assert view.status_code == 200
        assert view.json() == names_json("s1", None, False)

    def test_rules_only_with_fake_models(self, client: TestClient) -> None:
        queued = client.post("/api/sessions/s1/names")
        assert queued.status_code == 202, queued.text
        view = settled(client)
        assert view["model"] is None and view["auto"] == []
        assert "No local model" in view["note"]
        reasons = {(s["speaker"], s["reason"]) for s in view["suggestions"]}
        assert ("S2", "rules_only") in reasons

    def test_search_name_and_undo(
        self, client: TestClient, container: TranscriberContainer
    ) -> None:
        container.find_names._model = FakeChatModel(understands)
        assert client.post("/api/sessions/s1/names").status_code == 202
        view = settled(client)
        assert [(a["speaker"], a["name"]) for a in view["auto"]] == [
            ("S2", "Ana"), ("S3", "Pedro"),
        ]
        assert view["model"] == "fake-qwen"
        assert view["searched_at"].startswith("20")
        speakers = client.get("/api/sessions/s1/transcript").json()["speakers"]
        assert [s["name"] for s in speakers] == [None, "Ana", "Pedro"]
        undone = client.delete("/api/sessions/s1/names/S2")
        assert undone.status_code == 200, undone.text
        assert undone.json()["speakers"][1]["name"] is None
        again = client.delete("/api/sessions/s1/names/S2")
        assert again.json()["code"] == "invalid_input"
        view = client.get("/api/sessions/s1/names").json()
        assert [a["speaker"] for a in view["auto"]] == ["S3"]
        assert [s["reason"] for s in view["suggestions"]] == ["undone"]

    def test_errors(self, client: TestClient) -> None:
        assert client.get("/api/sessions/nope/names").status_code == 404
        assert client.post("/api/sessions/nope/names").status_code == 404
        assert client.delete(
            "/api/sessions/s1/names/S1"
        ).json()["code"] == "not_found"
        anonymous = TestClient(client.app)
        assert anonymous.get("/api/sessions/s1/names").status_code == 401
