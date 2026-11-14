"""The lab's HTTP routes and CLI over real adapters, fake models and runner."""

import io
import json
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.integration.transcriber.audio import scene, sound_file, tone
from tests.integration.transcriber.lab_sandbox import make_lab
from transcriber.adapters.inbound.cli import Output, build_parser, main
from transcriber.adapters.inbound.http_api import create_app
from transcriber.adapters.inbound.lab_cli import (
    LabCommands,
    summarise,
    summary_lines,
)
from transcriber.application.ports import RecordedAudio
from transcriber.bootstrap import TranscriberContainer, build_container
from transcriber.config import TranscriberSettings
from transcriber.domain.lab import Segment
from transcriber.domain.lab_reference import CORPORA, make_reference

CODE = "a-long-enough-access-code"
AUTH = {"X-Access-Code": CODE}
WAIT = 30.0
REQUEST = {
    "denoisers": ["none", "fake-denoise"],
    "diarizers": ["builtin", "fake-diarize", "fake-split"],
    "recognisers": ["project-asr", "fake-asr"],
}


def settings(tmp_path: Path) -> TranscriberSettings:
    """Fake models, the sandbox lab, data under ``tmp_path``."""
    return TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
        lab=make_lab(tmp_path / "lab"),
    )


@pytest.fixture(autouse=True)
def no_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide any Hugging Face token of the machine from the lab."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


@pytest.fixture()
def container(tmp_path: Path) -> Iterator[TranscriberContainer]:
    """A container with an enrolled person and an imported session."""
    built = build_container(settings(tmp_path))
    built.enrol_speaker(
        "ana", "Ana", "es",
        RecordedAudio(sound_file(tone(30.0, hz=220.0), 16000, "wav"), "wav"),
        True, built.settings.api.consent_version,
    )
    audio = scene([("220", 6.0), ("-", 1.6), ("880", 6.0), ("-", 1.6),
                   ("220", 4.8)])
    built.import_recording(
        "s1", "es", RecordedAudio(sound_file(audio, 16000, "wav"), "wav"),
    )
    yield built
    built.runner.close(WAIT)


@pytest.fixture()
def client(container: TranscriberContainer) -> TestClient:
    """A client with the access code."""
    return TestClient(create_app(container, CODE), headers=AUTH)


def wait(client: TestClient, run_id: str) -> Dict[str, Any]:
    """Poll the run until it ends."""
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        view: Dict[str, Any] = client.get(
            f"/api/lab/sessions/s1/runs/{run_id}"
        ).json()
        if view["state"] not in ("queued", "running"):
            return view
        time.sleep(0.05)
    raise AssertionError("the run did not end")


def test_catalogue(client: TestClient) -> None:
    body = client.get("/api/lab/catalogue").json()
    assert body["token_present"] is False
    states = {m["id"]: m["state"] for m in body["models"]}
    assert states["fake-diarize"] == "ready"
    assert states["fake-gated"] == "needs_token"
    assert states["missing-model"] == "not_installed"


def test_a_run_from_start_to_scores(client: TestClient) -> None:
    started = client.post("/api/lab/sessions/s1/runs", json=REQUEST)
    assert started.status_code == 202, started.text
    run = started.json()
    assert run["pipeline_count"] == 12 and run["state"] == "queued"
    assert run["steps"][0]["name"] == "denoise:fake-denoise"
    assert "worker_pid" not in run
    view = wait(client, run["id"])
    assert view["state"] == "done", view
    assert view["done_steps"] == view["total_steps"] == 11
    assert {p["state"] for p in view["pipelines"]} == {"done"}
    listed = client.get("/api/lab/sessions/s1/runs").json()["runs"]
    assert [r["id"] for r in listed] == [run["id"]]
    assert "steps" not in listed[0]
    base = f"/api/lab/sessions/s1/runs/{run['id']}"
    results = client.get(f"{base}/results").json()
    assert len(results["pipelines"]) == 12
    assert "fragments" not in results and results["fragment_count"] > 0
    split = [p for p in results["pipelines"] if "fake-split" in p["id"]]
    assert not split[0]["licence_commercial"]
    tracks = client.get(f"{base}/tracks").json()
    assert "none+fake-diarize" in tracks["tracks"]
    page = client.get(f"{base}/fragments",
                      params={"limit": 3, "sort": "time"}).json()
    assert page["total"] == results["fragment_count"]
    assert len(page["fragments"]) == 3
    assert page["fragments"][0]["start"] == 0.0
    assert client.get(f"{base}/fragments",
                      params={"sort": "loud"}).status_code == 400
    assert client.get(f"{base}/fragments",
                      params={"limit": 0}).status_code == 422
    labels = client.put("/api/lab/sessions/s1/labels", json={"labels": [
        {"start": 0, "end": 6, "kind": "person", "speakers": ["ana"]},
        {"start": 6.2, "end": 7.4, "kind": "nobody"},
    ]})
    assert labels.status_code == 200, labels.text
    assert labels.json()["speakers"] == ["ana"]
    assert len(client.get("/api/lab/sessions/s1/labels").json()["labels"]) \
        == 2
    spans = client.get(f"{base}/fragment-spans").json()["spans"]
    assert len(spans) == results["fragment_count"]
    assert spans[0]["start"] == 0.0 and "pipelines" not in spans[0]
    left = client.get(f"{base}/fragments", params={
        "sort": "time", "unlabelled": "true", "limit": 200}).json()
    assert 0 < left["total"] < results["fragment_count"]
    assert all(f["label"] is None for f in left["fragments"])
    bad = client.put("/api/lab/sessions/s1/labels", json={"labels": [
        {"start": 0, "end": 1, "kind": "person", "speakers": ["eve"]},
    ]})
    assert bad.json()["code"] == "invalid_input"
    scores = client.get(f"{base}/scores").json()
    assert scores["labelled_seconds"] == pytest.approx(7.2)
    assert scores["scores"][0]["rank"] == 1
    assert client.post(f"{base}/cancel").json()["code"] == "wrong_state"
    resumed = client.post(f"{base}/resume")
    assert resumed.status_code == 202
    assert wait(client, run["id"])["state"] == "done"


def test_audio_of_a_run(client: TestClient) -> None:
    run = client.post("/api/lab/sessions/s1/runs", json={
        "denoisers": ["none", "fake-denoise"], "diarizers": ["fake-diarize"],
    }).json()
    wait(client, run["id"])
    base = f"/api/lab/sessions/s1/runs/{run['id']}/audio"
    original = client.get(f"{base}/none")
    cleaned = client.get(f"{base}/fake-denoise",
                         headers={"Range": "bytes=0-99"})
    assert original.status_code == 200
    assert original.headers["content-type"] == "audio/wav"
    assert cleaned.status_code == 206 and len(cleaned.content) == 100
    assert client.get(f"{base}/other").status_code == 404
    cookie = client.post("/api/access/cookie").headers["set-cookie"]
    value = cookie.split(";")[0].split("=", 1)[1]
    plain = TestClient(client.app)
    played = plain.get(f"{base}/none",
                       headers={"Cookie": f"transcriber_playback={value}"})
    assert played.status_code == 200
    assert plain.get(f"/api/lab/sessions/s1/runs/{run['id']}").status_code \
        == 401


def test_overview_and_excerpts(client: TestClient) -> None:
    assert client.get("/api/lab/overview").json() == {"sessions": []}
    run = client.post("/api/lab/sessions/s1/runs", json={
        "denoisers": ["none", "fake-denoise"], "diarizers": ["fake-diarize"],
    }).json()
    wait(client, run["id"])
    entry, = client.get("/api/lab/overview").json()["sessions"]
    assert (entry["session_id"], entry["corpus"], entry["run_id"]) == (
        "s1", None, run["id"],
    )
    assert entry["run_state"] == "done" and entry["results_run"] == run["id"]
    steps = entry["steps"]
    assert steps["done"] == 3 and steps["failed"] == 0
    assert steps["open"] == [] and steps["failed_codes"] == {}
    base = f"/api/lab/sessions/s1/runs/{run['id']}/excerpt"
    cut = client.get(f"{base}/none", params={"start": 1, "seconds": 2})
    assert cut.status_code == 200, cut.text
    assert cut.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(cut.content)) as reader:
        assert (reader.getframerate(), reader.getnframes()) == (16000, 32000)
    end = client.get(f"{base}/fake-denoise", params={"start": 19})
    with wave.open(io.BytesIO(end.content)) as reader:
        assert reader.getnframes() == 16000 * 1.0
    ranged = client.get(f"{base}/none", params={"start": 0},
                        headers={"Range": "bytes=0-43"})
    assert ranged.status_code == 206 and ranged.content[:4] == b"RIFF"
    assert client.get(f"{base}/none", params={"start": 20}).json()[
        "code"] == "invalid_input"
    for params in ({}, {"start": -1}, {"start": 0, "seconds": 61},
                   {"start": 0, "seconds": 0}):
        assert client.get(f"{base}/none", params=params).status_code == 422
    assert client.get(f"{base}/other", params={"start": 0}).status_code \
        == 404
    cookie = client.post("/api/access/cookie").headers["set-cookie"]
    value = cookie.split(";")[0].split("=", 1)[1]
    plain = TestClient(client.app)
    assert plain.get(f"{base}/none", params={"start": 0}).status_code == 401
    assert plain.get(
        f"{base}/none", params={"start": 0},
        headers={"Cookie": f"transcriber_playback={value}"},
    ).status_code == 200
    assert plain.get("/api/lab/overview").status_code == 401


def test_reference_summary_route(
    client: TestClient, container: TranscriberContainer
) -> None:
    assert client.get("/api/lab/sessions/s1/reference-turns").json()[
        "code"] == "not_found"
    empty = client.get("/api/lab/reference-summary").json()
    assert empty["recordings"] == [] and empty["models"] == []
    assert client.get("/api/lab/reference-summary",
                      params={"corpus": "nope"}).json()["code"] \
        == "invalid_input"
    assert container.lab is not None
    container.lab.repository.save_reference("s1", make_reference(
        CORPORA["ami"], "test", "X1", 20.0,
        [Segment(start=0, end=6, speaker="MEE1")], None,
    ))
    summary = client.get("/api/lab/reference-summary",
                         params={"corpus": "ami"}).json()
    assert summary["corpora"] == ["ami"] and summary["unscored"] == ["s1"]
    assert client.get("/api/lab/reference-summary",
                      params={"corpus": "callhome"}).json()["unscored"] \
        == []
    turns = client.get("/api/lab/sessions/s1/reference-turns").json()
    assert turns == {"turns": [{"start": 0.0, "end": 6.0, "speaker": "MEE1"}]}
    entry, = client.get("/api/lab/overview").json()["sessions"]
    assert (entry["corpus"], entry["licence_commercial"]) == ("ami", True)
    assert entry["run_id"] is None and entry["steps"] is None


def test_the_audio_route_only_reads(
    client: TestClient, container: TranscriberContainer
) -> None:
    lab = container.settings.data_dir / "lab" / "s1"
    run = client.post("/api/lab/sessions/s1/runs", json={
        "diarizers": ["fake-diarize"],
    }).json()
    stored = json.loads((lab / "original.json").read_text())["key"]
    assert (lab / "audio" / f"{stored}.wav").is_file()
    wait(client, run["id"])
    base = f"/api/lab/sessions/s1/runs/{run['id']}/audio"
    assert client.get(f"{base}/none").status_code == 200
    (lab / "original.json").unlink()
    (lab / "audio" / f"{stored}.wav").unlink()
    before = sorted(str(p) for p in lab.rglob("*"))
    missing = client.get(f"{base}/none")
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"
    assert sorted(str(p) for p in lab.rglob("*")) == before
    assert client.post(
        f"/api/lab/sessions/s1/runs/{run['id']}/resume"
    ).status_code == 202
    assert (lab / "original.json").is_file()
    assert client.get(f"{base}/none").status_code == 200
    wait(client, run["id"])


def test_refusals(client: TestClient) -> None:
    def code(response: Any) -> str:
        return str(response.json()["code"])

    assert code(client.post("/api/lab/sessions/s1/runs",
                            json={"diarizers": ["nope"]})) == "invalid_input"
    assert code(client.post("/api/lab/sessions/s1/runs",
                            json={"diarizers": []})) == "invalid_request"
    for value in (0, -1, {"n": 2}, "auto", 1.5, True):
        bad = client.post("/api/lab/sessions/s1/runs", json={
            "diarizers": ["fake-diarize"],
            "params": {"fake-diarize": {"num_speakers": value}},
        })
        assert (bad.status_code, code(bad)) == (400, "invalid_input")
    assert client.get("/api/lab/sessions/s1/runs").json()["runs"] == []
    assert code(client.post("/api/lab/sessions/zz/runs", json={})) \
        == "not_found"
    assert code(client.get("/api/lab/sessions/s1/runs/bad")) \
        == "invalid_input"
    assert code(client.get(f"/api/lab/sessions/s1/runs/{'0' * 32}")) \
        == "not_found"
    big = {"labels": [{"start": n, "end": n + 0.5, "kind": "nobody"}
                      for n in range(5001)]}
    assert code(client.put("/api/lab/sessions/s1/labels", json=big)) \
        == "invalid_request"


def test_a_failing_and_a_cancelled_run(client: TestClient,
                                       container: TranscriberContainer
                                       ) -> None:
    failed = client.post("/api/lab/sessions/s1/runs",
                         json={"diarizers": ["fake-crash"]}).json()
    view = wait(client, failed["id"])
    assert view["state"] == "failed"
    assert view["pipelines"][0]["error"]["code"] == "failed"
    assert client.get(
        f"/api/lab/sessions/s1/runs/{failed['id']}/results"
    ).json()["pipelines"] == []
    slow = client.post("/api/lab/sessions/s1/runs",
                       json={"diarizers": ["fake-slow"]}).json()
    deadline = time.monotonic() + WAIT
    while client.get(f"/api/lab/sessions/s1/runs/{slow['id']}").json()[
        "state"
    ] != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    cancelled = client.post(f"/api/lab/sessions/s1/runs/{slow['id']}/cancel")
    assert cancelled.status_code == 200
    assert wait(client, slow["id"])["state"] == "cancelled"


def recording() -> RecordedAudio:
    """A short recording to import as a session."""
    return RecordedAudio(sound_file(tone(5.0), 16000, "wav"), "wav")


def test_deleting_a_session_deletes_its_lab_data(
    client: TestClient, container: TranscriberContainer
) -> None:
    run = client.post("/api/lab/sessions/s1/runs", json={
        "denoisers": ["none", "fake-denoise"], "diarizers": ["fake-diarize"],
        "recognisers": ["fake-asr"],
    }).json()
    assert wait(client, run["id"])["state"] == "done"
    labelled = client.put("/api/lab/sessions/s1/labels", json={"labels": [
        {"start": 0, "end": 6, "kind": "person", "speakers": ["ana"]},
    ]})
    assert labelled.status_code == 200, labelled.text
    audio = f"/api/lab/sessions/s1/runs/{run['id']}/audio/none"
    assert client.get(audio).status_code == 200
    lab_dir = container.settings.data_dir / "lab" / "s1"
    assert (lab_dir / "original.json").is_file()
    assert client.delete("/api/sessions/s1").status_code == 204
    assert not lab_dir.exists()
    container.import_recording("s1", "es", recording())
    assert client.get("/api/lab/sessions/s1/runs").json()["runs"] == []
    assert client.get("/api/lab/sessions/s1/labels").json()["labels"] == []
    assert client.get(audio).status_code == 404
    assert client.get(
        f"/api/lab/sessions/s1/runs/{run['id']}/results"
    ).status_code == 404


def test_a_session_with_an_active_lab_run_is_not_deleted(
    client: TestClient, container: TranscriberContainer
) -> None:
    slow = client.post("/api/lab/sessions/s1/runs",
                       json={"diarizers": ["fake-slow"]}).json()
    refused = client.delete("/api/sessions/s1")
    assert refused.status_code == 409
    assert refused.json()["code"] == "wrong_state"
    assert "lab run" in refused.json()["message"]
    assert client.get(f"/api/lab/sessions/s1/runs/{slow['id']}").json()[
        "state"
    ] in ("queued", "running")
    client.post(f"/api/lab/sessions/s1/runs/{slow['id']}/cancel")
    assert wait(client, slow["id"])["state"] == "cancelled"
    assert client.delete("/api/sessions/s1").status_code == 204
    assert not (container.settings.data_dir / "lab" / "s1").exists()


class TestCli:
    def run_cli(self, tmp_path: Path, *argv: str) -> "tuple[int, str]":
        """Run the CLI with the sandbox settings."""
        config = tmp_path / "config.json"
        config.write_text(settings(tmp_path).model_dump_json())
        out, err = io.StringIO(), io.StringIO()
        code = main(list(argv), {"TRANSCRIBER_CONFIG": str(config)}, out,
                    err)
        return code, out.getvalue()

    def test_catalogue_run_runs_summary(
        self, tmp_path: Path, container: TranscriberContainer
    ) -> None:
        code, text = self.run_cli(tmp_path, "lab", "catalogue")
        assert code == 0 and "fake-gated" in text and "gated" in text
        code, text = self.run_cli(tmp_path, "lab", "catalogue", "--json")
        assert json.loads(text)["token_present"] is False
        code, text = self.run_cli(
            tmp_path, "lab", "run", "s1", "--denoise", "fake-denoise",
            "--diarize", "fake-diarize,fake-split", "--recognise",
            "fake-asr", "--params", '{"fake-asr": {"language": "es"}}',
        )
        assert code == 0, text
        assert "Pipelines (agreement, not accuracy)" in text
        assert "[non-commercial]" in text
        code, text = self.run_cli(tmp_path, "lab", "runs", "s1", "--json")
        run_id = json.loads(text)["runs"][0]["id"]
        code, text = self.run_cli(tmp_path, "lab", "summary", "s1", run_id,
                                  "--json")
        data = json.loads(text)
        assert data["state"] == "done" and data["mean_track_agreement"]
        assert "text" not in json.dumps(data["pipelines"])
        code, text = self.run_cli(tmp_path, "lab", "runs", "s1")
        assert run_id in text
        code, text = self.run_cli(tmp_path, "lab", "resume", "s1", run_id)
        assert code == 0

    def test_all_ready_models_and_failures(
        self, tmp_path: Path, container: TranscriberContainer
    ) -> None:
        code, text = self.run_cli(
            tmp_path, "lab", "run", "s1", "--all", "--diarize",
            "fake-diarize,fake-crash", "--recognise", "fake-asr",
            "--speakers", "2", "--language", "es",
        )
        assert code == 0, text
        assert "Failed diarize:none+fake-crash: failed" in text
        code, _ = self.run_cli(tmp_path, "lab", "run", "s1", "--diarize",
                               "fake-crash")
        assert code == 1
        code, _ = self.run_cli(tmp_path, "lab", "run", "s1", "--params",
                               "[1]")
        assert code == 2
        code, _ = self.run_cli(tmp_path, "lab", "run", "s1", "--diarize",
                               "a,a")
        assert code == 2
        code, text = self.run_cli(tmp_path, "lab", "runs", "s9")
        assert code == 1

    def test_all_leaves_out_weights_not_downloaded(
        self, tmp_path: Path, container: TranscriberContainer
    ) -> None:
        assert container.lab is not None
        err = io.StringIO()
        commands = LabCommands(container.lab, Output(io.StringIO(), err))
        args = build_parser().parse_args(["lab", "run", "s1", "--all"])
        chosen = commands._chosen(args)
        assert "fake-heavy" not in chosen.recognisers
        assert "fake-asr" in chosen.recognisers
        assert "fake-gated" not in chosen.diarizers
        assert "not downloaded" in err.getvalue()

    def test_no_runs(self, tmp_path: Path,
                     container: TranscriberContainer) -> None:
        assert self.run_cli(tmp_path, "lab", "runs", "s1")[1] == (
            "No lab runs.\n"
        )


def test_summary_lines_without_results() -> None:
    lines = summary_lines({"state": "failed", "failures": [
        {"step": "x", "code": "needs_token", "message": "m"}]})
    assert lines == ["State: failed", "Failed x: needs_token: m"]
    assert summarise({
        "duration": 1.0, "consensus": {"coverage": None}, "tracks": [],
        "transcripts": [], "pipelines": [], "diarization": [],
        "recognition": [], "denoiser_effect": [], "fragment_count": 0,
    })["mean_track_agreement"] is None
