"""The command line, end to end with the fakes (in process and as a module)."""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import jsonschema
import pytest

from orchestrator.adapters.inbound import cli
from tests.unit.orchestrator.fixtures import REPO, transcript


@pytest.fixture()
def transcript_file(tmp_path: Path) -> Path:
    path = tmp_path / "unit.transcript.v1.json"
    path.write_text(transcript().model_dump_json(), encoding="utf-8")
    return path


def replay(path: Path, out: Path, *extra: str) -> int:
    return cli.main([
        "replay", str(path), "--product", "live_assist.support",
        "--out", str(out), *extra,
    ])


def test_check_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check-config"]) == 0
    assert capsys.readouterr().out.count("ok ") == 4


def test_check_config_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "x.yaml").write_text("product: nope\n", encoding="utf-8")
    assert cli.main(["check-config", "--dir", str(tmp_path)]) == 1
    assert "ERR" in capsys.readouterr().err


def test_replay_writes_suggestions_and_events(
    transcript_file: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out, events = tmp_path / "s.jsonl", tmp_path / "ev.jsonl"
    assert replay(transcript_file, out, "--events-out", str(events)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["calls"] == 2 and summary["roles"]["c"] == "customer"
    records = [json.loads(x) for x in out.read_text("utf-8").splitlines()]
    assert [r["trigger"]["utterance"] for r in records] == ["t1.s0", "t4.s0"]
    assert all({"knowledge_ms", "reasoner_ms", "total_ms"}
               <= set(r["stages"]) for r in records)
    schema = json.loads(
        (REPO / "contracts/live.event.v1.schema.json").read_text("utf-8"))
    for line in events.read_text("utf-8").splitlines():
        jsonschema.validate(json.loads(line), schema)


def test_replay_in_real_time_with_a_fast_clock(
    transcript_file: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert replay(transcript_file, tmp_path / "s.jsonl", "--speed", "100",
                  "--fake-delay-ms", "5") == 0
    assert json.loads(capsys.readouterr().out)["calls"] >= 1


def test_roles_option_overrides_the_order(
    transcript_file: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert replay(transcript_file, tmp_path / "s.jsonl",
                  "--roles", "a=customer,c=agent") == 0
    assert json.loads(capsys.readouterr().out)["roles"] == {
        "a": "customer", "c": "agent"}


@pytest.mark.parametrize("extra", [
    ["--speed", "0"], ["--speed", "fast"], ["--roles", "oops"],
    ["--knowledge", "carrier-pigeon"], ["--reasoner", "nonsense"],
])
def test_bad_options_exit_2(
    transcript_file: Path, tmp_path: Path, extra: List[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert replay(transcript_file, tmp_path / "s.jsonl", *extra) == 2
    assert "error:" in capsys.readouterr().err


def test_post_call_is_not_replayable(
    transcript_file: Path, tmp_path: Path
) -> None:
    code = cli.main(["replay", str(transcript_file), "--product",
                     "post_call", "--out", str(tmp_path / "s.jsonl")])
    assert code == 2


def test_missing_product_or_transcript(tmp_path: Path) -> None:
    assert cli.main(["replay", str(tmp_path / "no.json"),
                     "--product", "ghost"]) == 2


def test_default_output_path(
    transcript_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["replay", str(transcript_file), "--product",
                     "live_assist.support"]) == 0
    assert (tmp_path / "data/live/unit."
            "live_assist.support.suggestions.jsonl").exists()


def test_http_adapters_are_selectable(
    transcript_file: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nothing listens on the port: both services fail, replay goes on."""
    code = replay(transcript_file, tmp_path / "s.jsonl", "--knowledge",
                  "http", "--reasoner", "http", "--knowledge-url",
                  "http://127.0.0.1:9", "--reasoner-url",
                  "http://127.0.0.1:9")
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["by_status"] == {"error": 2}


def test_reasoner_factory_by_name(
    transcript_file: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = replay(transcript_file, tmp_path / "s.jsonl", "--reasoner",
                  "orchestrator.adapters.outbound.fakes:FakeReasoner")
    assert code == 0 and json.loads(capsys.readouterr().out)["calls"] == 2


def test_module_entry_point(transcript_file: Path, tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "orchestrator", "check-config"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0 and "live_assist.support" in done.stdout
