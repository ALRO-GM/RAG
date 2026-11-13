"""Integration tests: the site builder and its CLI on real directories."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from results_site import SITE_FORMAT_VERSION
from results_site.__main__ import git_commit, main
from results_site.builder import SiteBuilder
from results_site.summaries import SiteError

REPO = Path(__file__).resolve().parents[2]


def write(path: Path, text: str) -> None:
    """Write ``text`` creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def jsonl(*rows: Any) -> str:
    """Serialize rows as JSON Lines."""
    return "".join(json.dumps(r) + "\n" for r in rows)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A minimal repository with every kind of source."""
    root = tmp_path / "repo"
    write(root / "web/index.html", "<!doctype html><title>t</title>")
    write(root / "web/app.js", "// app")
    write(root / "README.md", "# Project\ntext")
    write(root / "docs/ARCHITECTURE.md", "no heading")
    write(root / "docs/skills/THEORY.md", "# Theory")
    write(root / "experiments/REPORT.md", "# Report")
    trial = {
        "trial_id": "b", "suite": "s", "name": "base", "params": {},
        "baseline_id": "b", "ranks": {},
        "metrics": {"docs": {"recall@5": 0.9}},
    }
    write(root / "experiments/results/trials.jsonl", jsonl(trial))
    write(root / "experiments/results/tuning_tune_all.json", "{}")
    write(root / "experiments/profiles/speed.json", "{}")
    write(root / "experiments/benchmarks/beir.jsonl", jsonl(
        {"dataset": "scifact", "config": "c", "ndcg10": 0.7},
    ))
    write(root / "knowledge/rubrics/core.es.json", json.dumps({
        "name": "Core", "language": "es",
        "competencies": [{"indicators": [{"polarity": "positive"}]}],
    }))
    return root


NOW = datetime(2026, 11, 13, 1, 2, 3, 456, tzinfo=timezone.utc)


class TestSiteBuilder:
    def test_writes_front_end_summaries_docs_and_raw(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        manifest = SiteBuilder(repo, out).build(commit="abc123", now=NOW)

        assert (out / "index.html").is_file()
        assert (out / "app.js").is_file()
        assert manifest["format"] == SITE_FORMAT_VERSION
        assert manifest["generated"] == "2026-11-13T01:02:03+00:00"
        assert manifest["commit"] == "abc123"
        assert manifest["sections"] == {
            "lab": True, "beir": True, "skills": False, "hardware": False,
        }
        lab = json.loads((out / "data/lab.json").read_text())
        assert lab["suites"][0]["trials"][0]["headline"] == 0.9
        assert (out / "data/beir.json").is_file()
        assert not (out / "data/skills.json").exists()
        on_disk = json.loads((out / "data/manifest.json").read_text())
        assert on_disk == manifest

    def test_documentation_index_groups_titles_and_copies(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        manifest = SiteBuilder(repo, out).build(now=NOW)
        groups = {g["key"]: g for g in manifest["docs"]}
        assert list(groups) == ["project", "design", "lab", "skills"]
        design = groups["design"]["items"]
        assert design == [{
            "path": "docs/ARCHITECTURE.md", "title": "ARCHITECTURE",
            "bytes": len("no heading"),
        }]
        assert groups["project"]["items"][0]["title"] == "Project"
        assert (out / "docs/docs/skills/THEORY.md").read_text() == "# Theory"

    def test_raw_files_and_rubrics_are_published(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        manifest = SiteBuilder(repo, out).build(now=NOW)
        assert manifest["raw"] == [
            "experiments/results/trials.jsonl",
            "experiments/results/tuning_tune_all.json",
            "experiments/profiles/speed.json",
            "experiments/benchmarks/beir.jsonl",
        ]
        for rel in manifest["raw"]:
            assert (out / "raw" / rel).is_file()
        rubrics = json.loads((out / "data/rubrics.json").read_text())
        assert rubrics[0]["indicators"] == 1
        assert manifest["rubrics"] == 1
        assert (out / "raw/knowledge/rubrics/core.es.json").is_file()

    def test_rebuild_removes_stale_files(
        self, repo: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        write(out / "stale.txt", "old")
        SiteBuilder(repo, out).build(now=NOW)
        assert not (out / "stale.txt").exists()

    def test_hardware_history_is_summarized(
        self, repo: Path, tmp_path: Path
    ) -> None:
        write(repo / "experiments/hardware/runtime.jsonl", jsonl({
            "created": "2026-11-13", "results": [
                {"plan": {"device": "cpu"}, "cost": 4.0},
                {"plan": {"device": "cuda"}, "cost": 1.0},
            ],
        }))
        out = tmp_path / "out"
        manifest = SiteBuilder(repo, out).build(now=NOW)
        assert manifest["sections"]["hardware"] is True
        hardware = json.loads((out / "data/hardware.json").read_text())
        assert hardware["latest"]["gpu_speedup"] == 4.0

    def test_broken_jsonl_is_reported(
        self, repo: Path, tmp_path: Path
    ) -> None:
        write(repo / "experiments/benchmarks/beir.jsonl", "{bad\n")
        with pytest.raises(SiteError, match="beir.jsonl:1"):
            SiteBuilder(repo, tmp_path / "out").build(now=NOW)

    @pytest.mark.parametrize("content, message", [
        ("{bad", "invalid JSON"),
        ("[]", "expected a JSON object"),
    ])
    def test_broken_rubric_is_reported(
        self, repo: Path, tmp_path: Path, content: str, message: str
    ) -> None:
        write(repo / "knowledge/rubrics/core.es.json", content)
        with pytest.raises(SiteError, match=message):
            SiteBuilder(repo, tmp_path / "out").build(now=NOW)

    def test_missing_front_end_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(SiteError, match="No front end"):
            SiteBuilder(tmp_path, tmp_path / "out").build()

    @pytest.mark.parametrize("target", [".", ".."])
    def test_refuses_to_build_over_the_repository(
        self, repo: Path, target: str
    ) -> None:
        with pytest.raises(SiteError, match="Refusing"):
            SiteBuilder(repo, repo / target)

    def test_generated_time_defaults_to_now(
        self, repo: Path, tmp_path: Path
    ) -> None:
        manifest = SiteBuilder(repo, tmp_path / "out").build()
        generated = datetime.fromisoformat(manifest["generated"])
        assert abs((datetime.now(timezone.utc) - generated).seconds) < 60


class TestRealRepository:
    def test_the_repository_builds_and_every_reference_exists(
        self, tmp_path: Path
    ) -> None:
        """What CI deploys: the checked-in results must all build."""
        out = tmp_path / "site"
        manifest = SiteBuilder(REPO, out).build(now=NOW)
        assert manifest["sections"]["lab"] is True
        assert manifest["sections"]["beir"] is True
        for group in manifest["docs"]:
            for doc in group["items"]:
                assert (out / "docs" / doc["path"]).is_file()
        for rel in manifest["raw"]:
            assert (out / "raw" / rel).is_file()
        for name, ready in manifest["sections"].items():
            assert (out / f"data/{name}.json").is_file() == ready
        html = (out / "index.html").read_text(encoding="utf-8")
        for asset in ("app.js", "style.css"):
            assert asset in html
            assert (out / asset).is_file()

    def test_private_server_notes_are_never_published(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "site"
        SiteBuilder(REPO, out).build(now=NOW)
        names = {p.name for p in out.rglob("*")}
        assert "SERVIDOR_ORACLE.md" not in names


class TestCommandLine:
    def test_build_command(
        self, repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        code = main(["build", "--root", str(repo), "--out", str(out)])
        assert code == 0
        assert (out / "data/manifest.json").is_file()
        printed = capsys.readouterr().out
        assert "4 documents" in printed
        assert "sections: lab, beir" in printed

    def test_explicit_commit_wins(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        code = main(["build", "--root", str(repo), "--out", str(out),
                     "--commit", "deadbee"])
        assert code == 0
        manifest = json.loads((out / "data/manifest.json").read_text())
        assert manifest["commit"] == "deadbee"

    def test_build_error_exit_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["build", "--root", str(tmp_path),
                     "--out", str(tmp_path / "out")])
        assert code == 1
        assert "error: No front end" in capsys.readouterr().err

    def test_no_sections_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write(tmp_path / "web/index.html", "x")
        code = main(["build", "--root", str(tmp_path),
                     "--out", str(tmp_path / "out")])
        assert code == 0
        assert "sections: none" in capsys.readouterr().out

    def test_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_module_entry_point(self, repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        done = subprocess.run(
            ["python", "-m", "results_site", "build",
             "--root", str(repo), "--out", str(out)],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        assert done.returncode == 0, done.stderr
        assert (out / "index.html").is_file()

    def test_git_commit_inside_and_outside_a_checkout(
        self, tmp_path: Path
    ) -> None:
        assert git_commit(tmp_path) == ""
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=t",
             "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
             "-m", "x"],
            check=True,
        )
        assert len(git_commit(tmp_path)) >= 7

    def test_git_missing_gives_empty_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "")
        assert git_commit(tmp_path) == ""
