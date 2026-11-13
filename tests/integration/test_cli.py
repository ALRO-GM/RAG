"""The Fire CLI run as ``python -m src`` in a subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from tests.samples import (
    MD_PATH,
    PY_PATH,
    TXT_PATH,
    write_corpus,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]


def run_cli(args: List[str], cwd: Path) -> "subprocess.CompletedProcess[str]":
    """Run ``python -m src *args`` from ``cwd`` without any model."""
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    return subprocess.run(
        [sys.executable, "-m", "src", *args],
        cwd=cwd, capture_output=True, text=True, timeout=120, env=env,
    )


def source(path: str, first: int, last: int) -> dict[str, object]:
    """Return a source as JSON-ready data."""
    return {"file_path": path, "first_character_index": first,
            "last_character_index": last}


@pytest.fixture(scope="module")
def cli_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Index the tiny corpus with the CLI once and add dataset files.

    Shared by the module: tests only read the index (the cache and the
    output directories are the only things they write).
    """
    project = tmp_path_factory.mktemp("cli")
    write_corpus(project)
    result = run_cli(["index", "--semantic=False", "--max_chunk_size=300"],
                     project)
    assert result.returncode == 0, result.stderr
    write_json(project / "data/datasets/questions.json", {"rag_questions": [
        {"question_id": "q1", "question": "How to load a LoRA adapter?",
         "answer": "x", "sources": [source(PY_PATH, 0, 200)]},
        {"question_id": "q2", "question": "What does speculative decoding"
         " use?", "answer": "y", "sources": [source(TXT_PATH, 0, 58)]},
        {"question_id": "q3", "question": "FP8 quantization memory"},
    ]})
    (project / "bad.json").write_text("{not valid json")
    write_json(project / "wrong.json", {"rag_questions": "nope"})
    write_json(project / "list.json", ["mode", "bm25"])
    write_json(project / "boost.json", {"file_boost": 0.5})
    write_json(project / "neutral.json", {"file_boost": 0.0})
    write_json(project / "semantic.json", {"mode": "semantic"})
    write_json(project / "negative.json", {"file_boost": -1})
    write_json(project / "experiments/profiles/named.json",
               {"mode": "semantic"})
    return project


def test_index_reports_and_is_incremental(project: Path) -> None:
    """The first run is full, the second incremental."""
    first = run_cli(["index", "--semantic=False"], project)
    assert first.returncode == 0, first.stderr
    assert "Ingestion complete (full): 3 files, 3 indexed" in first.stdout
    assert (project / "data/processed/manifest.json").is_file()
    second = run_cli(["index", "--semantic=False"], project)
    assert "(incremental): 3 files, 0 indexed, 3 unchanged" in second.stdout


def test_happy_path(cli_project: Path) -> None:
    """search → search_dataset → evaluate on the tiny corpus."""
    search = run_cli(["search", "load_lora_adapter", "--k=2"], cli_project)
    assert search.returncode == 0, search.stderr
    out = json.loads(search.stdout)
    assert out["mode"] == "bm25"
    assert out["retrieved_sources"][0]["file_path"] == PY_PATH
    assert len(out["scores"]) == len(out["retrieved_sources"]) <= 2

    dataset = run_cli([
        "search_dataset", "--dataset_path=data/datasets/questions.json",
        "--k=3", "--save_directory=data/output/search",
    ], cli_project)
    assert dataset.returncode == 0, dataset.stderr
    assert "Searched 3 questions" in dataset.stdout
    results_path = cli_project / "data/output/search/questions.json"
    results = json.loads(results_path.read_text())
    assert results["k"] == 3
    assert [r["question_id"] for r in results["search_results"]] == [
        "q1", "q2", "q3",
    ]

    evaluate = run_cli([
        "evaluate",
        "--student_search_results_path=data/output/search/questions.json",
        "--dataset_path=data/datasets/questions.json", "--k=3",
    ], cli_project)
    assert evaluate.returncode == 0, evaluate.stderr
    assert "Recall@1:" in evaluate.stdout and "Recall@3:" in evaluate.stdout
    assert "Questions: 2 (without results: 0)" in evaluate.stdout


def test_search_no_match_is_success(cli_project: Path) -> None:
    """A query without matches prints an empty list and a notice."""
    result = run_cli(["search", "zzzqqqxxx"], cli_project)
    assert result.returncode == 0
    assert json.loads(result.stdout)["retrieved_sources"] == []
    assert "No matching sources" in result.stderr


def test_numeric_query_is_text(cli_project: Path) -> None:
    """Fire parses ``42`` as an int; it is still searched as text."""
    result = run_cli(["search", "42"], cli_project)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["question"] == "42"


def test_clear_cache(cli_project: Path) -> None:
    """Cached searches are counted and removed."""
    run_cli(["clear_cache"], cli_project)
    run_cli(["search", "lora"], cli_project)
    result = run_cli(["clear_cache"], cli_project)
    assert result.returncode == 0
    assert "Removed 1 cached entries." in result.stdout


@pytest.mark.parametrize("args, message", [
    (["search", ""], "query is empty"),
    (["search", "   "], "query is empty"),
    (["search", "lora", "--k=0"], "k must be a positive integer"),
    (["search", "lora", "--k=-1"], "k must be a positive integer"),
    (["search", "lora", "--k=abc"], "k must be a positive integer"),
    (["search", "lora", "--k=2.5"], "k must be a positive integer"),
    (["search", "lora", "--mode=fuzzy"], "Unknown mode"),
    (["search", "lora", "--fusion=max", "--mode=hybrid"], "Error"),
    (["search", "lora", "--mode=semantic"], "no embeddings"),
    (["search", "lora", "--index_dir=missing_dir"], "No index found"),
    (["search", "lora", "--index_dir=bad.json/sub"], "No index found"),
    (["search", "lora", "--context_tokens=abc"], "context_tokens"),
    (["search", "lora", "--rerank", "--rerank_weight=2"], "rerank_weight"),
    (["search", "lora", "--rerank", "--rerank_depth=0"], "rerank_depth"),
    (["search", "lora", "--rerank_depth=abc"], "rerank_depth"),
    (["search_dataset", "--dataset_path=missing.json"], "File not found"),
    (["search_dataset", "--dataset_path=bad.json"], "Malformed JSON"),
    (["search_dataset", "--dataset_path=wrong.json"], "RagDataset format"),
    (["search_dataset", "--dataset_path=data/datasets/questions.json",
      "--k=0"], "k must be"),
    (["evaluate", "--student_search_results_path=bad.json",
      "--dataset_path=data/datasets/questions.json"], "Malformed JSON"),
    (["evaluate", "--student_search_results_path=missing.json",
      "--dataset_path=data/datasets/questions.json"], "File not found"),
    (["answer_dataset", "--student_search_results_path=missing.json"],
     "File not found"),
    (["answer", "", "--k=3"], "query is empty"),
    (["index", "--max_chunk_size=5000", "--semantic=False"],
     "between 50 and 2000"),
    (["index", "--max_chunk_size=abc", "--semantic=False"],
     "must be an integer"),
    (["index", "--raw_dir=nowhere", "--semantic=False"],
     "Corpus directory not found"),
    (["index", "--extensions=,", "--semantic=False"], "--extensions"),
    (["index", "--extensions=.json,.yaml", "--raw_dir=data/raw/proj/docs",
      "--semantic=False"], "No indexable files"),
    (["serve", "--port=70000"], "--port"),
])
def test_degenerate_inputs(
    cli_project: Path, args: List[str], message: str
) -> None:
    """Bad inputs exit non-zero with one clean message, no traceback."""
    result = run_cli(args, cli_project)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr + result.stdout
    assert message in result.stderr


def test_missing_index_dir_without_project(tmp_path: Path) -> None:
    """Searching before indexing tells the user to run ``index``."""
    result = run_cli(["search", "lora"], tmp_path)
    assert result.returncode == 1
    assert "Run the 'index' command first" in result.stderr
    assert "Traceback" not in result.stderr


def test_index_files_are_relative(cli_project: Path) -> None:
    """Indexed chunk paths are relative to the working directory."""
    lines = (cli_project / "data/processed/chunks.jsonl").read_text()
    paths = {json.loads(line)["file_path"] for line in lines.splitlines()}
    assert paths == {PY_PATH, MD_PATH, TXT_PATH}


# ------------------------------------------------------------ profiles


def search_json(cwd: Path, *args: str) -> dict[str, object]:
    """Run an uncached search and return its JSON output."""
    result = run_cli(["search", "KVCacheManager", "--no_cache", *args], cwd)
    assert result.returncode == 0, result.stderr
    out: dict[str, object] = json.loads(result.stdout)
    return out


def test_profile_settings_are_applied(cli_project: Path) -> None:
    """A JSON profile changes the settings used by the command."""
    plain = search_json(cli_project, "--profile=neutral.json")
    boosted = search_json(cli_project, "--profile=boost.json")
    plain_scores, boosted_scores = plain["scores"], boosted["scores"]
    assert isinstance(plain_scores, list)
    assert isinstance(boosted_scores, list)
    assert len(boosted_scores) > len(plain_scores) == 1
    assert boosted_scores[0] == pytest.approx(1.5 * plain_scores[0],
                                              rel=1e-3)


def test_profile_is_found_by_name(cli_project: Path) -> None:
    """A bare name is looked up in experiments/profiles."""
    result = run_cli(["search", "lora", "--profile=named"], cli_project)
    assert result.returncode == 1
    assert "no embeddings" in result.stderr


def test_explicit_option_wins_over_the_profile(cli_project: Path) -> None:
    """An option given on the command line overrides the profile."""
    refused = run_cli(["search", "lora", "--profile=semantic.json"],
                      cli_project)
    assert refused.returncode == 1
    assert "no embeddings" in refused.stderr
    out = search_json(cli_project, "--profile=semantic.json", "--mode=bm25")
    assert out["mode"] == "bm25"


@pytest.mark.parametrize("profile, message", [
    ("nope", "Error: Profile not found: nope"),
    ("missing/dir/p.json", "Error: Profile not found: missing/dir/p.json"),
    ("bad.json", "Error: Profile bad.json is not valid JSON."),
    ("list.json", "Error: Profile list.json must be an object."),
    ("wrong.json",
     "Error: Profile wrong.json: unknown setting(s) rag_questions."),
])
def test_profile_errors_are_one_clean_line(
    cli_project: Path, profile: str, message: str
) -> None:
    """A bad profile exits 1 with a single message and no traceback."""
    result = run_cli(["search", "lora", f"--profile={profile}"], cli_project)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == message
    assert "Traceback" not in result.stderr


def test_profile_errors_are_reported_by_lab_commands(
    cli_project: Path
) -> None:
    """The lab sub-commands validate the profile the same way."""
    result = run_cli(["lab", "analyze", "--profile=nope"], cli_project)
    assert result.returncode == 1
    assert result.stderr.strip() == "Error: Profile not found: nope"


def test_profile_with_invalid_value_is_rejected(cli_project: Path) -> None:
    """A value refused by the settings is an error, not a traceback."""
    result = run_cli(["search", "lora", "--profile=negative.json"],
                     cli_project)
    assert result.returncode == 1
    assert result.stderr.startswith("Error: Invalid option:")
    assert "file_boost" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("setting", ["max_chunk_size", "target_chunk_size"])
def test_index_applies_the_chunk_sizes_of_a_profile(
    project: Path, setting: str
) -> None:
    """Chunk sizes tuned in the lab are used by ``index --profile``."""
    write_json(project / "small.json", {setting: 100})
    result = run_cli(["index", "--semantic=False", "--profile=small.json"],
                     project)
    assert result.returncode == 0, result.stderr
    lines = (project / "data/processed/chunks.jsonl").read_text()
    sizes = [len(json.loads(line)["text"]) for line in lines.splitlines()]
    # target_chunk_size is a soft target: a section is never cut below
    # its natural boundaries, it only stops merging small pieces.
    assert max(sizes) <= (100 if setting == "max_chunk_size" else 200)
