"""Inbound adapter: the Python Fire command-line interface.

Usage: ``uv run python -m src <command> [options]``. This module only
parses and validates arguments, calls the use cases and prints results;
all logic lives in the application layer.
"""

import functools
import json
import shutil
import sys
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Set, TypeVar, cast,
)

from src.adapters.inbound.lab_cli import LabCLI, load_suite
from src.adapters.inbound.skills_cli import SkillsCLI
from src.adapters.outbound.torch_runtime import (
    append_tuning,
    save_measured,
)
from src.application.benchmarking import BenchmarkResult
from src.application.generalist import (
    GeneralistAnalysis,
    render_generalist,
)
from src.config import BENCHMARK_CONFIGS, SkillsSettings
from src.skills_bootstrap import SkillsContainer
from src.bootstrap import Container
from src.config import Settings
from src.domain.errors import InvalidInputError, RagError, short_reason
from src.domain.experiments import params_id
from src.domain.models import MinimalSearchResults
from src.domain.runtime import gpu_diagnosis
from src.domain.validation import validate_k, validate_query

F = TypeVar("F", bound=Callable[..., Any])


def safe_command(func: F) -> F:
    """Report errors as one clean message and a non-zero exit code."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except RagError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except MemoryError:
            print("Error: out of memory.", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:  # last-resort guard: never a traceback
            print(
                f"Unexpected error ({type(exc).__name__}): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    return cast(F, wrapper)


def as_text(value: Any, name: str) -> str:
    """Coerce a Fire argument to text (Fire parses ``42`` as an int)."""
    if value is None or isinstance(value, (dict, list, bool)):
        raise InvalidInputError(f"{name} must be a string.")
    if isinstance(value, tuple):
        return " ".join(str(v) for v in value)
    return str(value)


def as_int(value: Any, name: str) -> int:
    """Validate an integer option."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(
            f"--{name} must be an integer (got {value!r})."
        )
    return value


def as_float(value: Any, name: str) -> float:
    """Validate a numeric option."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(f"--{name} must be a number (got {value!r}).")
    return float(value)


def parse_extensions(value: Any) -> tuple[str, ...]:
    """Parse ``.py,.md`` (or a Fire tuple) into normalised extensions."""
    raw = value if isinstance(value, (tuple, list)) else str(value).split(",")
    exts = tuple(
        e if e.startswith(".") else f".{e}"
        for e in (str(x).strip().lower() for x in raw) if e
    )
    if not exts:
        raise InvalidInputError("--extensions must list an extension.")
    return exts


def select_configs(
    available: Mapping[str, Mapping[str, Any]], wanted: Any
) -> Dict[str, Dict[str, Any]]:
    """Keep the configurations whose name contains a wanted substring.

    ``wanted`` is a comma-separated string or a sequence (Fire passes
    one for ``a,b``); empty keeps everything.
    """
    parts = wanted if isinstance(wanted, (tuple, list)) else (
        str(wanted or "").split(",")
    )
    keys = [str(c).strip().lower() for c in parts if str(c).strip()]
    return {
        name: dict(params) for name, params in available.items()
        if not keys or any(k in name.lower() for k in keys)
    }


def run_key(
    dataset: str, suite: str, config: str, params: Mapping[str, Any]
) -> str:
    """Identity of one benchmark run, for resuming."""
    return f"{dataset}|{suite}|{config}|{params_id(params)}"


def completed_runs(path: Path, suite: str) -> Set[str]:
    """Run keys already stored in ``path`` for ``suite``.

    Only suite runs resume: the built-in configurations (suite ``""``)
    are always measured again, as before. Unreadable lines are ignored.
    """
    if not suite or not path.is_file():
        return set()
    done: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("suite") == suite:
            done.add(run_key(
                str(row.get("dataset")), suite, str(row.get("config")),
                row.get("params") or {},
            ))
    return done


def remove_tree(path: Path) -> None:
    """Delete a directory tree if it exists."""
    if path.is_dir():
        shutil.rmtree(path)


def read_benchmark_rows(path: Path) -> List[BenchmarkResult]:
    """Benchmark rows stored as JSON Lines; a missing file has none.

    Raises:
        InvalidInputError: a line is not a valid benchmark row.
    """
    if not path.is_file():
        return []
    rows: List[BenchmarkResult] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(BenchmarkResult.model_validate_json(line))
        except ValueError as exc:
            raise InvalidInputError(
                f"{path}:{number}: {short_reason(exc)}"
            )
    return rows


class RagCLI:
    """RAG against the machine: index, search and answer over vLLM."""

    def __init__(
        self,
        index_dir: Optional[str] = None,
        mode: Optional[str] = None,
        semantic_weight: Optional[float] = None,
        fusion: Optional[str] = None,
        llm: Optional[str] = None,
        context_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        no_cache: Optional[bool] = None,
        rerank: Optional[bool] = None,
        rerank_depth: Optional[int] = None,
        rerank_weight: Optional[float] = None,
        profile: str = "",
        device: Optional[str] = None,
        torch_threads: Optional[int] = None,
        cpu_cores: Optional[str] = None,
    ):
        """Options shared by every command.

        An option given on the command line wins over the profile,
        which wins over the built-in default (see ``src/config.py``).

        Args:
            index_dir: Directory holding the index (data/processed).
            mode: Retrieval mode: auto, bm25, semantic or hybrid. ``auto``
                uses hybrid when the index has embeddings, else bm25.
            semantic_weight: Weight of the semantic score in hybrid mode.
            fusion: Hybrid fusion strategy: linear or rrf.
            llm: Hugging Face model used to generate answers.
            context_tokens: Token budget for retrieved context.
            max_new_tokens: Maximum length of a generated answer.
            no_cache: Disable the query/answer cache.
            rerank: Re-score the best candidates with a cross-encoder.
            rerank_depth: Number of candidates re-scored per query.
            rerank_weight: Share of the cross-encoder score (0..1).
            profile: JSON file of settings (or the name of one in
                experiments/profiles) applied on top of the defaults.
            device: auto (GPU when usable, else CPU), cpu, cuda or mps.
            torch_threads: Threads of the model runtime (default: one
                per performance core, or the measured best).
            cpu_cores: auto, all (no pinning) or performance.
        """
        self._options: dict[str, Any] = {
            "index_dir": index_dir, "mode": mode,
            "semantic_weight": semantic_weight, "fusion": fusion,
            "llm": llm, "context_tokens": context_tokens,
            "max_new_tokens": max_new_tokens, "no_cache": no_cache,
            "rerank": rerank, "rerank_depth": rerank_depth,
            "rerank_weight": rerank_weight, "profile": profile,
            "device": device, "torch_threads": torch_threads,
            "cpu_cores": cpu_cores,
        }
        self._container: Optional[Container] = None

    def _explicit(self) -> dict[str, Any]:
        """Validate and return the options given on the command line."""
        converters: dict[str, Callable[[Any, str], Any]] = {
            "index_dir": as_text, "mode": as_text, "fusion": as_text,
            "llm": as_text, "semantic_weight": as_float,
            "rerank_weight": as_float, "context_tokens": as_int,
            "max_new_tokens": as_int, "rerank_depth": as_int,
            "device": as_text, "torch_threads": as_int,
            "cpu_cores": as_text,
        }
        o = self._options
        values: dict[str, Any] = {
            name: convert(o[name], name)
            for name, convert in converters.items() if o[name] is not None
        }
        if o["rerank"] is not None:
            values["rerank"] = bool(o["rerank"])
        if o["no_cache"] is not None:
            values["use_cache"] = not bool(o["no_cache"])
        return values

    def _app(self, **overrides: Any) -> Container:
        """Validate the shared options and build the container."""
        if self._container is None or overrides:
            values = {**self._profile(), **self._explicit(), **overrides}
            try:
                settings = Settings.model_validate(values)
            except ValueError as exc:
                raise InvalidInputError(
                    f"Invalid option: {short_reason(exc)}"
                )
            self._container = Container(settings)
        return self._container

    def _profile(self) -> dict[str, Any]:
        """Read the settings profile named by ``--profile``, if any."""
        name = self._options["profile"]
        if not name:
            return {}
        path = Path(as_text(name, "profile"))
        if not path.is_file():
            path = Path("experiments/profiles") / f"{name}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            raise InvalidInputError(f"Profile not found: {name}")
        except ValueError:
            raise InvalidInputError(f"Profile {path} is not valid JSON.")
        if not isinstance(data, dict):
            raise InvalidInputError(f"Profile {path} must be an object.")
        unknown = sorted(set(data) - set(Settings.model_fields))
        if unknown:
            raise InvalidInputError(
                f"Profile {path}: unknown setting(s) {', '.join(unknown)}."
            )
        return data

    def lab(self) -> LabCLI:
        """Experiment lab: experiment, tune, analyze, train_router."""
        return LabCLI(self._app, safe_command)

    def skills(
        self,
        *,
        catalogue: Optional[str] = None,
        rubric_dir: Optional[str] = None,
        rubric_language: Optional[str] = None,
        transcripts_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        judge: Optional[str] = None,
        local_model: Optional[str] = None,
        api_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        dense_weight: Optional[float] = None,
        candidates: Optional[int] = None,
        min_incidents: Optional[int] = None,
        min_episodes: Optional[int] = None,
        min_participant_words: Optional[int] = None,
        min_similarity: Optional[float] = None,
        lab_dir: Optional[str] = None,
        speech_model: Optional[str] = None,
        voice_model: Optional[str] = None,
        voice_threshold: Optional[float] = None,
        tags_per_sentence: Optional[int] = None,
    ) -> SkillsCLI:
        """Soft-skills analyzer: assess, evidence, agreement, evaluate, lab.

        Options go between ``skills`` and the command, e.g.
        ``skills --judge api assess --transcript FILE``. The global
        ``--no_cache`` also disables the cache of judge replies.

        Args:
            catalogue: Rubric catalogue name (core).
            rubric_dir: Directory of the catalogues (knowledge/rubrics).
            rubric_language: auto (the transcript's language), en or es.
            transcripts_dir: Where the lab finds sessions by identifier.
            output_dir: Where scorecards and forms are written.
            judge: local (nothing leaves the machine) or api (Anthropic).
            local_model: Hugging Face model of the local judge.
            api_model: Model of the API judge.
            embedding_model: Multilingual sentence-embedding model.
            dense_weight: Weight of the semantic score in retrieval.
            candidates: Turns shown to the judge per competency.
            min_incidents: Verified incidents needed to give a rating.
            min_episodes: Separate exchanges those incidents must span.
            min_participant_words: Words a participant must have said
                to be rated at all.
            min_similarity: Drop quotes whose cosine with their
                indicator is below this value (0 keeps all).
            lab_dir: Directory of the skills lab (experiments/skills).
            speech_model: Speech-recognition model for ``transcribe``.
            voice_model: Voice-embedding model that tells speakers apart.
            voice_threshold: Cosine below which two voices are different
                people (used when --speakers is not given).
            tags_per_sentence: Most candidate tags kept per sentence.
        """
        options = dict(locals())
        options.pop("self")
        options["llm"] = options.pop("local_model")
        options["semantic_weight"] = options.pop("dense_weight")
        for shared in ("no_cache", "device", "torch_threads", "cpu_cores"):
            options[shared] = self._options[shared]
        return SkillsCLI(safe_command, options)

    @safe_command
    def benchmark(
        self, datasets: str = "scifact", configs: str = "",
        max_queries: int = 0, output: str = "", suite: str = "",
        clean_indexes: bool = False,
    ) -> None:
        """Score the pipeline on public BEIR datasets (downloaded once).

        Args:
            datasets: Comma-separated BEIR names (scifact, nfcorpus,
                fiqa, arguana, scidocs, ...) or CoIR ones (coir:cosqa,
                coir:stackoverflow-qa).
            configs: Comma-separated substrings selecting the
                configurations to run (default: all).
            max_queries: Use only the first N test queries (0 = all).
            output: JSON file to append the results to (default:
                experiments/benchmarks/beir.jsonl).
            suite: Run the trials of a lab suite (name or path) instead
                of the built-in configurations. Results already in
                ``output`` for the same suite, trial, parameters and
                dataset are skipped, so an interrupted run resumes.
            clean_indexes: Delete the benchmark indexes after each
                dataset (the results and the embedding cache are kept).
                Large studies otherwise need tens of gigabytes.
        """
        app = self._app()
        if str(suite or "").strip():
            spec = load_suite(suite, app.settings.lab_dir)
            available = {t.name: t.params for t in spec.expand()}
            suite_name = spec.name
        else:
            available, suite_name = dict(BENCHMARK_CONFIGS), ""
        chosen = select_configs(available, configs)
        target = Path(str(output) or "experiments/benchmarks/beir.jsonl")
        done = completed_runs(target, suite_name)
        raw = datasets if isinstance(datasets, (tuple, list)) else (
            str(datasets).split(",")
        )
        for dataset in [str(d).strip() for d in raw if str(d).strip()]:
            pending = {
                name: params for name, params in chosen.items()
                if run_key(dataset, suite_name, name, params) not in done
            }
            skipped = len(chosen) - len(pending)
            if not pending:
                print(f"{dataset}: all {skipped} configurations done")
                continue
            results = app.benchmarks.run(
                dataset, pending, as_int(max_queries, "max_queries")
            )
            first = results[0]
            print(f"{dataset}: {first.documents} documents, "
                  f"{first.queries} queries"
                  + (f" ({skipped} already done)" if skipped else ""))
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as handle:
                for r in results:
                    r = r.model_copy(update={"suite": suite_name})
                    handle.write(r.model_dump_json() + "\n")
                    print(f"  {r.config:<52} nDCG@10 {r.ndcg10:.3f}  "
                          f"R@100 {r.recall100:.3f}  hit@1/3/5/10 "
                          f"{r.hit[1]:.2f}/{r.hit[3]:.2f}/{r.hit[5]:.2f}/"
                          f"{r.hit[10]:.2f}  index {r.index_seconds:.0f}s"
                          f"  search {r.search_seconds:.1f}s")
            if clean_indexes:
                app.benchmark_factory.forget()
                remove_tree(Path(app.settings.benchmark_dir) / "work"
                            / "indexes")

    @safe_command
    def general(
        self, suite: str,
        benchmarks: str = "experiments/general/benchmarks.jsonl",
        output: str = "", max_loss: float = 0.02,
    ) -> None:
        """Rank a suite's configurations across the 42 lab and benchmarks.

        Run ``lab experiment --suite S`` and ``benchmark --suite S
        --output <benchmarks>`` first; this combines both.

        Args:
            suite: Suite name or path.
            benchmarks: JSON Lines file with the suite's benchmark rows.
            output: Markdown report (default:
                experiments/general/<suite>.md).
            max_loss: Largest relative loss on any dataset allowed for
                the robust choice.
        """
        app = self._app()
        spec = load_suite(suite, app.settings.lab_dir)
        rows = read_benchmark_rows(Path(str(benchmarks)))
        report = GeneralistAnalysis(
            max_loss=as_float(max_loss, "max_loss")
        ).analyse(
            spec.name, app.trial_store.all(), rows,
            order=[t.name for t in spec.expand()],
        )
        target = Path(
            str(output) or f"{app.settings.lab_dir}/general/{spec.name}.md"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_generalist(report), encoding="utf-8")
        missing = [c.name for c in report.configs if not c.complete]
        print(f"{spec.name}: {len(report.configs)} configurations on "
              f"{len(report.datasets)} datasets "
              f"({', '.join(report.datasets)})")
        for c in sorted(report.configs_complete(),
                        key=lambda c: -(c.mean_relative or 0.0)):
            print(f"  {c.name:<44} mean {100 * (c.mean_relative or 0):+5.1f}%"
                  f"  worst {100 * (c.worst_relative or 0):+5.1f}%"
                  f"  W/L {c.wins}/{c.losses}")
        if missing:
            print(f"  partial (not ranked): {', '.join(missing)}")
        print(f"Best mean: {report.best}; robust: {report.robust}")
        print(f"Report: {target}")

    @safe_command
    def system(self, benchmark: bool = False, repeats: int = 2) -> None:
        """Show the hardware found and how the models will run on it.

        Args:
            benchmark: Measure every candidate CPU configuration with
                the real models (about a minute) and remember the
                fastest for this machine.
            repeats: Measurements per configuration (the best counts).
        """
        app = self._app()
        profile, plan = app.hardware
        fast = len(profile.performance_cores)
        kind = (
            f"{fast} performance + {len(profile.cores) - fast} efficiency"
            if profile.is_hybrid() else "all equal"
        )
        print(f"CPU: {len(profile.cores)} cores ({kind}), "
              f"{len(profile.logical_cpus())} threads")
        print(f"Memory: {profile.memory_gb:.0f} GB "
              f"({profile.available_memory_gb:.0f} GB available)")
        gpus = ", ".join(
            f"{g.name} ({g.memory_gb:.0f} GB)" for g in profile.gpus
        )
        print(f"GPU: {gpus or 'none usable'}")
        for card in profile.cards:
            print(f"  card: {card.vendor}, kernel driver "
                  f"{card.driver or 'none'}")
        for problem in gpu_diagnosis(profile):
            print(f"  not usable: {problem}")
        if benchmark:
            tuner = app.runtime_tuner
            tuner.repeats = as_int(repeats, "repeats")
            tuning = tuner.tune()
            print("Measured (fastest first):")
            for r in tuning.results:
                print(f"  {r.plan.name:<40} embed {r.embed_seconds:6.2f} s"
                      f"  generate {r.tokens_per_second:5.1f} tok/s")
            save_measured(
                app.settings.runtime_plan_path, profile, tuning.best()
            )
            append_tuning(
                app.settings.runtime_history_path, profile, tuning
            )
            print(f"History: {app.settings.runtime_history_path}")
            self._container = None
            profile, plan = self._app().hardware
            print(f"Saved to {app.settings.runtime_plan_path}")
        pinned = (
            f"pinned to CPUs {','.join(map(str, plan.cpus))}"
            if plan.cpus else "not pinned"
        )
        print(f"Plan: device {plan.device}, {plan.threads} threads, "
              f"{pinned}")
        print(f"  {plan.reason}")
        choice = SkillsContainer(SkillsSettings(
            device=app.settings.device,
            torch_threads=app.settings.torch_threads,
            cpu_cores=app.settings.cpu_cores,
            runtime_plan_path=app.settings.runtime_plan_path,
        )).model_choice
        speed = (
            "" if choice.tokens_per_second is None
            else f", about {choice.tokens_per_second:g} tokens/s"
        )
        print(f"Skills judge (auto): {choice.model} "
              f"(~{choice.memory_gb:g} GB{speed})")
        for name, why in choice.rejected.items():
            print(f"  not {name}: {why}")

    @safe_command
    def index(
        self,
        max_chunk_size: Optional[int] = None,
        raw_dir: Optional[str] = None,
        semantic: Optional[bool] = None,
        force: bool = False,
        extensions: Optional[str] = None,
    ) -> None:
        """Ingest raw_dir and build (or incrementally update) the index.

        Args:
            max_chunk_size: Maximum chunk length in characters
                (default 2000, the maximum allowed).
            raw_dir: Corpus directory (default data/raw).
            semantic: Also build the embedding index (default True).
            force: Rebuild everything instead of only changed files.
            extensions: Comma-separated file extensions to index
                (default .py,.md,.txt,.rst).
        """
        overrides: dict[str, Any] = {}
        if max_chunk_size is not None:
            overrides["max_chunk_size"] = as_int(
                max_chunk_size, "max_chunk_size")
        if raw_dir is not None:
            overrides["raw_dir"] = as_text(raw_dir, "raw_dir")
        if semantic is not None:
            overrides["semantic"] = bool(semantic)
        if extensions is not None:
            overrides["extensions"] = parse_extensions(extensions)
        app = self._app(**overrides)
        settings = app.settings
        report = app.indexing.run(
            settings.max_chunk_size, settings.target_chunk_size,
            force=bool(force),
        )
        kind = "incremental" if report.incremental else "full"
        print(
            f"Ingestion complete ({kind}): {report.files_total} files, "
            f"{report.files_reindexed} indexed, {report.files_reused} "
            f"unchanged, {report.files_removed} removed, {report.chunks} "
            f"chunks in {report.seconds}s.\n"
            f"Indices saved under {app.settings.index_dir}/"
        )

    @safe_command
    def search(self, query: Any, k: int = 10) -> None:
        """Print the top-k sources for a single query as JSON."""
        text = validate_query(as_text(query, "query"))
        search = self._app().search
        hits = search.search(text, validate_k(k))
        out = MinimalSearchResults(
            question_id="",
            question=text,
            retrieved_sources=[h.chunk.to_source() for h in hits],
        ).model_dump()
        out["scores"] = [round(h.score, 4) for h in hits]
        out["mode"] = search.mode
        print(json.dumps(out, indent=2))
        if not hits:
            print("No matching sources found.", file=sys.stderr)

    @safe_command
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search every question of a dataset and save the results."""
        report = self._app().datasets.search_dataset(
            as_text(dataset_path, "dataset_path"),
            validate_k(k),
            as_text(save_directory, "save_directory"),
        )
        print(
            f"Searched {report.n_questions} questions in "
            f"{report.seconds}s (mode: {report.mode}).\n"
            f"Saved student_search_results to {report.output_path}"
        )

    @safe_command
    def answer(self, query: Any, k: int = 10) -> None:
        """Answer a single query from retrieved context."""
        text = validate_query(as_text(query, "query"))
        result = self._app().answers.answer(text, validate_k(k))
        print(result.answer)
        print("\nSources:")
        for i, src in enumerate(result.retrieved_sources, start=1):
            print(
                f"  [{i}] {src.file_path} "
                f"[{src.first_character_index}:{src.last_character_index}]"
            )

    @safe_command
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        k: Optional[int] = None,
    ) -> None:
        """Generate answers for saved search results.

        Args:
            student_search_results_path: Output of ``search_dataset``.
            save_directory: Where to write the answers.
            k: Use only the first k sources of each question as context
                (default: all of them).
        """
        path = as_text(student_search_results_path, "path")
        print(f"Loading questions from {path}")
        report = self._app().datasets.answer_dataset(
            path,
            as_text(save_directory, "save_directory"),
            validate_k(k) if k is not None else None,
        )
        print(
            f"Processed {report.n_questions} questions in "
            f"{report.seconds}s\n"
            f"Saved student_search_results_and_answer to "
            f"{report.output_path}"
        )

    @safe_command
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        """Report recall@1/3/5/10 against a ground-truth dataset."""
        report, warnings = self._app().datasets.evaluate(
            as_text(student_search_results_path, "path"),
            as_text(dataset_path, "dataset_path"),
            validate_k(k),
            as_int(max_context_length, "max_context_length"),
        )
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        print("Evaluation Results")
        print("=" * 40)
        print(" ".join(
            f"Recall@{x}: {v:.3f}" for x, v in report.recall.items()
        ))
        print(f"Questions: {report.n_questions} "
              f"(without results: {report.n_missing})")

    @safe_command
    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the local HTTP API (OpenAPI docs at /docs)."""
        import os

        import uvicorn

        from src.adapters.inbound.http_api import TranscriptRoutes, create_app

        port = as_int(port, "port")
        if not 0 < port < 65536:
            raise InvalidInputError("--port must be between 1 and 65535.")
        app = self._app()
        snapshot = app.search.snapshot
        print(f"Index ready: {len(snapshot.chunks)} chunks, "
              f"mode {app.search.mode}")
        token = app.transcripts_token(os.environ)
        variable = app.settings.transcripts_token_env
        if token:
            print(f"Transcripts: POST /transcripts into "
                  f"{app.settings.transcripts_dir} (token from ${variable})")
        else:
            print(f"Transcripts: routes off (set ${variable} to receive "
                  "them)")
        routes = TranscriptRoutes(
            app.ingest_transcript, app.show_transcript, token
        )
        uvicorn.run(create_app(app.search, app.answers, routes),
                    host=as_text(host, "host"), port=port)

    @safe_command
    def clear_cache(self) -> None:
        """Delete every cached search result and answer."""
        print(f"Removed {self._app().cache.clear()} cached entries.")
