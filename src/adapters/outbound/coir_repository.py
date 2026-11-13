"""Benchmark source adapter: CoIR code-retrieval tasks (Hugging Face)."""

from pathlib import Path
from typing import Any, Dict, List

from src.adapters.outbound.beir_repository import write_documents
from src.application.benchmarking import BenchmarkData
from src.domain.errors import CorruptedDataError, NotFoundError

REPOSITORY = "CoIR-Retrieval/{name}"
MEMBERS = {
    "corpus": "corpus/corpus-00000-of-00001.parquet",
    "queries": "queries/queries-00000-of-00001.parquet",
    "qrels": "data/test-00000-of-00001.parquet",
}
EXTENSIONS = {"cosqa": ".py"}
"""Tasks whose documents are Python code go through the AST chunker."""


class CoirRepository:
    """Download a CoIR task and expose it as files to index.

    The whole corpus is searched and only the test queries are scored,
    as the benchmark does. Documents of a Python task are written as
    ``.py`` files so that the code chunker handles them, which is the
    path a real code corpus takes.
    """

    def __init__(self, directory: str = "data/benchmarks/coir"):
        """Keep document files under ``directory``."""
        self.directory = Path(directory)

    def _rows(self, name: str, member: str) -> List[Dict[str, Any]]:
        """Download one parquet member and return its rows."""
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import HfHubHTTPError
            import pyarrow.parquet as parquet
        except ImportError:
            raise NotFoundError(
                "Reading CoIR needs pyarrow; run `uv sync --group bench`."
            )
        try:
            path = hf_hub_download(
                REPOSITORY.format(name=name), MEMBERS[member],
                repo_type="dataset",
            )
            rows: List[Dict[str, Any]] = parquet.read_table(path).to_pylist()
            return rows
        except (HfHubHTTPError, OSError) as exc:
            raise NotFoundError(
                f"Cannot download CoIR task {name}: {type(exc).__name__}"
            )
        except ValueError as exc:
            raise CorruptedDataError(f"CoIR task {name} is unreadable: {exc}")

    def load(self, name: str) -> BenchmarkData:
        """Return the corpus files, test queries and judgements."""
        if not name.replace("-", "").replace("_", "").isalnum():
            raise NotFoundError(f"Invalid dataset name {name!r}.")
        try:
            qrels: Dict[str, Dict[str, int]] = {}
            for row in self._rows(name, "qrels"):
                qrels.setdefault(str(row["query-id"]), {})[
                    str(row["corpus-id"])
                ] = int(row["score"])
            queries = {
                str(q["_id"]): str(q["text"])
                for q in self._rows(name, "queries")
                if str(q["_id"]) in qrels
            }
            doc_ids = write_documents(
                self.directory / name / "files",
                (
                    (str(d["_id"]), str(d.get("title") or ""),
                     str(d.get("text") or ""))
                    for d in self._rows(name, "corpus")
                ),
                EXTENSIONS.get(name, ".txt"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptedDataError(
                f"CoIR task {name} has an unexpected layout: {exc}"
            )
        return BenchmarkData(
            name=f"coir:{name}",
            raw_dir=str(self.directory / name / "files"),
            doc_ids=doc_ids,
            queries={q: queries[q] for q in qrels if q in queries},
            qrels=qrels,
        )


class BenchmarkCatalogue:
    """Route a dataset name to its source: ``coir:<task>`` or BEIR."""

    def __init__(self, beir: Any, coir: Any):
        """Wire the two sources."""
        self.beir = beir
        self.coir = coir

    def load(self, name: str) -> BenchmarkData:
        """Return the dataset ``name``."""
        if name.startswith("coir:"):
            data: BenchmarkData = self.coir.load(name[len("coir:"):])
            return data
        loaded: BenchmarkData = self.beir.load(name)
        return loaded
