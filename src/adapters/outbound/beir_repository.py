"""Benchmark source adapter: BEIR datasets downloaded from the web."""

import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.application.benchmarking import BenchmarkData
from src.domain.errors import CorruptedDataError, NotFoundError

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
    "{name}.zip"
)


def write_documents(
    directory: Path,
    documents: Iterable[Tuple[str, str, str]],
    extension: str = ".txt",
) -> Dict[str, str]:
    """Write (id, title, text) documents as files; return stem → id.

    Files are named by position, so document ids never become paths.
    A directory already written completely is not rewritten.
    """
    doc_ids: Dict[str, str] = {}
    done = directory / ".complete"
    fresh = not done.is_file()
    if fresh:
        directory.mkdir(parents=True, exist_ok=True)
    for position, (doc_id, title, text) in enumerate(documents):
        stem = f"d{position:07d}"
        doc_ids[stem] = doc_id
        if fresh:
            title, text = title.strip(), text.strip()
            (directory / f"{stem}{extension}").write_text(
                f"{title}\n\n{text}" if title else text,
                encoding="utf-8", errors="replace",
            )
    done.write_text("ok", encoding="utf-8")
    return doc_ids


class BeirRepository:
    """Download a BEIR dataset once and expose it as files to index.

    The downloaded archive is data, never code: only the three known
    members (``corpus.jsonl``, ``queries.jsonl``, ``qrels/test.tsv``)
    are read, straight from the zip. Every document is written as
    ``<position>.txt`` holding its title and text, which is how the
    reference systems concatenate them.
    """

    def __init__(self, directory: str = "data/benchmarks/beir"):
        """Keep archives and document files under ``directory``."""
        self.directory = Path(directory)

    def _archive(self, name: str) -> Path:
        """Return the local archive, downloading it when missing."""
        target = self.directory / f"{name}.zip"
        if target.is_file():
            return target
        self.directory.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        try:
            urllib.request.urlretrieve(  # nosec: fixed https host
                BEIR_URL.format(name=name), partial
            )
        except (urllib.error.URLError, OSError) as exc:
            raise NotFoundError(f"Cannot download BEIR dataset {name}: {exc}")
        partial.rename(target)
        return target

    def load(self, name: str) -> BenchmarkData:
        """Return the corpus files, test queries and judgements."""
        if not name.replace("-", "").replace("_", "").isalnum():
            raise NotFoundError(f"Invalid dataset name {name!r}.")
        files = self.directory / name / "files"
        try:
            with zipfile.ZipFile(self._archive(name)) as archive:
                corpus = self._lines(archive, f"{name}/corpus.jsonl")
                queries = {
                    str(q["_id"]): str(q["text"])
                    for q in self._lines(archive, f"{name}/queries.jsonl")
                }
                qrels = self._qrels(archive, f"{name}/qrels/test.tsv")
        except (zipfile.BadZipFile, KeyError, ValueError) as exc:
            raise CorruptedDataError(
                f"BEIR dataset {name} is not readable: {exc}"
            )
        doc_ids = write_documents(files, (
            (str(d["_id"]), str(d.get("title") or ""),
             str(d.get("text") or ""))
            for d in corpus
        ))
        return BenchmarkData(
            name=name,
            raw_dir=str(files),
            doc_ids=doc_ids,
            queries={q: queries[q] for q in qrels if q in queries},
            qrels=qrels,
        )

    @staticmethod
    def _lines(
        archive: zipfile.ZipFile, member: str
    ) -> List[Dict[str, object]]:
        """Read a JSON-lines member of the archive."""
        with archive.open(member) as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @staticmethod
    def _qrels(
        archive: zipfile.ZipFile, member: str
    ) -> Dict[str, Dict[str, int]]:
        """Read the judgements: query id → document id → grade."""
        qrels: Dict[str, Dict[str, int]] = {}
        with archive.open(member) as handle:
            for number, raw in enumerate(handle):
                fields = raw.decode("utf-8").rstrip("\n").split("\t")
                if number == 0 or len(fields) < 3:
                    continue
                qrels.setdefault(fields[0], {})[fields[1]] = int(fields[2])
        return qrels
