"""Public labelled corpora read from Hugging Face parquet shards.

The corpora of :data:`~transcriber.domain.lab_reference.CORPORA` are
datasets prepared by ``diarizers`` (``diarizers-community/ami``,
``.../callhome``): each config holds ``<config>/<split>-NNNNN-of-MMMMM
.parquet`` shards whose rows are one recording each, with the columns
``audio`` (``{bytes, path}``: the file and its name), and
``timestamps_start``, ``timestamps_end`` and ``speakers`` (one value
per speaker turn; turns of different people may overlap). Neither
corpus has a transcript, so no reference words are given. A recording
is named after its file (``ES2004a.Array1-01.wav`` → ``ES2004a``);
CALLHOME's rows have no file name, so its recordings are numbered in
corpus order (``data-0001``...) and their format is told from the
bytes.

:class:`HuggingFaceCorpus` downloads only the shards it reads, in order,
into ``<directory>/<corpus>/`` (``<data_dir>/lab-corpora`` by default,
ignored by git), and stops at the shard that holds the last recording
asked for. A gated corpus is checked first: until the account whose
token is used (``huggingface-cli login`` or ``HF_TOKEN``) has accepted
its terms, it raises :class:`CorpusAccessError` saying where to accept
them. The network calls are injectable, so the tests read a tiny local
corpus with no network.
"""

from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

from transcriber.application.lab_ports import CorpusRecording
from transcriber.application.ports import RecordedAudio
from transcriber.domain.errors import CorpusAccessError
from transcriber.domain.lab_reference import CorpusSpec, recording_name

CORPORA_DIR = "lab-corpora"
"""Where the shards go, under the data directory (ignored by git)."""

COLUMNS = ["audio", "timestamps_start", "timestamps_end", "speakers"]

ListFiles = Callable[[str], List[str]]
"""Repo id → the files of the dataset repo."""
Download = Callable[[str, str, Path], Path]
"""(repo id, file in the repo, local directory) → the local file."""
CheckAccess = Callable[[str], None]
"""Raise ``GatedRepoError`` unless the account may read the repo."""


def _list_files(repo: str) -> List[str]:
    """List a dataset repo's files on the Hub."""
    from huggingface_hub import list_repo_files

    return list_repo_files(repo, repo_type="dataset")


def _download(repo: str, filename: str, directory: Path) -> Path:
    """Download one file of a dataset repo into ``directory``."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo, filename, repo_type="dataset", local_dir=directory,
    ))


def _check_access(repo: str) -> None:
    """Raise unless the logged-in account may read the dataset repo."""
    from huggingface_hub import auth_check

    auth_check(repo, repo_type="dataset")


def _gated(exc: BaseException) -> bool:
    """Tell whether ``exc`` is the Hub refusing a gated repo."""
    from huggingface_hub.errors import GatedRepoError

    return isinstance(exc, GatedRepoError)


class HuggingFaceCorpus:
    """Read corpus recordings from the Hub's parquet shards (see module)."""

    def __init__(
        self,
        directory: Path,
        list_files: ListFiles = _list_files,
        download: Download = _download,
        check_access: CheckAccess = _check_access,
    ) -> None:
        """Keep the shards under ``directory``; nothing is read yet."""
        self._directory = Path(directory).expanduser()
        self._list_files = list_files
        self._download = download
        self._check_access = check_access

    def recordings(
        self, spec: CorpusSpec, split: str, limit: Optional[int]
    ) -> Iterator[CorpusRecording]:
        """Yield up to ``limit`` recordings of ``split``, in shard order."""
        if spec.gated:
            self._call(spec, lambda: self._check_access(spec.repo))
        local = self._directory / spec.id
        given = 0
        for shard in self._shards(spec, split):
            if limit is not None and given >= limit:
                return
            path = self._call(
                spec, lambda: self._download(spec.repo, shard, local)
            )
            for number, row in enumerate(_rows(path)):
                if limit is not None and given >= limit:
                    return
                given += 1
                yield _recording(row, f"{split}-{given:04d}", number)

    def _shards(self, spec: CorpusSpec, split: str) -> List[str]:
        """Return the shards of ``split`` in order (listed on the Hub).

        Without the Hub (offline), the shards already downloaded.
        """
        prefix = f"{spec.config}/{split}-"
        try:
            files = self._call(spec, lambda: self._list_files(spec.repo))
        except CorpusAccessError:
            folder = self._directory / spec.id / spec.config
            files = [
                f"{spec.config}/{p.name}"
                for p in folder.glob(f"{split}-*.parquet")
            ]
            if not files:
                raise
        shards = sorted(
            f for f in files
            if f.startswith(prefix) and f.endswith(".parquet")
        )
        if not shards:
            raise CorpusAccessError(
                f"{spec.display_name} has no shard for split {split!r}."
            )
        return shards

    def _call(self, spec: CorpusSpec, work: Callable[[], Any]) -> Any:
        """Run a Hub call; turn its failures into ``CorpusAccessError``."""
        try:
            return work()
        except CorpusAccessError:
            raise
        except Exception as exc:
            if _gated(exc):
                raise CorpusAccessError(
                    f"{spec.display_name} is gated: accept its terms at"
                    f" {spec.source_url} with the Hugging Face account you"
                    " log in with (huggingface-cli login, or HF_TOKEN),"
                    " then import again."
                ) from exc
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() \
                else type(exc).__name__
            raise CorpusAccessError(
                f"Cannot read {spec.display_name} from {spec.source_url}:"
                f" {reason}"
            ) from exc


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the rows of a parquet shard one at a time."""
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(path)
    for batch in shard.iter_batches(batch_size=1, columns=COLUMNS):
        yield from batch.to_pylist()


MAGIC = {b"RIFF": "wav", b"fLaC": "flac", b"OggS": "ogg"}
"""File signatures of the formats a nameless row may hold."""


def _sniff(head: bytes) -> str:
    """Return the format of audio bytes without a file name (WAV if unknown).

    CALLHOME's rows have no file name; the decoder reads WAV, FLAC and
    OGG with libsndfile and anything else with PyAV.
    """
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return "mp3"
    return MAGIC.get(head, "wav")


def _recording(
    row: dict[str, Any], fallback: str, number: int
) -> CorpusRecording:
    """Return one parquet row as a corpus recording."""
    audio = row.get("audio") or {}
    data = audio.get("bytes") if isinstance(audio, dict) else None
    name = audio.get("path") if isinstance(audio, dict) else None
    if not data:
        raise CorpusAccessError(
            f"Row {number} of the corpus shard has no audio."
        )
    suffix = Path(str(name or "")).suffix.lstrip(".").lower() or _sniff(
        bytes(data[:4])
    )
    return CorpusRecording(
        recording=recording_name(name, fallback),
        audio=RecordedAudio(bytes(data), suffix),
        starts=list(row.get("timestamps_start") or []),
        ends=list(row.get("timestamps_end") or []),
        speakers=[str(s) for s in row.get("speakers") or []],
    )
