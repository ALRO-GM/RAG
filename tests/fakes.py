"""In-memory fakes of the ports in :mod:`src.application.ports`.

They keep no state on disk and load no model, so the use cases can be
unit-tested quickly and deterministically. Each fake records the calls
the tests need to assert on.
"""

import hashlib
import json
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np
from pydantic import BaseModel, ValidationError

from src.application.ports import (
    FileRecord,
    FloatArray,
    IndexSnapshot,
)
from src.domain.chunking import default_registry
from src.domain.errors import CorruptedDataError, NotFoundError
from src.domain.experiments import Ranks, TrialResult, rank_metrics
from src.domain.models import (
    AnsweredQuestion,
    Chunk,
    IndexManifest,
    IndexReport,
    MinimalSource,
    ProcessedDocument,
    RagDataset,
    ScoredChunk,
)
from src.domain.priors import RouterModel
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.transcript import Transcript, build_transcript
from src.domain.text_analysis import tokenize

ModelT = TypeVar("ModelT", bound=BaseModel)
T = TypeVar("T")
RankRule = Callable[[Mapping[str, Any], str], int]
"""(params, query) → 1-based rank of the reference chunk (0 = absent)."""


def sha(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeCorpus:
    """A corpus held in a ``{path: text}`` dictionary."""

    def __init__(self, files: Mapping[str, str]):
        """Serve ``files``; mutate :attr:`files` to simulate edits."""
        self.files: Dict[str, str] = dict(files)

    def list_files(self) -> List[str]:
        """Return the paths, sorted."""
        return sorted(self.files)

    def file_hash(self, path: str) -> str:
        """Return the hash of the file text."""
        return sha(self.files[path])


class FakeProcessor:
    """Chunk documents of a :class:`FakeCorpus` in-process."""

    def __init__(self, corpus: FakeCorpus):
        """Read texts from ``corpus``."""
        self.corpus = corpus
        self.calls: List[List[str]] = []

    def process(
        self, paths: Sequence[str], max_size: int, target: Optional[int]
    ) -> Iterator[ProcessedDocument]:
        """Yield one processed document per path."""
        self.calls.append(list(paths))
        registry = default_registry()
        for path in paths:
            text = self.corpus.files[path]
            chunks = registry.chunk(path, text, max_size, target)
            yield ProcessedDocument(
                file_path=path,
                sha256=sha(text),
                chunks=chunks,
                term_counts=[
                    dict(Counter(tokenize(c.search_text()))) for c in chunks
                ],
            )


class FakeLexicalIndex:
    """Score = sum of the query-term counts of each document."""

    def __init__(self, term_counts: Sequence[Mapping[str, int]]):
        """Index the given documents."""
        self.term_counts = [dict(tc) for tc in term_counts]
        self.calls = 0

    def scores(self, terms: Sequence[str]) -> FloatArray:
        """Return one score per document."""
        self.calls += 1
        return np.asarray(
            [sum(tc.get(t, 0) for t in terms) for tc in self.term_counts],
            dtype=np.float32,
        )


class FakeLexicalBuilder:
    """Build :class:`FakeLexicalIndex` instances and count builds."""

    def __init__(self) -> None:
        """Start with no build."""
        self.builds = 0

    def build(
        self, term_counts: Sequence[Mapping[str, int]]
    ) -> FakeLexicalIndex:
        """Return an index over ``term_counts``."""
        self.builds += 1
        return FakeLexicalIndex(term_counts)


class FakeEmbedder:
    """Deterministic bag-of-hashed-words embeddings, L2-normalised."""

    def __init__(self, model_name: str = "fake-embedder", dim: int = 32):
        """Embed into ``dim`` dimensions."""
        self._model_name = model_name
        self.dim = dim
        self.encoded: List[str] = []
        self.calls = 0

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def encode(
        self, texts: Sequence[str], progress: bool = False
    ) -> FloatArray:
        """Return one normalised row per text."""
        self.calls += 1
        self.encoded.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in tokenize(text):
                out[row, int(sha(term)[:8], 16) % self.dim] += 1.0
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class FakeReranker:
    """Score 1 when the passage contains ``prefer``, else 0."""

    def __init__(self, prefer: str = "", model_name: str = "fake-reranker"):
        """Favour passages containing ``prefer``."""
        self.prefer = prefer
        self._model_name = model_name
        self.calls: List[List[Tuple[str, str]]] = []

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return one score per pair and record the call."""
        self.calls.append(list(pairs))
        return [
            1.0 if self.prefer and self.prefer in passage else 0.0
            for _, passage in pairs
        ]


class InMemoryIndexRepository:
    """Keep the last saved snapshot and records in memory."""

    def __init__(self) -> None:
        """Start empty."""
        self.snapshot: Optional[IndexSnapshot] = None
        self.records: Dict[str, FileRecord] = {}
        self.params: Dict[str, object] = {}
        self.loads = 0
        self.saves = 0

    def load_records(
        self, params: Mapping[str, object]
    ) -> Dict[str, FileRecord]:
        """Return the records if they were built with ``params``."""
        if dict(params) != self.params:
            return {}
        return dict(self.records)

    def save(
        self,
        snapshot: IndexSnapshot,
        records: Mapping[str, FileRecord],
        params: Mapping[str, object],
    ) -> None:
        """Store the snapshot and its records."""
        self.saves += 1
        self.snapshot = snapshot
        self.records = dict(records)
        self.params = dict(params)

    def load(self) -> IndexSnapshot:
        """Return the stored snapshot."""
        self.loads += 1
        if self.snapshot is None:
            raise NotFoundError("No index found.")
        return self.snapshot


class FakeGenerator:
    """Echo the question and the number of context chunks."""

    def __init__(self, model_name: str = "fake-llm"):
        """Answer with model ``model_name``."""
        self._model_name = model_name
        self.calls: List[Tuple[str, int]] = []

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def generate(self, question: str, context: Sequence[Chunk]) -> str:
        """Return a deterministic answer."""
        self.calls.append((question, len(context)))
        return f"answer to {question!r} from {len(context)} chunk(s)"


class InMemoryCache:
    """A dictionary-backed cache counting hits and writes."""

    def __init__(self) -> None:
        """Start empty."""
        self.data: Dict[str, str] = {}
        self.hits = 0
        self.sets = 0

    def get(self, key: str) -> Optional[str]:
        """Return the value stored under ``key``."""
        value = self.data.get(key)
        if value is not None:
            self.hits += 1
        return value

    def set(self, key: str, value: str) -> None:
        """Store ``value``."""
        self.sets += 1
        self.data[key] = value

    def clear(self) -> int:
        """Remove everything."""
        count = len(self.data)
        self.data.clear()
        return count


class InMemoryDatasetRepository:
    """JSON documents kept in a ``{path: json}`` dictionary."""

    def __init__(self, files: Optional[Mapping[str, str]] = None):
        """Serve ``files``."""
        self.files: Dict[str, str] = dict(files or {})

    def read(self, path: str, model: Type[ModelT]) -> ModelT:
        """Validate the stored JSON as ``model``."""
        if path not in self.files:
            raise NotFoundError(f"File not found: {path}")
        try:
            return model.model_validate_json(self.files[path])
        except ValidationError as exc:
            raise CorruptedDataError(str(exc))

    def write(self, directory: str, filename: str, data: BaseModel) -> str:
        """Store ``data`` under ``directory/filename``."""
        path = f"{directory}/{filename}"
        self.files[path] = data.model_dump_json()
        return path

    def json(self, path: str) -> object:
        """Return the stored document at ``path`` parsed."""
        return json.loads(self.files[path])


class FakeSourceReader:
    """Resolve sources against a ``{path: text}`` dictionary."""

    def __init__(self, files: Mapping[str, str]):
        """Serve ``files``."""
        self.files = dict(files)
        self.calls: List[List[MinimalSource]] = []

    def read(self, sources: Sequence[MinimalSource]) -> List[Chunk]:
        """Return the readable sources, skipping unknown paths."""
        self.calls.append(list(sources))
        out: List[Chunk] = []
        for src in sources:
            text = self.files.get(src.file_path)
            if text is None:
                continue
            out.append(Chunk(
                file_path=src.file_path,
                first_character_index=src.first_character_index,
                last_character_index=src.last_character_index,
                kind="source",
                text=text[
                    src.first_character_index:src.last_character_index
                ],
            ))
        return out


class RecordingProgress:
    """A progress reporter remembering every tracked loop."""

    def __init__(self) -> None:
        """Start with no loop."""
        self.loops: List[Tuple[str, int]] = []

    def track(
        self, items: Iterable[T], total: int, description: str
    ) -> Iterator[T]:
        """Yield ``items`` unchanged."""
        self.loops.append((description, total))
        yield from items


def make_chunk(
    path: str, start: int, text: str, context: str = "", kind: str = "text"
) -> Chunk:
    """Return a chunk of ``text`` starting at ``start`` in ``path``."""
    return Chunk(
        file_path=path,
        first_character_index=start,
        last_character_index=start + len(text),
        kind=kind,
        context=context,
        text=text,
    )


def make_snapshot(
    chunks: Sequence[Chunk],
    embedder: Optional[FakeEmbedder] = None,
    fingerprint: str = "fp",
) -> IndexSnapshot:
    """Build an in-memory snapshot, with embeddings if ``embedder``."""
    lexical = FakeLexicalIndex(
        [Counter(tokenize(c.search_text())) for c in chunks]
    )
    embeddings = None
    if embedder is not None:
        embeddings = embedder.encode([c.search_text() for c in chunks])
    manifest = IndexManifest(
        version=2,
        max_chunk_size=2000,
        semantic=embeddings is not None,
        embedding_model=embedder.model_name if embedder else "",
        fingerprint=fingerprint,
        n_chunks=len(chunks),
        n_files=len({c.file_path for c in chunks}),
    )
    return IndexSnapshot(manifest, list(chunks), lexical, embeddings)


class FakePipeline:
    """A pipeline whose results are scripted by its factory.

    For every query the reference chunk registered in the factory is
    placed at the rank given by the factory rule; the other positions
    are filler chunks of another file.
    """

    def __init__(self, factory: "FakePipelineFactory", params: Dict[str, Any]):
        """Serve the scripted results of ``factory`` for ``params``."""
        self.factory = factory
        self.params = params

    def index(self) -> Tuple[IndexReport, bool]:
        """Pretend to build the index (always a cold build)."""
        self.factory.index_calls += 1
        report = IndexReport(
            files_total=2, files_reindexed=2, files_reused=0,
            files_removed=0, chunks=self.factory.n_chunks,
            seconds=self.factory.index_seconds, incremental=False,
        )
        return report, True

    def search(
        self, queries: Sequence[str], k: int
    ) -> List[List[ScoredChunk]]:
        """Return ``k`` hits per query with the reference at its rank."""
        self.factory.search_calls.append(list(queries))
        results: List[List[ScoredChunk]] = []
        for query in queries:
            hits = [
                ScoredChunk(
                    chunk=make_chunk("filler.md", 100 * i, f"filler {i}"),
                    score=float(k - i),
                )
                for i in range(k)
            ]
            rank = self.factory.rule(self.params, query)
            target = self.factory.targets.get(query)
            if target is not None and 0 < rank <= k:
                hits[rank - 1] = ScoredChunk(
                    chunk=target, score=hits[rank - 1].score
                )
            results.append(hits)
        return results


class FakePipelineFactory:
    """Build :class:`FakePipeline` objects and count what they do."""

    def __init__(
        self,
        targets: Mapping[str, Chunk],
        rule: Optional[RankRule] = None,
        n_chunks: int = 7,
        index_seconds: float = 1.5,
    ):
        """Script the rank of ``targets`` (query → chunk) with ``rule``."""
        self.targets = dict(targets)
        self.rule: RankRule = rule or (lambda params, query: 1)
        self.n_chunks = n_chunks
        self.index_seconds = index_seconds
        self.created: List[Dict[str, Any]] = []
        self.index_calls = 0
        self.search_calls: List[List[str]] = []

    def create(self, params: Mapping[str, Any]) -> FakePipeline:
        """Return a pipeline for ``params`` and record the request."""
        self.created.append(dict(params))
        return FakePipeline(self, dict(params))


class InMemoryTrialStore:
    """Trial results in a dictionary, one per (suite, trial)."""

    def __init__(self, results: Sequence[TrialResult] = ()):
        """Start with ``results``."""
        self.results: Dict[Tuple[str, str], TrialResult] = {}
        self.appended = 0
        for result in results:
            self.results[(result.suite, result.trial_id)] = result

    def append(self, result: TrialResult) -> None:
        """Store ``result`` as the latest of its (suite, trial)."""
        self.appended += 1
        key = (result.suite, result.trial_id)
        self.results.pop(key, None)
        self.results[key] = result

    def get(self, trial_id: str) -> Optional[TrialResult]:
        """Return the latest result of ``trial_id`` in any suite."""
        for result in reversed(list(self.results.values())):
            if result.trial_id == trial_id:
                return result
        return None

    def all(self) -> List[TrialResult]:
        """Return every stored result, oldest first."""
        return list(self.results.values())


class InMemoryRouterRepository:
    """Keep the saved router model in memory."""

    def __init__(self, model: Optional[RouterModel] = None):
        """Start with ``model`` (or nothing saved)."""
        self.model = model
        self.saves = 0

    def load(self) -> Optional[RouterModel]:
        """Return the saved model, if any."""
        return self.model

    def save(self, model: RouterModel) -> str:
        """Remember ``model``."""
        self.saves += 1
        self.model = model
        return "memory://router"


def make_question(
    question: str, *chunks: Chunk, question_id: str = ""
) -> AnsweredQuestion:
    """Return a question whose references are the spans of ``chunks``."""
    return AnsweredQuestion(
        question_id=question_id or question,
        question=question,
        sources=[c.to_source() for c in chunks],
        answer="reference answer",
    )


def dataset_json(questions: Sequence[AnsweredQuestion]) -> str:
    """Return ``questions`` serialised as a RagDataset document."""
    return RagDataset(rag_questions=list(questions)).model_dump_json()


def found_ranks(found: Iterable[int], total: int, rank: int = 1) -> Ranks:
    """Return ranks of ``total`` questions: ``rank`` for ``found``, else 0."""
    hits = set(found)
    return [[rank if i in hits else 0] for i in range(total)]


def make_trial(
    trial_id: str,
    ranks: Mapping[str, Ranks],
    params: Optional[Mapping[str, Any]] = None,
    suite: str = "suite",
    name: str = "",
    baseline_id: str = "",
    seconds: float = 1.0,
    index_seconds: float = 0.0,
    index_cold: bool = False,
    n_chunks: int = 10,
) -> TrialResult:
    """Return a trial result whose metrics follow from ``ranks``."""
    return TrialResult(
        trial_id=trial_id,
        suite=suite,
        name=name or trial_id,
        params=dict(params or {}),
        baseline_id=baseline_id or trial_id,
        metrics={n: rank_metrics(r) for n, r in ranks.items()},
        ranks={n: [list(q) for q in r] for n, r in ranks.items()},
        search_seconds_per_100=seconds,
        index_seconds=index_seconds,
        index_cold=index_cold,
        n_chunks=n_chunks,
    )


# ------------------------------------------------- soft-skills analyzer

Reply = Callable[[str, str], str]
"""(system, user) → the scripted reply of a :class:`FakeChatModel`."""


class FakeChatModel:
    """A judge answering scripted replies and recording its calls.

    ``replies`` are served in order and the last one is repeated;
    ``responder``, when given, computes the reply from the prompts.
    """

    def __init__(
        self,
        replies: Sequence[str] = ("{}",),
        model_name: str = "fake-judge",
        responder: Optional[Reply] = None,
    ):
        """Script the replies of the model."""
        self.replies = list(replies)
        self._model_name = model_name
        self.responder = responder
        self.calls: List[Tuple[str, str, int]] = []

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the next scripted reply and record the call."""
        self.calls.append((system, user, max_tokens))
        if self.responder is not None:
            return self.responder(system, user)
        position = min(len(self.calls), len(self.replies)) - 1
        return self.replies[position]


class InMemoryTranscriptRepository:
    """Transcripts kept in a ``{session id: transcript}`` dictionary."""

    def __init__(self, transcripts: Sequence[Transcript] = ()):
        """Serve ``transcripts`` by session identifier (or as a path)."""
        self.transcripts: Dict[str, Transcript] = {
            t.session_id: t for t in transcripts
        }
        self.finds: List[str] = []

    def read(self, path: str) -> Transcript:
        """Return the transcript whose identifier is ``path``."""
        if path not in self.transcripts:
            raise NotFoundError(f"Transcript not found: {path}")
        return self.transcripts[path]

    def find(self, session_id: str) -> Transcript:
        """Return the transcript of ``session_id`` and record the lookup."""
        self.finds.append(session_id)
        if session_id not in self.transcripts:
            raise NotFoundError(f"No transcript for session {session_id!r}.")
        return self.transcripts[session_id]

    def sessions(self) -> List[str]:
        """Return the stored session identifiers, sorted."""
        return sorted(self.transcripts)


class InMemoryTranscriptStore:
    """``transcript.v1`` documents kept in a ``{session id: doc}`` dict."""

    def __init__(self) -> None:
        """Start empty."""
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.puts: List[str] = []

    def put(self, session_id: str, document: Mapping[str, Any]) -> bool:
        """Keep a copy of ``document``; tell whether one was replaced."""
        self.puts.append(session_id)
        replaced = session_id in self.documents
        self.documents[session_id] = json.loads(json.dumps(document))
        return replaced

    def get(self, session_id: str) -> Dict[str, Any]:
        """Return the document of ``session_id``."""
        if session_id not in self.documents:
            raise NotFoundError(f"No transcript for session {session_id!r}.")
        return self.documents[session_id]


class InMemoryCatalogueRepository:
    """Rubric catalogues kept by (name, language)."""

    def __init__(self, catalogues: Sequence[RubricCatalogue] = ()):
        """Serve ``catalogues`` and record every load request."""
        self.catalogues: Dict[Tuple[str, str], RubricCatalogue] = {
            (c.name, c.language): c for c in catalogues
        }
        self.loads: List[Tuple[str, str]] = []

    def load(self, name: str, language: str) -> RubricCatalogue:
        """Return the catalogue ``name`` in ``language``."""
        self.loads.append((name, language))
        key = (name, language[:2].lower())
        if key not in self.catalogues:
            raise NotFoundError(
                f"Catalogue {name!r} is not available in {language!r}."
            )
        return self.catalogues[key]

    def available(self) -> List[Tuple[str, str]]:
        """Return the stored (name, language) pairs, sorted."""
        return sorted(self.catalogues)


CATALOGUE_TEXTS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "en": {
        "listening": ("Active listening",
                      "Attends to what the others say and checks it."),
        "L1": ("Asks a clarifying question", "what do you mean exactly"),
        "L2": ("Summarises what others said",
               "so if I understood correctly"),
        "L3": ("Dismisses others", "that is nonsense"),
        "planning": ("Planning", "Organises the work and commits to it."),
        "P1": ("Commits to a deadline", "I will finish it by Wednesday"),
        "P2": ("Refuses to plan", "planning is a waste of time"),
    },
    "es": {
        "listening": ("Escucha activa",
                      "Atiende a lo que dicen los demás y lo comprueba."),
        "L1": ("Hace una pregunta aclaratoria",
               "qué quieres decir exactamente"),
        "L2": ("Resume lo que dijeron los demás",
               "si lo he entendido bien"),
        "L3": ("Descalifica a los demás", "eso es una tontería"),
        "planning": ("Planificación",
                     "Organiza el trabajo y se compromete con él."),
        "P1": ("Se compromete con un plazo", "lo termino el miércoles"),
        "P2": ("Se niega a planificar",
               "planificar es perder el tiempo"),
    },
}
"""Texts of the test catalogue: competency (name, definition) and
indicator (text, example utterance), per language."""

NEGATIVE_INDICATORS = ("L3", "P2")


def catalogue_data(
    language: str = "en", name: str = "core", version: str = "1.0"
) -> Dict[str, Any]:
    """Return the test catalogue as JSON-ready data.

    Two competencies: ``listening`` (L1 +, L2 +, L3 -) and ``planning``
    (P1 +, P2 -), on a 1-5 scale, with the same ids in every language.
    """
    texts = CATALOGUE_TEXTS[language]

    def indicator(indicator_id: str) -> Dict[str, Any]:
        """Return one indicator of the catalogue."""
        text, example = texts[indicator_id]
        negative = indicator_id in NEGATIVE_INDICATORS
        return {
            "id": indicator_id, "text": text, "examples": [example],
            "polarity": "negative" if negative else "positive",
        }

    def competency(competency_id: str, ids: Sequence[str]) -> Dict[str, Any]:
        """Return one competency of the catalogue."""
        title, definition = texts[competency_id]
        return {
            "id": competency_id, "name": title, "definition": definition,
            "indicators": [indicator(i) for i in ids],
            "anchors": {1: "rarely shown", 3: "shown at times",
                        5: "shown consistently"},
            "sources": ["test framework"],
            "limits": "tone of voice",
        }

    return {
        "name": name, "version": version, "language": language,
        "scale": {"minimum": 1, "maximum": 5,
                  "labels": {2: "below", 4: "above"}},
        "competencies": [
            competency("listening", ["L1", "L2", "L3"]),
            competency("planning", ["P1", "P2"]),
        ],
        "excluded": [{"name": "Empathy", "reason": "needs the voice"}],
        "references": ["a reference"],
    }


def make_catalogue(
    language: str = "en", name: str = "core", version: str = "1.0"
) -> RubricCatalogue:
    """Return the small two-competency test catalogue (see above)."""
    return RubricCatalogue.model_validate(
        catalogue_data(language, name, version)
    )


TurnSpec = Tuple[Any, ...]
"""``(speaker, text)`` or ``(speaker, text, start, end)``."""

MEETING_EN: Tuple[Tuple[str, str], ...] = (
    ("Maya", "What do you mean exactly by streaming the file, Priya?"),
    ("Tom", "That is nonsense, nobody will test with the big file."),
    ("Priya", "I measured the files yesterday and the largest one is "
              "twelve gigabytes."),
    ("Maya", "So if I understood correctly, the big file must work and "
             "we stream it."),
    ("Tom", "Planning is a waste of time, we should be coding already."),
    ("Priya", "I will finish it by Wednesday afternoon, the reader "
              "I mean."),
    ("Maya", "Okay."),
    ("Tom", "Fine, I will finish it by Friday then, my part of the "
            "parser."),
)
"""A short English meeting whose turns echo the test catalogue:
turn 0 → L1, 1 → L3, 3 → L2, 4 → P2, 5 and 7 → P1; 6 is too short."""

MEETING_ES: Tuple[Tuple[str, str], ...] = (
    ("Ana", "¿Qué quieres decir exactamente con leer el fichero por "
            "partes, Marta?"),
    ("Luis", "Eso es una tontería, nadie va a probar con el fichero "
             "grande."),
    ("Marta", "Ayer medí los ficheros y el más grande ocupa doce "
              "gigas."),
    ("Ana", "Entonces, si lo he entendido bien, el fichero grande "
            "tiene que funcionar."),
    ("Luis", "Planificar es perder el tiempo, deberíamos estar "
             "programando ya."),
    ("Marta", "Lo termino el miércoles por la tarde, el lector quiero "
              "decir."),
    ("Ana", "Vale."),
    ("Luis", "Bueno, lo termino el viernes entonces, mi parte."),
)
"""The same meeting in Spanish, turn for turn."""


def make_transcript(
    turns: Sequence[TurnSpec] = MEETING_EN,
    session_id: str = "s1",
    language: Optional[str] = None,
) -> Transcript:
    """Return a transcript with one turn per ``turns`` item, unmerged."""
    segments: List[Dict[str, Any]] = []
    for position, spec in enumerate(turns):
        segment: Dict[str, Any] = {"speaker": spec[0], "text": spec[1]}
        if len(spec) >= 4:
            segment.update(start=spec[2], end=spec[3])
        segments.append(segment)
        if position and turns[position - 1][0] == spec[0]:
            raise ValueError("consecutive turns of one speaker would merge")
    return build_transcript(session_id, segments, language)


def judge_reply(
    evidence: Sequence[Tuple[int, str, str]] = (),
    rating: Optional[int] = None,
    rationale: str = "because",
) -> str:
    """Return a well-formed judge reply: (turn, quote, indicator) items."""
    return json.dumps({
        "evidence": [
            {"turn": turn, "quote": quote, "indicator": indicator}
            for turn, quote, indicator in evidence
        ],
        "rating": rating,
        "rationale": rationale,
    })
