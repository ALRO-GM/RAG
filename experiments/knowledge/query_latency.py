"""Measure ``POST /knowledge/query`` latency on a synthetic domain (PL-8).

Usage (from an empty working folder; it writes a domain and its index
there)::

    PYTHONPATH=<repo> HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= \\
        python <repo>/experiments/knowledge/query_latency.py <folder> \\
        [files=45] [example.json]

The domain ``interview.backend`` gets ``files`` Markdown guides of
eleven sections (random Spanish technical words, front matter with a
script phase and an intent) and 40 JSONL questions; with 45 files it
has about a thousand chunks. The real knowledge embedder indexes it,
then 200 queries without and 200 with a ``script_phase`` filter go
through the FastAPI app (in process, bearer token) and the p50, p90
and maximum are printed. The text is synthetic: the numbers are about
speed, never about retrieval quality.
"""

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from src.adapters.inbound.http_api import (
    KnowledgeRoutes,
    TranscriptRoutes,
    create_app,
)
from src.adapters.outbound.fs_transcript_store import (
    FileSystemTranscriptStore,
)
from src.application.skills.ingest import (
    IngestTranscript,
    ShowIngestedTranscript,
)
from src.bootstrap import Container
from src.config import Settings

WORDS = (
    "migración monolito servicios transacciones base de datos colas "
    "eventos latencia caché despliegue contenedores kubernetes pruebas "
    "integración contrato api versión seguridad autenticación token "
    "observabilidad métricas trazas alertas equipo revisión código "
    "arquitectura dominio agregado consistencia eventual réplica "
    "partición escalado coste incidencia postmortem cliente"
).split()
TOKEN = "latency-token"
QUERIES = 200


def sentence(rng: random.Random, words: int) -> str:
    """Return ``words`` random words as a sentence."""
    return " ".join(rng.choice(WORDS) for _ in range(words)).capitalize() \
        + "."


def write_domain(rng: random.Random, files: int) -> None:
    """Write the synthetic domain under ``knowledge/domains``."""
    root = Path("knowledge/domains/interview.backend")
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.yaml").write_text(
        "id: interview.backend\ntitle: Entrevistas backend\nlanguage: es\n"
        "vocabularies:\n  script_phase: [intro, technical, closing]\n"
        "  intent: [architecture, operations, team]\n", encoding="utf-8",
    )
    for f in range(files):
        parts = [
            f"---\nscript_phase: "
            f"{rng.choice(['intro', 'technical', 'closing'])}\n"
            f"intent: {rng.choice(['architecture', 'operations', 'team'])}"
            "\n---\n"
        ]
        for s in range(11):
            parts.append(f"# Tema {f}.{s}\n\n{sentence(rng, 60)}\n\n"
                         f"{sentence(rng, 60)}\n")
        (root / f"guia-{f:03d}.md").write_text("\n".join(parts),
                                               encoding="utf-8")
    with open(root / "preguntas.jsonl", "w", encoding="utf-8") as handle:
        for q in range(40):
            handle.write(json.dumps({
                "id": f"q-arch-{q:02d}",
                "text": "¿" + sentence(rng, 14)[:-1] + "?",
                "intent": "architecture", "category": "pregunta",
                "script_phase": "technical", "tags": ["microservicios"],
            }, ensure_ascii=False) + "\n")


def query(
    rng: random.Random, n: int, filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return a ``knowledge.query.v1`` body of six random words."""
    body: Dict[str, Any] = {
        "schema": "knowledge.query.v1", "request_id": f"r-{n}",
        "domain": "interview.backend", "top_k": 4,
        "text": " ".join(rng.choice(WORDS) for _ in range(6)),
    }
    if filters:
        body["filters"] = filters
    return body


def main(folder: str, files: int = 45, example: str = "") -> None:
    """Build the domain in ``folder``, index it and time the queries."""
    os.chdir(folder)
    rng = random.Random(7)
    write_domain(rng, files)
    container = Container(Settings(index_dir="no-index"))
    start = time.perf_counter()
    done = container.knowledge_registry.index("interview.backend")
    print(f"indexed: {done.report.files_total} files, {done.report.chunks}"
          f" chunks in {time.perf_counter() - start:.1f}s on "
          f"{container.runtime.device}")
    store = FileSystemTranscriptStore("transcripts")
    client = TestClient(create_app(
        container.search, container.answers,
        TranscriptRoutes(
            IngestTranscript(store), ShowIngestedTranscript(store), TOKEN,
            knowledge=KnowledgeRoutes(container.knowledge_query),
        ),
    ))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    for n in range(5):  # loads the model
        client.post("/knowledge/query", json=query(rng, n), headers=headers)
    for label, filters in (
        ("no filter", None),
        ("filter script_phase", {"script_phase": ["technical"]}),
    ):
        times: List[float] = []
        for n in range(QUERIES):
            begin = time.perf_counter()
            answer = client.post("/knowledge/query", headers=headers,
                                 json=query(rng, n, filters))
            times.append((time.perf_counter() - begin) * 1000)
            assert answer.status_code == 200, answer.text
        times.sort()
        print(f"{label}: n={QUERIES} p50 {statistics.median(times):.1f} ms"
              f", p90 {times[int(QUERIES * 0.9) - 1]:.1f} ms, max "
              f"{times[-1]:.1f} ms")
    if example:
        answer = client.post("/knowledge/query", headers=headers, json={
            "schema": "knowledge.query.v1", "request_id": "r-91",
            "domain": "interview.backend",
            "text": "migración de monolito a microservicios",
            "language": "es", "top_k": 1,
            "filters": {"script_phase": ["technical"],
                        "category": ["pregunta"]},
        })
        Path(example).write_text(
            json.dumps(answer.json(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 45,
         sys.argv[3] if len(sys.argv) > 3 else "")
