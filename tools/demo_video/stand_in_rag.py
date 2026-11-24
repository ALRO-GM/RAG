"""A stand-in for the RAG's analysis read routes, for the demo recording.

The recording needs only the result page of the scripted interview
(``#/analysis/demo-entrevista-lucia-marin``). The real RAG loads its
models at start-up; this small app installs the demo results of
``examples/demo`` into a temporary store with the RAG's own code
(``install_demos``) and serves the two read routes the page calls,
so the result page shows exactly what the real RAG would show.

Usage: ``python -m tools.demo_video.stand_in_rag STORE_DIR PORT``.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException

from src.adapters.outbound.fs_analysis_store import FileSystemAnalysisStore
from src.adapters.outbound.fs_demo_analyses import read_demo_analyses
from src.application.skills.demo_analysis import install_demos
from src.application.skills.session_analysis import AnalysisResults
from src.domain.errors import NotFoundError

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "examples" / "demo"


def build_app(store_dir: Path) -> FastAPI:
    """Return the app serving the demo results installed in ``store_dir``."""
    store = FileSystemAnalysisStore(str(store_dir))
    installed = install_demos(store, read_demo_analyses(DEMO_DIR),
                              report=print)
    print(f"demo analyses installed: {installed}", flush=True)
    results = AnalysisResults(store)
    app = FastAPI()

    @app.get("/analysis/sessions/{session_id}")
    def status(session_id: str) -> Dict[str, Any]:
        """Return the stored run, never active."""
        try:
            run = results.status(session_id)
        except NotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {**run.model_dump(mode="json"), "active": False}

    @app.get("/analysis/sessions/{session_id}/steps/{step}")
    def step(session_id: str, step: str) -> Any:
        """Return one stored step result."""
        try:
            return results.result(session_id, step)
        except NotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app


def main(argv: List[str]) -> int:
    """Serve the stand-in on loopback: ``STORE_DIR PORT``."""
    if len(argv) != 2:
        print("usage: python -m tools.demo_video.stand_in_rag STORE_DIR PORT",
              file=sys.stderr)
        return 2
    uvicorn.run(build_app(Path(argv[0])), host="127.0.0.1",
                port=int(argv[1]), log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
