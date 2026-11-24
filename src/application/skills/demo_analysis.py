"""Use case: install the prepared results of the scripted demos (#232).

The RAG installs every ``analysis.demo.v1`` result in its analysis store
when it starts (:func:`install_demos`), so the live page's demo can end
on its candidate's result page. A result already installed with the
same ``started_at`` is left alone; a newer file replaces it. Only
``demo-`` sessions are written, never over a run that is not itself
synthetic (a real analysis under that id is kept), and only while the
session's analysis lock is free. A demo that cannot be installed is
reported, not raised, so the RAG still starts.
"""

from typing import Callable, Dict, List, Optional, Sequence

from src.application.skills.ports import AnalysisLock, AnalysisStore
from src.application.skills.session_analysis import dumps
from src.domain.errors import RagError
from src.domain.skills.analysis_run import STEP_OUTPUTS
from src.domain.skills.demo_analysis import DemoAnalysis

OUTPUTS: Dict[str, str] = {str(k): v for k, v in STEP_OUTPUTS.items()}
"""The file of each step, by plain name."""


def install_demo(
    store: AnalysisStore, bundle: DemoAnalysis,
    lock: Optional[AnalysisLock] = None,
) -> bool:
    """Store one demo result; return whether anything was written."""
    current = store.load(bundle.session_id)
    if current is not None and (
        current.started_at == bundle.started_at or not current.synthetic
    ):
        return False
    if lock is not None and lock.acquire(bundle.session_id) is None:
        return False
    try:
        store.clear(bundle.session_id)
        for name, content in bundle.files.items():
            store.write(bundle.session_id, OUTPUTS[name], dumps(content))
        store.write(bundle.session_id, STEP_OUTPUTS["synthesis"],
                    dumps(bundle.synthesis))
        store.save(bundle.run())
    finally:
        if lock is not None:
            lock.release(bundle.session_id)
    return True


def install_demos(
    store: AnalysisStore, bundles: Sequence[DemoAnalysis],
    lock: Optional[AnalysisLock] = None,
    report: Callable[[str], object] = lambda message: None,
) -> List[str]:
    """Install every demo result; return the sessions written now.

    A demo that fails (unreadable store, no space...) is passed to
    ``report`` and skipped.
    """
    written: List[str] = []
    for bundle in bundles:
        try:
            if install_demo(store, bundle, lock):
                written.append(bundle.session_id)
        except (RagError, OSError) as exc:
            report(f"demo result {bundle.session_id} not installed: {exc}")
    return written
