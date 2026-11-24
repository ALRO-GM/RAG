"""Use case: install the prepared results of the scripted demos (#232).

The RAG installs every ``analysis.demo.v1`` result in its analysis store
when it starts (:func:`install_demos`), so the live page's demo can end
on its candidate's result page. A result already installed with the
same ``started_at`` is left alone; a newer file replaces it. Only
``demo-`` sessions are ever written.
"""

from typing import List, Sequence

from src.application.skills.ports import AnalysisStore
from src.application.skills.session_analysis import dumps
from src.domain.skills.analysis_run import STEP_OUTPUTS
from src.domain.skills.demo_analysis import DemoAnalysis


def install_demo(store: AnalysisStore, bundle: DemoAnalysis) -> bool:
    """Store one demo result; return whether anything was written."""
    current = store.load(bundle.session_id)
    if current is not None and current.started_at == bundle.started_at:
        return False
    store.clear(bundle.session_id)
    store.write(bundle.session_id, STEP_OUTPUTS["synthesis"],
                dumps(bundle.synthesis))
    store.save(bundle.run())
    return True


def install_demos(
    store: AnalysisStore, bundles: Sequence[DemoAnalysis]
) -> List[str]:
    """Install every demo result; return the sessions written now."""
    return [b.session_id for b in bundles if install_demo(store, b)]
