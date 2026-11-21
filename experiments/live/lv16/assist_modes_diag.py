"""Why interview calls came back empty (LV-16): two of them again, raw.

Re-asks two interview triggers that gave ``no_suggestion`` in
``assist_modes.py`` (``pincelada``) and prints the raw model output next
to what survived parsing; 3 real calls (one priming). The output of the
run cited in ``docs/live/ASSIST_MODES.md`` is kept in
``assist_modes_diag.jsonl``::

    PYTHONPATH=. python experiments/live/lv16/assist_modes_diag.py
"""

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from assist_modes import (  # noqa: E402
    CASES,
    Counted,
    build,
    wait_warm,
)
from src.application.reasoning import Reasoner  # noqa: E402
from src.domain.reasoning.templates import get_template  # noqa: E402


def main() -> None:
    """Ask twice and print the raw replies."""
    backend = Counted("haiku", effort="low", pool_size=1, timeout=20.0)
    reasoner = Reasoner(lambda a: backend, live_models=lambda a: backend)
    case = CASES[1]
    template = get_template(case["template"], "suggester")
    probe = build(case, "t9.s0", "pincelada")
    reasoner.warm_up(case["template"], "haiku", "es", 3)
    wait_warm(backend, template.system(probe), template.max_output_tokens)
    for trigger in ("t9.s0", "t15.s0"):
        pieces: List[str] = []
        request = build(case, trigger, "pincelada")
        answer = reasoner.reason(request, on_text=pieces.append)
        print(json.dumps({
            "trigger": trigger, "status": answer.status,
            "raw": "".join(pieces),
            "texts": [s.text for s in answer.suggestions],
            "retrieved": [r.id for r in request.retrieved],
        }, ensure_ascii=False))
    print("primes", Counted.primes)
    backend.close()


if __name__ == "__main__":
    main()
