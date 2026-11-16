"""Render a fairness audit report from a results document (#86).

::

    python experiments/skills/fairness_audit.py results.fairness.json \\
        report.md [--language es|en]

The results document is ``fairness.results.v1``
(:class:`src.application.skills.fairness_report.AuditResults`): the
audit's context and, per measure, the observations (group and outcome
or value per person). The report has one table per measure (group, n,
rate or mean, confidence interval, ratio to the reference group and
its interval, status) and the rules used (``docs/skills/FAIRNESS.md``
§6).
"""

import argparse
from pathlib import Path
from typing import List, Literal, Optional, cast

from src.application.skills.fairness_report import (
    AuditResults,
    fill_audit,
    render_audit,
)


def main(argv: Optional[List[str]] = None) -> None:
    """Read the results, fill the tables and write the Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--language", choices=("es", "en"), default="es")
    args = parser.parse_args(argv)
    results = AuditResults.model_validate_json(
        args.results.read_text(encoding="utf-8")
    )
    filled = fill_audit(results)
    language = cast(Literal["es", "en"], args.language)
    args.report.write_text(render_audit(results, filled, language),
                           encoding="utf-8")
    for audit in filled:
        flagged = audit.table.flagged()
        print(f"{audit.table.metric}: "
              f"{'flagged ' + ', '.join(flagged) if flagged else 'ok'}")


if __name__ == "__main__":
    main()
