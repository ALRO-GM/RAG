"""Fairness audit report: fill the LL144 / four-fifths template (#86).

``FAIRNESS.md`` §4.3: one table per measure with group, n, rate (or
mean), confidence interval, ratio to the reference group and its
interval, plus the rules (threshold, small groups) and where the data
and the groups come from, so that on the day of a pilot the report is
filled from the results rather than written. ``fill_audit`` turns a
results document (``fairness.results.v1``, see ``AuditResults``) into
filled tables; ``render_audit`` writes them as Markdown in Spanish or
English. Nothing is read or written here; the caller does the I/O.
"""

from typing import Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from src.domain.skills.fairness import (
    MIN_SHARE,
    AuditKind,
    AuditPolicy,
    AuditRow,
    AuditTable,
    Direction,
    Measurement,
    Outcome,
    above_median,
    mean_audit,
    rate_audit,
)

SCHEMA_NAME = "fairness.results.v1"
Language = Literal["es", "en"]


class AuditSpec(BaseModel):
    """One measure of a results document and its observations.

    ``kind`` ``rate`` takes ``selected`` per person; ``mean`` takes
    ``value`` per measurement; ``scoring`` ``above_median`` makes a rate
    table from ``value`` (LL144's scoring rate). ``threshold`` defaults
    to 0.80 for higher-is-better and 1.25 for lower-is-better.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    description: str = ""
    kind: AuditKind
    scoring: Optional[Literal["above_median"]] = None
    direction: Direction = "higher_is_better"
    threshold: Optional[float] = Field(default=None, gt=0)
    min_units: int = Field(default=5, ge=1)
    min_share: float = Field(default=MIN_SHARE, ge=0, lt=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    resamples: int = Field(default=2000, ge=100)
    seed: int = 0
    observations: List[Union[Outcome, Measurement]]

    def policy(self) -> AuditPolicy:
        """Return the table's rules with the default threshold filled."""
        return AuditPolicy.model_validate({
            "direction": self.direction, "threshold": self.threshold,
            "min_units": self.min_units, "min_share": self.min_share,
            "confidence": self.confidence, "resamples": self.resamples,
            "seed": self.seed,
        })


class AuditResults(BaseModel):
    """A results document: context of the audit and its measures.

    ``system``: what was audited (models, versions, thresholds);
    ``data``: which recordings or sessions and when; ``groups_source``:
    where the group of each person comes from (declared with consent,
    corpus metadata...; never inferred by the system); ``synthetic``
    marks simulated data, shown at the top of the report.
    """

    model_config = ConfigDict(extra="forbid")

    schema_name: Literal["fairness.results.v1"] = "fairness.results.v1"
    title: str
    date: str
    system: str
    data: str
    groups_source: str
    synthetic: bool = False
    notes: List[str] = Field(default_factory=list)
    audits: List[AuditSpec]


class FilledAudit(BaseModel):
    """One measure with its filled table."""

    model_config = ConfigDict(frozen=True)

    description: str
    table: AuditTable


def _outcomes(spec: AuditSpec) -> List[Outcome]:
    """Return the yes/no outcomes of a rate measure."""
    if spec.scoring == "above_median":
        return above_median(_measurements(spec))
    found = [o for o in spec.observations if isinstance(o, Outcome)]
    if len(found) != len(spec.observations):
        raise ValueError(f"{spec.metric}: a rate needs 'selected' per person")
    return found


def _measurements(spec: AuditSpec) -> List[Measurement]:
    """Return the numbers of a mean (or scoring-rate) measure."""
    found = [o for o in spec.observations if isinstance(o, Measurement)]
    if len(found) != len(spec.observations):
        raise ValueError(f"{spec.metric}: needs 'value' per measurement")
    return found


def fill_audit(results: AuditResults) -> List[FilledAudit]:
    """Return one filled table per measure of ``results``."""
    filled: List[FilledAudit] = []
    for spec in results.audits:
        if spec.kind == "rate":
            table = rate_audit(_outcomes(spec), spec.metric, spec.policy())
        else:
            if spec.scoring is not None:
                raise ValueError(f"{spec.metric}: scoring needs kind 'rate'")
            table = mean_audit(_measurements(spec), spec.metric,
                               spec.policy())
        filled.append(FilledAudit(description=spec.description, table=table))
    return filled


TEXTS: Dict[str, Dict[str, str]] = {
    "es": {
        "synthetic": (
            "**Datos simulados.** Este informe no describe a personas "
            "reales: sirve para comprobar la plantilla."
        ),
        "date": "Fecha", "system": "Sistema auditado", "data": "Datos",
        "groups": "Origen de los grupos", "notes": "Notas",
        "summary": "Resumen",
        "none_flagged": "Ningún grupo pasa del umbral.",
        "summary_flagged": "Pasan del umbral: {groups}.",
        "header": (
            "| Grupo | n | % del total | {value} | IC {conf} % | "
            "Cociente | IC del cociente | Estado |"
        ),
        "rate": "Tasa", "mean": "Media",
        "reference": "referencia", "ok": "dentro", "flagged": "pasa el umbral",
        "described_only": "sólo descrito",
        "zero_reference": (
            "sin cociente: la referencia es 0 (diferencia {difference})"
        ),
        "summary_zero": (
            "Sin cociente porque la referencia es 0, peores que ella: "
            "{groups}."
        ),
        "uncertain": " (el IC cruza el umbral)",
        "rules_rate_higher": (
            "Cociente = tasa del grupo / tasa del grupo de referencia (el "
            "de mayor tasa). Umbral {threshold}: por debajo es indicio de "
            "impacto adverso (regla de los 4/5, EEOC), no una prueba."
        ),
        "rules_rate_lower": (
            "Cociente = tasa del grupo / tasa del grupo de referencia (el "
            "de menor tasa). Umbral {threshold} (nuestro, provisional): "
            "por encima se marca."
        ),
        "rules_mean_higher": (
            "Cociente = media del grupo / media del grupo de referencia (el "
            "de mayor media). Umbral {threshold}: por debajo se marca."
        ),
        "rules_mean_lower": (
            "Cociente = media del grupo / media del grupo de referencia (el "
            "de menor media). Umbral {threshold} (nuestro, provisional): "
            "por encima se marca."
        ),
        "rules_small": (
            "Grupos con menos de {units} personas o menos del {share} % del "
            "total: se dan n, valor e intervalo, sin cociente."
        ),
        "rules_ci_rate": (
            "IC de la tasa: Wilson; IC del cociente: Katz (logaritmo), "
            "con 0,5 por celda si una está vacía."
        ),
        "rules_ci_mean": (
            "Cada persona cuenta una vez (media de sus medidas). IC: "
            "bootstrap de percentiles sobre personas, {resamples} "
            "remuestreos, semilla {seed}. Con pocas personas por grupo "
            "los intervalos son sólo orientativos; un grupo con menos de "
            "dos valores distintos no tiene intervalo (—)."
        ),
        "scoring": (
            "Tasa de puntuación (LL144): proporción del grupo por encima "
            "de la mediana de toda la muestra."
        ),
        "people": "{units} personas.",
        "no_reference": (
            "Ningún grupo tiene tamaño suficiente para servir de "
            "referencia: sólo se describe."
        ),
    },
    "en": {
        "synthetic": (
            "**Simulated data.** This report describes no real people: it "
            "is a check of the template."
        ),
        "date": "Date", "system": "Audited system", "data": "Data",
        "groups": "Source of the groups", "notes": "Notes",
        "summary": "Summary",
        "none_flagged": "No group is past the threshold.",
        "summary_flagged": "Past the threshold: {groups}.",
        "header": (
            "| Group | n | % of total | {value} | {conf} % CI | "
            "Ratio | Ratio CI | Status |"
        ),
        "rate": "Rate", "mean": "Mean",
        "reference": "reference", "ok": "within", "flagged": "past threshold",
        "described_only": "described only",
        "zero_reference": (
            "no ratio: the reference is 0 (difference {difference})"
        ),
        "summary_zero": (
            "No ratio as the reference is 0, worse than it: {groups}."
        ),
        "uncertain": " (CI crosses the threshold)",
        "rules_rate_higher": (
            "Ratio = group rate / reference group rate (the highest). "
            "Threshold {threshold}: below it is evidence of adverse impact "
            "(EEOC four-fifths rule), not a finding."
        ),
        "rules_rate_lower": (
            "Ratio = group rate / reference group rate (the lowest). "
            "Threshold {threshold} (ours, provisional): above it is flagged."
        ),
        "rules_mean_higher": (
            "Ratio = group mean / reference group mean (the highest). "
            "Threshold {threshold}: below it is flagged."
        ),
        "rules_mean_lower": (
            "Ratio = group mean / reference group mean (the lowest). "
            "Threshold {threshold} (ours, provisional): above it is flagged."
        ),
        "rules_small": (
            "Groups with fewer than {units} people or under {share} % of "
            "the total: n, value and interval are given, no ratio."
        ),
        "rules_ci_rate": (
            "Rate CI: Wilson; ratio CI: Katz (log), with 0.5 per cell when "
            "one is empty."
        ),
        "rules_ci_mean": (
            "Each person counts once (mean of their measurements). CI: "
            "percentile bootstrap over people, {resamples} resamples, "
            "seed {seed}. With few people per group the intervals are "
            "only indicative; a group with fewer than two distinct values "
            "has no interval (—)."
        ),
        "scoring": (
            "Scoring rate (LL144): share of the group above the median of "
            "the whole sample."
        ),
        "people": "{units} people.",
        "no_reference": (
            "No group is large enough to be the reference: description only."
        ),
    },
}


def _number(value: Optional[float], language: Language) -> str:
    """Return a number with three decimals in the report's style."""
    if value is None:
        return "—"
    text = f"{value:.3f}"
    return text.replace(".", ",") if language == "es" else text


def _short(value: float, language: Language) -> str:
    """Return a number without trailing zeros (0.8, 1.25, 95)."""
    return _number(value, language).rstrip("0").rstrip(",.")


def _signed(value: Optional[float], language: Language) -> str:
    """Return a difference with its sign (+0,500)."""
    if value is None:
        return "—"
    return ("+" if value >= 0 else "") + _number(value, language)


def _row(row: AuditRow, language: Language) -> str:
    """Return one Markdown table row."""
    t = TEXTS[language]
    status = t[row.status].format(
        difference=_signed(row.difference, language)
    )
    if row.certain is False and row.status in ("ok", "flagged"):
        status += t["uncertain"]
    value = _number(row.value, language)
    if row.selected is not None:
        value += f" ({row.selected}/{row.n})"
    interval = (
        "—" if row.low is None or row.high is None else
        f"{_number(row.low, language)}–{_number(row.high, language)}"
    )
    ratio_ci = (
        "—" if row.ratio_low is None or row.ratio_high is None else
        f"{_number(row.ratio_low, language)}–"
        f"{_number(row.ratio_high, language)}"
    )
    return (
        f"| {row.group} | {row.n} | {_number(100 * row.share, language)} | "
        f"{value} | {interval} | {_number(row.ratio, language)} | "
        f"{ratio_ci} | {status} |"
    )


def _rules(table: AuditTable, language: Language) -> List[str]:
    """Return the sentences that state a table's rules."""
    t = TEXTS[language]
    p = table.policy
    side = "higher" if p.direction == "higher_is_better" else "lower"
    lines = [
        t[f"rules_{table.kind}_{side}"].format(
            threshold=_short(p.threshold, language)
        ),
        t["rules_small"].format(
            units=p.min_units,
            share=_short(100 * p.min_share, language),
        ),
        t["rules_ci_rate"] if table.kind == "rate" else
        t["rules_ci_mean"].format(resamples=p.resamples, seed=p.seed),
        t["people"].format(units=table.units),
    ]
    if table.reference_group is None:
        lines.append(t["no_reference"])
    return lines


def render_audit(
    results: AuditResults, filled: Sequence[FilledAudit],
    language: Language = "es",
) -> str:
    """Return the audit report as Markdown."""
    t = TEXTS[language]
    out = [f"# {results.title}", ""]
    if results.synthetic:
        out += [t["synthetic"], ""]
    out += [
        f"- {t['date']}: {results.date}",
        f"- {t['system']}: {results.system}",
        f"- {t['data']}: {results.data}",
        f"- {t['groups']}: {results.groups_source}",
        "",
        f"## {t['summary']}", "",
    ]
    for audit in filled:
        parts = []
        flagged = audit.table.flagged()
        zero = audit.table.zero_reference()
        if flagged:
            parts.append(
                t["summary_flagged"].format(groups=", ".join(flagged)))
        if zero:
            parts.append(t["summary_zero"].format(groups=", ".join(zero)))
        verdict = " ".join(parts) if parts else t["none_flagged"]
        out.append(f"- `{audit.table.metric}`: {verdict}")
    for audit, spec in zip(filled, results.audits):
        table = audit.table
        out += ["", f"## `{table.metric}`", ""]
        if audit.description:
            out += [audit.description, ""]
        if spec.scoring == "above_median":
            out += [t["scoring"], ""]
        out.append(t["header"].format(
            value=t[table.kind],
            conf=_short(100 * table.policy.confidence, language),
        ))
        out.append("|---|---|---|---|---|---|---|---|")
        out += [_row(row, language) for row in table.rows]
        out += [""] + [f"- {line}" for line in _rules(table, language)]
    if results.notes:
        out += ["", f"## {t['notes']}", ""]
        out += [f"- {note}" for note in results.notes]
    return "\n".join(out) + "\n"
