"""Build the published data of the soft-skills research page (issue #27).

The research itself lives in a private, git-ignored folder (default
``data/research/softskills``): source records, scores, notes and the
saved papers. This script publishes only what may be public, in two
steps:

``extract`` (needs the private folder)
    Write ``docs/skills/research/``: ``scores.json`` (as it is, except
    one phrase about the project's owner, see ``SCRUB``), ``RUBRIC.md``
    and ``coverage.json`` as they are, and ``sources.json`` with only the
    bibliographic fields of the sources cited by the scores, by the
    hand-reviewed ``figures.json``, by ``docs/skills/RESEARCH.md`` or by
    the page. Then runs ``page``.
``page`` (committed data only)
    Write ``transcriber/adapters/inbound/web/research/data.json`` (what
    the page's charts read) and rewrite the generated blocks of
    ``research.html`` (the tables that are the page's no-JS fallback and
    its reference list) from the files in ``docs/skills/research/``.
``check`` (committed data only)
    Exit 1 if ``page`` would change ``data.json`` or ``research.html``.

Usage::

    python experiments/skills/build_research_data.py extract
    python experiments/skills/build_research_data.py page
    python experiments/skills/build_research_data.py check

Standard library only; it never imports ``src`` or ``transcriber``.
"""

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data" / "research" / "softskills"
DOCS = ROOT / "docs" / "skills" / "research"
RESEARCH_MD = ROOT / "docs" / "skills" / "RESEARCH.md"
WEB = ROOT / "transcriber" / "adapters" / "inbound" / "web"
PAGE = WEB / "research.html"
PAGE_DATA = WEB / "research" / "data.json"

Json = Dict[str, Any]

VERIFIED_RANK = {"full-text": 0, "abstract-only": 1, "secondary": 2}
"""Best verification first: a record read in full beats an abstract."""

SOURCE_FIELDS = ("id", "authors", "year", "title", "venue", "doi", "url",
                 "verified", "kind")

FAMILY_ES = {
    "selection-method": "Métodos de selección",
    "observational-scheme": "Esquemas de observación",
    "peer/other-report": "Valoración por otros",
    "automatic-method": "Métodos automáticos",
    "questionnaire": "Cuestionarios y tests",
    "framework": "Marcos de competencias",
}
"""Family labels in Spanish, in the order the page shows them."""

VERDICT_ES = {
    "use": "usar",
    "use-as-reference": "usar como referencia",
    "cite-only": "solo citar",
    "avoid": "evitar",
}
CONFIDENCE_ES = {"High": "alta", "Medium": "media", "Low": "baja"}
VERIFIED_ES = {
    "full-text": "texto completo",
    "abstract-only": "solo resumen",
    "secondary": "fuente secundaria",
}
MODE_ES = {
    "behaviour": "observa conducta",
    "simulation": "simulación escrita o en vídeo",
    "interview": "entrevista",
    "test": "test",
    "self-report": "autoinforme",
}
CHANNEL_ES = {"text": "solo texto", "audio": "solo audio",
              "av": "conducta natural (audio y vídeo)"}

BLOCK = re.compile(
    r"(<!-- build:(?P<name>[a-z-]+) -->)(?P<body>.*?)"
    r"(<!-- /build:(?P=name) -->)",
    re.S,
)
REF_LINK = re.compile(r'href="#ref-([a-z0-9_]+)"')
BACKTICK_ID = re.compile(r"`([a-z][a-z0-9_]+)`")


# --------------------------------------------------------------------------
# Reading


def read_json(path: Path) -> Any:
    """Parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented UTF-8 JSON with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


def read_records(private: Path) -> List[Json]:
    """Every source record of the private ``sources/*.jsonl`` files."""
    records: List[Json] = []
    for path in sorted((private / "sources").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------
# Sources


def https_url(value: Any) -> Optional[str]:
    """``value`` if it is an https URL, else None."""
    text = str(value or "").strip()
    return text if text.startswith("https://") and " " not in text else None


def merge_sources(records: Iterable[Json]) -> Dict[str, Json]:
    """One public record per id, from its best-verified record.

    Only the bibliographic fields are kept. The DOI is lower-cased. The
    recorded URL is kept only when there is no DOI (the page links the
    DOI resolver then, never a copy of the paper) and it is https.
    """
    best: Dict[str, Json] = {}
    for record in records:
        current = best.get(record["id"])
        rank = VERIFIED_RANK.get(record.get("verified", ""), 9)
        if current is None or rank < VERIFIED_RANK.get(
                current.get("verified", ""), 9):
            best[record["id"]] = record
    merged: Dict[str, Json] = {}
    for key, record in sorted(best.items()):
        doi = str(record.get("doi") or "").strip().lower() or None
        out = {field: record.get(field) for field in SOURCE_FIELDS}
        out["doi"] = doi
        out["url"] = None if doi else https_url(record.get("url"))
        merged[key] = out
    return merged


def link_for(source: Json) -> Optional[str]:
    """The page's link for a source: the DOI resolver, else its URL."""
    if source.get("doi"):
        return "https://doi.org/" + str(source["doi"])
    url = source.get("url")
    return str(url) if url else None


CITE_NAME = {"conocer_ec0554": "CONOCER"}
"""Author fields that the generic rule below would garble."""

INITIALS = re.compile(r"^(?:[A-Z]\.\s*)+$")


def short_cite(source: Json) -> str:
    """``Arthur et al. (2003)``, ``Woehr y Arthur (2003)``, ``Lance (2008)``.

    The year is ``s. f.`` when unknown; without authors, the title's
    head stands in for them.
    """
    raw = CITE_NAME.get(source["id"], str(source.get("authors") or "")).strip()
    if ";" in raw:
        names = [part.split(",")[0].strip() for part in raw.split(";")]
    else:
        names = [part.strip() for part in raw.split(",")]
    names = [re.sub(r"\s*\(.*$", "", n) for n in names]
    names = [n for n in names
             if n and n != "et al." and not INITIALS.match(n)]
    year = source.get("year") or "s. f."
    if not names:
        title = str(source.get("title") or source["id"]).split(":")[0]
        return f"{title} ({year})"
    first = names[0].replace(" et al.", "")
    if " et al." in names[0] or len(names) > 2:
        who = f"{first} et al."
    elif len(names) == 2:
        who = f"{first} y {names[1]}"
    else:
        who = first
    return f"{who} ({year})"


def score_citations(scores: List[Json]) -> Dict[str, List[str]]:
    """Source id -> ids of the scored entries whose lines cite it."""
    cited: Dict[str, Set[str]] = {}
    for entry in scores:
        for part in "ABC":
            for line in entry[part]["lines"].values():
                for source in line.get("sources", []):
                    cited.setdefault(source, set()).add(entry["id"])
    return {key: sorted(value) for key, value in cited.items()}


def page_citations(page_html: str) -> Set[str]:
    """Source ids the page links to with ``href="#ref-<id>"``."""
    return set(REF_LINK.findall(page_html))


def doc_citations(markdown: str, known: Set[str]) -> Set[str]:
    """Known source ids written in backticks in a Markdown document."""
    return {m for m in BACKTICK_ID.findall(markdown) if m in known}


# --------------------------------------------------------------------------
# extract

SCRUB = (
    ("the owner has already excluded it from scoring",
     "the repository's rubric already excludes it"),
)
"""Phrases of the private scores that refer to the project's owner."""


def scrub(value: Any) -> Any:
    """``value`` with every SCRUB phrase replaced in its strings."""
    if isinstance(value, str):
        for old, new in SCRUB:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def extract(private: Path, docs: Path = DOCS, page: Path = PAGE,
            research_md: Path = RESEARCH_MD) -> List[str]:
    """Publish the private folder's public part into ``docs``."""
    scores: List[Json] = scrub(read_json(private / "scores.json"))
    figures: Json = read_json(docs / "figures.json")
    sources = merge_sources(read_records(private))
    known = set(sources)
    wanted = set(score_citations(scores))
    wanted |= {f["source"] for f in figures["figures"]}
    if page.exists():
        wanted |= page_citations(page.read_text(encoding="utf-8"))
    if research_md.exists():
        wanted |= doc_citations(research_md.read_text(encoding="utf-8"),
                                known)
    missing = sorted(wanted - known)
    if missing:
        raise ValueError(f"cited ids without a source record: {missing}")
    docs.mkdir(parents=True, exist_ok=True)
    write_json(docs / "scores.json", scores)
    write_json(docs / "sources.json", [sources[k] for k in sorted(wanted)])
    write_json(docs / "coverage.json", read_json(private / "coverage.json"))
    shutil.copyfile(private / "RUBRIC.md", docs / "RUBRIC.md")
    return ["scores.json", "sources.json", "coverage.json", "RUBRIC.md"]


# --------------------------------------------------------------------------
# page data


def entry_summary(entry: Json) -> Json:
    """What the page shows of one scored entry."""
    return {
        "id": entry["id"],
        "name": entry["name"],
        "family": entry["family"],
        "A": entry["A"]["total"],
        "B": entry["B"]["total"],
        "C": entry["C"]["total"],
        "confidence": entry["confidence"],
        "verdict": entry["verdict"]["label"],
        "popular_but_weak": bool(entry.get("popular_but_weak")),
    }


def spearman_brown_single(reliability: float, situations: int) -> float:
    """One situation's reliability, given that of a mean of several."""
    return reliability / (situations - (situations - 1) * reliability)


def spearman_brown(single: float, situations: int) -> float:
    """Reliability of the mean of ``situations`` parallel situations."""
    return situations * single / (1 + (situations - 1) * single)


def projection(figures: List[Json], up_to: int = 10) -> Json:
    """The analyst's Spearman-Brown projection from the marked base figure."""
    base = next(f for f in figures if f.get("projection_base"))
    single = spearman_brown_single(base["value"], base["situations"])
    points = [{"situations": n, "value": round(spearman_brown(single, n), 3)}
              for n in range(1, up_to + 1)]
    return {"from": base["id"], "single": round(single, 3),
            "points": points, "derived": True}


def build_page_data(docs: Path = DOCS) -> Json:
    """The JSON the page reads, from the published files only."""
    scores: List[Json] = read_json(docs / "scores.json")
    sources: List[Json] = read_json(docs / "sources.json")
    figures: Json = read_json(docs / "figures.json")
    coverage: Json = read_json(docs / "coverage.json")
    pending: Json = read_json(docs / "pending.json")
    by_score = score_citations(scores)
    figure_ids: Dict[str, List[str]] = {}
    for figure in figures["figures"]:
        figure_ids.setdefault(figure["source"], []).append(figure["id"])
    entry_family = {e["id"]: e["family"] for e in scores}
    by_id = {s["id"]: s for s in sources}
    published = []
    for source in sources:
        cited_by = by_score.get(source["id"], [])
        published.append({
            **source,
            "cite": short_cite(source),
            "link": link_for(source),
            "cited_by": cited_by,
            "families": sorted({entry_family[e] for e in cited_by}),
            "figures": figure_ids.get(source["id"], []),
        })
    shown = []
    for figure in figures["figures"]:
        shown.append({**figure, "cite": short_cite(by_id[figure["source"]])})
    return {
        "generated": coverage.get("generated"),
        "panels": figures["panels"],
        "figures": shown,
        "projection": projection(figures["figures"]),
        "entries": [entry_summary(e) for e in scores],
        "sources": published,
        "coverage": coverage,
        "pending": pending["papers"],
        "labels": {
            "family": FAMILY_ES, "verdict": VERDICT_ES,
            "confidence": CONFIDENCE_ES, "verified": VERIFIED_ES,
            "mode": MODE_ES, "channel": CHANNEL_ES,
        },
    }


# --------------------------------------------------------------------------
# HTML fallback blocks


def esc(value: Any) -> str:
    """HTML-escape any value."""
    return html.escape(str(value), quote=True)


def num(value: float) -> str:
    """``.80``, ``.286``, ``-.26``: the APA style of the sources."""
    text = f"{value:.3f}".rstrip("0")
    if len(text.split(".")[1]) < 2:
        text = f"{value:.2f}"
    return text.replace("0.", ".", 1) if abs(value) < 1 else text


def ref(source_id: str, text: str) -> str:
    """A link to the reference list entry of ``source_id``."""
    return f'<a href="#ref-{esc(source_id)}">{esc(text)}</a>'


def table(caption: str, head: Sequence[str], rows: Iterable[Sequence[str]],
          css: str = "data") -> str:
    """An HTML table; cells are already-escaped HTML."""
    parts = [f'<table class="{css}"><caption>{esc(caption)}</caption>',
             "<thead><tr>"]
    parts += [f'<th scope="col">{esc(h)}</th>' for h in head]
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def figure_rows(data: Json, chart: str) -> List[Tuple[Json, Json]]:
    """(panel, figure) pairs of one chart, in file order."""
    panels = {p["id"]: p for p in data["panels"] if p["chart"] == chart}
    return [(panels[f["panel"]], f) for f in data["figures"]
            if f["panel"] in panels]


def block_reliability(data: Json) -> str:
    """Table alternative of chart 1."""
    rows = []
    for panel, f in figure_rows(data, "reliability"):
        rows.append([esc(panel["title"]), esc(f.get("label", "")),
                     esc(num(f["value"])), esc(f["statistic"]),
                     esc(f.get("raters", "")), esc(f["sample"]),
                     ref(f["source"], f["cite"]) + " · " + esc(f["location"]),
                     esc(VERIFIED_ES[f["verified"]])])
    return table("Cifras de fiabilidad y de acuerdo máquina–humano",
                 ["Panel", "Qué se mide", "Valor", "Estadístico",
                  "Quién puntúa", "Muestra", "Fuente y lugar", "Leído"], rows)


def block_validity(data: Json) -> str:
    """Table alternative of chart 2."""
    rows = []
    for _, f in figure_rows(data, "validity"):
        rows.append([esc(f["method"]), esc(MODE_ES[f["mode"]]),
                     esc(num(f["value"])), esc(num(f["sd"])),
                     ref(f["source"], f["cite"]) + " · " + esc(f["location"]),
                     esc(VERIFIED_ES[f["verified"]])])
    return table("Validez operativa frente al rendimiento laboral (ρ)",
                 ["Método", "Modo", "ρ", "DT de ρ", "Fuente y lugar",
                  "Leído"], rows)


def block_map(data: Json) -> str:
    """Table alternative of chart 3: the 70 scored entries."""
    entries = sorted(data["entries"], key=lambda e: (
        list(FAMILY_ES).index(e["family"]), -e["C"], e["name"]))
    rows = []
    for e in entries:
        name = esc(e["name"]) + (" (popular pero débil)"
                                 if e["popular_but_weak"] else "")
        rows.append([name, esc(FAMILY_ES[e["family"]]), esc(e["A"]),
                     esc(e["B"]), esc(e["C"]),
                     esc(CONFIDENCE_ES[e["confidence"]]),
                     esc(VERDICT_ES[e["verdict"]])])
    return table("Las 70 entradas puntuadas",
                 ["Modelo, instrumento o método", "Familia",
                  "A credibilidad", "B aceptación", "C encaje", "Confianza",
                  "Veredicto"], rows)


def block_situations(data: Json) -> str:
    """Table alternative of chart 4, the projection marked as such."""
    rows = []
    for panel, f in figure_rows(data, "situations"):
        rows.append([esc(panel["title"]), esc(f["label"]),
                     esc(f["situations"]), esc(num(f["value"])),
                     esc(f["statistic"]),
                     ref(f["source"], f["cite"]) + " · " + esc(f["location"]),
                     esc(VERIFIED_ES[f["verified"]])])
    proj = data["projection"]
    for point in proj["points"]:
        rows.append(["Proyección propia (Spearman-Brown)",
                     "Puntuación global, otras situaciones",
                     esc(point["situations"]), esc(num(point["value"])),
                     "Eρ² proyectado",
                     "Cálculo propio a partir de .65 con 3 ejercicios",
                     "no es un dato publicado"])
    return table("Cuántas situaciones hacen falta",
                 ["Panel", "Qué se mide", "Situaciones", "Valor",
                  "Estadístico", "Fuente y lugar", "Leído"], rows)


def block_references(data: Json) -> str:
    """The reference list: one item per published source."""
    names = {e["id"]: e["name"] for e in data["entries"]}
    items = []
    for s in sorted(data["sources"], key=lambda s: s["cite"].lower()):
        fams = " ".join(s["families"])
        bits = [f'<li id="ref-{esc(s["id"])}" class="ref" '
                f'data-verified="{esc(s["verified"])}" '
                f'data-families="{esc(fams)}">',
                f'<span class="ref-cite">{esc(s["cite"])}</span> ',
                f'<span class="ref-title">{esc(s["title"])}</span>']
        if s.get("venue"):
            bits.append(f'. <span class="ref-venue">{esc(s["venue"])}</span>')
        bits.append(". ")
        if s.get("link"):
            label = ("doi:" + s["doi"]) if s.get("doi") else "enlace"
            bits.append(f'<a href="{esc(s["link"])}" rel="noopener noreferrer"'
                        f'>{esc(label)}</a>. ')
        bits.append(f'<span class="ref-level ref-{esc(s["verified"])}">'
                    f'{esc(VERIFIED_ES[s["verified"]])}</span>')
        if s["cited_by"]:
            bits.append('<span class="ref-used"> · Lo citan: '
                        + esc("; ".join(names[i] for i in s["cited_by"]))
                        + "</span>")
        if s["figures"]:
            bits.append('<span class="ref-used"> · Cifras en los gráficos: '
                        + esc(len(s["figures"])) + "</span>")
        bits.append("</li>")
        items.append("".join(bits))
    return '<ol class="refs" id="ref-list">' + "".join(items) + "</ol>"


def block_pending(data: Json) -> str:
    """The ranked list of papers still to obtain."""
    rows = []
    for p in data["pending"]:
        doi = (f'<a href="https://doi.org/{esc(p["doi"])}" '
               f'rel="noopener noreferrer">{esc(p["doi"])}</a>'
               if p["doi"] else "sin DOI")
        rows.append([esc(p["rank"]), esc(p["paper"]), doi, esc(p["why"])])
    return table("Artículos pendientes de conseguir",
                 ["#", "Artículo", "DOI", "Por qué importa"], rows)


def block_coverage(data: Json) -> str:
    """Counts per area, computed from coverage.json."""
    cov = data["coverage"]
    names = {"behaviour": "conducta", "computational": "métodos automáticos",
             "frameworks": "marcos", "instruments": "instrumentos",
             "practice": "práctica profesional", "spanish": "España"}
    rows = []
    for area, c in cov["areas"].items():
        rows.append([esc(names.get(area, area)), esc(c["sources"]),
                     esc(c["full_text"]), esc(c["abstract_only"]),
                     esc(c["secondary"]), esc(c["records_with_file"])])
    t = cov["totals"]
    rows.append(["<strong>total</strong>", esc(t["records"]),
                 esc(t["records_full_text"]), esc(t["records_abstract_only"]),
                 esc(t["records_secondary"]),
                 esc(sum(c["records_with_file"]
                         for c in cov["areas"].values()))])
    return table("Registros de fuentes por área",
                 ["Área", "Registros", "Texto completo", "Solo resumen",
                  "Secundaria", "Con archivo guardado"], rows)


def block_coverage_facts(data: Json) -> str:
    """Sentence-level counts for the limits section."""
    cov = data["coverage"]
    t, s = cov["totals"], cov["scores"]
    read = t["records_full_text"]
    share = round(100 * read / t["records"])
    ver = s["cited_ids_verification"]
    return (
        "<ul>"
        f"<li>{esc(t['records'])} registros de fuentes, {esc(t['unique_ids'])}"
        f" identificadores distintos ({esc(t['duplicate_records_same_id'])}"
        " registros repiten un identificador en otra área y "
        f"{esc(len(t['same_paper_under_two_ids']))} artículos aparecen con dos"
        " identificadores).</li>"
        f"<li>Leídos a texto completo: {esc(read)} registros ({esc(share)} %);"
        f" solo el resumen: {esc(t['records_abstract_only'])}; solo a través"
        f" de otra fuente: {esc(t['records_secondary'])}.</li>"
        f"<li>{esc(s['entries'])} entradas puntuadas citan"
        f" {esc(s['cited_source_ids_count'])} fuentes: {esc(ver['full-text'])}"
        f" leídas a texto completo, {esc(ver['abstract-only'])} solo por el"
        f" resumen y {esc(ver['secondary'])} de segunda mano.</li>"
        "</ul>"
    )


BLOCKS = {
    "reliability-table": block_reliability,
    "validity-table": block_validity,
    "map-table": block_map,
    "situations-table": block_situations,
    "references": block_references,
    "pending": block_pending,
    "coverage": block_coverage,
    "coverage-facts": block_coverage_facts,
}
"""Generated blocks of research.html, by marker name."""


def render_page(page_html: str, data: Json) -> str:
    """``page_html`` with every generated block rewritten from ``data``."""
    seen: Set[str] = set()

    def fill(match: "re.Match[str]") -> str:
        name = match.group("name")
        if name not in BLOCKS:
            raise ValueError(f"unknown block {name!r}")
        seen.add(name)
        return f"{match.group(1)}{BLOCKS[name](data)}{match.group(4)}"

    out = BLOCK.sub(fill, page_html)
    missing = sorted(set(BLOCKS) - seen)
    if missing:
        raise ValueError(f"research.html lacks the blocks {missing}")
    return out


def page_outputs(docs: Path = DOCS, page: Path = PAGE) -> Tuple[str, str]:
    """(data.json text, research.html text) as ``page`` would write them."""
    data = build_page_data(docs)
    data_text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    return data_text, render_page(page.read_text(encoding="utf-8"), data)


def write_page(docs: Path = DOCS, page: Path = PAGE,
               page_data: Path = PAGE_DATA) -> None:
    """Write the page's data and its generated blocks."""
    data_text, page_text = page_outputs(docs, page)
    page_data.parent.mkdir(parents=True, exist_ok=True)
    page_data.write_text(data_text, encoding="utf-8")
    page.write_text(page_text, encoding="utf-8")


def stale(docs: Path = DOCS, page: Path = PAGE,
          page_data: Path = PAGE_DATA) -> List[str]:
    """Files that ``page`` would change."""
    data_text, page_text = page_outputs(docs, page)
    out = []
    if not page_data.exists() or page_data.read_text(
            encoding="utf-8") != data_text:
        out.append(str(page_data.relative_to(ROOT)))
    if page.read_text(encoding="utf-8") != page_text:
        out.append(str(page.relative_to(ROOT)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one step from the command line; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["extract", "page", "check"])
    parser.add_argument("--private", type=Path, default=PRIVATE)
    args = parser.parse_args(argv)
    if args.step == "extract":
        written = extract(args.private)
        print("wrote " + ", ".join(written) + f" to {DOCS.relative_to(ROOT)}")
    if args.step in ("extract", "page"):
        write_page()
        print(f"wrote {PAGE_DATA.relative_to(ROOT)} and "
              f"{PAGE.relative_to(ROOT)}")
        return 0
    changed = stale()
    for path in changed:
        print(f"out of date: {path}", file=sys.stderr)
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
