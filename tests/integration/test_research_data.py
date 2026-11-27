"""The published data of the research page, checked on the committed files.

CI has no access to the private research folder, so everything here
reads only what is committed: ``docs/skills/research/``, the page
(``research.html``) and its ``research/data.json``. A small fake private
folder exercises the ``extract`` step.
"""

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Set, Tuple
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "skills" / "build_research_data.py"
DOCS = ROOT / "docs" / "skills" / "research"
WEB = ROOT / "transcriber" / "adapters" / "inbound" / "web"
PAGE = WEB / "research.html"
PAGE_DATA = WEB / "research" / "data.json"
RESEARCH_MD = ROOT / "docs" / "skills" / "RESEARCH.md"


def load_builder() -> ModuleType:
    """Import the build script, which is not in a package."""
    spec = importlib.util.spec_from_file_location("build_research_data",
                                                  SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def read(path: Path) -> Any:
    """Parse a committed JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


SOURCES: Dict[str, Dict[str, Any]] = {
    s["id"]: s for s in read(DOCS / "sources.json")}
FIGURES: Dict[str, Any] = read(DOCS / "figures.json")
SCORES: List[Dict[str, Any]] = read(DOCS / "scores.json")
DATA: Dict[str, Any] = read(PAGE_DATA)
HTML = PAGE.read_text(encoding="utf-8")

PUBLISHED = [*DOCS.iterdir(), PAGE, PAGE_DATA, RESEARCH_MD]

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"https?://[^\s\"'<>)\]]+")
PRIVATE = re.compile(
    r"/home/|file://|data/research|softskills/|\bpapers/|\bnotes/|"
    r"PROTOCOL\.md|SYNTHESIS\.md|\badviser\b|C:\\\\",
    re.IGNORECASE,
)


def strings(value: Any) -> Iterator[str]:
    """Every string inside a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)


def test_the_page_and_its_data_are_up_to_date() -> None:
    assert builder.stale() == []


def test_every_figure_has_a_published_source_and_location() -> None:
    panels = {p["id"] for p in FIGURES["panels"]}
    for figure in FIGURES["figures"]:
        assert figure["panel"] in panels, figure["id"]
        source = SOURCES.get(figure["source"])
        assert source is not None, figure["id"]
        assert figure["verified"] == source["verified"], figure["id"]
        assert isinstance(figure["value"], float), figure["id"]
        assert 0 <= figure["value"] <= 1, figure["id"]
        for field in ("statistic", "sample", "location"):
            assert str(figure.get(field, "")).strip(), (figure["id"], field)


def test_figure_ids_are_unique_and_every_panel_is_drawn() -> None:
    ids = [f["id"] for f in FIGURES["figures"]]
    assert len(ids) == len(set(ids))
    used = {f["panel"] for f in FIGURES["figures"]}
    assert used == {p["id"] for p in FIGURES["panels"]}


def test_each_chart_figure_is_of_its_panels_kind() -> None:
    """Reliability panels never mix statistics of different kinds."""
    kinds = {p["id"]: p["kind"] for p in FIGURES["panels"]}
    by_kind: Dict[str, Set[str]] = {}
    for figure in FIGURES["figures"]:
        by_kind.setdefault(kinds[figure["panel"]], set()).add(
            figure["statistic"])
    assert by_kind["kappa"] <= {"κ de Cohen", "κ de Fleiss"}
    assert by_kind["icc"] <= {"ICC", "ICC1", "ICC2", "ICC(1,1)"}
    assert by_kind["auroc"] == {"AUROC (micro-media)"}
    assert by_kind["rho_rr"] == {"ρrr"}
    assert by_kind["rho_op"] == {"ρ"}


def test_score_totals_equal_the_sum_of_their_lines() -> None:
    assert len(SCORES) == 70
    for entry in SCORES:
        for part in "ABC":
            block = entry[part]
            raw = sum(line["score"] for line in block["lines"].values())
            assert block["total"] == max(0, raw), (entry["id"], part)
            if raw < 0:
                assert "floored" in block["note"], entry["id"]


def test_every_cited_source_is_published() -> None:
    cited = set(builder.score_citations(SCORES))
    cited |= builder.page_citations(HTML)
    assert cited <= set(SOURCES)
    anchors = set(re.findall(r'id="ref-([a-z0-9_]+)"', HTML))
    assert builder.page_citations(HTML) <= anchors


def test_nothing_private_is_published() -> None:
    for path in PUBLISHED:
        text = path.read_text(encoding="utf-8")
        without_urls = URL.sub(" ", text)
        assert not EMAIL.search(without_urls), path.name
        found = PRIVATE.search(without_urls)
        assert found is None, (path.name, found and found.group(0))
    for source in SOURCES.values():
        assert set(source) == set(builder.SOURCE_FIELDS)


def recorded_hosts() -> Set[str]:
    """Hosts a page link may point to: the DOI resolver and the URLs."""
    hosts = {"doi.org"}
    for source in SOURCES.values():
        if source["url"]:
            hosts.add(urlparse(source["url"]).netloc)
    return hosts


def test_every_link_is_https_to_doi_org_or_a_recorded_host() -> None:
    hosts = recorded_hosts()
    for source in DATA["sources"]:
        link = source["link"]
        if link is None:
            continue
        parsed = urlparse(link)
        assert parsed.scheme == "https", source["id"]
        assert parsed.netloc in hosts, source["id"]
        if source["doi"]:
            assert link == "https://doi.org/" + source["doi"]
        else:
            assert link == source["url"]
    for href in re.findall(r'href="([^"#][^"]*)"', HTML):
        # The pages' own files (the shell's links and icon sprite).
        if href.split("#")[0] in ("style.css", "icon.svg", "icons.svg",
                                  "research.html", "study.html",
                                  "competition.html", "index.html"):
            continue
        parsed = urlparse(href)
        assert parsed.scheme == "https", href
        assert parsed.netloc in hosts, href
    external = re.findall(r'<a href="https://[^"]*"([^>]*)>', HTML)
    assert external and all('rel="noopener noreferrer"' in a
                            for a in external)


def test_the_projection_is_marked_as_derived() -> None:
    projection = DATA["projection"]
    assert projection["derived"] is True
    base = next(f for f in FIGURES["figures"] if f.get("projection_base"))
    assert projection["from"] == base["id"]
    four = next(p for p in projection["points"] if p["situations"] == 4)
    assert four["value"] >= 0.70
    three = next(p for p in projection["points"] if p["situations"] == 3)
    assert three["value"] == pytest.approx(base["value"], abs=1e-3)
    assert "Proyección propia" in HTML


def test_coverage_counts_on_the_page_come_from_coverage_json() -> None:
    totals = DATA["coverage"]["totals"]
    assert f"{totals['records']} registros de fuentes" in HTML
    assert f"Leídos a texto completo: {totals['records_full_text']}" in HTML


@pytest.mark.parametrize("value, text", [
    (0.8, ".80"), (0.286, ".286"), (0.54, ".54"), (-0.26, "-.26"),
    (0.0, ".00"), (1.0, "1.00"),
])
def test_num(value: float, text: str) -> None:
    assert builder.num(value) == text


@pytest.mark.parametrize("authors, year, cite", [
    ("Arthur, Day, McNelly, Edens", 2003, "Arthur et al. (2003)"),
    ("Woehr, Arthur", 2003, "Woehr y Arthur (2003)"),
    ("Lance", 2008, "Lance (2008)"),
    ("Highhouse, S.", 2008, "Highhouse (2008)"),
    ("Fisher, P.A.; Risavy, S.D.; Robie, C.", None, "Fisher et al. (s. f.)"),
    ("Rupp et al.", 2015, "Rupp et al. (2015)"),
    ("SIOP (President G. Chao)", 2021, "SIOP (2021)"),
    ("", 2020, "Some title (2020)"),
])
def test_short_cite(authors: str, year: Any, cite: str) -> None:
    source = {"id": "x", "authors": authors, "year": year,
              "title": "Some title: subtitle"}
    assert builder.short_cite(source) == cite


def test_merge_keeps_the_best_read_record_and_no_copy_links() -> None:
    records = [
        {"id": "a", "verified": "secondary", "doi": "10.1/X",
         "url": "https://mirror.example/a.pdf", "title": "T", "authors": "A",
         "year": 2000, "venue": "V", "kind": "review", "pdf": "a.pdf"},
        {"id": "a", "verified": "full-text", "doi": "10.1/X",
         "url": "https://mirror.example/a.pdf", "title": "T2",
         "authors": "A", "year": 2000, "venue": "V", "kind": "review"},
        {"id": "b", "verified": "abstract-only", "doi": "",
         "url": "http://insecure.example", "title": "U", "authors": "B",
         "year": 2001, "venue": "W", "kind": "other"},
    ]
    merged = builder.merge_sources(records)
    assert merged["a"]["title"] == "T2"
    assert merged["a"]["doi"] == "10.1/x" and merged["a"]["url"] is None
    assert "pdf" not in merged["a"]
    assert merged["b"]["url"] is None and merged["b"]["doi"] is None
    assert builder.link_for(merged["a"]) == "https://doi.org/10.1/x"
    assert builder.link_for(merged["b"]) is None


def test_scrub_removes_the_owner_phrase() -> None:
    value = {"reason": ["No validity; the owner has already excluded it "
                        "from scoring."], "n": 3}
    out = builder.scrub(value)
    assert out == {"reason": ["No validity; the repository's rubric "
                              "already excludes it."], "n": 3}


def test_render_page_needs_every_block() -> None:
    with pytest.raises(ValueError, match="lacks the blocks"):
        builder.render_page("<p>no blocks</p>", DATA)
    with pytest.raises(ValueError, match="unknown block"):
        builder.render_page(
            "<!-- build:nope --><!-- /build:nope -->", DATA)


def fake_private(tmp: Path) -> Tuple[Path, Path, Path]:
    """A tiny private folder, docs folder and page for ``extract``."""
    private = tmp / "private"
    (private / "sources").mkdir(parents=True)
    record = {"id": "s1", "verified": "full-text", "doi": "10.1/a",
              "url": "https://x.example", "title": "T", "authors": "A, B",
              "year": 2020, "venue": "V", "kind": "review",
              "numbers": ["r = .5"], "pdf": "s1.pdf"}
    other = {**record, "id": "s2", "doi": None}
    (private / "sources" / "a.jsonl").write_text(
        json.dumps(record) + "\n" + json.dumps(other) + "\n",
        encoding="utf-8")
    line = {"score": 5, "why": "x", "sources": ["s1"]}
    entry = {"id": "e1", **{p: {"total": 5, "lines": {"L": line}}
                            for p in "ABC"}}
    (private / "scores.json").write_text(json.dumps([entry]),
                                         encoding="utf-8")
    (private / "coverage.json").write_text("{}", encoding="utf-8")
    (private / "RUBRIC.md").write_text("# Rubric\n", encoding="utf-8")
    docs = tmp / "docs"
    docs.mkdir()
    (docs / "figures.json").write_text(json.dumps(
        {"figures": [{"source": "s2"}]}), encoding="utf-8")
    page = tmp / "page.html"
    page.write_text('<a href="#ref-s1">x</a>', encoding="utf-8")
    return private, docs, page


def test_extract_publishes_only_cited_bibliographic_fields(
        tmp_path: Path) -> None:
    private, docs, page = fake_private(tmp_path)
    md = tmp_path / "R.md"
    md.write_text("see `s1` and `unknown_id`", encoding="utf-8")
    written = builder.extract(private, docs=docs, page=page, research_md=md)
    assert written == ["scores.json", "sources.json", "coverage.json",
                       "RUBRIC.md"]
    sources = read(docs / "sources.json")
    assert [s["id"] for s in sources] == ["s1", "s2"]
    assert all(set(s) == set(builder.SOURCE_FIELDS) for s in sources)
    assert (docs / "RUBRIC.md").read_text() == "# Rubric\n"


def test_extract_refuses_a_citation_without_a_record(tmp_path: Path) -> None:
    private, docs, page = fake_private(tmp_path)
    page.write_text('<a href="#ref-ghost">x</a>', encoding="utf-8")
    with pytest.raises(ValueError, match="ghost"):
        builder.extract(private, docs=docs, page=page,
                        research_md=tmp_path / "none.md")
