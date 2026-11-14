"""The static pages: self-contained, CSP friendly, and their logic tested.

The pages under ``transcriber/adapters/inbound/web`` are served by the
workstation's API in development and by the server's Caddy in
production, under a strict Content-Security-Policy and with no request
to any other origin. These tests check those properties on the files
themselves, and run the pure-logic tests of ``tests/web`` with node.
"""

import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "transcriber" / "adapters" / "inbound" / "web"
TEXT_SUFFIXES = {".html", ".css", ".js", ".json"}

EXTERNAL = re.compile(
    r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I
)
FORBIDDEN_JS = [
    (re.compile(r"\beval\s*\("), "eval"),
    (re.compile(r"\bnew\s+Function\b"), "new Function"),
    (re.compile(r"\binnerHTML\b|\bouterHTML\b|insertAdjacentHTML"),
     "HTML injection"),
    (re.compile(r"document\.write"), "document.write"),
    (re.compile(r"setAttribute\(\s*[\"']style[\"']"), "inline style"),
    (re.compile(r"javascript:", re.I), "javascript: URL"),
    (re.compile(r"\bXMLHttpRequest\b|\bWebSocket\b|\bsendBeacon\b"),
     "other network API"),
]
IMPORT = re.compile(
    r"(?:import\s[^;]*?from\s*|import\s*\(\s*|import\s+)[\"'](\.[^\"']+)[\"']"
)
URL_REF = re.compile(
    r"new URL\(\s*[\"'](\.[^\"']+)[\"']\s*,\s*import\.meta\.url"
)


def web_files() -> List[Path]:
    """Every text file the pages are made of."""
    return sorted(
        p for p in WEB.rglob("*") if p.suffix in TEXT_SUFFIXES and p.is_file()
    )


class Page(HTMLParser):
    """Collect what an HTML page loads and any inline code in it."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        super().__init__()
        self.references: List[str] = []
        self.problems: List[str] = []
        self._script_src: Optional[bool] = None

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        """Record references and inline handlers, styles and scripts."""
        names = dict(attrs)
        for name, value in attrs:
            if name.startswith("on"):
                self.problems.append(f"inline handler {name} on <{tag}>")
            if name == "style":
                self.problems.append(f"inline style on <{tag}>")
            if name in ("src", "href") and value is not None:
                self.references.append(value)
            if name == "action" and value:
                self.problems.append("form action")
        if tag == "script":
            self._script_src = "src" in names
            if "src" not in names:
                self.problems.append("<script> without src")
        if tag == "style":
            self.problems.append("<style> element")
        if tag == "link" and names.get("rel") not in (
            "stylesheet", "icon", "manifest"
        ):
            self.problems.append(f"unexpected <link rel={names.get('rel')}>")


def parse(path: Path) -> Page:
    """Parse one HTML file."""
    page = Page()
    page.feed(path.read_text(encoding="utf-8"))
    return page


def test_the_web_directory_has_the_pages() -> None:
    names = {p.relative_to(WEB).as_posix() for p in web_files()}
    for needed in ("index.html", "style.css", "js/main.js",
                   "js/worklet/recorder-processor.js"):
        assert needed in names


def test_no_external_urls_anywhere() -> None:
    found = []
    for path in web_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if EXTERNAL.search(line):
                found.append(f"{path.relative_to(WEB)}:{number}: {line}")
    assert found == []


@pytest.mark.parametrize(
    "page", sorted(WEB.rglob("*.html")), ids=lambda p: p.name
)
def test_html_has_no_inline_code_and_loads_only_local_files(
    page: Path,
) -> None:
    parsed = parse(page)
    assert parsed.problems == []
    assert parsed.references, "a page loads at least its script"
    for ref in parsed.references:
        if ref.startswith("#"):
            continue
        assert not re.match(r"^(?:[a-z]+:)?//|^[a-z]+:", ref, re.I), ref
        target = (page.parent / ref.split("?")[0].split("#")[0]).resolve()
        assert target.is_file(), f"{page.name} references missing {ref}"
        assert WEB.resolve() in target.parents


def test_every_relative_import_exists() -> None:
    missing = []
    for path in web_files():
        if path.suffix != ".js":
            continue
        text = path.read_text(encoding="utf-8")
        for ref in IMPORT.findall(text) + URL_REF.findall(text):
            if not (path.parent / ref).resolve().is_file():
                missing.append(f"{path.relative_to(WEB)} -> {ref}")
    assert missing == []


def test_javascript_stays_csp_friendly() -> None:
    found = []
    for path in web_files():
        if path.suffix != ".js":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            code = line.split("//")[0] if "//" in line and "://" not in line \
                else line
            for pattern, what in FORBIDDEN_JS:
                if pattern.search(code):
                    found.append(f"{path.relative_to(WEB)}:{number}: {what}")
    assert found == []


def test_css_loads_nothing() -> None:
    css = (WEB / "style.css").read_text(encoding="utf-8")
    assert "@import" not in css
    assert "@font-face" not in css
    assert not re.search(r"url\(", css)


def test_the_worklet_is_its_own_file() -> None:
    worklet = (WEB / "js" / "worklet" / "recorder-processor.js").read_text(
        encoding="utf-8"
    )
    assert "registerProcessor(" in worklet
    assert "import " not in worklet, "worklets are loaded standalone"


def test_dictionaries_ask_for_no_emoji_icons() -> None:
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿]")
    for path in web_files():
        assert not emoji.search(path.read_text(encoding="utf-8")), path.name


def node_major() -> Optional[int]:
    """Return node's major version, or ``None`` when it is not installed."""
    node = shutil.which("node")
    if node is None:
        return None
    out = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    match = re.match(r"v(\d+)\.", out)
    return int(match.group(1)) if match else None


def test_the_pure_logic_passes_its_node_tests() -> None:
    major = node_major()
    if major is None or major < 18:
        pytest.skip("node >= 18 is needed to run tests/web (node --test)")
    tests = sorted((ROOT / "tests" / "web").glob("*.test.mjs"))
    assert tests, "tests/web has no tests"
    done = subprocess.run(
        ["node", "--test", *map(str, tests)],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=300,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
