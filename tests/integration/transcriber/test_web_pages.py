"""The static pages: self-contained, CSP friendly, and their logic tested.

The pages under ``transcriber/adapters/inbound/web`` are served by the
workstation's API in development and by the server's Caddy in
production, under a strict Content-Security-Policy and with no request
to any other origin. These tests check those properties on the files
themselves, and run the pure-logic tests of ``tests/web`` with node.
"""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
LINK = re.compile(
    r'<a href="https://[^"\s]+" rel="noopener noreferrer">'
)
"""An outbound link the reader may follow (the research page's
references): a navigation, not a request the page makes."""
SVG_NS = '"http://www.w3.org/2000/svg"'
"""The SVG namespace name, an identifier and never fetched."""
LINK_FIELDS = {"link", "url"}
"""Fields of research/data.json that hold a reference's address."""
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
        outbound = (
            tag == "a" and names.get("rel") == "noopener noreferrer"
            and str(names.get("href")).startswith("https://")
        )
        for name, value in attrs:
            if name.startswith("on"):
                self.problems.append(f"inline handler {name} on <{tag}>")
            if name == "style":
                self.problems.append(f"inline style on <{tag}>")
            if name in ("src", "href") and value is not None \
                    and not outbound:
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
                   "js/worklet/recorder-processor.js", "js/lib/lab.js",
                   "js/views/lab.js", "js/views/lab-run.js",
                   "js/views/lab-timeline.js", "js/views/lab-review.js"):
        assert needed in names


def test_no_external_urls_anywhere() -> None:
    """Nothing loads from another origin.

    Allowed: outbound https links with ``rel="noopener noreferrer"`` in
    the HTML, the SVG namespace name in scripts, and the reference
    addresses of ``research/data.json`` (shown as such links).
    """
    found = []
    for path in web_files():
        if path.suffix == ".json":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if EXTERNAL.search(LINK.sub("", line).replace(SVG_NS, "")):
                found.append(f"{path.relative_to(WEB)}:{number}: {line}")
    assert found == []


def json_strings(value: object, key: str = "") -> List[Tuple[str, str]]:
    """Every (key, string) pair inside a JSON value."""
    if isinstance(value, str):
        return [(key, value)]
    if isinstance(value, list):
        return [p for v in value for p in json_strings(v, key)]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in json_strings(v, k)]
    return []


def test_json_holds_addresses_only_as_reference_links() -> None:
    for path in web_files():
        if path.suffix != ".json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, text in json_strings(data):
            if EXTERNAL.search(text):
                assert key in LINK_FIELDS, (path.name, key, text)
                assert text.startswith("https://"), (path.name, text)


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


SHARE_SOCKET_MODULE = "js/share.js"
"""«Compartir pantalla» (#278): its socket to ``/api/share``; the code
goes in the first message (``js/lib/share.js``), never in the address."""
SOCKET_MODULE = "js/live-audio.js"
"""The one module allowed a WebSocket: the live page's microphone goes
to the transcriber's audio socket (``docs/live/LIVE_SESSION.md``)."""


def test_javascript_stays_csp_friendly() -> None:
    found = []
    for path in web_files():
        if path.suffix != ".js":
            continue
        name = path.relative_to(WEB).as_posix()
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            code = line.split("//")[0] if "//" in line and "://" not in line \
                else line
            if name in (SOCKET_MODULE, SHARE_SOCKET_MODULE):
                code = re.sub(r"\bnew WebSocket\(", "", code)
            for pattern, what in FORBIDDEN_JS:
                if pattern.search(code):
                    found.append(f"{path.relative_to(WEB)}:{number}: {what}")
    assert found == []


def test_only_the_live_microphone_and_the_share_open_a_socket() -> None:
    users = sorted(
        path.relative_to(WEB).as_posix() for path in web_files()
        if path.suffix == ".js"
        and "WebSocket" in path.read_text(encoding="utf-8")
    )
    assert users == [SOCKET_MODULE, SHARE_SOCKET_MODULE]
    link = (WEB / "js" / "lib" / "share.js").read_text(encoding="utf-8")
    assert "access_code: this.code" in link
    share = (WEB / SHARE_SOCKET_MODULE).read_text(encoding="utf-8")
    assert "new WebSocket(url)" in share and "socketUrl(location)" in share
    text = (WEB / SOCKET_MODULE).read_text(encoding="utf-8")
    # The code goes in the first message, never in the socket's address.
    assert "access_code: this.code" in text
    assert "/api/live/${encodeURIComponent(id)}/audio`;" in text


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
    failing = [
        line for line in done.stdout.splitlines()
        if line.lstrip().startswith("not ok")
    ]
    assert done.returncode == 0, (
        "\n".join(failing) + "\n" + done.stdout[-3000:]
        + done.stderr[-2000:]
    )


def test_the_lab_tab_is_reachable_and_keeps_the_code_out_of_urls() -> None:
    """The nav links to the tab; the lab views never put the code in a URL.

    The audio address carries only the playback cookie (set by
    ``POST /api/access/cookie``), never the access code.
    """
    assert 'href="#/lab"' in (WEB / "index.html").read_text(encoding="utf-8")
    assert "labView" in (WEB / "js" / "main.js").read_text(encoding="utf-8")
    for path in sorted((WEB / "js" / "views").glob("lab*.js")):
        text = path.read_text(encoding="utf-8")
        assert "getCode" not in text and "X-Access-Code" not in text, path


class Header(HTMLParser):
    """Collect the header's links and language buttons of one page."""

    def __init__(self) -> None:
        """Start outside the header."""
        super().__init__()
        self.inside = False
        self.links: List[Dict[str, Optional[str]]] = []
        self.buttons: List[Dict[str, Optional[str]]] = []
        self.groups: List[Dict[str, Optional[str]]] = []
        self.title: Dict[str, Optional[str]] = {}

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        """Note the header's parts and the page's <title>."""
        names = dict(attrs)
        if tag == "title":
            self.title = names
        if tag == "header":
            self.inside = True
        if not self.inside:
            return
        if tag == "a":
            self.links.append(names)
        if tag == "button":
            self.buttons.append(names)
        if names.get("role") == "group":
            self.groups.append(names)

    def handle_endtag(self, tag: str) -> None:
        """Leave the header."""
        if tag == "header":
            self.inside = False


def dictionary_keys(language: str) -> Set[str]:
    """The keys of one of the pages' dictionaries."""
    text = (WEB / "js" / "i18n" / f"{language}.js").read_text(
        encoding="utf-8"
    )
    return set(re.findall(r'^\s*"([\w.]+)":', text, re.M))


@pytest.mark.parametrize(
    "page", ["index.html", "research.html", "study.html", "competition.html"]
)
def test_every_page_has_the_language_switch(page: str) -> None:
    """ES | EN buttons in the header; the header and title translatable."""
    header = Header()
    header.feed((WEB / page).read_text(encoding="utf-8"))
    switch = [b for b in header.buttons if b.get("id") != "logout"]
    assert [b.get("data-lang") for b in switch] == ["es", "en"]
    assert all(b.get("type") == "button" for b in header.buttons)
    assert all(b.get("aria-pressed") in ("true", "false") for b in switch)
    switches = [g for g in header.groups if g.get("class") != "nav-admin"]
    assert [g.get("data-i18n-label") for g in switches] == ["lang.switch"]
    admin = [g for g in header.groups if g.get("class") == "nav-admin"]
    assert [g.get("aria-labelledby") for g in admin] == ["nav-admin-title"]
    keys = dictionary_keys("es") & dictionary_keys("en")
    assert header.title.get("data-i18n") in keys
    assert header.links
    for link in header.links:
        assert link.get("data-i18n") in keys, link


@pytest.mark.parametrize(
    "module",
    ["js/main.js", "js/research/main.js", "js/study.js", "js/competition.js"],
)
def test_every_entry_point_installs_the_switch(module: str) -> None:
    """Each page's script wires the switch and translates the header."""
    text = (WEB / module).read_text(encoding="utf-8")
    assert "installLanguageSwitch(" in text
    assert "translateStatic(document)" in text


def test_the_chosen_language_is_kept_safely() -> None:
    """localStorage only behind try/catch; switch off while recording."""
    i18n = (WEB / "js" / "i18n.js").read_text(encoding="utf-8")
    assert i18n.count("localStorage") == 2
    assert i18n.count("try {") == 2
    switch = (WEB / "js" / "lang-switch.js").read_text(encoding="utf-8")
    assert "isBusy()" in switch


@pytest.mark.parametrize(
    "page", ["index.html", "research.html", "study.html", "competition.html"]
)
def test_every_page_links_to_the_study(page: str) -> None:
    """The study page is reachable from the header of every page."""
    header = Header()
    header.feed((WEB / page).read_text(encoding="utf-8"))
    links = {link.get("href"): link for link in header.links}
    assert links["study.html"].get("data-i18n") == "nav.study"


@pytest.mark.parametrize(
    "page,module",
    [("study.html", "js/study.js"), ("research.html", "js/research/main.js"),
     ("competition.html", "js/competition.js")],
)
def test_the_standalone_pages_carry_the_share(page: str, module: str) -> None:
    """#284: «Compartir pantalla» on the pages that are other documents:
    the same button as the application's, the same component installed."""
    html = (WEB / page).read_text(encoding="utf-8")
    app = (WEB / "index.html").read_text(encoding="utf-8")
    button = re.search(r'<button[^>]*id="share-open"[^>]*>', app)
    assert button and button.group(0) in html
    text = (WEB / module).read_text(encoding="utf-8")
    assert "installPageShare()" in text and "share?.relabel()" in text
    shared = (WEB / "js" / "page-share.js").read_text(encoding="utf-8")
    assert "installShare(" in shared
    main = (WEB / "js" / "main.js").read_text(encoding="utf-8")
    assert 'import { CODE_KEY } from "./page-share.js";' in main
