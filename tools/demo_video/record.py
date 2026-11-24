"""DM-1 (#233): record the two-minute demo video, reproducibly.

It starts its own instances on spare loopback ports (CPU only, a
temporary data directory, its own GPU-lease folder, never the services
of the workstation): the transcriber's web with the fake runtime, the
orchestrator with fake knowledge and reasoner (the scripted demo needs
neither), and :mod:`tools.demo_video.stand_in_rag` for the result page.
Headless Google Chrome plays the path over DevTools while its
screencast is kept: the home with the three services, the scripted
interview in real time (``#/live/demo``), then Lucía Marín's result.

The frames are resampled to a constant 30 fps on their capture times
(:func:`tools.demo_video.timeline.resample`), so the video runs in real
time, and encoded by a static ffmpeg kept outside the repository
(``make demo-video-tools``) to H.264 yuv420p 1920x1080, with the
captions of :data:`tools.demo_video.timeline.CAPTIONS` in a band under
the page. Nothing is written to the repository: the video and the
frames checked go to ``~/.local/share/ragputin/demo-video/``.

Run: ``make demo-video`` (``uv run --no-sync python -m
tools.demo_video.record --help``).
"""

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from tools.demo_video.cdp import CdpPage
from tools.demo_video.timeline import (
    CAPTIONS,
    MAX_DURATION_S,
    Shown,
    fade_alpha,
    holds,
    jpeg_size,
    longest_hold,
    resample,
    resolve,
)

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_JS = Path(__file__).with_name("overlay.js")
TOOL_ENV = Path("~/.local/share/ragputin/tools/demo-video").expanduser()
OUT_DIR = Path("~/.local/share/ragputin/demo-video").expanduser()
CHROME = Path("/usr/bin/google-chrome")
FONT = Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")

CSS_WIDTH, CSS_HEIGHT, SCALE = 1536, 792, 1.25
"""The page as Chrome lays it out; times ``SCALE`` it is 1920x990."""
VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
PAGE_HEIGHT = int(CSS_HEIGHT * SCALE)
BAND = VIDEO_HEIGHT - PAGE_HEIGHT
"""The caption band under the page: captions never cover the page."""
FPS = 30
CAPTION_SIZE = 34
FADE_S = 0.35
JPEG_QUALITY = 92
DEMO_ROUTE = "#/live/demo"
DEMO_LENGTH_S = 69.5

TITLE = ("Entrevistas con apoyo en tiempo real",
         "Sugerencias mientras entrevistas y un resumen claro al terminar",
         "Demostración · datos ficticios")
CLOSING = ("Entrevistas más claras y más justas",
           "El asistente sugiere; la decisión es siempre de quien entrevista",
           "Demostración · datos ficticios")


T = TypeVar("T")


class RecordingError(RuntimeError):
    """The recording cannot go on (a tool, a port or the page failed)."""


@dataclass
class Recording:
    """What a run kept: frames, scene marks, start and end (wall time)."""

    started: float
    ended: float = 0.0
    marks: Dict[str, float] = field(default_factory=dict)
    frames: List[Tuple[float, Path]] = field(default_factory=list)
    colours: Dict[str, str] = field(default_factory=dict)


def find_ffmpeg(tool_env: Path = TOOL_ENV) -> Path:
    """Return the static ffmpeg of the tool environment, or fail clearly."""
    found = sorted(tool_env.glob(
        "lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
    ))
    if not found:
        raise RecordingError(
            f"no ffmpeg in {tool_env}: run `make demo-video-tools` (it"
            " installs the imageio-ffmpeg wheel, a static ffmpeg, in that"
            " separate environment; nothing goes to the project's .venv)"
        )
    return found[-1]


def check_tools(ffmpeg: Path) -> None:
    """Fail clearly when Chrome or the caption font is missing."""
    for path, what in ((CHROME, "Google Chrome"), (FONT, "the Noto Sans font"),
                       (ffmpeg, "ffmpeg")):
        if not path.exists():
            raise RecordingError(f"{what} not found at {path}")


def check_port_free(port: int) -> None:
    """Fail when something already listens on ``port`` (loopback)."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RecordingError(
                f"port {port} is taken; pass another --base-port"
            ) from exc


def wait_until(check: Callable[[], T], timeout: float, what: str) -> T:
    """Call ``check`` until it returns something true, for ``timeout`` s."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.2)
    raise RecordingError(f"timed out waiting for {what}")


def port_open(port: int) -> bool:
    """Return whether something accepts connections on ``port``."""
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@contextmanager
def dev_stack(work: Path, base_port: int) -> Iterator[Tuple[str, str]]:
    """Run the three dev instances; yield the page's URL and access code."""
    web, orch, rag = base_port, base_port + 1, base_port + 2
    for port in (web, orch, rag):
        check_port_free(port)
    token = work / "token"
    token.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    token.chmod(0o600)
    config = work / "transcriber.json"
    config.write_text(json.dumps({
        "publisher": "http",
        "publish_http": {"url": f"http://127.0.0.1:{rag}",
                         "token_file": str(token)},
        "rag_token_file": str(token),
    }), encoding="utf-8")
    for name in ("store", "data", "logs", "run", "interview", "voiceprints"):
        (work / name).mkdir()
    env = {
        **_base_env(), "XDG_RUNTIME_DIR": str(work / "run"),
        "CUDA_VISIBLE_DEVICES": "",
        "TRANSCRIBER_CONFIG": str(config),
        "TRANSCRIBER_DATA_DIR": str(work / "data"),
        "TRANSCRIBER_RUNTIME": "fake", "TRANSCRIBER_DEVICE": "cpu",
        "TRANSCRIBER_ORCHESTRATOR_URL": f"http://127.0.0.1:{orch}",
    }
    commands = {
        "rag": ["-m", "tools.demo_video.stand_in_rag", str(work / "store"),
                str(rag)],
        "orchestrator": [
            "-m", "orchestrator", "serve", "--port", str(orch),
            "--token-file", str(token), "--rag-token-file", str(token),
            "--knowledge", "fake", "--reasoner", "fake", "--tts", "none",
            "--natural-tts", "off", "--gpu-lease-dir", str(work / "run"),
            "--log-dir", str(work / "logs"),
            "--transcriber-url", f"http://127.0.0.1:{web}",
            "--transcriber-code-file", str(work / "no-code"),
            "--interview-dir", str(work / "interview"),
            "--voiceprints-dir", str(work / "voiceprints"),
        ],
        "transcriber": ["-m", "transcriber", "serve", "--port", str(web),
                        "--no-tunnel", "--dev-pages"],
    }
    processes: List[subprocess.Popen[bytes]] = []
    try:
        for name, args in commands.items():
            log = (work / f"{name}.log").open("wb")
            processes.append(subprocess.Popen(
                [sys.executable, *args], cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT,
            ))
        for port in (rag, orch, web):
            wait_until(lambda: port_open(port), 60, f"port {port}")
        log_file = work / "transcriber.log"
        code = wait_until(lambda: access_code(log_file), 30,
                          "the transcriber's access code")
        yield f"http://127.0.0.1:{web}", code
    finally:
        _stop(processes)


def access_code(log_file: Path) -> str:
    """Return the access code the transcriber printed ("" until then)."""
    found = re.search(r"#k=([A-Za-z0-9_-]+)",
                      log_file.read_text(encoding="utf-8"))
    return found.group(1) if found else ""


def _base_env() -> Dict[str, str]:
    """Return the environment the dev instances inherit."""
    return dict(os.environ)


def _stop(processes: Sequence["subprocess.Popen[bytes]"]) -> None:
    """Terminate the processes, then kill what did not stop."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@contextmanager
def chrome(work: Path, port: int) -> Iterator[str]:
    """Run headless Chrome; yield its page's DevTools WebSocket URL."""
    check_port_free(port)
    process = subprocess.Popen([
        str(CHROME), "--headless=new", f"--remote-debugging-port={port}",
        f"--user-data-dir={work / 'chrome'}", "--no-first-run",
        "--no-default-browser-check", "--hide-scrollbars", "--mute-audio",
        "--lang=es-ES", "--force-color-profile=srgb",
        # The screencast keeps device pixels only with this flag (the
        # emulated scale factor gives frames in CSS pixels).
        f"--force-device-scale-factor={SCALE}",
        f"--window-size={CSS_WIDTH},{CSS_HEIGHT + 100}", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=(work / "chrome.log").open("wb"))

    def page_url() -> Optional[str]:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json", timeout=1
            ) as answer:
                targets = json.load(answer)
        except OSError:
            return None
        for target in targets:
            if target.get("type") == "page":
                return str(target["webSocketDebuggerUrl"])
        return None

    try:
        yield str(wait_until(page_url, 20, "Chrome's DevTools"))
    finally:
        _stop([process])


class Director:
    """Plays the path on the page and keeps the scene marks."""

    def __init__(self, page: CdpPage, recording: Recording) -> None:
        """Drive ``page``; marks go to ``recording``."""
        self.page = page
        self.recording = recording

    def mark(self, name: str) -> None:
        """Mark that scene ``name`` starts now."""
        self.recording.marks[name] = time.time() - self.recording.started

    async def at(self, name: str, offset: float) -> None:
        """Sleep until ``offset`` seconds after mark ``name``."""
        due = self.recording.started + self.recording.marks[name] + offset
        await asyncio.sleep(max(0.0, due - time.time()))

    async def wait_for(self, expression: str, timeout: float,
                       what: str) -> Any:
        """Poll ``expression`` in the page until it is true."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = await self.page.evaluate(expression)
            if value:
                return value
            await asyncio.sleep(0.1)
        raise RecordingError(f"the page never showed {what}")

    async def find(self, text: Optional[str], scope: str = "main",
                   selector: str = "*") -> Dict[str, float]:
        """Return the box of an element (see ``__demo.find``)."""
        expression = (f"__demo.find({json.dumps(text)}, {json.dumps(scope)},"
                      f" {json.dumps(selector)})")
        box = None
        deadline = time.monotonic() + 4.0
        while not box and time.monotonic() < deadline:
            box = await self.page.evaluate(expression)
            if not box:
                await asyncio.sleep(0.1)
        if not box:
            raise RecordingError(f"no element {text or selector!r} found")
        return dict(box)

    async def glide(self, box: Dict[str, float], ms: int,
                    dx: float = 0.0, dy: float = 0.0) -> None:
        """Move the pointer to the box's centre (plus an offset)."""
        await self.page.evaluate(
            f"__demo.move({box['x'] + dx}, {box['y'] + dy}, {ms})"
        )

    async def click(self, box: Dict[str, float]) -> None:
        """Show the click ring and click the box's centre for real."""
        await self.page.evaluate("__demo.ring()")
        await asyncio.sleep(0.12)
        await self.page.click(box["x"], box["y"])

    async def scroll_to(self, text: str, top: float, ms: int,
                        selector: str = "*") -> None:
        """Scroll gently until element ``text`` is ``top`` px from the top."""
        await self.page.evaluate(
            "(() => { const b = __demo.find(" + json.dumps(text)
            + ", 'main', " + json.dumps(selector) + ");"
            " const box = document.scrollingElement;"
            f" return __demo.scrollTo(document.querySelector('main'),"
            f" box.scrollTop + b.top - {top}, {ms}); }})()"
        )


async def prepare(page: CdpPage, base_url: str, code: str) -> None:
    """Open the home in Spanish, light theme, with the overlay loaded."""
    await page.send("Page.enable")
    await page.send("Page.setBypassCSP", enabled=True)
    await page.send("Emulation.setDeviceMetricsOverride", width=CSS_WIDTH,
                    height=CSS_HEIGHT, deviceScaleFactor=0, mobile=False)
    await page.send("Emulation.setEmulatedMedia", features=[
        {"name": "prefers-color-scheme", "value": "light"},
    ])
    await page.send("Page.navigate", url=f"{base_url}/#k={code}")
    await asyncio.sleep(1.5)
    await page.evaluate(
        "localStorage.setItem('transcriber.lang', 'es');"
        " localStorage.setItem('transcriber.theme', 'light');"
        " localStorage.removeItem('transcriber.nav'); 1"
    )
    await page.send("Page.navigate", url=f"{base_url}/#/")
    await asyncio.sleep(0.5)
    await page.send("Page.reload")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        ready = await page.evaluate(
            "!!document.querySelector('a[href=\"#/live/demo\"]')"
        )
        if ready:
            break
        await asyncio.sleep(0.2)
    else:
        raise RecordingError("the home never showed the example interview"
                             " link (login page or wrong build?)")
    await asyncio.sleep(1.0)
    await page.evaluate(OVERLAY_JS.read_text(encoding="utf-8"))


async def play(page: CdpPage, recording: Recording, frames: Path) -> None:
    """Play the whole path while the screencast keeps frames."""
    act = Director(page, recording)
    colours = await page.evaluate("__demo.colours()")
    recording.colours = {key: str(value) for key, value in colours.items()}
    await page.evaluate("__demo.cardNow() && __demo.card(true, "
                        + ", ".join(json.dumps(x) for x in TITLE) + ")")
    await asyncio.sleep(0.6)
    recording.started = time.time()
    await page.start_screencast(frames, VIDEO_WIDTH, PAGE_HEIGHT,
                                JPEG_QUALITY)
    act.mark("start")
    await act.at("start", 2.4)
    centre = {"x": CSS_WIDTH * 0.62, "y": CSS_HEIGHT * 0.78}
    await page.evaluate(f"__demo.show({centre['x']}, {centre['y']})")
    await page.evaluate("__demo.card(false)")
    act.mark("home")

    # Home: the three services, then the example interview.
    await act.at("home", 1.6)
    await act.glide(await act.find("En directo"), 1400)
    await act.at("home", 5.4)
    link = await act.find(None, "main", 'a[href="#/live/demo"]')
    await act.glide(link, 1300)
    await act.at("home", 9.0)
    await act.click(link)
    await act.wait_for("!!document.querySelector('.lv-now')", 15,
                       "the live page")
    act.mark("live")

    # Live: the demo plays on its own; the pointer follows the story.
    await act.at("live", 1.5)
    rest = await act.find("Ahora")
    await act.glide(rest, 1500, dx=90)
    await act.at("live", 30.2)
    chip = await act.find("Mejor evitar")
    await act.glide(chip, 1200, dx=120, dy=-10)
    await act.at("live", 43.0)
    await act.glide(await act.find("Temas del puesto"), 1400, dx=150)
    await act.at("live", 57.0)
    await act.glide(rest, 1500, dx=90)
    await act.wait_for(
        "document.querySelector('.lv-now')?.innerText"
        ".includes('Entrevista terminada')",
        DEMO_LENGTH_S + 25, "the end of the demo",
    )
    act.mark("ended")

    # The end: open the result from the «Ahora» card.
    await act.at("ended", 0.8)
    button = await act.find(None, ".lv-now", 'a[href^="#/analysis/"]')
    await act.glide(button, 1300)
    await act.at("ended", 3.8)
    await act.click(button)
    act.mark("result")
    await act.wait_for("document.querySelector('main')?.innerText"
                       ".includes('Puntos fuertes')", 15, "the result page")

    # Result: ring, strengths and improvements, bars, «Más información».
    await act.at("result", 1.2)
    ring = await act.find(None, "main", 'svg[viewBox="0 0 120 120"]')
    await act.glide(ring, 1300, dx=95, dy=40)
    # Each caption starts once its part is in view: the marks are set
    # when the gentle scroll has finished.
    await act.at("result", 6.6)
    await act.scroll_to("Puntos fuertes", 120, 2200)
    act.mark("strengths")
    await act.glide(await act.find("Puntos fuertes"), 1200, dx=-40, dy=60)
    await act.at("strengths", 5.6)
    await act.scroll_to("Lo que pide el puesto", 150, 2200)
    act.mark("bars")
    await act.glide(await act.find("Lo que pide el puesto"), 1200, dx=140,
                    dy=50)
    await act.at("bars", 5.2)
    more = await act.page.evaluate(
        "(() => { const s = [...document.querySelectorAll('main summary')]"
        ".find((x) => x.innerText.trim().startsWith('Más información'));"
        " if (!s) return null; const r = s.getBoundingClientRect();"
        " return {x: r.left + 80, y: r.top + r.height / 2}; })()"
    )
    if not more:
        raise RecordingError("no «Más información» on the result page")
    await act.glide(more, 900)
    await act.at("bars", 6.3)
    await act.click(more)
    act.mark("more")
    await act.at("more", 1.0)
    await page.evaluate("__demo.scrollTo(document.querySelector('main'),"
                        " document.scrollingElement.scrollTop + 300, 1800)")
    await act.at("more", 5.4)
    await page.evaluate("__demo.card(true, "
                        + ", ".join(json.dumps(x) for x in CLOSING) + ")")
    act.mark("closing")
    await act.at("closing", 2.9)
    recording.ended = time.time()
    await page.stop_screencast()
    recording.frames = list(page.frames)


def _hex(css: str, fallback: str) -> str:
    """Return ``rgb(r, g, b)`` as ``RRGGBB`` (``fallback`` otherwise)."""
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", css)
    if not match:
        return fallback
    return "".join(f"{int(part):02x}" for part in match.groups())


def caption_filter(shown: Sequence[Shown], texts: Path,
                   colours: Dict[str, str], marks: Dict[str, float]) -> str:
    """Return the ffmpeg filter: page, caption band and captions.

    The band's top line is drawn only between the title cards.
    """
    background = _hex(colours.get("bg", ""), "f3f4f6")
    line = _hex(colours.get("line", ""), "e5e6ea")
    ink = _hex(colours.get("text", ""), "23252b")
    parts = [
        f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:0:0:color=0x{background}",
        f"drawbox=x=0:y={PAGE_HEIGHT}:w={VIDEO_WIDTH}:h=2"
        f":color=0x{line}:t=fill"
        f":enable='between(t,{marks['home']:.3f},{marks['closing']:.3f})'",
    ]
    y = PAGE_HEIGHT + (BAND - CAPTION_SIZE) // 2 - 4
    for number, item in enumerate(shown):
        path = texts / f"caption-{number:02d}.txt"
        path.write_text(item.text, encoding="utf-8")
        parts.append(
            f"drawtext=fontfile={FONT}:textfile={path}:expansion=none"
            f":fontsize={CAPTION_SIZE}:fontcolor=0x{ink}"
            f":x=(w-text_w)/2:y={y}"
            f":alpha='{fade_alpha(item, FADE_S)}'"
        )
    # JPEG frames are full range: convert to the limited-range BT.709
    # that phones and QuickTime expect.
    parts.append("scale=in_range=pc:out_range=tv:out_color_matrix=bt709")
    parts.append("format=yuv420p")
    return ",".join(parts)


def encode(recording: Recording, ffmpeg: Path, out: Path,
           work: Path) -> Dict[str, Any]:
    """Write the mp4; return what was measured while writing it."""
    # Chrome's capture times can step back by a few milliseconds once in
    # a while: order the frames by time (a stable sort keeps ties).
    frames = sorted(recording.frames, key=lambda frame: frame[0])
    stamps = [stamp for stamp, _ in frames]
    paths = [path for _, path in frames]
    size = jpeg_size(paths[0].read_bytes())
    if size != (VIDEO_WIDTH, PAGE_HEIGHT):
        raise RecordingError(f"frames are {size}, not"
                             f" {(VIDEO_WIDTH, PAGE_HEIGHT)}")
    indices = resample(stamps, recording.started, recording.ended, FPS)
    duration = len(indices) / FPS
    if duration > MAX_DURATION_S:
        raise RecordingError(f"the video would last {duration:.1f} s")
    shown = resolve(CAPTIONS, recording.marks, duration)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg",
        "-i", "-",
        "-vf", caption_filter(shown, work, recording.colours,
                              recording.marks),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", "-an", str(out),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    last_index, last_bytes = -1, b""
    for index in indices:
        if index != last_index:
            last_index, last_bytes = index, paths[index].read_bytes()
        encoder.stdin.write(last_bytes)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RecordingError("ffmpeg failed to encode the video")
    return {
        "frames_captured": len(stamps),
        "frames_written": len(indices),
        "duration_s": round(duration, 3),
        "captured_fps": round(len(stamps) / (stamps[-1] - stamps[0]), 1)
        if len(stamps) > 1 else 0.0,
        "longest_hold_s": round(longest_hold(indices, FPS), 3),
        "holds_over_0_2_s": [[round(at, 2), round(length, 2)]
                             for at, length in holds(indices, FPS, 0.2)],
        "marks": {key: round(value, 2)
                  for key, value in recording.marks.items()},
        "captions": [[round(s.start, 2), round(s.end, 2), s.text]
                     for s in shown],
    }


def extract_frames(ffmpeg: Path, video: Path, duration: float,
                   count: int) -> List[Path]:
    """Save ``count`` frames spread over the video as PNG next to it."""
    for old in video.parent.glob(f"{video.stem}-frame-*.png"):
        old.unlink()
    saved: List[Path] = []
    for number in range(count):
        at = duration * (number + 0.5) / count
        path = video.with_name(f"{video.stem}-frame-{number + 1:02d}"
                               f"-{at:05.1f}s.png")
        subprocess.run([
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
            str(path),
        ], check=True)
        saved.append(path)
    return saved


async def record(base_url: str, code: str, websocket: str,
                 frames: Path) -> Recording:
    """Prepare the page and play the path; return what was kept."""
    page = await CdpPage.open(websocket)
    try:
        await prepare(page, base_url, code)
        recording = Recording(started=time.time())
        await play(page, recording, frames)
        return recording
    finally:
        await page.close()


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.demo_video.record",
        description="Record the demo video (home, live interview, result).",
    )
    parser.add_argument("--out", type=Path,
                        default=OUT_DIR / "demo-entrevista.mp4",
                        help="the mp4 to write (outside the repository)")
    parser.add_argument("--base-port", type=int, default=18841,
                        help="first of four free loopback ports")
    parser.add_argument("--check-frames", type=int, default=8,
                        help="PNG frames saved next to the video")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the temporary folder (logs, frames)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Record, encode and report; return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        ffmpeg = find_ffmpeg()
        check_tools(ffmpeg)
        with ExitStack() as stack:
            work = Path(tempfile.mkdtemp(prefix="demo-video-"))
            if not args.keep_work:
                stack.callback(_remove, work)
            base_url, code = stack.enter_context(
                dev_stack(work, args.base_port)
            )
            websocket = stack.enter_context(
                chrome(work, args.base_port + 3)
            )
            recording = asyncio.run(
                record(base_url, code, websocket, work / "frames")
            )
            report = encode(recording, ffmpeg, args.out, work)
            report["video"] = str(args.out)
            report["checked_frames"] = [
                str(path) for path in extract_frames(
                    ffmpeg, args.out, report["duration_s"],
                    args.check_frames,
                )
            ]
            if args.keep_work:
                report["work"] = str(work)
    except (RecordingError, ValueError) as exc:
        print(f"demo video: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _remove(path: Path) -> None:
    """Delete the temporary folder."""
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
