"""Image captions with Claude Code (``claude -p``), an opt-in backend.

Tested on 2026-11-21: ``claude -p`` in the same isolated mode as the
judge (``--safe-mode``, no tools, no settings, no session) accepts an
image when the prompt is sent as ``--input-format stream-json`` with a
base64 ``image`` block. It runs on the subscription, never on the paid
API (as the judge). The image leaves the machine, so this backend is
off by default (``KnowledgeSettings.caption_backend``).
"""

import base64
import io
import json
import re
import subprocess
from typing import Callable, List, Optional

from src.application.knowledge.ports import ImageDescription
from src.domain.errors import RagError

PROMPT = (
    "Describe esta imagen de un documento en una o dos frases, en español, "
    "y copia el texto que aparezca en ella. Responde exactamente con dos "
    "líneas:\nDESCRIPCIÓN: <frase>\nTEXTO: <texto de la imagen, o vacío>"
)
SYSTEM = "You describe images of documents for a search index."
Runner = Callable[[List[str], str], str]


def run_command(command: List[str], stdin: str) -> str:
    """Run ``command`` with ``stdin`` and return its standard output."""
    done = subprocess.run(
        command, input=stdin, capture_output=True, text=True, timeout=180,
        check=False,
    )
    if done.returncode != 0:
        raise RagError(f"claude failed ({done.returncode}): "
                       f"{done.stderr.strip()[:200]}")
    return done.stdout


def media_type(image: bytes) -> str:
    """Return the media type Claude accepts for ``image`` (PNG fallback)."""
    from PIL import Image

    with Image.open(io.BytesIO(image)) as opened:
        found = (opened.format or "PNG").lower()
    return {"jpeg": "image/jpeg", "webp": "image/webp"}.get(
        found, "image/png"
    )


def converted(image: bytes) -> bytes:
    """Return ``image`` as PNG unless it is already PNG, JPEG or WebP."""
    from PIL import Image

    with Image.open(io.BytesIO(image)) as opened:
        if (opened.format or "") in ("PNG", "JPEG", "WEBP"):
            return image
        out = io.BytesIO()
        opened.convert("RGB").save(out, "PNG")
        return out.getvalue()


def parse(answer: str) -> ImageDescription:
    """Read the two labelled lines of Claude's answer."""
    caption = re.search(r"DESCRIPCI[ÓO]N:\s*(.*?)\s*(?:\nTEXTO:|$)",
                        answer, re.DOTALL)
    text = re.search(r"TEXTO:\s*(.*)$", answer, re.DOTALL)
    return ImageDescription(
        caption=" ".join(caption.group(1).split()) if caption else
        " ".join(answer.split()),
        text=" ".join(text.group(1).split()) if text else "",
    )


class ClaudeCodeCaptioner:
    """Describe images with ``claude -p`` in isolated mode.

    Args:
        model: Claude alias (``haiku`` is enough for captions).
        runner: Runs the command; replaced in tests.
    """

    def __init__(
        self, model: str = "haiku", runner: Optional[Runner] = None
    ) -> None:
        """Keep the model and the runner."""
        self.model = model
        self.runner = runner or run_command

    def describe(self, image: bytes) -> ImageDescription:
        """Return the caption and the text of ``image``."""
        png = converted(image)
        message = {"type": "user", "message": {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": media_type(png),
                "data": base64.b64encode(png).decode("ascii")}},
            {"type": "text", "text": PROMPT},
        ]}}
        command = [
            "claude", "-p", "--safe-mode", "--setting-sources", "",
            "--strict-mcp-config", "--tools", "", "--disable-slash-commands",
            "--no-session-persistence", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            "--model", self.model, "--system-prompt", SYSTEM,
        ]
        output = self.runner(command, json.dumps(message) + "\n")
        for line in reversed(output.splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                if event.get("is_error"):
                    raise RagError("claude returned an error")
                return parse(str(event.get("result", "")))
        raise RagError("claude gave no result")
