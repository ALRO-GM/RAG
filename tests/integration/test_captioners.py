"""Caption backends: Claude Code (fake runner), Florence (fake model)."""

import json
import sys
import types
from typing import Any, Dict, List

import pytest

from src.adapters.outbound import claude_captioner, florence_captioner
from src.adapters.outbound.claude_captioner import (
    ClaudeCodeCaptioner,
    converted,
    media_type,
    parse,
    run_command,
)
from src.adapters.outbound.florence_captioner import FlorenceCaptioner
from src.domain.errors import RagError
from tests.knowledge_samples import make_image


def result(text: str, error: bool = False) -> str:
    """Return the stream-json lines of a finished call."""
    return "\n".join([
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "result", "result": text, "is_error": error}),
    ])


def test_claude_answer_is_parsed_and_the_command_is_isolated() -> None:
    """The image goes as a base64 block; the answer's two lines are read."""
    seen: Dict[str, Any] = {}

    def runner(command: List[str], stdin: str) -> str:
        seen["command"], seen["stdin"] = command, stdin
        return result("DESCRIPCIÓN: Un gráfico de barras.\nTEXTO: Ventas 2025")

    found = ClaudeCodeCaptioner("haiku", runner).describe(make_image("png"))
    assert found.caption == "Un gráfico de barras."
    assert found.text == "Ventas 2025"
    command = seen["command"]
    assert command[:2] == ["claude", "-p"]
    assert command[command.index("--tools") + 1] == ""
    assert "--safe-mode" in command and "--no-session-persistence" in command
    block = json.loads(seen["stdin"])["message"]["content"][0]
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"


def test_claude_errors() -> None:
    """An error result and an empty stream are reported."""
    with pytest.raises(RagError, match="error"):
        ClaudeCodeCaptioner(runner=lambda c, s: result("x", True)).describe(
            make_image())
    with pytest.raises(RagError, match="no result"):
        ClaudeCodeCaptioner(runner=lambda c, s: "not json\n").describe(
            make_image())


def test_parse_without_labels_keeps_the_answer() -> None:
    """Free text becomes the caption."""
    assert parse("Una foto\ncualquiera").caption == "Una foto cualquiera"
    assert parse("DESCRIPCION: a\nTEXTO:").text == ""


def test_media_types_and_conversion() -> None:
    """TIFF is converted to PNG; JPEG and WebP are sent as they are."""
    assert media_type(make_image("jpg")) == "image/jpeg"
    assert media_type(make_image("webp")) == "image/webp"
    assert media_type(make_image("png")) == "image/png"
    tiff = make_image("tiff")
    assert media_type(converted(tiff)) == "image/png"
    jpg = make_image("jpg")
    assert converted(jpg) is jpg


def test_run_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real runner returns stdout and raises on a failure."""
    assert run_command([sys.executable, "-c", "print('hi')"], "").strip() \
        == "hi"
    with pytest.raises(RagError, match="failed"):
        run_command([sys.executable, "-c", "raise SystemExit(3)"], "")
    assert claude_captioner.SYSTEM


class Gate:
    """A gate with a fixed answer."""

    def __init__(self, free: bool) -> None:
        """Say ``free`` always."""
        self._free = free

    def free(self) -> bool:
        """Return the answer."""
        return self._free


class FakeProcessor:
    """Stands in for the Florence processor."""

    def __call__(self, text: str, images: Any, return_tensors: str) -> Any:
        """Return tensors-like inputs carrying the task."""
        import torch

        return {"input_ids": torch.tensor([[1]]),
                "pixel_values": torch.zeros(1, 3, 2, 2), "task": text}

    def batch_decode(self, ids: Any, skip_special_tokens: bool) -> List[str]:
        """Return a placeholder text."""
        return ["raw"]

    def post_process_generation(
        self, raw: str, task: str, image_size: Any
    ) -> Dict[str, Any]:
        """Answer by task."""
        if "OCR" in task:
            return {task: {"labels": ["HOLA</s>", "42"], "quad_boxes": []}}
        return {task: "a  red square"}


class FakeModel:
    """Stands in for ``Florence2ForConditionalGeneration``."""

    loaded: List[str] = []

    def __init__(self) -> None:
        """Pretend to be a float32 model."""
        import torch

        self.dtype = torch.float32

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: Any) -> "FakeModel":
        """Record the load."""
        cls.loaded.append(model_id)
        return cls()

    def to(self, device: str) -> "FakeModel":
        """Ignore the device."""
        return self

    def eval(self) -> "FakeModel":
        """Return itself."""
        return self

    def generate(self, **kwargs: Any) -> Any:
        """Return dummy ids."""
        return [[0]]


@pytest.fixture()
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake ``transformers`` classes."""
    FakeModel.loaded.clear()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoProcessor=types.SimpleNamespace(
            from_pretrained=lambda model_id: FakeProcessor()),
        Florence2ForConditionalGeneration=FakeModel,
    ))


def test_florence_describes_with_two_tasks(fake_transformers: None) -> None:
    """Caption and OCR come from the two Florence tasks, spaces tidied."""
    captioner = FlorenceCaptioner()
    found = captioner.describe(make_image("png"))
    assert found.caption == "a red square" and found.text == "HOLA 42"
    captioner.describe(make_image("png"))
    assert FakeModel.loaded == [florence_captioner.MODEL]  # loaded once


def test_florence_follows_the_gpu_gate(
    fake_transformers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model is reloaded when the card is taken or given back."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(FlorenceCaptioner, "_run", lambda *args: "x")
    gate = Gate(True)
    captioner = FlorenceCaptioner(gate)
    assert FlorenceCaptioner()._device() == "cpu"
    captioner.describe(make_image())
    captioner.describe(make_image())
    assert FakeModel.loaded == [florence_captioner.MODEL]
    gate._free = False
    captioner.describe(make_image())
    gate._free = True
    captioner.describe(make_image())
    assert len(FakeModel.loaded) == 3


@pytest.mark.parametrize("ocr, backend, layout, captioner", [
    (False, "florence", None, None),
    (True, "off", "DoclingPages", None),
    (True, "florence", "DoclingPages", "FlorenceCaptioner"),
    (True, "claude", "DoclingPages", "ClaudeCodeCaptioner"),
])
def test_the_container_wires_ocr_and_captions(
    monkeypatch: pytest.MonkeyPatch, ocr: bool, backend: str,
    layout: Any, captioner: Any,
) -> None:
    """OCR needs the group and the setting; captions follow the backend."""
    from src import bootstrap
    from src.bootstrap import Container
    from src.config import KnowledgeSettings, Settings

    monkeypatch.setattr(bootstrap, "docling_available", lambda: True)
    knowledge = KnowledgeSettings.model_validate(
        {"ocr": ocr, "caption_backend": backend})
    container = Container(Settings(), knowledge=knowledge)
    extractors = container.document_extractors
    names = [None if o is None else type(o).__name__
             for o in (extractors.layout, extractors.captioner)]
    assert names == [layout, captioner]
    monkeypatch.setattr(bootstrap, "docling_available", lambda: False)
    missing = Container(Settings(), knowledge=KnowledgeSettings())
    assert missing.document_extractors.layout is None
