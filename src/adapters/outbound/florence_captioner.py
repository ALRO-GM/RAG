"""Image captions and in-image text with Florence-2 (MIT).

Implements :class:`~src.application.knowledge.ports.ImageCaptioner` with
``florence-community/Florence-2-base`` (0.23 B parameters, 463 MB of
fp16 weights; ``-large`` is 1.55 GB) through the ``transformers`` native
class: ``<DETAILED_CAPTION>`` for the description and
``<OCR_WITH_REGION>`` for the text in the image, one label per region
(``<OCR>`` runs the words of a chart together; ``<MORE_DETAILED_CAPTION>``
invented more: both measured in ``experiments/knowledge/OCR.md``).
Captions come out in English (the model has no other language); the OCR
text keeps the image's language.

Like the layout reader it is a batch job: before each image the GPU gate
is asked and the model lives on the GPU only while no live session holds
it, else on the CPU (the other copy is dropped). A model that cannot be
loaded (not downloaded, offline) raises once; the caller keeps going
without captions.
"""

import gc
import io
import logging
from typing import Any, Optional, Tuple

from src.application.knowledge.ports import GpuGate, ImageDescription

LOG = logging.getLogger(__name__)

MODEL = "florence-community/Florence-2-base"
CAPTION_TASK = "<DETAILED_CAPTION>"
OCR_TASK = "<OCR_WITH_REGION>"


class FlorenceCaptioner:
    """Describe images with Florence-2.

    Args:
        gpu: Tells whether the GPU is free; ``None`` means CPU only.
        model: Hugging Face id of the Florence-2 checkpoint.
        max_new_tokens: Longest generated text per task.
    """

    def __init__(
        self,
        gpu: Optional[GpuGate] = None,
        model: str = MODEL,
        max_new_tokens: int = 160,
    ) -> None:
        """Keep the settings; the model loads on first use."""
        self.gpu = gpu
        self.model_id = model
        self.max_new_tokens = max_new_tokens
        self._loaded: Optional[Tuple[Any, Any, str]] = None

    def describe(self, image: bytes) -> ImageDescription:
        """Return the caption and the text of an image (any PIL format)."""
        from PIL import Image

        model, processor, device = self._model()
        with Image.open(io.BytesIO(image)) as opened:
            picture = opened.convert("RGB")
        return ImageDescription(
            caption=self._run(model, processor, device, picture,
                              CAPTION_TASK),
            text=self._run(model, processor, device, picture, OCR_TASK),
        )

    def _device(self) -> str:
        """Return ``cuda`` while the card is free, else ``cpu``."""
        import torch

        if (self.gpu is not None and self.gpu.free()
                and torch.cuda.is_available()):
            return "cuda"
        return "cpu"

    def _model(self) -> Tuple[Any, Any, str]:
        """Return the model, processor and device; reload on a move."""
        import torch
        import transformers

        auto_processor: Any = transformers.AutoProcessor
        florence: Any = transformers.Florence2ForConditionalGeneration

        device = self._device()
        if self._loaded is not None and self._loaded[2] == device:
            return self._loaded
        if self._loaded is not None:
            LOG.info("Captioner moves from %s to %s", self._loaded[2], device)
            self._loaded = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = auto_processor.from_pretrained(self.model_id)
        model = florence.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device).eval()
        self._loaded = (model, processor, device)
        return self._loaded

    def _run(
        self, model: Any, processor: Any, device: str, picture: Any,
        task: str,
    ) -> str:
        """Run one Florence task on a PIL image; return its text."""
        import torch

        inputs = processor(text=task, images=picture, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device, model.dtype)
        with torch.inference_mode():
            ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=pixels, max_new_tokens=self.max_new_tokens,
                num_beams=3, do_sample=False,
            )
        raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            raw, task=task, image_size=picture.size
        )
        value = parsed.get(task, "")
        if isinstance(value, dict):  # regions: one label per text box
            value = " ".join(str(v) for v in value.get("labels", []))
        return " ".join(str(value).replace("</s>", " ").split())
