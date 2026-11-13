"""Voice encoder adapter: WavLM x-vectors, one per window of audio.

Moved from ``src/adapters/outbound/xvector_diarizer.py`` and reduced to
the embedding: grouping the vectors into voices is domain logic
(:mod:`transcriber.domain.voices`). The model is public (no licence
gate), runs in float16 on CUDA, is loaded on first use and unloaded by
:meth:`WavLMVoiceEncoder.release` after a job.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from transcriber.adapters.outbound.torch_runtime import (
    Device,
    ModelLoadError,
    model_kwargs,
    pick_device,
    release,
)
from transcriber.application.ports import Audio
from transcriber.config import DEFAULT_VOICE_MODEL
from transcriber.domain.errors import InvalidInputError
from transcriber.domain.voices import FloatArray

MODEL_SAMPLE_RATE = 16000
"""The rate WavLM was trained on."""


class WavLMVoiceEncoder:
    """Map windows of audio to L2-normalised voice embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_VOICE_MODEL,
        batch_size: int = 32,
        device: Device = "auto",
    ) -> None:
        """Configure the encoder without loading the model yet."""
        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._loaded: Optional[Tuple[Any, Any]] = None
        self._device = "cpu"

    @property
    def model_name(self) -> str:
        """Return the identifier of the voice model."""
        return self._model_name

    @property
    def loaded(self) -> bool:
        """Tell whether the model is in memory."""
        return self._loaded is not None

    def _load(self) -> Tuple[Any, Any]:  # pragma: no cover - real model
        """Load the feature extractor and the model on first use."""
        if self._loaded is None:
            import transformers

            self._device = pick_device(self.device)
            loader: Any = transformers
            try:
                extractor = loader.AutoFeatureExtractor.from_pretrained(
                    self._model_name
                )
                model = loader.WavLMForXVector.from_pretrained(
                    self._model_name, **model_kwargs(self._device)
                )
            except (OSError, ValueError) as exc:
                raise ModelLoadError(
                    f"Cannot load voice model {self._model_name}: {exc}"
                ) from exc
            self._loaded = (extractor, model.to(self._device).eval())
        return self._loaded

    def embed(
        self, audio: Audio, windows: Sequence[Tuple[float, float]]
    ) -> FloatArray:
        """Return one embedding row per window (seconds) of ``audio``.

        No windows give a ``(0, 0)`` array without loading the model.
        """
        if not windows:
            return np.zeros((0, 0), dtype=np.float32)
        if audio.sample_rate != MODEL_SAMPLE_RATE:
            raise InvalidInputError(
                f"The voice model needs {MODEL_SAMPLE_RATE} Hz audio,"
                f" got {audio.sample_rate} Hz."
            )
        clips = [audio.cut(start, end) for start, end in windows]
        if any(clip.size == 0 for clip in clips):
            raise InvalidInputError("A voice window lies outside the audio.")
        return self._vectors(clips)

    def _vectors(
        self, clips: Sequence[FloatArray]
    ) -> FloatArray:  # pragma: no cover - runs a real model
        """Embed ``clips`` in batches; return unit rows in float32."""
        import torch

        extractor, model = self._load()
        dtype = next(model.parameters()).dtype
        rows: List[FloatArray] = []
        for start in range(0, len(clips), self.batch_size):
            batch = [
                np.asarray(c) for c in clips[start:start + self.batch_size]
            ]
            inputs = extractor(
                batch, sampling_rate=MODEL_SAMPLE_RATE, padding=True,
                return_tensors="pt",
            )
            tensors = {
                k: (v.to(self._device, dtype) if v.is_floating_point()
                    else v.to(self._device))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                output = model(**tensors)
            vectors = torch.nn.functional.normalize(
                output.embeddings.float(), dim=-1
            )
            rows.append(vectors.cpu().numpy().astype(np.float32))
        stacked: FloatArray = np.vstack(rows)
        return stacked

    def release(self) -> None:
        """Unload the model and give its memory back."""
        if self._loaded is None:
            return
        self._loaded = None
        release(self._device)
