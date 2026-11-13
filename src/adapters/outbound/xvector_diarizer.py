"""Diarization adapter: voice embeddings (x-vectors) plus clustering."""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from src.adapters.outbound.torch_runtime import prepare
from src.application.skills.ports import Audio
from src.domain.errors import ModelError
from src.domain.runtime import RuntimePlan
from src.domain.skills.speech import (
    SpeakerSpan,
    analysis_windows,
    cluster_speakers,
    spans_from_windows,
)

FloatArray = npt.NDArray[np.float32]

DEFAULT_VOICE_MODEL = "microsoft/wavlm-base-plus-sv"
"""Speaker-verification model: one voice embedding per audio window."""


class XVectorDiarizer:
    """Tell who spoke when by clustering voice embeddings.

    Speech regions are cut into overlapping windows; each window gets
    a voice embedding and the embeddings are clustered into speakers.
    The model is public (no licence gate) and needs no extra package.
    It does not detect overlapping speech: when two people talk at
    once the window goes to one of them.

    Args:
        threshold: Cosine similarity below which two groups of windows
            are different people, used when the number of speakers is
            not given. It depends on the model and must be re-measured
            if the model changes.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VOICE_MODEL,
        window: float = 1.5,
        step: float = 0.75,
        threshold: float = 0.5,
        batch_size: int = 32,
        plan: Optional[RuntimePlan] = None,
    ):
        """Configure the diarizer without loading the model yet."""
        self._model_name = model_name
        self.window = window
        self.step = step
        self.threshold = threshold
        self.batch_size = batch_size
        self.plan = plan
        self._loaded: Optional[Tuple[Any, Any, str]] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the voice model."""
        return self._model_name

    def _load(self) -> Tuple[Any, Any, str]:
        """Load the feature extractor and the model on first use."""
        if self._loaded is None:
            import transformers

            device = prepare(self.plan)
            loader: Any = transformers
            try:
                extractor = loader.AutoFeatureExtractor.from_pretrained(
                    self._model_name
                )
                model = loader.WavLMForXVector.from_pretrained(
                    self._model_name
                )
            except (OSError, ValueError) as exc:
                raise ModelError(
                    f"Cannot load voice model {self._model_name}: {exc}"
                )
            self._loaded = (extractor, model.to(device).eval(), device)
        return self._loaded

    def embed(self, clips: Sequence[FloatArray], rate: int) -> FloatArray:
        """Return one L2-normalised voice embedding per audio clip."""
        import torch

        extractor, model, device = self._load()
        rows: List[FloatArray] = []
        for start in range(0, len(clips), self.batch_size):
            batch = [
                np.asarray(c) for c in clips[start:start + self.batch_size]
            ]
            inputs = extractor(
                batch, sampling_rate=rate, padding=True, return_tensors="pt"
            )
            with torch.no_grad():
                output = model(**{k: v.to(device) for k, v in inputs.items()})
            vectors = torch.nn.functional.normalize(output.embeddings, dim=-1)
            rows.append(vectors.float().cpu().numpy().astype(np.float32))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(rows)

    def diarize(
        self,
        audio: Audio,
        regions: Sequence[Tuple[float, float]],
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSpan]:
        """Return speaker spans over the speech ``regions`` of ``audio``."""
        windows = analysis_windows(regions, self.window, self.step)
        if not windows:
            return []
        vectors = self.embed(
            [audio.cut(start, end) for start, end in windows],
            audio.sample_rate,
        )
        labels = cluster_speakers(vectors, num_speakers, self.threshold)
        return spans_from_windows(windows, labels)
