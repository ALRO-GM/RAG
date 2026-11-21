"""Streaming recogniser adapter: faster-whisper (CTranslate2), int8, warm.

``large-v3-turbo`` quantised to int8 (``docs/platform/ARCHITECTURE.md``
§5, §9), greedy decoding (beam 1), no previous-text conditioning and no
VAD of its own (the live stream has one). The model is loaded once and
kept: :meth:`FasterWhisperStream.warm_up` loads it and decodes a second
of silence, so the first real segment does not pay for the loading or
CUDA's first kernels. Partials are decoded without word times (faster);
finals with them, and their confidence is the mean word probability.

CTranslate2 needs cuBLAS 12 and cuDNN 9 on CUDA; the ones the CUDA
build of torch installs (the ``nvidia-*`` wheels) are loaded first, so
no system CUDA toolkit is needed. Install with the ``live`` dependency
group.
"""

import ctypes
import importlib
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import numpy.typing as npt

from transcriber.domain.live import SAMPLE_RATE, Hypothesis
from transcriber.domain.models import Word

FloatArray = npt.NDArray[np.float32]

DEFAULT_MODEL = "large-v3-turbo"
"""Short name of ``mobiuslabsgmbh/faster-whisper-large-v3-turbo``."""

ModelFactory = Callable[[str, str, str], Any]

CUDA_LIBRARIES = (
    ("nvidia.cublas", ("libcublasLt.so.12", "libcublas.so.12")),
    ("nvidia.cudnn", ("libcudnn*.so.9",)),
)
"""The pip packages and libraries CTranslate2 loads on CUDA."""


def preload_cuda_libraries() -> List[str]:  # pragma: no cover - CUDA only
    """Load cuBLAS and cuDNN from the ``nvidia-*`` wheels, if installed.

    Returns the files loaded. Missing packages are skipped: the system
    libraries are then used, if any.
    """
    loaded: List[str] = []
    for package, patterns in CUDA_LIBRARIES:
        try:
            module = importlib.import_module(package)
        except ImportError:
            continue
        for root in getattr(module, "__path__", []):
            for pattern in patterns:
                for path in sorted(Path(root, "lib").glob(pattern)):
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                    loaded.append(path.name)
    return loaded


def whisper_model(
    name: str, device: str, compute_type: str
) -> Any:  # pragma: no cover - loads a real model
    """Load a faster-whisper model."""
    if device == "cuda":
        preload_cuda_libraries()
    from faster_whisper import WhisperModel

    return WhisperModel(name, device=device, compute_type=compute_type)


def _mean(values: List[float]) -> Optional[float]:
    """Return the mean kept inside [0, 1], or ``None`` for no values."""
    if not values:
        return None
    return min(max(float(sum(values) / len(values)), 0.0), 1.0)


class FasterWhisperStream:
    """The :class:`~transcriber.application.ports.AsrStream` adapter."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cuda",
        compute_type: str = "int8",
        language: Optional[str] = "es",
        factory: ModelFactory = whisper_model,
    ) -> None:
        """Keep the settings; the model loads on first use."""
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._factory = factory
        self._model: Any = None
        self._lock = threading.Lock()

    def _loaded(self) -> Any:
        """Return the model, loading it once."""
        with self._lock:
            if self._model is None:
                self._model = self._factory(
                    self.model_name, self.device, self.compute_type
                )
            return self._model

    def warm_up(self) -> None:
        """Load the model and decode one second of silence."""
        self.decode(np.zeros(SAMPLE_RATE, dtype=np.float32), True)

    def decode(self, samples: FloatArray, final: bool) -> Hypothesis:
        """Return what is said in ``samples`` (mono, 16 kHz)."""
        model = self._loaded()
        segments, info = model.transcribe(
            samples.astype(np.float32),
            language=self.language,
            beam_size=1,
            word_timestamps=final,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        texts: List[str] = []
        words: List[Word] = []
        for segment in segments:
            texts.append(str(segment.text))
            for word in getattr(segment, "words", None) or []:
                text = str(word.word)
                if not text.strip():
                    continue
                start = max(float(word.start), 0.0)
                words.append(Word(
                    text=text.strip(), start=start,
                    end=max(float(word.end), start),
                    probability=min(max(float(word.probability), 0.0), 1.0),
                ))
        language = getattr(info, "language", None) or self.language
        return Hypothesis(
            text=" ".join(" ".join(texts).split()),
            words=words,
            language=language,
            confidence=_mean([
                w.probability for w in words if w.probability is not None
            ]),
        )
