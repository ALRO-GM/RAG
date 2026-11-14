"""Speech recognition adapter: Whisper through transformers.

Moved from ``src/adapters/outbound/whisper_recogniser.py``: greedy
decoding, the encoder's attention maps switched off, word timestamps,
30-second windows two at a time, float16 on CUDA. The model is loaded
on first use and :meth:`WhisperRecogniser.release` unloads it after a
job.
"""

import threading
from typing import Any, Dict, List, Optional

from transcriber.adapters.outbound.torch_runtime import (
    Device,
    ModelLoadError,
    model_kwargs,
    pick_device,
    release,
)
from transcriber.application.ports import Audio, RecognisedSpeech
from transcriber.config import DEFAULT_RECOGNISER
from transcriber.domain.models import Word


def words_from_chunks(
    chunks: List[Dict[str, Any]], duration: float
) -> List[Word]:
    """Convert the pipeline's word chunks to timed words.

    Whisper sometimes leaves the end of the last word of a window
    undefined; it is then closed at the start of the next word, or at
    the end of the recording. Chunks without a start time are dropped
    (they cannot be placed) and times are kept inside the recording
    and in order.
    """
    words: List[Word] = []
    for position, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        start, end = (tuple(chunk.get("timestamp") or ()) + (None, None))[:2]
        if start is None or not text.strip():
            continue
        if end is None:
            following = [
                c["timestamp"][0] for c in chunks[position + 1:]
                if c.get("timestamp") and c["timestamp"][0] is not None
            ]
            end = following[0] if following else duration
        start = min(max(float(start), 0.0), duration)
        end = min(max(float(end), start), max(duration, start))
        words.append(Word(text=text, start=start, end=end))
    return words


def without_encoder_attentions(model: Any) -> None:
    """Stop the encoder from returning its attention maps.

    Word timestamps come from the decoder's cross-attention. Asking
    for them makes transformers request *every* attention map, and the
    encoder's (32 layers x 20 heads x 1500 x 1500 for large-v3) take
    about 3 GB per 30-second window for nothing. This keeps the request
    away from the encoder only.
    """
    encoder = model.get_encoder()
    forward = encoder.forward

    def lean_forward(*args: Any, **kwargs: Any) -> Any:
        """Call the encoder without attention maps."""
        kwargs["output_attentions"] = False
        return forward(*args, **kwargs)

    encoder.forward = lean_forward


class WhisperRecogniser:
    """Transcribe audio with word timestamps.

    Long recordings are processed in 30-second windows, ``batch_size``
    at a time (measured under ``src`` on an 8 GB GPU with
    large-v3-turbo in float16: 2 windows need 3.3 GB and run at about
    24x real time; 1 window needs 1.9 GB).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RECOGNISER,
        batch_size: int = 2,
        device: Device = "auto",
    ) -> None:
        """Configure the recogniser without loading the model yet."""
        self._model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._pipeline: Optional[Any] = None
        self._device = "cpu"
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Return the identifier of the recognition model."""
        return self._model_name

    @property
    def loaded(self) -> bool:
        """Tell whether the model is in memory."""
        return self._pipeline is not None

    def _load(self) -> Any:  # pragma: no cover - loads a real model
        """Build the recognition pipeline on first use.

        Under a lock, so two first uses at once load one model.
        """
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> Any:  # pragma: no cover - real model
        """Build the pipeline unless it exists (hold ``_lock``)."""
        if self._pipeline is None:
            from transformers import pipeline

            self._device = pick_device(self.device)
            try:
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self._model_name,
                    device=self._device,
                    **model_kwargs(self._device),
                )
            except (OSError, ValueError) as exc:
                raise ModelLoadError(
                    f"Cannot load speech model {self._model_name}: {exc}"
                ) from exc
            without_encoder_attentions(self._pipeline.model)
        return self._pipeline

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:  # pragma: no cover - runs a real model
        """Return the words spoken in ``audio`` with their times.

        With ``language=None`` Whisper detects the language of each
        window and the reported language is ``None``.
        """
        # Greedy decoding: beam search multiplies memory with the
        # attention maps that word timestamps need.
        generate: Dict[str, Any] = {"task": "transcribe", "num_beams": 1}
        if language:
            generate["language"] = language
        result = self._load()(
            {"raw": audio.samples, "sampling_rate": audio.sample_rate},
            chunk_length_s=30,
            batch_size=self.batch_size,
            return_timestamps="word",
            generate_kwargs=generate,
        )
        words = words_from_chunks(
            list(result.get("chunks") or []), audio.duration
        )
        return RecognisedSpeech(words=words, language=language)

    def release(self) -> None:
        """Unload the model and give its memory back."""
        with self._lock:
            if self._pipeline is None:
                return
            self._pipeline = None
            release(self._device)
