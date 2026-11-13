"""Speech recognition adapter: Whisper through transformers."""

from typing import Any, Dict, List, Optional

from src.adapters.outbound.torch_runtime import model_kwargs, prepare
from src.application.skills.ports import Audio, RecognisedSpeech
from src.domain.errors import ModelError
from src.domain.languages import detect_language
from src.domain.runtime import RuntimePlan
from src.domain.skills.transcript import Word

DEFAULT_SPEECH_MODEL = "openai/whisper-large-v3-turbo"
"""Multilingual, 809M parameters: near large-v3 accuracy, far faster."""


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
        kwargs["output_attentions"] = False
        return forward(*args, **kwargs)

    encoder.forward = lean_forward


class WhisperRecogniser:
    """Transcribe audio with word timestamps.

    The model is loaded lazily. Long recordings are processed in
    30-second windows, ``batch_size`` at a time (measured on an 8 GB
    GPU with large-v3-turbo in float16: 2 windows need 3.3 GB and run
    at about 24x real time; 1 window needs 1.9 GB).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SPEECH_MODEL,
        batch_size: int = 2,
        plan: Optional[RuntimePlan] = None,
    ):
        """Configure the recogniser without loading the model yet."""
        self._model_name = model_name
        self.batch_size = batch_size
        self.plan = plan
        self._pipeline: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the recognition model."""
        return self._model_name

    def _load(self) -> Any:
        """Build the recognition pipeline on first use."""
        if self._pipeline is None:
            from transformers import pipeline

            device = prepare(self.plan)
            try:
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self._model_name,
                    device=device,
                    model_kwargs=model_kwargs(device),
                )
            except (OSError, ValueError) as exc:
                raise ModelError(
                    f"Cannot load speech model {self._model_name}: {exc}"
                )
            without_encoder_attentions(self._pipeline.model)
        return self._pipeline

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        """Return the words spoken in ``audio`` with their times.

        With ``language=None`` Whisper detects the language of each
        window; the reported language is then detected from the text.
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
        text = " ".join(w.text.strip() for w in words)
        spoken = language or (detect_language(text) if text else None)
        return RecognisedSpeech(words=words, language=spoken)
