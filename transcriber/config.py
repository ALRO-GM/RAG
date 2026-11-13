"""Settings of the transcriber: one validated object, read at start-up."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATA_DIR = Path("data/transcriber")
DEFAULT_RECOGNISER = "openai/whisper-large-v3-turbo"
DEFAULT_VOICE_MODEL = "microsoft/wavlm-base-plus-sv"


class TranscriberSettings(BaseModel):
    """Configuration consumed by the composition root.

    ``data_dir`` holds the audio, voiceprints and transcripts (ignored
    by git); ``recogniser`` and ``voice_model`` are the Hugging Face ids
    of the speech recogniser and the voice encoder.
    """

    model_config = ConfigDict(frozen=True)

    data_dir: Path = DEFAULT_DATA_DIR
    recogniser: str = Field(default=DEFAULT_RECOGNISER, min_length=1)
    voice_model: str = Field(default=DEFAULT_VOICE_MODEL, min_length=1)
