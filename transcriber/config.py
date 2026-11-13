"""Settings of the transcriber: one validated object, read at start-up."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.enrolment import SampleRules
from transcriber.domain.parameters import TranscriptionParameters

DEFAULT_DATA_DIR = Path("data/transcriber")
DEFAULT_RECOGNISER = "openai/whisper-large-v3-turbo"
DEFAULT_VOICE_MODEL = "microsoft/wavlm-base-plus-sv"


class TranscriberSettings(BaseModel):
    """Configuration consumed by the composition root.

    ``data_dir`` holds the audio, voiceprints and transcripts (ignored
    by git); ``recogniser`` and ``voice_model`` are the Hugging Face ids
    of the speech recogniser and the voice encoder. ``transcription``
    and ``enrolment`` hold every threshold the use cases apply (none is
    hard-coded in them; the defaults are untuned, see TR-05).
    """

    model_config = ConfigDict(frozen=True)

    data_dir: Path = DEFAULT_DATA_DIR
    recogniser: str = Field(default=DEFAULT_RECOGNISER, min_length=1)
    voice_model: str = Field(default=DEFAULT_VOICE_MODEL, min_length=1)
    transcription: TranscriptionParameters = TranscriptionParameters()
    enrolment: SampleRules = SampleRules()
