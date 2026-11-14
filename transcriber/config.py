"""Settings of the transcriber: one validated object, read at start-up."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.enrolment import SampleRules
from transcriber.domain.parameters import TranscriptionParameters

DEFAULT_DATA_DIR = Path("data/transcriber")
DEFAULT_RECOGNISER = "openai/whisper-large-v3-turbo"
DEFAULT_VOICE_MODEL = "microsoft/wavlm-base-plus-sv"
PUBLISHED_DIR = "published"

Runtime = Literal["real", "fake"]
Device = Literal["auto", "cpu", "cuda"]

MIB = 1024 * 1024


class ApiSettings(BaseModel):
    """Limits and policies of the HTTP API.

    Body limits are in bytes and are enforced while a body is read:
    ``chunk_max_bytes`` for one recorded chunk (a few seconds of 16 kHz
    16-bit PCM is about 32 KB per second), ``upload_max_bytes`` for a
    whole recording uploaded at once, ``enrolment_max_bytes`` for an
    enrolment sample and ``json_max_bytes`` for a JSON body.

    Wrong access codes are throttled per client: after
    ``failed_code_limit`` failures within ``failed_code_window``
    seconds the client gets ``429`` for ``failed_code_cooldown``
    seconds. ``consent_version`` is the version of the consent
    statement the pages show, recorded with every enrolment.
    ``audio_cache_bytes`` bounds the memory kept for audio rendered for
    the player. ``static_dir`` serves the pages at ``/`` (development
    only; in production the server's Caddy serves them).
    """

    model_config = ConfigDict(frozen=True)

    chunk_max_bytes: int = Field(default=1 * MIB, gt=0)
    upload_max_bytes: int = Field(default=256 * MIB, gt=0)
    enrolment_max_bytes: int = Field(default=20 * MIB, gt=0)
    json_max_bytes: int = Field(default=64 * 1024, gt=0)
    failed_code_limit: int = Field(default=10, ge=1)
    failed_code_window: float = Field(default=60.0, gt=0)
    failed_code_cooldown: float = Field(default=300.0, gt=0)
    consent_version: str = Field(default="consent-1", min_length=1)
    audio_cache_bytes: int = Field(default=256 * MIB, ge=0)
    static_dir: Optional[Path] = None


class TranscriberSettings(BaseModel):
    """Configuration consumed by the composition root.

    ``data_dir`` holds the audio, voiceprints and transcripts (ignored
    by git); ``recogniser`` and ``voice_model`` are the Hugging Face ids
    of the speech recogniser and the voice encoder. ``transcription``
    and ``enrolment`` hold every threshold the use cases apply (none is
    hard-coded in them; the defaults are untuned, see TR-05).

    ``runtime`` selects the models: ``real`` loads ``recogniser`` and
    ``voice_model`` on ``device`` (``auto``: CUDA when available);
    ``fake`` uses the deterministic stand-ins of
    :mod:`transcriber.adapters.outbound.fake_models`, which need no
    torch (for tests that run the CLI or the API in a subprocess).
    ``publish_dir`` receives published transcripts (default
    ``<data_dir>/published``). ``api`` holds the HTTP API's limits.
    """

    model_config = ConfigDict(frozen=True)

    data_dir: Path = DEFAULT_DATA_DIR
    recogniser: str = Field(default=DEFAULT_RECOGNISER, min_length=1)
    voice_model: str = Field(default=DEFAULT_VOICE_MODEL, min_length=1)
    transcription: TranscriptionParameters = TranscriptionParameters()
    enrolment: SampleRules = SampleRules()
    runtime: Runtime = "real"
    device: Device = "auto"
    publish_dir: Optional[Path] = None
    api: ApiSettings = ApiSettings()

    def published(self) -> Path:
        """Return the directory published transcripts are written to."""
        if self.publish_dir is not None:
            return self.publish_dir
        return self.data_dir / PUBLISHED_DIR
