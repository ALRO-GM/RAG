"""Settings of the transcriber: one validated object, read at start-up.

:func:`load_settings` reads them from the environment: a JSON file
named by ``TRANSCRIBER_CONFIG`` (any field of
:class:`TranscriberSettings`, nested as in the model), then the
variables ``TRANSCRIBER_DATA_DIR``, ``TRANSCRIBER_RUNTIME`` and
``TRANSCRIBER_DEVICE``, which win over the file.
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

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

DEFAULT_PORT = 8770
DEFAULT_PUBLIC_HOST = "rag.peer4u.app"

CONFIG_VARIABLE = "TRANSCRIBER_CONFIG"
ENVIRONMENT_FIELDS = {
    "TRANSCRIBER_DATA_DIR": "data_dir",
    "TRANSCRIBER_RUNTIME": "runtime",
    "TRANSCRIBER_DEVICE": "device",
}
"""Environment variable → top-level setting it overrides."""


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
    ``cookie_secure`` marks the playback cookie ``Secure``; turn it off
    only for local development over plain http (a browser does not
    send a ``Secure`` cookie to ``http://127.0.0.1``).
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
    cookie_secure: bool = True
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)


class TunnelSettings(BaseModel):
    """The SSH reverse forward that puts the API behind the server.

    ``transcriber serve`` runs ``ssh -N -R <remote_socket>:127.0.0.1:
    <port>`` to ``target`` with the key ``identity_file`` (``~`` is
    expanded), checking the server's host key strictly (against
    ``known_hosts`` when given, else the user's own file). Before every
    connection it runs ``ssh <target> true``, which the server's forced
    command for that key uses to remove a stale socket.

    When the forward ends it is opened again after ``backoff_initial``
    seconds, doubled (``backoff_factor``) after every failure up to
    ``backoff_max``; a connection that lasted ``stable_after`` seconds
    resets the delay. A forward still alive ``connect_grace`` seconds
    after it started is reported ``connected`` (``ExitOnForwardFailure``
    makes ssh exit when the forward is refused). ``cleanup_timeout``
    bounds the clean-up connection. ``public_host`` is the host of the
    join link. ``ssh`` is the client executable.
    """

    model_config = ConfigDict(frozen=True)

    target: str = Field(default="ubuntu@rag.peer4u.app", min_length=1)
    remote_socket: str = Field(
        default="/opt/ragputin/run/app.sock", min_length=1
    )
    identity_file: Path = Path("~/.ssh/ragputin_tunnel")
    known_hosts: Optional[Path] = None
    backoff_initial: float = Field(default=1.0, gt=0)
    backoff_factor: float = Field(default=2.0, ge=1)
    backoff_max: float = Field(default=60.0, gt=0)
    stable_after: float = Field(default=60.0, gt=0)
    connect_grace: float = Field(default=3.0, ge=0)
    cleanup_timeout: float = Field(default=30.0, gt=0)
    public_host: str = Field(default=DEFAULT_PUBLIC_HOST, min_length=1)
    ssh: str = Field(default="ssh", min_length=1)


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
    ``<data_dir>/published``). ``api`` holds the HTTP API's limits and
    ``tunnel`` the SSH forward of ``transcriber serve``.
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
    tunnel: TunnelSettings = TunnelSettings()

    def published(self) -> Path:
        """Return the directory published transcripts are written to."""
        if self.publish_dir is not None:
            return self.publish_dir
        return self.data_dir / PUBLISHED_DIR


def load_settings(environ: Mapping[str, str]) -> TranscriberSettings:
    """Return the settings the environment ``environ`` describes.

    Without any of the variables the defaults are returned. Raises
    :class:`OSError` if the config file cannot be read and
    :class:`ValueError` (pydantic's ``ValidationError`` included) if it
    is not a JSON object or a value is invalid.
    """
    fields: Dict[str, Any] = {}
    config = environ.get(CONFIG_VARIABLE)
    if config:
        loaded = json.loads(Path(config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{config}: the settings must be a JSON object.")
        fields.update(loaded)
    for variable, field in ENVIRONMENT_FIELDS.items():
        value = environ.get(variable)
        if value:
            fields[field] = value
    return TranscriberSettings.model_validate(fields)
