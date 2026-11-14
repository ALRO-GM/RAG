"""Settings of the transcriber: one validated object, read at start-up.

:func:`load_settings` reads them from the environment: a JSON file
named by ``TRANSCRIBER_CONFIG`` (any field of
:class:`TranscriberSettings`, nested as in the model), then the
variables ``TRANSCRIBER_DATA_DIR``, ``TRANSCRIBER_RUNTIME`` and
``TRANSCRIBER_DEVICE``, which win over the file.
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcriber.domain.enrolment import SampleRules
from transcriber.domain.parameters import TranscriptionParameters

DEFAULT_DATA_DIR = Path("data/transcriber")
DEFAULT_RECOGNISER = "openai/whisper-large-v3-turbo"
DEFAULT_VOICE_MODEL = "microsoft/wavlm-base-plus-sv"
PUBLISHED_DIR = "published"

Runtime = Literal["real", "fake"]
Publisher = Literal["directory", "http"]
Device = Literal["auto", "cpu", "cuda"]

MIB = 1024 * 1024

DEFAULT_PORT = 8770

UPLOAD_FORMATS: Tuple[str, ...] = (
    "wav", "flac", "ogg", "opus", "mp3", "m4a", "mp4", "aac", "webm",
    "mov", "3gp", "amr", "wma",
)
"""File extensions an uploaded recording may have (audio and video)."""
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
    seconds. The client is the direct peer, except for a request that
    arrives from loopback or a unix socket (the tunnel), whose client
    is the address in the ``client_ip_header`` header that the server's
    proxy sets (it overwrites whatever the caller sent);
    ``X-Forwarded-For`` is never read. A right code is never refused.

    At most ``max_pending_enrolments`` enrolments may be in progress or
    waiting for the models at once; a further one gets ``409 busy``
    with ``Retry-After: busy_retry_after`` seconds. ``upload_formats``
    are the file extensions an uploaded recording may have.

    ``consent_version`` is the version of the consent
    statement the pages show, recorded with every enrolment.
    ``static_dir`` serves the pages at ``/`` (development
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
    client_ip_header: str = Field(
        default="X-Client-IP", pattern=r"^[A-Za-z0-9-]{1,64}$"
    )
    max_pending_enrolments: int = Field(default=2, ge=1)
    busy_retry_after: int = Field(default=10, ge=1)
    upload_formats: Tuple[str, ...] = Field(
        default=UPLOAD_FORMATS, min_length=1
    )
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


class HttpPublishSettings(BaseModel):
    """Where and how ``publisher: "http"`` sends transcripts.

    ``url`` is the RAG's base URL; transcripts go to
    ``<url>/transcripts`` with the bearer token held by the environment
    variable ``token_env`` (the token itself is never a setting, so it
    cannot end up in a config file). Every attempt waits ``timeout``
    seconds at most; a network error, a time-out or a ``5xx`` answer is
    retried ``retries`` times, after ``backoff_initial`` seconds
    multiplied by ``backoff_factor`` after every retry. A ``4xx``
    answer is final.
    """

    model_config = ConfigDict(frozen=True)

    url: Optional[str] = Field(default=None, pattern=r"^https?://\S+$")
    token_env: str = Field(
        default="TRANSCRIBER_PUBLISH_TOKEN",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$",
    )
    timeout: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0, le=10)
    backoff_initial: float = Field(default=1.0, ge=0)
    backoff_factor: float = Field(default=2.0, ge=1)


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
    ``publisher`` chooses how ``publish`` hands a transcript over:
    ``directory`` (the default) writes it to ``publish_dir`` (default
    ``<data_dir>/published``), ``http`` posts it to the RAG as
    ``publish_http`` says. ``api`` holds the HTTP API's limits and
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
    publisher: Publisher = "directory"
    publish_http: HttpPublishSettings = HttpPublishSettings()
    api: ApiSettings = ApiSettings()
    tunnel: TunnelSettings = TunnelSettings()

    @model_validator(mode="after")
    def _http_needs_a_url(self) -> "TranscriberSettings":
        """Refuse ``publisher: "http"`` without ``publish_http.url``."""
        if self.publisher == "http" and self.publish_http.url is None:
            raise ValueError(
                'publisher "http" needs publish_http.url (the RAG\'s URL)'
            )
        return self

    def published(self) -> Path:
        """Return the directory published transcripts are written to."""
        if self.publish_dir is not None:
            return self.publish_dir
        return self.data_dir / PUBLISHED_DIR

    def publish_target(self) -> str:
        """Return where ``publish`` sends transcripts, for messages."""
        if self.publisher == "http" and self.publish_http.url is not None:
            return self.publish_http.url.rstrip("/") + "/transcripts"
        return str(self.published())


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
