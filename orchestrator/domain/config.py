"""The product config: one pydantic model of ``config/products/*.yaml``.

A product (post-call analysis, live assist, voice bot) is a composition
of pieces chosen by this file (``docs/platform/ARCHITECTURE.md`` §3).
Plain data and validation: no I/O (reading the YAML is an adapter's job).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal[
    "interviewer", "candidate", "agent", "customer", "user", "bot",
    "unknown",
]
PRODUCTS = (
    "post_call", "live_assist.interview", "live_assist.support", "voicebot",
)
"""The products the platform defines."""


class _Strict(BaseModel):
    """Base of the config models: unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")


class Channel(_Strict):
    """Who speaks on one audio channel."""

    speaker: str = Field(min_length=1)
    role: Role


class Source(_Strict):
    """Where the audio comes from."""

    kind: Literal["upload", "pipewire", "ws"]
    channels: Dict[int, Channel] = Field(default_factory=dict)


class Vad(_Strict):
    """Voice activity detection."""

    model: str
    hang_ms: int = Field(gt=0)


class Asr(_Strict):
    """Speech recognition."""

    kind: Literal["batch", "stream"]
    model: str
    partial_every_ms: Optional[int] = Field(default=None, gt=0)


class Attribution(_Strict):
    """How words get a speaker."""

    kind: Literal[
        "channels", "sortformer_stream", "single_speaker", "diarizer"
    ]


class Knowledge(_Strict):
    """Which knowledge domains to query."""

    domains: List[str] = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=50)
    filters_from: List[str] = Field(default_factory=list)


class Reasoner(_Strict):
    """Which reasoner role, model and prompt to use."""

    role: Literal["judge", "suggester", "conversational"]
    model: str = "haiku"
    template: Optional[str] = Field(
        default=None, pattern=r"^[a-z_.]+@[0-9]+$"
    )
    deadline_ms: Optional[int] = Field(default=None, gt=0)
    max_inflight: int = 1
    max_suggestions: int = Field(default=3, ge=0, le=5)
    window: int = Field(default=12, ge=1, le=40)
    expressed_signals: bool = False
    mode: Optional[Literal["pincelada", "guiado"]] = None
    """Assistance mode of the ``@2`` suggester templates (``None``: the
    template's default, ``pincelada``); the session may switch it."""

    @model_validator(mode="after")
    def _one_in_flight(self) -> "Reasoner":
        if self.max_inflight != 1:
            raise ValueError("max_inflight must be 1 (one call in flight)")
        return self


class Trigger(_Strict):
    """When a conversation event asks the reasoner for something."""

    on: Literal["final", "turn_end"]
    speaker_roles: List[Role] = Field(min_length=1)
    min_words: int = Field(default=1, ge=1)
    debounce_ms: int = Field(default=0, ge=0)
    cancel_stale: bool = True


class Tts(_Strict):
    """Text to speech."""

    kind: Literal["piper"]
    voice: str


class Ui(_Strict):
    """Where the result is shown.

    ``script_phases`` are the phases of the conversation's script, in
    order, whose coverage the live page shows (interviews); they are
    values of the knowledge domain's ``script_phase`` vocabulary.
    """

    kind: Literal["live_page", "speaker", "hr_page"]
    script_phases: List[str] = Field(default_factory=list)


class Privacy(_Strict):
    """What is kept."""

    store_audio: bool = False
    store_text_days: Optional[int] = Field(default=None, ge=0)
    banner: Optional[str] = None


class Handoff(_Strict):
    """What the product hands to the post-call analysis."""

    transcript_v1: bool = False
    run_post_call: bool = False


class ProductConfig(_Strict):
    """The composition of one product."""

    product: Literal[
        "post_call", "live_assist.interview", "live_assist.support",
        "voicebot",
    ]
    version: int = Field(ge=1)
    language: str = Field(min_length=2)
    source: Source
    vad: Optional[Vad] = None
    asr: Asr
    attribution: Attribution
    knowledge: Optional[Knowledge] = None
    reasoner: Optional[Reasoner] = None
    trigger: Optional[Trigger] = None
    tts: Optional[Tts] = None
    ui: Optional[Ui] = None
    output: Optional[str] = None
    privacy: Privacy = Field(default_factory=Privacy)
    handoff: Handoff = Field(default_factory=Handoff)

    @property
    def is_live(self) -> bool:
        """Tell whether the product follows a conversation as it happens."""
        return self.product != "post_call"

    @model_validator(mode="after")
    def _consistent(self) -> "ProductConfig":
        live = self.is_live
        if live and (self.reasoner is None or self.trigger is None
                     or self.vad is None):
            raise ValueError(
                "a live product needs vad, reasoner and trigger"
            )
        if live and self.asr.kind != "stream":
            raise ValueError("a live product needs a streaming asr")
        if not live and self.trigger is not None:
            raise ValueError("post_call has no live trigger")
        if self.privacy.store_audio and live:
            raise ValueError("a live product keeps no audio by default")
        if self.reasoner is not None:
            self._check_reasoner(self.reasoner)
        if self.tts is not None and self.product != "voicebot":
            raise ValueError("only the voicebot speaks")
        if self.source.kind == "pipewire" and not self.source.channels:
            raise ValueError("a pipewire source declares its channels")
        return self

    def _check_reasoner(self, reasoner: Reasoner) -> None:
        """Reject a reasoner section that breaks the design's rules."""
        wanted = {
            "live_assist.interview": "suggester",
            "live_assist.support": "suggester",
            "voicebot": "conversational",
            "post_call": "judge",
        }[self.product]
        if self.is_live and reasoner.template is None:
            raise ValueError("a live product names its prompt template")
        if reasoner.role != wanted:
            raise ValueError(f"{self.product} uses the {wanted} role")
        if reasoner.expressed_signals and (
            self.product != "live_assist.support"
        ):
            raise ValueError(
                "expressed signals exist only in live_assist.support"
            )
