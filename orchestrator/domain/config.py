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
    "voicebot.interview",
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
    """Text to speech.

    ``voice`` is a Piper voice; ``length_scale``, ``noise_scale``,
    ``noise_w`` and ``sentence_silence`` are Piper's controls (``None``:
    the voice's own). ``engine`` says who speaks: ``piper`` (``voice``),
    ``pocket`` (Kyutai Pocket TTS, catalogue voice ``pocket_voice`` on
    ``pocket_device``) or ``qwen3`` (Qwen3-TTS cloning the default
    reference voice); Piper's ``voice`` is the last fallback of the
    others (#214, ``docs/live/NATURAL_TTS.md``). ``clone_of`` names an
    enrolled person whose voice is wanted instead, cloned by
    ``clone_engine``; ``match_rate_of`` one whose speaking rate sets
    the length scale (``orchestrator.domain.voices``).
    """

    kind: Literal["piper"]
    voice: str
    engine: Literal["piper", "pocket", "qwen3"] = "piper"
    pocket_voice: str = Field(default="lola", pattern=r"^[a-z_]{1,32}$")
    pocket_device: Literal["cpu", "cuda"] = "cpu"
    clone_engine: Literal["qwen3", "openvoice"] = "qwen3"
    length_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    noise_scale: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    noise_w: Optional[float] = Field(default=None, ge=0.0, le=1.5)
    sentence_silence: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    clone_of: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    )
    match_rate_of: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    )


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


class InterviewSection(_Strict):
    """The AI interviewer of ``voicebot.interview`` (#208).

    ``corpus`` is the interview-method domain the planner and the
    interviewer read; ``filler`` the slot filler after each answer
    (``haiku``: ``filler_template`` on the reasoner; ``heuristic``: no
    model); ``max_probes`` the follow-ups per slot before the
    clarification offer; ``minutes`` the default time budget of a new
    plan; ``report`` whether the slots are scored by ``report_template``
    when the interview ends (else a coarse estimate from each slot's
    status); ``post_call`` whether the analysis starts by itself.
    The turn-taking thresholds (VB-10, #213,
    :mod:`~orchestrator.domain.turn_taking`) are seconds:
    ``turn_silence_s`` ends an answer, ``think_s`` is kept after a
    question, ``max_wait_s`` before re-asking or ending an unfinished
    answer, ``stuck_s`` before a hint; ``backchannel`` is ``voice``,
    ``visual`` or ``off``.
    """

    corpus: str = "interview.generic"
    corpus_top_k: int = Field(default=6, ge=1, le=10)
    planner_template: str = Field(
        default="interview.planner@2", pattern=r"^[a-z_.]+@[0-9]+$"
    )
    filler_template: str = Field(
        default="interview.slots@2", pattern=r"^[a-z_.]+@[0-9]+$"
    )
    report_template: str = Field(
        default="interview.report@2", pattern=r"^[a-z_.]+@[0-9]+$"
    )
    filler: Literal["haiku", "heuristic"] = "haiku"
    max_probes: int = Field(default=1, ge=0, le=3)
    minutes: int = Field(default=20, ge=5, le=90)
    report: bool = True
    post_call: bool = False
    turn_silence_s: float = Field(default=3.5, ge=2.0, le=6.0)
    think_s: float = Field(default=6.0, ge=3.0, le=15.0)
    max_wait_s: float = Field(default=10.0, ge=5.0, le=20.0)
    stuck_s: float = Field(default=7.0, ge=4.0, le=15.0)
    backchannel: Literal["voice", "visual", "off"] = "voice"


VOICEBOTS = ("voicebot", "voicebot.interview")
"""The products that speak (``tts``)."""


class ProductConfig(_Strict):
    """The composition of one product."""

    product: Literal[
        "post_call", "live_assist.interview", "live_assist.support",
        "voicebot", "voicebot.interview",
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
    interview: Optional[InterviewSection] = None

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
        if self.tts is not None and self.product not in VOICEBOTS:
            raise ValueError("only the voicebot speaks")
        if (self.interview is not None) != (
            self.product == "voicebot.interview"
        ):
            raise ValueError(
                "voicebot.interview, and only it, has an interview section"
            )
        if self.source.kind == "pipewire" and not self.source.channels:
            raise ValueError("a pipewire source declares its channels")
        return self

    def _check_reasoner(self, reasoner: Reasoner) -> None:
        """Reject a reasoner section that breaks the design's rules."""
        wanted = {
            "live_assist.interview": "suggester",
            "live_assist.support": "suggester",
            "voicebot": "conversational",
            "voicebot.interview": "conversational",
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
