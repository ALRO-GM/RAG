"""Apply the «Ajustes» page's live choices to a product (``settings.v1``).

The product file says which reasoner model, deadline and attribution a
live session uses; the pipeline settings saved from the web may replace
them when a session starts: ``live.suggester_model`` (haiku, sonnet,
opus), ``live.deadline`` (``product`` keeps the file's, or a number of
milliseconds), ``live.diarization`` (``off`` hears a one-microphone
session as a single person instead of telling the voices apart) and
``live.assist_mode`` (``pincelada`` or ``guiado``, LV-16). Every
value is matched against an allow-list; anything else keeps the
product's value. The voice bot has its own: ``voicebot.reasoner_model``,
``voicebot.voice`` (a Piper voice of :data:`VOICES`, or ``clone:<person>``),
Piper's controls (``voicebot.length_scale``, ``noise_scale``, ``noise_w``,
``sentence_silence``, numbers within :data:`PIPER_RANGES`) and
``voicebot.match_rate`` (a person whose pace sets the length scale); the
suggester's choices do not apply to it. Plain data: no I/O.
"""

import re
from typing import Any, Dict, Mapping, Optional, Tuple

from orchestrator.domain.config import VOICEBOTS, ProductConfig

MODELS = ("haiku", "sonnet", "opus")
MODES = ("pincelada", "guiado")
"""Assistance modes of the ``@2`` suggester templates (LV-16)."""
DEADLINES = {"1800": 1800, "2800": 2800}
VOICES = {
    "davefx": "es_ES-davefx-medium", "sharvard": "es_ES-sharvard-medium",
}
"""Voice option → Piper voice (``--voices-dir``)."""
CLONE = re.compile(r"clone:([A-Za-z0-9][A-Za-z0-9._-]{0,63})")
PERSON = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CLONE_BASE = "davefx"
"""The Piper voice of a clone (its fallback, and the measured one)."""
PIPER_RANGES: Dict[str, Tuple[float, float]] = {
    "length_scale": (0.7, 1.5), "noise_scale": (0.0, 1.0),
    "noise_w": (0.0, 1.0), "sentence_silence": (0.0, 1.0),
}
"""Piper control → accepted range (``settings.v1``)."""
NUMBER = re.compile(r"[0-9]{1,2}(?:\.[0-9]{1,3})?")


def _number(text: Optional[str], low: float, high: float) -> Optional[float]:
    """Return ``text`` as a number within ``[low, high]``, else None."""
    if not isinstance(text, str) or not NUMBER.fullmatch(text):
        return None
    value = float(text)
    return value if low <= value <= high else None


def voice_choices(choices: Mapping[str, str]) -> Dict[str, Any]:
    """Return the ``Tts`` fields the voice bot's choices ask for.

    Only valid values are returned; anything else keeps the product's.
    """
    update: Dict[str, Any] = {}
    option = choices.get("voicebot.voice", "")
    clone = CLONE.fullmatch(option) if isinstance(option, str) else None
    if clone is not None:
        update["voice"] = VOICES[CLONE_BASE]
        update["clone_of"] = clone.group(1)
    elif option in VOICES:
        update["voice"] = VOICES[option]
    for name, (low, high) in PIPER_RANGES.items():
        value = _number(choices.get(f"voicebot.{name}"), low, high)
        if value is not None:
            update[name] = value
    rate = choices.get("voicebot.match_rate", "")
    if isinstance(rate, str) and rate != "off" and PERSON.fullmatch(rate):
        update["match_rate_of"] = rate
    return update


def apply_live_choices(
    config: ProductConfig, choices: Optional[Mapping[str, str]]
) -> ProductConfig:
    """Return ``config`` with the saved live choices applied."""
    if not choices or config.reasoner is None:
        return config
    if config.product in VOICEBOTS:
        return _voicebot(config, choices)
    reasoner: Dict[str, Any] = {}
    model = choices.get("live.suggester_model")
    if model in MODELS:
        reasoner["model"] = model
    deadline = DEADLINES.get(choices.get("live.deadline", ""))
    if deadline is not None:
        reasoner["deadline_ms"] = deadline
    mode = choices.get("live.assist_mode")
    if mode in MODES:
        reasoner["mode"] = mode
    update: Dict[str, Any] = {}
    if reasoner:
        update["reasoner"] = config.reasoner.model_copy(update=reasoner)
    if (choices.get("live.diarization") == "off"
            and config.attribution.kind == "sortformer_stream"):
        update["attribution"] = config.attribution.model_copy(
            update={"kind": "single_speaker"}
        )
    return config.model_copy(update=update) if update else config


def _voicebot(
    config: ProductConfig, choices: Mapping[str, str]
) -> ProductConfig:
    """Return the voice bot with its own saved model and voice."""
    update: Dict[str, Any] = {}
    model = choices.get("voicebot.reasoner_model")
    if model in MODELS and config.reasoner is not None:
        update["reasoner"] = config.reasoner.model_copy(
            update={"model": model}
        )
    voice = voice_choices(choices)
    if voice and config.tts is not None:
        update["tts"] = config.tts.model_copy(update=voice)
    if config.interview is not None:
        update.update(_interview(config, choices))
    return config.model_copy(update=update) if update else config


PROBES = {"0": 0, "1": 1, "2": 2}
MINUTES = {"15": 15, "20": 20, "30": 30, "45": 45}
FILLERS = ("haiku", "heuristic")
SWITCH = {"on": True, "off": False}


def _interview(
    config: ProductConfig, choices: Mapping[str, str]
) -> Dict[str, Any]:
    """Return the interviewer's own saved choices (#208), if any."""
    assert config.interview is not None
    section: Dict[str, Any] = {}
    filler = choices.get("interview.slot_filler")
    if filler in FILLERS:
        section["filler"] = filler
    probes = PROBES.get(choices.get("interview.max_probes", ""))
    if probes is not None:
        section["max_probes"] = probes
    minutes = MINUTES.get(choices.get("interview.minutes", ""))
    if minutes is not None:
        section["minutes"] = minutes
    post = SWITCH.get(choices.get("interview.post_call", ""))
    if post is not None:
        section["post_call"] = post
    if not section:
        return {}
    return {"interview": config.interview.model_copy(update=section)}
