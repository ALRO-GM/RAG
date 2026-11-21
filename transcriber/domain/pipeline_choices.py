"""What the transcriber takes from the pipeline settings (``settings.v1``).

The web's «Ajustes» page saves one choice per step in the RAG
(``contracts/settings.v1.schema.json``); the transcriber reads the
choices of its own steps when a job or a live session starts and turns
them into its settings here. Every value is matched against an
allow-list; an unknown or missing one keeps the service's configuration
(the defaults before the page existed). Plain data: no I/O.
"""

from typing import Dict, Mapping, Optional, Tuple

from transcriber.domain.diarization import (
    DEFAULT_DIARIZER,
    TWO_SPEAKER_DIARIZER,
    DiarizerRule,
)
from transcriber.domain.lab import BUILTIN_DIARIZER

Choices = Mapping[str, str]

RECOGNISERS: Dict[str, str] = {
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-large-v3": "openai/whisper-large-v3",
}
"""Recogniser option → Hugging Face model (loaded at start-up)."""

DIARIZERS: Dict[str, Optional[DiarizerRule]] = {
    "auto": None,
    "diarizen": DiarizerRule(model=DEFAULT_DIARIZER, by_speakers={}),
    "nemotron": DiarizerRule(model=TWO_SPEAKER_DIARIZER, by_speakers={}),
    "pyannote": DiarizerRule(model="pyannote-community-1", by_speakers={}),
    "builtin": DiarizerRule(model=BUILTIN_DIARIZER, by_speakers={}),
}
"""Diarizer option → rule (``None``: the configured rule)."""

NAME_MODELS: Dict[str, Optional[Tuple[str, ...]]] = {
    "qwen3-8b": None,
    "qwen3-4b": ("Qwen/Qwen3-4B",),
    "rules": (),
}
"""Name finder option → local models (``None``: the configured ones;
``()``: rules only, names are only suggested)."""


def recogniser(choices: Optional[Choices], configured: str) -> str:
    """Return the recogniser to load: the chosen one, or ``configured``.

    The first option keeps ``configured`` (which may be another model
    set in the service's config file).
    """
    option = (choices or {}).get("transcription.recogniser", "")
    if option in RECOGNISERS and option != "whisper-large-v3-turbo":
        return RECOGNISERS[option]
    return configured


def diarizer_rule(
    choices: Optional[Choices], configured: DiarizerRule
) -> DiarizerRule:
    """Return the diarizer rule of a job starting now."""
    rule = DIARIZERS.get((choices or {}).get("transcription.diarizer", ""))
    return configured if rule is None else rule


def name_models(
    choices: Optional[Choices], configured: Tuple[str, ...]
) -> Tuple[str, ...]:
    """Return the name finder's models (``()``: rules only)."""
    models = NAME_MODELS.get((choices or {}).get("transcription.names", ""))
    return configured if models is None else models


def live_cpu_only(choices: Optional[Choices]) -> bool:
    """Tell whether live sessions must run on the CPU (no GPU lease)."""
    return (choices or {}).get("live.asr") == "cpu"
