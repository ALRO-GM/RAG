"""The scripted voice-assistant example of a demo (``live.demo.v1``, #250).

A short exchange between a customer and the voice assistant of the same
fictitious company: what the customer says, what the assistant answers
and the knowledge sources the answer comes from. It is played by the
page (the assistant's lines can be spoken by the browser's own voice);
nothing is asked of a model, the voice service or the GPU, and it says
it is a scripted example. Texts are Spanish with an ``en`` mapping.
Plain data and rules: no I/O.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VoiceExchange(BaseModel):
    """One customer question and the assistant's answer."""

    model_config = ConfigDict(extra="forbid")

    user: str = Field(min_length=1)
    bot: str = Field(min_length=1)
    cites: List[str] = Field(min_length=1)
    """Ids of the demo's ``sources`` the answer comes from."""
    en: Optional[Dict[str, str]] = None
    """``user`` and ``bot`` in English (the page can speak either)."""


class VoiceDemo(BaseModel):
    """The scripted conversation with the voice assistant."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    intro: str = Field(min_length=1)
    exchanges: List[VoiceExchange] = Field(min_length=1)
    en: Optional[Dict[str, str]] = None
    """``title`` and ``intro`` in English."""
    user: str = "cliente"
    """The cast id of the customer who asks (#254: whose voice reads it)."""
    bot: str = "asistente"
    """The cast id of the assistant who answers."""
