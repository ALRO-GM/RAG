"""The voice bot's voice: Piper's controls, rate matching and cloning.

Plain data and functions: no I/O (``docs/live/VOICE_CLONING.md``).

- **Rate matching** (``voicebot.match_rate``): an enrolled operator's
  speaking rate (words per second, measured by the transcriber on the
  enrolment reading) sets Piper's length scale, so the bot speaks at
  the operator's pace: ``length_scale = piper_rate / operator_rate``,
  where Piper's rate is :data:`PIPER_WORDS_PER_SECOND` (measured at
  length scale 1 by ``experiments/live/piper_rate.py``), clamped to
  :data:`LENGTH_SCALE_RANGE`.
- **Cloning** (``voicebot.voice = clone:<person>``): the bot speaks with
  the person's timbre only when their voice profile says they agreed
  to cloning *and* a reference clip of their voice is kept; otherwise
  it speaks with Piper (:func:`plan_voice`). There is no spoken notice
  when the bot speaks with a cloned voice (owner's decision, #207;
  tracked as a compliance item in #151).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

PIPER_WORDS_PER_SECOND = {
    "es_ES-davefx-medium": 4.065, "es_ES-sharvard-medium": 3.332,
}
"""Words per second of each Piper voice at length scale 1.

Ten fixed Spanish sentences, words over voiced seconds
(``experiments/live/piper_rate_results.json``)."""
LENGTH_SCALE_RANGE: Tuple[float, float] = (0.7, 1.5)
"""Lowest and highest length scale rate matching may choose."""
MIN_OPERATOR_RATE = 0.5
"""Below this many words per second a measured rate is not trusted."""
SAMPLE_TEXT = (
    "Hola, soy el asistente de voz. Así sonaré cuando te responda;"
    " dime si prefieres que hable más despacio o más deprisa."
)
"""The fixed sentence of the «Escuchar ejemplo» button."""


@dataclass(frozen=True)
class VoiceProfile:
    """What the voice bot may use of an enrolled person's voice.

    ``words_per_second`` is their measured speaking rate (``None``:
    not measured); ``clone_consent`` whether they agreed to their voice
    being cloned (revocable); ``reference`` the clip of their voice a
    cloner imitates (only kept while that consent stands).
    """

    speaker_id: str
    words_per_second: Optional[float] = None
    clone_consent: bool = False
    reference: Optional[str] = None

    @property
    def clonable(self) -> bool:
        """Whether this voice may be cloned now (consent and a clip)."""
        return self.clone_consent is True and bool(self.reference)


def matched_length_scale(
    operator_rate: Optional[float], voice: str
) -> Optional[float]:
    """Return the length scale that matches ``operator_rate``, or None.

    ``None`` when the rate is missing or implausible, or Piper's own
    rate of ``voice`` is not known.
    """
    piper = PIPER_WORDS_PER_SECOND.get(voice)
    if (piper is None or operator_rate is None
            or not operator_rate >= MIN_OPERATOR_RATE):
        return None
    low, high = LENGTH_SCALE_RANGE
    return round(min(high, max(low, piper / operator_rate)), 3)


@dataclass(frozen=True)
class VoicePlan:
    """How a voice-bot session speaks.

    ``clone`` is the profile whose voice is cloned (``None``: Piper's
    ``voice``); ``reason`` says why a wanted clone or rate was not used
    (empty when everything wanted is used).
    """

    voice: str
    length_scale: float
    clone: Optional[VoiceProfile] = None
    reason: str = ""


def plan_voice(
    voice: str,
    length_scale: float,
    clone_of: Optional[str],
    match_rate_of: Optional[str],
    profile: Callable[[str], Optional[VoiceProfile]],
) -> VoicePlan:
    """Decide the session's voice from the settings and the profiles.

    ``voice`` and ``length_scale`` are Piper's; ``clone_of`` the person
    whose voice is wanted, ``match_rate_of`` the person whose pace is
    wanted; ``profile`` looks a person up. Without a profile, without
    consent or without a clip there is no clone: Piper speaks.
    """
    reasons = []
    clone: Optional[VoiceProfile] = None
    if clone_of:
        found = profile(clone_of)
        if found is None:
            reasons.append("no voice profile")
        elif not found.clone_consent:
            reasons.append("no consent to cloning")
        elif not found.clonable:
            reasons.append("no reference clip")
        else:
            clone = found
    if match_rate_of:
        found = profile(match_rate_of)
        matched = matched_length_scale(
            found.words_per_second if found is not None else None, voice
        )
        if matched is None:
            reasons.append("no measured speaking rate")
        else:
            length_scale = matched
    return VoicePlan(voice=voice, length_scale=length_scale, clone=clone,
                     reason="; ".join(reasons))
