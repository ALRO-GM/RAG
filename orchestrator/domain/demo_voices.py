"""The voices of the scripted examples (DM-3, #254): rules, no I/O.

Every line of a demo (``live.demo.v1``) can be heard: the sentences of
the call (or interview) and both sides of the voice-assistant example.
The clips are rendered ahead of time by an operator who says explicitly
which voice reads each scripted person (``--voice agente=pocket:lola``);
they are kept outside the repository with a manifest
(``demo.voices.v1``, :class:`DemoVoiceManifest`), and the page plays
them in step with the text.

- **Voices** (:func:`parse_voice`): a stock voice of the catalogue
  (:data:`STOCK_VOICES`: Pocket TTS ``lola``, Piper ``davefx`` and the
  two speakers of ``sharvard``), whose gender is known, or
  ``clone:<person>`` (optionally ``clone:<person>:female``), the cloned
  voice of an enrolled person who agreed to cloning.
- **Checks** (:func:`check_voices`): every person who speaks has a
  voice, two people of one example never share one, a voice's gender
  (when known) is the person's, and a clone needs the person's consent
  and reference clip, as the voice bot's does
  (``docs/live/VOICE_CLONING.md`` §2).
- **What to render** (:func:`stale_lines`): a line whose text, voice or
  engine changed since its clip, or that has none.
- **What to play** (:func:`playable`): an example is heard only when
  every one of its lines has a clip of its current text and voice whose
  person (for a clone) still agrees; otherwise it falls back to text.
- **Loudness** (:func:`even_loudness`): every clip is brought to one
  level, so voices of different engines sound alike side by side.
- **Timing** (:func:`retime`): the call's sentences take the clips' real
  lengths (the pauses between them kept), so the text types in while
  the voice says it, with the sound on or off.
"""

import hashlib
import math
import re
from array import array
import unicodedata
from dataclasses import dataclass
from typing import (
    Callable, Dict, List, Literal, Mapping, Optional, Set, Tuple,
)

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.domain.demo import DemoScript
from orchestrator.domain.events import Transcript
from orchestrator.domain.voices import VoiceProfile

SCHEMA = "demo.voices.v1"
CLIP_KEY = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$"
"""A line key (an utterance id, ``vb0.user``): also the clip's URL."""
PERSON_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CLONE_ENGINES = ("qwen3", "openvoice")
"""The cloners a clone may be rendered with (``NATURAL_TTS.md`` §6)."""
Example = Literal["call", "voicebot"]
EXAMPLES: Tuple[Example, ...] = ("call", "voicebot")


@dataclass(frozen=True)
class DemoVoice:
    """A voice that reads one scripted person.

    ``spec`` is how the operator names it (``pocket:lola``,
    ``piper:davefx``, ``clone:ana``); ``engine`` the family that speaks
    it (``pocket``, ``piper``, ``qwen3``, ``openvoice``); ``voice`` the
    engine's own voice (Piper's model, Pocket's catalogue voice; empty
    for a clone); ``speaker`` Piper's speaker of a multi-speaker model;
    ``person`` the enrolled person of a clone.
    """

    spec: str
    engine: str
    voice: str = ""
    speaker: Optional[int] = None
    gender: Optional[str] = None
    person: Optional[str] = None

    @property
    def kind(self) -> Literal["clone", "stock"]:
        """``clone`` or ``stock``."""
        return "clone" if self.person else "stock"


STOCK_VOICES: Dict[str, DemoVoice] = {
    "pocket:lola": DemoVoice("pocket:lola", "pocket", "lola",
                             gender="female"),
    "piper:davefx": DemoVoice("piper:davefx", "piper",
                              "es_ES-davefx-medium", gender="male"),
    "piper:sharvard-m": DemoVoice("piper:sharvard-m", "piper",
                                  "es_ES-sharvard-medium", 0, "male"),
    "piper:sharvard-f": DemoVoice("piper:sharvard-f", "piper",
                                  "es_ES-sharvard-medium", 1, "female"),
}
"""The stock Spanish voices a demo may use, with their gender (Pocket's
``lola`` is its one Spanish voice; Piper's ``sharvard`` model has a male
speaker 0 and a female speaker 1, its ``.onnx.json`` says)."""


def parse_voice(text: str, clone_engine: str = "qwen3") -> DemoVoice:
    """Read a voice as the operator writes it.

    ``clone:<person>`` is rendered by ``clone_engine``; a trailing
    ``:female`` / ``:male`` says the person's voice gender, which is
    otherwise unknown.

    Raises:
        ValueError: Not a stock voice nor a well-formed clone.
    """
    text = text.strip()
    if text in STOCK_VOICES:
        return STOCK_VOICES[text]
    kind, _, rest = text.partition(":")
    if kind != "clone" or not rest:
        raise ValueError(
            f"unknown voice {text!r}: one of {', '.join(STOCK_VOICES)}"
            " or clone:<person>[:female|:male]"
        )
    person, _, gender = rest.partition(":")
    if not PERSON_ID.fullmatch(person) or ".." in person:
        raise ValueError(f"invalid person id {person!r}")
    if gender not in ("", "female", "male"):
        raise ValueError(f"unknown gender {gender!r}: female or male")
    if clone_engine not in CLONE_ENGINES:
        raise ValueError(f"unknown cloner {clone_engine!r}")
    return DemoVoice(f"clone:{person}", clone_engine, gender=gender or None,
                     person=person)


@dataclass(frozen=True)
class DemoLine:
    """One line of a demo as it is heard.

    ``key`` names its clip (the utterance id; ``vb<n>.user`` and
    ``vb<n>.bot`` in the voice-assistant example); ``person`` is the
    cast id of who says it.
    """

    key: str
    example: Example
    person: str
    text: str

    @property
    def text_sha256(self) -> str:
        """The hash of the text the clip must say."""
        return text_hash(self.text)


def text_hash(text: str) -> str:
    """Return the SHA-256 of a line's text (Unicode NFC, trimmed)."""
    clean = unicodedata.normalize("NFC", " ".join(text.split()))
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def demo_lines(script: DemoScript) -> List[DemoLine]:
    """Every line of ``script`` that can be heard, in playing order.

    The voice-assistant example's lines are Spanish, as the call's.
    """
    lines = [DemoLine(u.id, "call", u.speaker, u.text)
             for u in script.transcript.utterances if u.text.strip()]
    if script.voicebot is not None:
        vb = script.voicebot
        for n, exchange in enumerate(vb.exchanges):
            lines.append(DemoLine(f"vb{n}.user", "voicebot", vb.user,
                                  exchange.user))
            lines.append(DemoLine(f"vb{n}.bot", "voicebot", vb.bot,
                                  exchange.bot))
    for line in lines:
        if not re.fullmatch(CLIP_KEY, line.key):
            raise ValueError(f"line id {line.key!r} cannot name a clip")
    return lines


def people_of(script: DemoScript) -> Dict[str, Optional[str]]:
    """Cast id → gender of everybody who says a line (``None``: any)."""
    genders = {c.id: c.gender for c in script.cast}
    return {line.person: genders.get(line.person)
            for line in demo_lines(script)}


def check_voices(
    script: DemoScript,
    voices: Dict[str, DemoVoice],
    profile: Callable[[str], Optional[VoiceProfile]],
) -> List[str]:
    """Return what is wrong with ``voices`` for ``script`` (empty: none).

    ``profile`` looks an enrolled person up. A clone needs the person's
    clone consent and reference clip; two people who talk in the same
    example need different voices; a voice of known gender reads only a
    person of that gender (rename the person in the cast to change it).
    """
    problems: List[str] = []
    people = people_of(script)
    for person in sorted(set(voices) - set(people)):
        problems.append(f"{person!r} says nothing in {script.id}")
    for person in sorted(set(people) - set(voices)):
        problems.append(f"no voice for {person!r}")
    names = {c.id: c.name for c in script.cast}
    for person, gender in sorted(people.items()):
        voice = voices.get(person)
        if voice is None:
            continue
        if gender and voice.gender and gender != voice.gender:
            problems.append(
                f"{voice.spec} is a {voice.gender} voice and "
                f"{names.get(person, person)!r} ({person}) is {gender}:"
                " choose another voice or rename the person in the cast"
            )
        if voice.person is not None:
            found = profile(voice.person)
            if found is None:
                problems.append(f"{voice.spec}: no enrolled person")
            elif not found.clone_consent:
                problems.append(f"{voice.spec}: no consent to cloning")
            elif not found.clonable:
                problems.append(f"{voice.spec}: no reference clip")
    for example in EXAMPLES:
        speaking = sorted({line.person for line in demo_lines(script)
                           if line.example == example})
        seen: Dict[str, str] = {}
        for person in speaking:
            voice = voices.get(person)
            if voice is None:
                continue
            if voice.spec in seen:
                problems.append(
                    f"{seen[voice.spec]!r} and {person!r} would share"
                    f" {voice.spec} in the {example}"
                )
            seen.setdefault(voice.spec, person)
    return problems


class DemoClip(BaseModel):
    """One rendered line in the manifest."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=CLIP_KEY)
    file: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.wav$")
    person: str = Field(min_length=1)
    """The cast id of who says the line."""
    voice: str = Field(min_length=1)
    """The voice's spec (``pocket:lola``, ``clone:ana``)."""
    kind: Literal["clone", "stock"]
    cloned_from: Optional[str] = None
    """The enrolled person whose voice was cloned (``kind`` clone)."""
    engine: str = Field(min_length=1)
    """What spoke it (``pocket-cpu``, ``piper``, ``qwen3-cuda``)."""
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seconds: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    rendered_at: str


class DemoVoiceManifest(BaseModel):
    """``demo.voices.v1``: the clips of one demo and who reads whom.

    Kept next to the clips, outside the repository, mode 600. The
    transcriber reads ``clips[*].cloned_from`` to delete a person's
    clips when their clone consent is withdrawn.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["demo.voices.v1"] = Field(
        default="demo.voices.v1", alias="schema")
    demo: str = Field(min_length=1)
    voices: Dict[str, str] = Field(default_factory=dict)
    """Cast id → the voice's spec, as the operator chose it."""
    clips: Dict[str, DemoClip] = Field(default_factory=dict)
    updated_at: Optional[str] = None


def clip_file(line: DemoLine, voice: DemoVoice, engine: str) -> str:
    """The file a line's clip is kept in: its key and a short hash.

    A new text, voice or engine gives a new name, so a render writes
    beside the old clips and only replaces them once it succeeded.
    """
    tag = hashlib.sha256(
        f"{line.text_sha256}|{voice.spec}|{engine}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{line.key}-{tag}.wav"


def stale_lines(
    lines: List[DemoLine],
    voices: Dict[str, DemoVoice],
    engine_of: Callable[[DemoVoice], str],
    manifest: Optional[DemoVoiceManifest],
    exists: Callable[[str], bool],
) -> List[DemoLine]:
    """The lines whose clip is missing or no longer fits.

    A clip fits while its text hash, voice and engine are the line's
    and its file is there.
    """
    clips = manifest.clips if manifest is not None else {}
    stale: List[DemoLine] = []
    for line in lines:
        voice = voices[line.person]
        clip = clips.get(line.key)
        if (clip is None or clip.text_sha256 != line.text_sha256
                or clip.voice != voice.spec
                or clip.engine != engine_of(voice)
                or clip.person != line.person
                or not exists(clip.file)):
            stale.append(line)
    return stale


def withdrawn(
    manifest: DemoVoiceManifest,
    profile: Callable[[str], Optional[VoiceProfile]],
) -> Set[str]:
    """The cloned people of ``manifest`` who may no longer be cloned."""
    people = {c.cloned_from for c in manifest.clips.values()
              if c.kind == "clone" and c.cloned_from}
    out: Set[str] = set()
    for person in people:
        found = profile(person)
        if found is None or not found.clonable:
            out.add(person)
    return out


def playable(
    lines: List[DemoLine],
    manifest: Optional[DemoVoiceManifest],
    exists: Callable[[str], bool],
    gone: Optional[Set[str]] = None,
) -> Dict[Example, Dict[str, DemoClip]]:
    """The clips the page may play, per example: all lines or none.

    A clip is played only when it says the line's current text in the
    voice chosen for its person, its file is there and (for a clone)
    its person is not in ``gone``. An example with one line short of
    that is not heard at all: half a conversation in voices and half in
    silence would be worse than the text.
    """
    out: Dict[Example, Dict[str, DemoClip]] = {}
    if manifest is None:
        return out
    for example in EXAMPLES:
        mine = [line for line in lines if line.example == example]
        found: Dict[str, DemoClip] = {}
        for line in mine:
            clip = manifest.clips.get(line.key)
            if (clip is None or clip.text_sha256 != line.text_sha256
                    or clip.person != line.person
                    or manifest.voices.get(line.person) != clip.voice
                    or (clip.cloned_from or "") in (gone or set())
                    or not exists(clip.file)):
                break
            found[line.key] = clip
        else:
            if mine:
                out[example] = found
    return out


def retime(transcript: Transcript, seconds: Dict[str, float]) -> Transcript:
    """Return ``transcript`` with its sentences as long as their clips.

    Each sentence keeps the pause before it (never an overlap) and takes
    ``seconds[id]`` (its own length when absent).
    """
    utterances = []
    shift_end = 0.0
    previous_end: Optional[float] = None
    for u in transcript.utterances:
        gap = u.start if previous_end is None else max(
            0.0, u.start - previous_end)
        start = round(shift_end + gap, 3)
        length = seconds.get(u.id, max(0.0, u.end - u.start))
        end = round(start + length, 3)
        utterances.append(u.model_copy(update={"start": start, "end": end}))
        previous_end, shift_end = u.end, end
    return transcript.model_copy(update={"utterances": utterances})


def page_view(
    script: DemoScript,
    clips: Mapping[str, Dict[str, DemoClip]],
) -> Dict[str, object]:
    """What the page gets: per example, whether it is heard and each
    clip's length; per person, whether the voice is cloned or stock.

    No person id of a clone and no file name leave the server.
    """
    kinds: Dict[str, str] = {}
    examples: Dict[str, object] = {}
    for example in EXAMPLES:
        heard = clips.get(example)
        examples[example] = {
            "available": bool(heard),
            "clips": {k: {"seconds": round(c.seconds, 3)}
                      for k, c in (heard or {}).items()},
        }
        for clip in (heard or {}).values():
            kinds[clip.person] = clip.kind
    return {
        "demo": True, "id": script.id, "examples": examples,
        "voices": dict(sorted(kinds.items())),
        "kinds": sorted(set(kinds.values())),
    }


TARGET_RMS = 3000.0
"""Loudness every clip is brought to (16-bit RMS), so two engines of
different levels sound alike side by side in one conversation."""
PEAK = 30000
"""Highest sample after the gain: the gain is lowered rather than clip."""


def even_loudness(pcm: bytes, target: float = TARGET_RMS) -> bytes:
    """Return 16-bit mono PCM scaled to ``target`` RMS, never clipping.

    Silence stays silence; a gain is capped so the loudest sample stays
    at or under :data:`PEAK`.
    """
    samples = array("h")
    samples.frombytes(pcm[:len(pcm) - len(pcm) % 2])
    if not samples:
        return b""
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    peak = max(abs(s) for s in samples)
    if rms < 1.0 or peak == 0:
        return samples.tobytes()
    gain = min(target / rms, PEAK / peak)
    return array("h", (int(round(s * gain)) for s in samples)).tobytes()
