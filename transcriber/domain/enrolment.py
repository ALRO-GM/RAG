"""Enrolment: the text a person reads and the checks on their sample.

To put a user id on a voice, each person records one sample reading a
fixed text that starts with their name (``docs/transcriber/
ARCHITECTURE.md`` §6). The voice model does not care about the words;
the text only guarantees enough varied speech, and the spoken name
lets a human check whose sample it is. The texts are versioned data:
the version read is stored with the consent.

:func:`enrolment_vector` turns the window embeddings of a sample into
one voiceprint vector and rejects a sample that is too short, mostly
silent, or holds more than one voice. Every limit is a field of
:class:`SampleRules`; the several-voices limits were measured in
ticket TR-05 and still let some two-person samples through.
"""

import enum
import math
from collections import Counter
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.errors import InvalidInputError, SampleRejectedError
from transcriber.domain.models import Voiceprint
from transcriber.domain.speakers import speech_seconds
from transcriber.domain.voices import FloatArray, cluster_speakers, unit_rows

NAME_PLACEHOLDER = "{name}"


class EnrolmentText(BaseModel):
    """One version of the text read at enrolment, in one language.

    ``template`` starts with :data:`NAME_PLACEHOLDER`, replaced by the
    person's name. It is written to take about 30 seconds read aloud
    at a calm pace; that reading time has not been timed.
    """

    model_config = ConfigDict(frozen=True)

    language: str = Field(min_length=2)
    version: str = Field(min_length=1)
    template: str = Field(min_length=1)

    def text(self, name: str) -> str:
        """Return the text to read, with ``name`` in place."""
        clean = " ".join(name.split())
        if not clean:
            raise InvalidInputError("The name to read cannot be empty.")
        return self.template.replace(NAME_PLACEHOLDER, clean)


ENROLMENT_TEXTS: Tuple[EnrolmentText, ...] = (
    EnrolmentText(
        language="es", version="es-1",
        template=(
            "Me llamo {name}. Hoy es un buen día para grabar mi voz. Leo"
            " este texto despacio y con mi tono normal, como si hablara"
            " con el equipo. ¿Qué tal sonará? Enseguida lo sabremos. El"
            " viejo reloj de la plaza marca las doce y cuarto; un perro"
            " marrón cruza la calle y un niño ríe junto a la fuente."
            " Cuento del uno al diez: uno, dos, tres, cuatro, cinco,"
            " seis, siete, ocho, nueve y diez. ¡Ya casi termino! Gracias"
            " por escuchar, y hasta la próxima reunión."
        ),
    ),
    EnrolmentText(
        language="en", version="en-1",
        template=(
            "My name is {name}. Today is a good day to record my voice."
            " I read this text slowly, in my usual tone, as if I were"
            " talking to the team. How will it sound? We will know soon."
            " The old clock in the square shows a quarter past twelve; a"
            " brown dog crosses the street and a child laughs by the"
            " fountain. I count from one to ten: one, two, three, four,"
            " five, six, seven, eight, nine and ten. I am nearly done!"
            " Thank you for listening, and see you at the next meeting."
        ),
    ),
)


def enrolment_text(
    language: str, version: Optional[str] = None
) -> EnrolmentText:
    """Return the enrolment text of ``language`` (``es``, ``en-GB``...).

    Without ``version`` the latest one of the language is returned.
    """
    base = language.strip().lower().split("-")[0].split("_")[0]
    candidates = [t for t in ENROLMENT_TEXTS if t.language == base]
    if version is not None:
        candidates = [t for t in candidates if t.version == version]
    if not candidates:
        wanted = f"{language} {version}" if version else language
        raise InvalidInputError(f"No enrolment text for {wanted}.")
    return candidates[-1]


class Rejection(str, enum.Enum):
    """Why an enrolment sample was rejected."""

    TOO_SHORT = "too_short"
    MOSTLY_SILENT = "mostly_silent"
    SEVERAL_VOICES = "several_voices"


class SampleRules(BaseModel):
    """Limits an enrolment sample must meet.

    ``same_voice_threshold`` and ``max_other_voice_share`` were set by
    TR-05 (``docs/transcriber/IDENTIFICATION.md`` §4): on 30 s samples
    of read speech the old 0.5 and 0.2 let 17 of 18 two-reader samples
    through (15 of 18 on held-out readers); 0.85 and 0.1 kept every
    genuine sample (20 of 20 on each set) and let 3 of 18 through on
    the readers used to choose them, 7 of 18 on the held-out readers.
    The other limits are reasoned starting values, not tuned; one
    genuine sample sat at their edge (15.3 s of speech in 30 s, 5
    voiced windows).

    ``min_duration``: seconds of recording. ``min_speech_ratio``: share
    of the recording covered by speech regions. ``min_windows``: windows
    that must carry a voice (a zero or near-zero embedding, norm at or
    below ``min_window_norm``, counts as non-speech). ``min_window_
    seconds``: seconds of the recording those voiced windows must cover,
    so a voiceprint never rests on one or two windows whatever the
    regions claim. ``same_voice_threshold``: the clustering threshold
    (:func:`cluster_speakers`) under which windows count as another
    voice. ``max_other_voice_share``: share of voiced windows that may
    fall outside the main voice (a cough, a word from someone nearby)
    before the sample counts as several voices; those windows are left
    out of the voiceprint.
    """

    model_config = ConfigDict(frozen=True)

    min_duration: float = Field(default=20.0, ge=0, allow_inf_nan=False)
    min_speech_ratio: float = Field(default=0.5, ge=0, le=1)
    min_windows: int = Field(default=5, ge=1)
    min_window_seconds: float = Field(default=10.0, ge=0, allow_inf_nan=False)
    min_window_norm: float = Field(default=1e-6, ge=0, allow_inf_nan=False)
    same_voice_threshold: float = Field(default=0.85, ge=-1, le=1)
    max_other_voice_share: float = Field(default=0.1, ge=0, le=1)


def clip_spans(
    spans: Sequence[Tuple[float, float]], duration: float, what: str
) -> List[Tuple[float, float]]:
    """Return ``spans`` (seconds) clipped to ``[0, duration]``.

    Every bound must be a finite number and no span may end before it
    starts; :class:`InvalidInputError` names ``what`` otherwise. Spans
    left empty by the clipping are kept with zero length, so the result
    stays parallel to the input.
    """
    clipped: List[Tuple[float, float]] = []
    for start, end in spans:
        if not (math.isfinite(start) and math.isfinite(end)):
            raise InvalidInputError(
                f"A {what} needs finite times, got ({start}, {end})."
            )
        if end < start:
            raise InvalidInputError(
                f"A {what} cannot end before it starts: ({start}, {end})."
            )
        low = min(max(start, 0.0), duration)
        clipped.append((low, min(max(end, low), duration)))
    return clipped


def enrolment_vector(
    vectors: FloatArray,
    windows: Sequence[Tuple[float, float]],
    regions: Sequence[Tuple[float, float]],
    duration: float,
    rules: SampleRules = SampleRules(),
) -> List[float]:
    """Return the voiceprint vector of an enrolment sample.

    ``vectors`` are the embeddings of the sample's analysis windows
    (one row per window, a 2-D array), ``windows`` their times in
    seconds, ``regions`` the stretches of speech (seconds) and
    ``duration`` the length of the recording. Regions and windows are
    clipped to the recording. Windows whose embedding is (nearly) zero
    carry no voice and count as non-speech.

    The checks run in order: too short; mostly silent (too little
    speech in the regions, too few voiced windows, or voiced windows
    covering too little of the recording); several voices. Voiced
    windows are clustered with no cap on the number of clusters: a cap
    would merge the closest clusters first, which in a two-voice sample
    with stray windows are the two real voices. The vector is the mean
    of the unit window vectors of the main voice, re-normalised.

    Raises :class:`InvalidInputError` for malformed input (a duration
    that is not a finite non-negative number, a reversed or non-finite
    span, a vector array that is not 2-D or does not match
    ``windows``) and :class:`SampleRejectedError` with the
    :class:`Rejection` reason for a sample that is well formed but
    unfit.
    """
    if not math.isfinite(duration) or duration < 0:
        raise InvalidInputError(
            f"The duration must be a finite number of seconds, got"
            f" {duration}."
        )
    if vectors.ndim != 2 or vectors.shape[0] != len(windows):
        raise InvalidInputError(
            "One embedding row is needed per analysis window."
        )
    if not bool(np.isfinite(vectors).all()):
        raise InvalidInputError("Window embeddings must be finite.")
    spoken = clip_spans(regions, duration, "speech region")
    timed = clip_spans(windows, duration, "window")
    if duration < rules.min_duration:
        raise SampleRejectedError(
            Rejection.TOO_SHORT.value,
            f"The sample lasts {duration:.1f} s; at least"
            f" {rules.min_duration:.1f} s are needed.",
        )
    speech = speech_seconds(spoken)
    if speech < rules.min_speech_ratio * duration:
        raise SampleRejectedError(
            Rejection.MOSTLY_SILENT.value,
            f"Only {speech:.1f} s of {duration:.1f} s contain speech.",
        )
    norms = np.linalg.norm(vectors, axis=1)
    voiced = [i for i in range(len(timed))
              if float(norms[i]) > rules.min_window_norm]
    covered = speech_seconds([timed[i] for i in voiced])
    if len(voiced) < rules.min_windows or covered < rules.min_window_seconds:
        raise SampleRejectedError(
            Rejection.MOSTLY_SILENT.value,
            f"Only {len(voiced)} windows ({covered:.1f} s) carry a voice;"
            f" at least {rules.min_windows} covering"
            f" {rules.min_window_seconds:.1f} s are needed.",
        )
    kept = vectors[voiced]
    labels = cluster_speakers(
        kept, threshold=rules.same_voice_threshold, max_speakers=len(voiced),
    )
    counts = Counter(labels)
    main = min(counts, key=lambda label: (-counts[label], label))
    other = 1.0 - counts[main] / len(labels)
    if other > rules.max_other_voice_share:
        raise SampleRejectedError(
            Rejection.SEVERAL_VOICES.value,
            f"{other:.0%} of the sample sounds like another voice.",
        )
    members = unit_rows(kept)[[i for i, v in enumerate(labels) if v == main]]
    mean = members.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= rules.min_window_norm:
        raise SampleRejectedError(
            Rejection.MOSTLY_SILENT.value,
            "The sample carries no voice information.",
        )
    return [float(x) for x in mean / norm]


def build_voiceprint(
    speaker_id: str,
    model: str,
    consent_version: str,
    enrolled_at: datetime,
    vectors: FloatArray,
    windows: Sequence[Tuple[float, float]],
    regions: Sequence[Tuple[float, float]],
    duration: float,
    rules: SampleRules = SampleRules(),
) -> Voiceprint:
    """Check an enrolment sample and return the person's voiceprint.

    See :func:`enrolment_vector` for the checks; ``model`` names the
    voice encoder that produced ``vectors``.
    """
    return Voiceprint(
        speaker_id=speaker_id,
        model=model,
        vector=enrolment_vector(vectors, windows, regions, duration, rules),
        consent_version=consent_version,
        enrolled_at=enrolled_at,
    )
