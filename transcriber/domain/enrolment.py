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
:class:`SampleRules` with a starting default that has not been measured
yet (ticket TR-05).
"""

import enum
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
    """Limits an enrolment sample must meet. Defaults are not tuned.

    ``min_duration``: seconds of recording. ``min_speech_ratio``: share
    of the recording covered by speech. ``same_voice_threshold``: the
    clustering threshold (:func:`cluster_speakers`) under which windows
    count as another voice. ``max_other_voice_share``: share of windows
    that may fall outside the main voice (a cough, a word from someone
    nearby) before the sample counts as several voices; those windows
    are left out of the voiceprint.
    """

    model_config = ConfigDict(frozen=True)

    min_duration: float = Field(default=20.0, ge=0, allow_inf_nan=False)
    min_speech_ratio: float = Field(default=0.5, ge=0, le=1)
    same_voice_threshold: float = Field(default=0.5, ge=-1, le=1)
    max_other_voice_share: float = Field(default=0.2, ge=0, le=1)


def enrolment_vector(
    vectors: FloatArray,
    regions: Sequence[Tuple[float, float]],
    duration: float,
    rules: SampleRules = SampleRules(),
) -> List[float]:
    """Return the voiceprint vector of an enrolment sample.

    ``vectors`` are the embeddings of the sample's analysis windows,
    ``regions`` the stretches of speech (seconds) and ``duration`` the
    length of the recording. The checks run in order: too short, mostly
    silent, several voices. The vector is the mean of the unit window
    vectors of the main voice, re-normalised. Raises
    :class:`SampleRejectedError` with the :class:`Rejection` reason.
    """
    if duration < rules.min_duration:
        raise SampleRejectedError(
            Rejection.TOO_SHORT.value,
            f"The sample lasts {duration:.1f} s; at least"
            f" {rules.min_duration:.1f} s are needed.",
        )
    speech = speech_seconds(regions)
    if vectors.shape[0] == 0 or speech < rules.min_speech_ratio * duration:
        raise SampleRejectedError(
            Rejection.MOSTLY_SILENT.value,
            f"Only {speech:.1f} s of {duration:.1f} s contain speech.",
        )
    labels = cluster_speakers(vectors, threshold=rules.same_voice_threshold)
    counts = Counter(labels)
    main = min(counts, key=lambda label: (-counts[label], label))
    other = 1.0 - counts[main] / len(labels)
    if other > rules.max_other_voice_share:
        raise SampleRejectedError(
            Rejection.SEVERAL_VOICES.value,
            f"{other:.0%} of the sample sounds like another voice.",
        )
    members = unit_rows(vectors)[[i for i, v in enumerate(labels)
                                  if v == main]]
    mean = members.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
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
        vector=enrolment_vector(vectors, regions, duration, rules),
        consent_version=consent_version,
        enrolled_at=enrolled_at,
    )
