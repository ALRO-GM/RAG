"""Consent notices: what a person agrees to before their voice is taken.

Two notices exist (``docs/transcriber/ARCHITECTURE.md`` §8): one before a
voice is enrolled and one before a meeting is recorded. They are
versioned data, like the enrolment texts: the pages show the text the
API gives, and the version shown is stored with the enrolment (the
consent record) or the session (``Session.consent_version``). A change
of wording is a new version; old versions stay, so a stored version can
always be looked up.
"""

import enum
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from transcriber.domain.errors import InvalidInputError


class NoticeKind(str, enum.Enum):
    """What a consent notice is for."""

    ENROLMENT = "enrolment"
    RECORDING = "recording"


class ConsentNotice(BaseModel):
    """One version of one notice in one language.

    ``points`` are short plain sentences shown as a list; ``accept`` is
    the sentence the person ticks to agree.
    """

    model_config = ConfigDict(frozen=True)

    kind: NoticeKind
    language: str = Field(min_length=2)
    version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    points: Tuple[str, ...] = Field(min_length=1)
    accept: str = Field(min_length=1)


_TRANSIT_ES = (
    "El audio pasa por el servidor de esta página de camino a ese"
    " ordenador; el servidor no lo guarda."
)
_TRANSIT_EN = (
    "The audio passes through the server of this page on its way to that"
    " computer; the server does not keep it."
)

CONSENT_NOTICES: Tuple[ConsentNotice, ...] = (
    ConsentNotice(
        kind=NoticeKind.ENROLMENT, language="es", version="consent-1",
        title="Consentimiento",
        points=(
            "Se graba tu voz unos 30 segundos mientras lees un texto, y con"
            " ella se calcula una huella numérica de tu voz para"
            " reconocerte en las reuniones.",
            "Eso se hace en el ordenador de quien organiza las reuniones."
            " La huella y esta aceptación se guardan solo allí; el audio"
            " del registro se borra en cuanto se calcula la huella.",
            _TRANSIT_ES,
            "Puedes hacer que se borren cuando quieras: pulsa «Olvidar»"
            " junto a tu nombre en esta página, o pídeselo a quien"
            " organiza.",
        ),
        accept="Acepto que se registre mi voz de esta manera.",
    ),
    ConsentNotice(
        kind=NoticeKind.ENROLMENT, language="en", version="consent-1",
        title="Consent",
        points=(
            "Your voice is recorded for about 30 seconds while you read a"
            " text, and a numeric print of your voice is made from it to"
            " recognise you in meetings.",
            "This is done on the computer of the person who runs the"
            " meetings. The voiceprint and this agreement are kept only"
            " there; the enrolment audio is deleted as soon as the"
            " voiceprint is made.",
            _TRANSIT_EN,
            "You can have them deleted at any time: press «Forget» next to"
            " your name on this page, or ask the person who runs the"
            " meetings.",
        ),
        accept="I agree to my voice being enrolled this way.",
    ),
    ConsentNotice(
        kind=NoticeKind.RECORDING, language="es", version="consent-1",
        title="Antes de empezar",
        points=(
            "Se graba el sonido del micrófono de este teléfono: las voces"
            " de todas las personas presentes. Se convierte en texto con"
            " el nombre de quien habla.",
            "La transcripción se hace en el ordenador de quien organiza la"
            " reunión. El audio y el texto se guardan solo allí.",
            _TRANSIT_ES,
            "Cualquier sesión se puede borrar cuando se quiera, con su"
            " audio y su texto: botón «Borrar» de la lista de sesiones, o"
            " pidiéndoselo a quien organiza.",
        ),
        accept=(
            "Las personas presentes lo saben y están de acuerdo en que se"
            " grabe."
        ),
    ),
    ConsentNotice(
        kind=NoticeKind.RECORDING, language="en", version="consent-1",
        title="Before you start",
        points=(
            "The sound of this phone's microphone is recorded: the voices"
            " of everyone present. It is turned into text with the name of"
            " who speaks.",
            "The transcription is made on the computer of the person who"
            " runs the meeting. The audio and the text are kept only"
            " there.",
            _TRANSIT_EN,
            "Any session can be deleted at any time, with its audio and its"
            " text: the «Delete» button of the sessions list, or by asking"
            " the person who runs the meeting.",
        ),
        accept="Everyone present knows and agrees to being recorded.",
    ),
)


def _kind(kind: str) -> NoticeKind:
    """Return ``kind`` as a :class:`NoticeKind` or raise an input error."""
    try:
        return NoticeKind(kind)
    except ValueError as exc:
        raise InvalidInputError(
            f"No consent notice of kind {kind!r}."
        ) from exc


def consent_notice(
    kind: str, language: str, version: Optional[str] = None
) -> ConsentNotice:
    """Return the notice of ``kind`` in ``language`` (``es``, ``en-GB``...).

    Without ``version`` the latest one is returned. An unknown kind,
    language or version is an :class:`InvalidInputError`.
    """
    wanted = _kind(kind)
    base = language.strip().lower().split("-")[0].split("_")[0]
    found = [
        n for n in CONSENT_NOTICES
        if n.kind is wanted and n.language == base
        and (version is None or n.version == version)
    ]
    if not found:
        raise InvalidInputError(
            f"No {wanted.value} consent notice for {language}"
            + (f" version {version}." if version else ".")
        )
    return found[-1]


def check_consent_version(kind: str, version: str) -> str:
    """Return ``version`` if a notice of ``kind`` has it, else raise."""
    wanted = _kind(kind)
    if not any(
        n.kind is wanted and n.version == version for n in CONSENT_NOTICES
    ):
        raise InvalidInputError(
            f"Unknown {wanted.value} consent version {version!r}."
        )
    return version
