"""Languages: the code a session is recorded in, and ``auto``.

A session's ``language`` is a language code (``es``, ``en-GB``...) or
:data:`AUTO` (let the recogniser detect the language of each of its
windows: a mixed-language conversation). The recogniser only knows
some base codes; :func:`check_language` refuses any other before a job
runs, so a typo is an input error and not a failed transcription.
"""

from typing import AbstractSet, Optional

from transcriber.domain.errors import InvalidInputError

AUTO = "auto"


def base_code(language: str) -> str:
    """Return the base code of ``language``: ``es-ES`` → ``es``.

    Lower case, without region or script (``_`` or ``-`` and what
    follows).
    """
    return language.strip().lower().replace("_", "-").split("-")[0]


def is_auto(language: str) -> bool:
    """Tell whether ``language`` asks for detection (:data:`AUTO`)."""
    return language.strip().lower() == AUTO


def recognition_language(language: str) -> Optional[str]:
    """Return the code given to the recogniser: ``None`` for ``auto``."""
    return None if is_auto(language) else base_code(language)


def check_language(
    language: str, known: Optional[AbstractSet[str]]
) -> str:
    """Return ``language`` if the recogniser can use it.

    ``known`` are the base codes the recogniser knows (``None`` accepts
    any code). :data:`AUTO` is always accepted. Raises
    :class:`InvalidInputError` naming the code otherwise.
    """
    if is_auto(language) or known is None or base_code(language) in known:
        return language
    raise InvalidInputError(
        f"Unknown language {language!r}: give a code the recogniser knows"
        f" (for example es or en) or {AUTO!r} to detect it."
    )
