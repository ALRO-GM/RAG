"""Languages: the code a session is recorded in, and ``auto``.

A session's ``language`` is a language code (``es``, ``en-GB``...) or
:data:`AUTO` (let the recogniser detect the language of each of its
windows: a mixed-language conversation). The recogniser only knows
some base codes; :func:`check_language` refuses any other before a job
runs, so a typo is an input error and not a failed transcription.
:func:`tag_languages` gives each sentence the language detected where
it was said, and flags it when that is not the session's fixed one.
"""

from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.models import Utterance

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


def tag_languages(
    utterances: Sequence[Utterance],
    detected: Sequence[Tuple[float, float, str]],
    session_language: str,
) -> List[Utterance]:
    """Give each utterance the language detected where it was said.

    ``detected`` are ``(start, end, code)`` stretches the recogniser
    told the language of. An utterance takes the code of the stretches
    it shares the most seconds with (an utterance of no length: the
    stretch holding it); none, and its ``language`` stays ``None``.
    ``language_differs`` is set only when ``session_language`` is a
    fixed code: ``True`` when the detected base code is another one.
    """
    fixed = None if is_auto(session_language) else base_code(
        session_language)
    out: List[Utterance] = []
    for utterance in utterances:
        shares: Dict[str, float] = {}
        for start, end, code in detected:
            shared = min(end, utterance.end) - max(start, utterance.start)
            if shared > 0 or (
                utterance.end <= utterance.start
                and start <= utterance.start <= end
            ):
                shares[code] = shares.get(code, 0.0) + max(shared, 0.0)
        if not shares:
            out.append(utterance.model_copy(update={
                "language": None, "language_differs": None,
            }))
            continue
        language = max(shares, key=lambda code: shares[code])
        out.append(utterance.model_copy(update={
            "language": language,
            "language_differs": (
                None if fixed is None else base_code(language) != fixed
            ),
        }))
    return out
