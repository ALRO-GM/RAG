"""The consent notice the pages show before recording or enrolling.

The notices are versioned data of the domain
(:mod:`transcriber.domain.consent`); the version in force is a setting
(``api.consent_version``), so the pages always show the text an
agreement made now is recorded against.
"""

from transcriber.domain.consent import ConsentNotice, consent_notice


class ShowConsentNotice:
    """The consent notice in force, of one kind, in one language."""

    def __init__(self, consent_version: str) -> None:
        """Keep the version of the notices in force."""
        self._consent_version = consent_version

    @property
    def consent_version(self) -> str:
        """Return the version of the notices in force."""
        return self._consent_version

    def __call__(self, kind: str, language: str) -> ConsentNotice:
        """Return the notice of ``kind`` (``enrolment`` or ``recording``).

        :class:`~transcriber.domain.errors.InvalidInputError` for an
        unknown kind or a language without that notice.
        """
        return consent_notice(kind, language, self._consent_version)
