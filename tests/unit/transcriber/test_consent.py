"""Consent notices: versioned data, looked up by kind and language."""

import pytest

from transcriber.application.consent import ShowConsentNotice
from transcriber.domain.consent import (
    CONSENT_NOTICES,
    ConsentNotice,
    NoticeKind,
    check_consent_version,
    consent_notice,
)
from transcriber.domain.errors import InvalidInputError


class TestNotices:
    def test_every_kind_in_spanish_and_english(self) -> None:
        found = {(n.kind, n.language) for n in CONSENT_NOTICES}
        assert found == {
            (kind, language)
            for kind in NoticeKind for language in ("es", "en")
        }

    @pytest.mark.parametrize("notice", CONSENT_NOTICES,
                             ids=lambda n: f"{n.kind.value}-{n.language}")
    def test_each_says_where_the_audio_goes_and_how_to_delete(
        self, notice: ConsentNotice
    ) -> None:
        text = " ".join(notice.points).lower()
        if notice.language == "es":
            words = ["graba", "ordenador", "solo allí", "servidor", "borr"]
        else:
            words = ["recorded", "computer", "only", "server", "delete"]
        for word in words:
            assert word in text, (notice.kind, notice.language, word)
        assert all(len(point) < 260 for point in notice.points)

    def test_lookup_by_language_tag_and_version(self) -> None:
        notice = consent_notice("recording", "en-GB")
        assert (notice.kind, notice.language) == (NoticeKind.RECORDING, "en")
        assert consent_notice("enrolment", "es", "consent-1").title

    @pytest.mark.parametrize("kind, language, version", [
        ("nope", "es", None), ("recording", "xx", None),
        ("recording", "es", "consent-99"),
    ])
    def test_unknown_notices(
        self, kind: str, language: str, version: object
    ) -> None:
        with pytest.raises(InvalidInputError):
            consent_notice(kind, language, version)  # type: ignore[arg-type]

    def test_check_consent_version(self) -> None:
        assert check_consent_version("recording", "consent-1") == "consent-1"
        with pytest.raises(InvalidInputError, match="Unknown recording"):
            check_consent_version("recording", "consent-0")
        with pytest.raises(InvalidInputError, match="kind"):
            check_consent_version("other", "consent-1")


class TestShowConsentNotice:
    def test_the_version_in_force(self) -> None:
        show = ShowConsentNotice("consent-1")
        assert show.consent_version == "consent-1"
        notice = show("recording", "es")
        assert notice.version == "consent-1"
        assert notice.accept.startswith("Las personas presentes")
