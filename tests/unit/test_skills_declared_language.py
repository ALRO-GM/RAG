"""A transcript recorded with language ``auto`` is analysed, not refused."""
from typing import Optional

import pytest

from src.domain.skills.transcript import declared_language


@pytest.mark.parametrize("given, expected", [
    ("es", "es"), ("EN", "en"), ("es-ES", "es"),
    ("auto", None), ("fr", None), ("", None), (None, None),
])
def test_only_supported_codes_are_kept(
    given: Optional[str], expected: Optional[str],
) -> None:
    assert declared_language(given) == expected
