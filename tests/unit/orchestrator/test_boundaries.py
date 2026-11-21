"""Import ban across ``src``, ``transcriber`` and ``orchestrator``.

Every ordered pair of the three packages is checked, and the
orchestrator's domain and application stay free of I/O libraries.
"""

from pathlib import Path

import pytest

from tests.unit.transcriber.test_boundaries import (
    REPO,
    crossing_imports,
    imports_of,
    write,
)

PACKAGES = ("src", "transcriber", "orchestrator")
PAIRS = [(a, b) for a in PACKAGES for b in PACKAGES if a != b]
DOMAIN_FORBIDDEN = {
    "torch", "transformers", "fastapi", "httpx", "requests", "yaml", "os",
    "pathlib", "subprocess", "socket", "threading", "src", "transcriber",
}
APPLICATION_FORBIDDEN = {"torch", "transformers", "fastapi", "httpx", "yaml"}


@pytest.mark.parametrize(("source", "target"), PAIRS)
def test_no_package_imports_another(source: str, target: str) -> None:
    assert crossing_imports(REPO, source, target) == []


def test_the_ban_sees_the_orchestrator() -> None:
    assert len(list((REPO / "orchestrator").rglob("*.py"))) >= 10


@pytest.mark.parametrize(("source", "target"), PAIRS)
def test_the_ban_catches_a_crossing(
    tmp_path: Path, source: str, target: str
) -> None:
    write(tmp_path, f"{source}/x.py", f"from {target}.domain import y\n")
    assert crossing_imports(tmp_path, source, target)


def _offences(layer: str, forbidden: "set[str]") -> "list[str]":
    found = []
    for path in sorted((REPO / "orchestrator" / layer).rglob("*.py")):
        for line, module in imports_of(path, REPO):
            if module.split(".")[0] in forbidden or module.startswith(
                "orchestrator.adapters"
            ):
                found.append(f"{path.name}:{line} {module}")
    return found


def test_domain_is_pure() -> None:
    assert _offences("domain", DOMAIN_FORBIDDEN) == []


def test_application_does_not_reach_adapters_or_libraries() -> None:
    assert _offences("application", APPLICATION_FORBIDDEN) == []


def test_inbound_does_not_import_torch() -> None:
    found = []
    for path in sorted((REPO / "orchestrator").rglob("*.py")):
        for line, module in imports_of(path, REPO):
            if module.split(".")[0] in {"torch", "torchaudio",
                                        "transformers"}:
                found.append(f"{path}:{line}")
    assert found == []
