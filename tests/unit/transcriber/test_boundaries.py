"""Import rules: ``src`` and ``transcriber`` stay apart; inner layers stay in.

The domain is pure and imports no other transcriber layer; the
application (use cases and ports) imports neither the adapters, the
composition root nor the libraries only adapters may use.

The rules are read from the source with :mod:`ast`, so a forbidden
import fails here even when it sits in a branch no test runs.
"""

import ast
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

REPO = Path(__file__).resolve().parents[3]

# What the transcriber domain may not import: models, web and audio
# libraries, and anything that touches files, processes or the network.
# numpy and scipy are allowed, as in ``src/domain``.
DOMAIN_FORBIDDEN = frozenset({
    "torch", "torchaudio", "transformers", "sentence_transformers",
    "huggingface_hub", "fastapi", "starlette", "uvicorn", "httpx",
    "requests", "urllib", "http", "socket", "soundfile", "av", "segno",
    "fire", "os", "pathlib", "io", "shutil", "subprocess", "tempfile",
    "sqlite3", "glob", "pickle", "threading", "multiprocessing", "anthropic",
})
# Layers of the transcriber the domain may not reach. The domain may
# only import ``transcriber.domain``; these are named for the messages,
# the rule below rejects any other ``transcriber`` module too.
DOMAIN_FORBIDDEN_INTERNAL = (
    "transcriber.application", "transcriber.adapters",
    "transcriber.config", "transcriber.bootstrap", "transcriber.__main__",
)
# What use cases and ports may not import: the adapters that implement
# the ports, the composition root that wires them, and the model, web
# and audio libraries only adapters use.
APPLICATION_FORBIDDEN = frozenset({
    "torch", "torchaudio", "transformers", "fastapi", "starlette",
    "uvicorn", "av", "soundfile",
})
APPLICATION_FORBIDDEN_INTERNAL = (
    "transcriber.adapters", "transcriber.bootstrap", "transcriber.__main__",
)


def module_name(path: Path, root: Path) -> str:
    """Return the dotted module name of ``path`` relative to ``root``."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(path: Path, root: Path) -> Iterator[Tuple[int, str]]:
    """Yield ``(line, module)`` for every import in ``path``.

    Relative imports are resolved; ``importlib.import_module("x")`` and
    ``__import__("x")`` with a literal name count too.
    """
    name = module_name(path, root)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - (node.level - 1)]
                tail = [node.module] if node.module else []
                target = ".".join(base + tail)
            else:
                target = node.module or ""
            yield node.lineno, target
            for alias in node.names:
                yield node.lineno, f"{target}.{alias.name}"
        elif (
            isinstance(node, ast.Call) and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _called(node) in {"import_module", "__import__"}
        ):
            yield node.lineno, node.args[0].value


def _called(node: ast.Call) -> str:
    """Return the bare name of the function called by ``node``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def opens_files(path: Path) -> List[int]:
    """Return the lines of ``path`` that call the built-in ``open``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "open"
    ]


def _within(module: str, prefix: str) -> bool:
    """Tell whether ``module`` is ``prefix`` or inside it."""
    return module == prefix or module.startswith(prefix + ".")


def crossing_imports(root: Path, source: str, target: str) -> List[str]:
    """Return ``file:line module`` for imports of ``target`` in ``source``."""
    found = []
    for path in sorted((root / source).rglob("*.py")):
        for line, module in imports_of(path, root):
            if _within(module, target):
                found.append(f"{path.relative_to(root)}:{line} {module}")
    return found


def impure_domain(root: Path) -> List[str]:
    """Return the violations of the domain rules under ``root``."""
    found = []
    for path in sorted((root / "transcriber" / "domain").rglob("*.py")):
        where = path.relative_to(root)
        for line, module in imports_of(path, root):
            top = module.split(".")[0]
            outside = _within(module, "transcriber") and not _within(
                module, "transcriber.domain"
            )
            if top in DOMAIN_FORBIDDEN or outside or any(
                _within(module, layer) for layer in DOMAIN_FORBIDDEN_INTERNAL
            ):
                found.append(f"{where}:{line} {module}")
        found.extend(f"{where}:{line} open()" for line in opens_files(path))
    return found


def leaky_application(root: Path) -> List[str]:
    """Return the violations of the application rules under ``root``."""
    found = []
    for path in sorted((root / "transcriber" / "application").rglob("*.py")):
        where = path.relative_to(root)
        for line, module in imports_of(path, root):
            if module.split(".")[0] in APPLICATION_FORBIDDEN or any(
                _within(module, layer)
                for layer in APPLICATION_FORBIDDEN_INTERNAL
            ):
                found.append(f"{where}:{line} {module}")
    return found


class TestRealTree:
    def test_transcriber_does_not_import_src(self) -> None:
        assert crossing_imports(REPO, "transcriber", "src") == []

    def test_src_does_not_import_transcriber(self) -> None:
        assert crossing_imports(REPO, "src", "transcriber") == []

    def test_transcriber_domain_is_pure(self) -> None:
        assert impure_domain(REPO) == []

    def test_transcriber_application_stays_inside(self) -> None:
        assert leaky_application(REPO) == []

    def test_the_rules_see_the_real_packages(self) -> None:
        """Guard against a vacuous pass (wrong root, no files found)."""
        domain = list((REPO / "transcriber" / "domain").rglob("*.py"))
        assert len(domain) >= 3
        assert any((REPO / "transcriber" / "application").rglob("*.py"))
        assert any((REPO / "src" / "domain").rglob("*.py"))


def write(root: Path, relative: str, text: str) -> None:
    """Write a Python file of a throw-away tree."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestRulesCatchViolations:
    @pytest.mark.parametrize("text", [
        "import src\n",
        "from src.domain import models\n",
        "from src import config\n",
        "import importlib\nimportlib.import_module('src.config')\n",
        "__import__('src')\n",
    ])
    def test_transcriber_importing_src(
        self, tmp_path: Path, text: str
    ) -> None:
        write(tmp_path, "transcriber/adapters/x.py", text)
        assert crossing_imports(tmp_path, "transcriber", "src")

    def test_src_importing_transcriber(self, tmp_path: Path) -> None:
        write(tmp_path, "src/domain/x.py",
              "from transcriber.domain.models import Word\n")
        assert crossing_imports(tmp_path, "src", "transcriber")

    def test_similar_names_are_not_crossings(self, tmp_path: Path) -> None:
        write(tmp_path, "transcriber/x.py", "import srcery\n")
        assert crossing_imports(tmp_path, "transcriber", "src") == []

    @pytest.mark.parametrize("text", [
        "import torch\n",
        "from transformers import pipeline\n",
        "from fastapi import FastAPI\n",
        "import soundfile as sf\n",
        "import av\n",
        "from pathlib import Path\n",
        "import os.path\n",
        "from transcriber.adapters.inbound import cli\n",
        "from ..application import ports\n",
        "from .. import config\n",
        "from transcriber import bootstrap\n",
        "import transcriber.config\n",
        "from transcriber.application.ports import Recogniser\n",
        "data = open('x').read()\n",
    ])
    def test_impure_domain(self, tmp_path: Path, text: str) -> None:
        write(tmp_path, "transcriber/domain/__init__.py", "")
        write(tmp_path, "transcriber/domain/x.py", text)
        assert impure_domain(tmp_path)

    @pytest.mark.parametrize("text", [
        "import numpy as np\n",
        "from scipy.cluster.hierarchy import linkage\n",
        "import re\nfrom pydantic import BaseModel\n",
        "from .models import Word\n",
        "from transcriber.domain.errors import ContractError\n",
    ])
    def test_allowed_in_domain(self, tmp_path: Path, text: str) -> None:
        write(tmp_path, "transcriber/domain/__init__.py", "")
        write(tmp_path, "transcriber/domain/x.py", text)
        assert impure_domain(tmp_path) == []


class TestApplicationRulesCatchViolations:
    @pytest.mark.parametrize("text", [
        "import torch\n",
        "import torchaudio\n",
        "from transformers import pipeline\n",
        "from fastapi import FastAPI\n",
        "import av\n",
        "import soundfile as sf\n",
        "from transcriber.adapters.outbound import recogniser\n",
        "from ..adapters.inbound import cli\n",
        "from transcriber import bootstrap\n",
        "from transcriber.bootstrap import build\n",
        "from .. import bootstrap\n",
        "import importlib\nimportlib.import_module('transcriber.adapters')\n",
    ])
    def test_leaky_application(self, tmp_path: Path, text: str) -> None:
        write(tmp_path, "transcriber/application/__init__.py", "")
        write(tmp_path, "transcriber/application/x.py", text)
        assert leaky_application(tmp_path)

    @pytest.mark.parametrize("text", [
        "from typing import Protocol\n",
        "from transcriber.domain.models import Transcript\n",
        "from ..domain.errors import ContractError\n",
        "from .ports import Recogniser\n",
        "import numpy as np\n",
    ])
    def test_allowed_in_application(
        self, tmp_path: Path, text: str
    ) -> None:
        write(tmp_path, "transcriber/application/__init__.py", "")
        write(tmp_path, "transcriber/application/x.py", text)
        assert leaky_application(tmp_path) == []
