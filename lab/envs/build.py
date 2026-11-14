"""Build one model-lab environment outside the repository.

Usage::

    python3 lab/envs/build.py FAMILY [--compile] [--envs-dir DIR]
    python3 lab/envs/build.py --all [--compile]
    python3 lab/envs/build.py --selftest [FAMILY ...]

Standard library only, so any Python 3 runs it. A family is a
directory ``lab/envs/<family>/`` with:

``requirements.in``
    the top-level packages, pinned.
``requirements.txt``
    every package the family needs *beyond the project's environment*,
    exactly pinned (written by ``--compile``, committed).
``requirements.lock``
    ``uv pip freeze`` of the built environment (written by every build,
    committed).
``overrides.txt`` (optional)
    versions the family needs that differ from the project's.
``build.json`` (optional)
    ``{"unresolved": [...], "no_build_isolation": [...],
    "build_constraints": [...],
    "post_install": [[argv...]]}``: ``unresolved`` are exact pins added
    to ``requirements.txt`` without resolving their dependencies (for
    packages whose declared pins conflict with the project's although
    the code runs with its versions; their real needs are listed in
    ``requirements.in``); ``{python}`` in a ``post_install`` command is
    the environment's interpreter, ``{models}`` the directory of
    converted models (``~/.local/share/ragputin/lab-models``).

The environment lives in ``~/.local/share/ragputin/lab-envs/<family>``.
It is *layered* on the project's virtual environment: same interpreter
(Python 3.10), and a ``zz-project.pth`` file that appends the project's
``site-packages`` to ``sys.path``. The family's own packages come first
and shadow the project's; the project's environment is only read, never
changed. This reuses the project's torch, CUDA libraries and numpy
(about 6 GB) instead of installing them again.

uv does not follow ``.pth`` files, so it would install torch again if
asked to resolve the family on its own. ``--compile`` therefore
resolves ``requirements.in`` with the project's frozen versions as
overrides (``uv pip compile --override``) and drops from the result
every package the project already has (and torch's CUDA wheels); the
rest is installed with ``--no-deps``.

After installing, the family's runner ``--selftest`` runs; on success
its JSON is written to ``<env>/.ragputin-lab-ready``, the marker the
service reads to call the family installed.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
ENVS_SPEC = REPO / "lab" / "envs"
RUNNERS = REPO / "lab" / "runners"
PROJECT_PYTHON = REPO / ".venv" / "bin" / "python"
DEFAULT_ENVS = Path.home() / ".local" / "share" / "ragputin" / "lab-envs"
MODELS = Path.home() / ".local" / "share" / "ragputin" / "lab-models"
"""Converted or fetched model files that are not in the hub's cache."""
READY = ".ragputin-lab-ready"
PTH = "zz-project.pth"
ALWAYS_DROP = re.compile(r"^(nvidia-.*|triton|torch|cuda-toolkit)$")
"""Packages that come with the project's torch build, never installed."""


def normal(name: str) -> str:
    """Return a distribution name in its normalised form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def run(argv: List[str]) -> str:
    """Run ``argv`` (no shell), echo it, return its standard output."""
    print("+ " + " ".join(argv), flush=True)
    done = subprocess.run(
        argv, check=True, text=True, stdout=subprocess.PIPE
    )
    return str(done.stdout)


def project_pins() -> Dict[str, str]:
    """Return ``name -> version`` of the project's environment."""
    text = run(["uv", "pip", "freeze", "--python", str(PROJECT_PYTHON)])
    pins: Dict[str, str] = {}
    for line in text.splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            pins[normal(name)] = version.split("+", 1)[0].strip()
    return pins


def families() -> List[str]:
    """Return every family that has a ``requirements.in``."""
    return sorted(
        p.parent.name for p in ENVS_SPEC.glob("*/requirements.in")
    )


def options(family: str) -> Dict[str, object]:
    """Return the family's ``build.json`` (empty when absent)."""
    path = ENVS_SPEC / family / "build.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def _list(opts: Dict[str, object], key: str) -> List[object]:
    """Return the list ``opts[key]`` (empty when absent)."""
    value = opts.get(key, [])
    assert isinstance(value, list)
    return value


def build_constraints(family: str, directory: Path) -> None:
    """Point uv at the family's ``build_constraints`` (if any).

    Old ``setup.py`` files need an old setuptools (``pkg_resources``)
    to build; the constraints apply only to the build environments.
    """
    wanted = [str(x) for x in _list(options(family), "build_constraints")]
    if wanted:
        path = directory / "build-constraints.txt"
        path.write_text("".join(f"{x}\n" for x in wanted))
        os.environ["UV_BUILD_CONSTRAINT"] = str(path)
    else:
        os.environ.pop("UV_BUILD_CONSTRAINT", None)


def compile_family(family: str) -> None:
    """Write ``requirements.txt`` of ``family`` (see the module doc)."""
    spec = ENVS_SPEC / family
    pins = project_pins()
    extra = spec / "overrides.txt"
    own: Dict[str, str] = {}
    if extra.is_file():
        for line in extra.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                own[normal(name)] = version.strip()
    with tempfile.TemporaryDirectory() as tmp:
        build_constraints(family, Path(tmp))
        override = Path(tmp) / "overrides.txt"
        merged = {**pins, **own}
        override.write_text("".join(
            f"{name}=={version}\n" for name, version in sorted(merged.items())
        ))
        out = Path(tmp) / "compiled.txt"
        run([
            "uv", "pip", "compile", str(spec / "requirements.in"),
            "--python-version", "3.10", "--override", str(override),
            "--no-header", "--no-annotate", "-o", str(out),
        ])
        kept = []
        for line in out.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = re.split(r"[=<>@ \[]", stripped, 1)[0]
            key = normal(name)
            if ALWAYS_DROP.match(key) or (key in pins and key not in own):
                continue
            kept.append(stripped)
    kept.extend(str(x) for x in _list(options(family), "unresolved"))
    header = (
        f"# {family}: packages beyond the project's environment.\n"
        "# Written by `python3 lab/envs/build.py --compile`; installed\n"
        "# with --no-deps into an env layered on the project's.\n"
    )
    (spec / "requirements.txt").write_text(
        header + "".join(f"{line}\n" for line in kept)
    )


def env_path(envs: Path, family: str) -> Path:
    """Return the directory of the family's environment."""
    return envs / family


def site_packages(python: Path) -> Path:
    """Return the ``site-packages`` directory of an interpreter."""
    text = subprocess.run(
        [str(python), "-I", "-c",
         "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    return Path(text.strip())


def build(family: str, envs: Path, recompile: bool) -> bool:
    """Create or update the family's environment; tell if it is ready."""
    spec = ENVS_SPEC / family
    if not (spec / "requirements.in").is_file():
        print(f"No family {family!r} in {ENVS_SPEC}.", file=sys.stderr)
        return False
    if recompile or not (spec / "requirements.txt").is_file():
        compile_family(family)
    env = env_path(envs, family)
    python = env / "bin" / "python"
    if not python.exists():
        envs.mkdir(parents=True, exist_ok=True)
        run(["uv", "venv", "--python", str(PROJECT_PYTHON), str(env)])
    (env / READY).unlink(missing_ok=True)
    project_site = site_packages(PROJECT_PYTHON)
    (site_packages(python) / PTH).write_text(f"{project_site}\n")
    opts = options(family)
    isolated = [str(x) for x in _list(opts, "no_build_isolation")]
    requirements = spec / "requirements.txt"
    lines = [
        line.strip() for line in requirements.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    normal_lines = [
        x for x in lines if normal(re.split(r"[=<>@ \[]", x, 1)[0])
        not in {normal(i) for i in isolated}
    ]
    with tempfile.TemporaryDirectory() as tmp:
        build_constraints(family, Path(tmp))
        plain = Path(tmp) / "plain.txt"
        plain.write_text("".join(f"{x}\n" for x in normal_lines))
        run(["uv", "pip", "install", "--python", str(python), "--no-deps",
             "-r", str(plain)])
    late = [x for x in lines if x not in normal_lines]
    if late:
        run(["uv", "pip", "install", "--python", str(python), "--no-deps",
             "--no-build-isolation", *late])
    for argv in _list(opts, "post_install"):
        assert isinstance(argv, list)
        run([
            str(a).replace("{python}", str(python)).replace(
                "{models}", str(MODELS))
            for a in argv
        ])
    lock = run(["uv", "pip", "freeze", "--python", str(python)])
    (spec / "requirements.lock").write_text(
        f"# {family}: `uv pip freeze` of the built environment (layered on\n"
        "# the project's; the project's packages are not listed).\n" + lock
    )
    return selftest(family, envs)


def selftest(family: str, envs: Path) -> bool:
    """Run the family's runner ``--selftest``; write the ready marker."""
    env = env_path(envs, family)
    python = env / "bin" / "python"
    runner = RUNNERS / f"{family}.py"
    if not python.exists() or not runner.is_file():
        print(f"{family}: not built", flush=True)
        return False
    done = subprocess.run(
        [str(python), str(runner), "--selftest"], text=True,
        stdout=subprocess.PIPE, env=_runner_env(),
    )
    last = (done.stdout.strip().splitlines() or [""])[-1]
    print(f"{family}: {last}", flush=True)
    try:
        ok = done.returncode == 0 and json.loads(last).get("ok") is True
    except (json.JSONDecodeError, AttributeError):
        ok = False
    if not ok:
        (env / READY).unlink(missing_ok=True)
        return False
    (env / READY).write_text(last + "\n")
    return True


def _runner_env() -> Dict[str, str]:
    """Return a small environment for the self-tests (offline)."""
    keep = ("PATH", "HOME", "LANG", "HF_HOME", "XDG_CACHE_HOME")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["HF_HUB_OFFLINE"] = "1"
    return env


def main(argv: Optional[List[str]] = None) -> int:
    """Parse the arguments and build, rebuild or test environments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("family", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--envs-dir", type=Path, default=DEFAULT_ENVS)
    args = parser.parse_args(argv)
    if shutil.which("uv") is None:
        print("uv is needed (https://docs.astral.sh/uv/).", file=sys.stderr)
        return 2
    chosen = families() if args.all or not args.family else args.family
    failed = []
    for family in chosen:
        try:
            ok = (
                selftest(family, args.envs_dir) if args.selftest
                else build(family, args.envs_dir, args.compile)
            )
        except subprocess.CalledProcessError as exc:
            print(f"{family}: failed: {exc}", file=sys.stderr)
            ok = False
        if not ok:
            failed.append(family)
    if failed:
        print("Not ready: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("Ready: " + ", ".join(chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
