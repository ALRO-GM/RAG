"""The workstation: the transcriber publishes to a RAG process on loopback.

``python -m src serve --token_file`` runs as its own process (as ``make
rag-serve`` starts it) on a free loopback port, with the token in a
private file. The transcriber, with fake models, publishes a finished
session to it with the token read from the same file, reads it back over
HTTP, deletes the session and finds the RAG's copy gone. The other tests
check that the tunnel forwards only the transcriber's port and that the
RAG's port and bind are not the tunnel's.
"""

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, NamedTuple

import httpx
import pytest

from src.adapters.outbound.token_file import make_token_file, read_token_file
from tests.integration.test_cli import run_cli
from tests.integration.transcriber.test_pipeline import (
    enrol_ana,
    record,
    transcribe,
)
from tests.samples import write_corpus
from transcriber.adapters.outbound.chisel_tunnel import client_command
from transcriber.bootstrap import build_container
from transcriber.config import (
    DEFAULT_PORT,
    LOCAL_RAG_URL,
    HttpPublishSettings,
    TranscriberSettings,
    TunnelSettings,
)

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"
WAIT = 60.0


class Rag(NamedTuple):
    """A running RAG process: its URL, folders and token file."""

    url: str
    transcripts: Path
    token_file: Path


def free_port() -> int:
    """Return a loopback port nothing listens on now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def rag(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rag]:
    """``python -m src serve --token_file`` on loopback, as a process."""
    project = tmp_path_factory.mktemp("rag")
    write_corpus(project)
    indexed = run_cli(["index", "--semantic=False"], project)
    assert indexed.returncode == 0, indexed.stderr
    token_file = project / "config" / "rag-token"
    make_token_file(token_file)
    port = free_port()
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"})
    env.pop("RAG_TRANSCRIPTS_TOKEN", None)
    log = project / "serve.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "src", "--mode", "bm25", "serve",
             "--host", "127.0.0.1", "--port", str(port),
             "--token_file", str(token_file)],
            cwd=project, env=env, stdout=output, stderr=subprocess.STDOUT,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + WAIT
        while True:
            assert process.poll() is None, log.read_text()
            try:
                if httpx.get(f"{url}/health", timeout=2.0).is_success:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, log.read_text()
            time.sleep(0.2)
        yield Rag(url, project / "data" / "transcripts", token_file)
    finally:
        process.terminate()
        process.wait(WAIT)
    assert read_token_file(token_file) not in log.read_text()


def test_publish_read_back_then_delete_with_the_session(
    rag: Rag, tmp_path: Path,
) -> None:
    settings = TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake", publisher="http",
        publish_http=HttpPublishSettings(
            url=rag.url, token_file=rag.token_file, retries=0,
        ),
    )
    container = build_container(settings, environ={})
    auth: Dict[str, str] = {
        "Authorization": f"Bearer {read_token_file(rag.token_file)}"
    }
    try:
        enrol_ana(container)
        record(container, "equipo-a")
        transcribe(container, "equipo-a")
        document = container.publish("equipo-a")
        container.publish("equipo-a")  # a repeat replaces the copy
        read = httpx.get(
            f"{rag.url}/transcripts/equipo-a", headers=auth, timeout=10.0
        )
        assert read.status_code == 200 and read.json() == document
        assert [p.name for p in rag.transcripts.iterdir()] == [
            "equipo-a.transcript.v1.json"
        ]
        container.delete_session("equipo-a")
    finally:
        container.runner.close(5.0)
    gone = httpx.get(
        f"{rag.url}/transcripts/equipo-a", headers=auth, timeout=10.0
    )
    assert gone.status_code == 404
    assert list(rag.transcripts.iterdir()) == []
    assert httpx.get(
        f"{rag.url}/transcripts/equipo-a", timeout=10.0
    ).status_code == 401


def test_deleting_an_unpublished_session_is_quiet(
    rag: Rag, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    settings = TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake", publisher="http",
        publish_http=HttpPublishSettings(
            url=rag.url, token_file=rag.token_file, retries=0,
        ),
    )
    container = build_container(settings, environ={})
    try:
        container.open_session("borrador", "es")
        with caplog.at_level("WARNING"):
            container.delete_session("borrador")
    finally:
        container.runner.close(5.0)
    assert container.sessions.get("borrador") is None
    assert caplog.records == []


def test_a_rag_that_is_down_does_not_stop_a_deletion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    token_file = tmp_path / "rag-token"
    make_token_file(token_file)
    settings = TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake", publisher="http",
        publish_http=HttpPublishSettings(
            url=f"http://127.0.0.1:{free_port()}", token_file=token_file,
            timeout=2.0,
        ),
    )
    container = build_container(settings, environ={})
    try:
        container.open_session("s1", "es")
        with caplog.at_level("WARNING"):
            container.delete_session("s1")
    finally:
        container.runner.close(5.0)
    assert container.sessions.get("s1") is None
    [warning] = caplog.records
    message = warning.getMessage()
    assert message.startswith("Session s1 was deleted, but not the RAG's")
    assert read_token_file(token_file) not in message


def test_the_directory_publisher_keeps_its_files(tmp_path: Path) -> None:
    container = build_container(TranscriberSettings(
        data_dir=tmp_path / "data", runtime="fake",
    ))
    try:
        enrol_ana(container)
        record(container, "equipo-a")
        transcribe(container, "equipo-a")
        container.publish("equipo-a")
        container.delete_session("equipo-a")
    finally:
        container.runner.close(5.0)
    published = tmp_path / "data" / "published"
    assert [p.name for p in published.iterdir()] == [
        "equipo-a.transcript.v1.json"
    ]


# -- the RAG is not behind the tunnel --------------------------------------

def make_variable(name: str) -> str:
    """Return the default of ``name ?= value`` in the Makefile."""
    found = re.search(
        rf"^{name}\s*\?=\s*(\S+)\s*$", MAKEFILE.read_text(), re.MULTILINE
    )
    assert found, name
    return found.group(1)


def recipe(target: str) -> str:
    """Return the recipe of ``target`` in the Makefile, lines joined."""
    text = MAKEFILE.read_text()
    found = re.search(
        rf"^{target}:\n((?:\t.*\n)+)", text, re.MULTILINE
    )
    assert found, target
    return found.group(1).replace("\\\n", " ")


def test_the_rag_port_is_not_the_tunnels() -> None:
    rag_port = int(make_variable("RAG_PORT"))
    tunnel = TunnelSettings()
    assert rag_port not in (DEFAULT_PORT, tunnel.remote_port)
    assert LOCAL_RAG_URL == f"http://127.0.0.1:{rag_port}"
    forwarded = client_command(tunnel, DEFAULT_PORT)[-1]
    assert forwarded == (
        f"R:0.0.0.0:{tunnel.remote_port}:127.0.0.1:{DEFAULT_PORT}"
    )
    assert not forwarded.endswith(f":{rag_port}")


def test_rag_serve_binds_loopback_with_the_token_file() -> None:
    serve = recipe("rag-serve")
    assert "--host 127.0.0.1" in serve
    assert "--port $(RAG_PORT)" in serve
    assert '--token_file "$(RAG_TOKEN_FILE)"' in serve
    assert "0.0.0.0" not in serve
    assert make_variable("RAG_TOKEN_FILE").startswith("$(HOME)/")
    assert "src.adapters.outbound.token_file" in recipe("rag-token")
