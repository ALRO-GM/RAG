"""Publishing to the RAG of this machine: ``TRANSCRIBER_LOCAL_RAG``.

With the variable on, no ``publisher`` in the config file and the token
file present, :func:`~transcriber.config.load_settings` points the HTTP
publisher at ``http://127.0.0.1:8000`` with the token read from that
file; otherwise nothing changes.
"""

import json
from pathlib import Path
from typing import Dict

import pytest

from transcriber.adapters.inbound.cli import local_rag_note
from transcriber.adapters.outbound.fs_transcript_publisher import (
    DirectoryTranscriptPublisher,
)
from transcriber.adapters.outbound.http_transcript_publisher import (
    HttpTranscriptPublisher,
)
from transcriber.bootstrap import build_publisher
from transcriber.config import (
    DEFAULT_RAG_TOKEN_FILE,
    LOCAL_RAG_URL,
    TranscriberSettings,
    load_settings,
)


def environment(tmp_path: Path, **fields: object) -> Dict[str, str]:
    """Variables that turn the local RAG on with a config in ``tmp_path``."""
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({
        "rag_token_file": str(tmp_path / "rag-token"), **fields,
    }))
    return {"TRANSCRIBER_CONFIG": str(config), "TRANSCRIBER_LOCAL_RAG": "1"}


def make_token(tmp_path: Path) -> Path:
    """Write a private token file as ``make rag-token`` does."""
    path = tmp_path / "rag-token"
    path.write_text("t" * 43 + "\n")
    path.chmod(0o600)
    return path


def test_the_default_token_file_is_outside_the_repository() -> None:
    assert str(DEFAULT_RAG_TOKEN_FILE).startswith("~/")
    assert LOCAL_RAG_URL == "http://127.0.0.1:8000"
    assert TranscriberSettings().local_rag is False


def test_the_token_file_points_publishing_at_the_local_rag(
    tmp_path: Path,
) -> None:
    path = make_token(tmp_path)
    settings = load_settings(environment(tmp_path, publish_http={
        "retries": 1,
    }))
    assert settings.publisher == "http"
    assert settings.publish_http.url == LOCAL_RAG_URL
    assert settings.publish_http.token_file == path
    assert settings.publish_http.retries == 1
    assert settings.publish_target() == f"{LOCAL_RAG_URL}/transcripts"
    built = build_publisher(settings, {})
    assert isinstance(built, HttpTranscriptPublisher)
    assert built.token_file == path
    assert local_rag_note(settings) is None


def test_the_local_rag_settings_are_validated(tmp_path: Path) -> None:
    settings = TranscriberSettings(rag_token_file=tmp_path / "t")
    local = settings.with_local_rag()
    assert local.publish_http.token_file == tmp_path / "t"
    assert local.publish_http.url == LOCAL_RAG_URL
    with pytest.raises(ValueError, match="rag_token_file must be"):
        load_settings(environment(tmp_path, rag_token_file="relative/t"))


def test_without_the_token_file_the_directory_stays(tmp_path: Path) -> None:
    settings = load_settings(environment(tmp_path))
    assert settings.local_rag and settings.publisher == "directory"
    assert isinstance(build_publisher(settings), DirectoryTranscriptPublisher)
    note = local_rag_note(settings)
    assert note is not None
    assert "rag-token does not exist (make rag-token)" in note
    assert str(settings.published()) in note


@pytest.mark.parametrize("fields", [
    {"publisher": "directory"},
    {"publisher": "http", "publish_http": {"url": "https://rag.example"}},
])
def test_an_explicit_publisher_wins(
    tmp_path: Path, fields: Dict[str, object],
) -> None:
    make_token(tmp_path)
    settings = load_settings(environment(tmp_path, **fields))
    assert settings.publisher == fields["publisher"]
    assert settings.publish_http.url != LOCAL_RAG_URL
    note = local_rag_note(settings)
    assert note is not None and "the config file chooses" in note


def test_off_by_default_even_with_the_token_file(tmp_path: Path) -> None:
    make_token(tmp_path)
    variables = environment(tmp_path)
    variables["TRANSCRIBER_LOCAL_RAG"] = "0"
    settings = load_settings(variables)
    assert settings.publisher == "directory"
    assert local_rag_note(settings) is None
    del variables["TRANSCRIBER_LOCAL_RAG"]
    assert load_settings(variables).publisher == "directory"
