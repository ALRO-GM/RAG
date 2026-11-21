"""The product config model and the four shipped YAMLs."""

from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from orchestrator.adapters.outbound.files import (
    ConfigError,
    check_dir,
    load_product,
    load_transcript,
)
from orchestrator.domain.config import PRODUCTS, ProductConfig
from tests.unit.orchestrator.fixtures import PRODUCTS as DIR
from tests.unit.orchestrator.fixtures import product


def live() -> Dict[str, Any]:
    """A valid live-support config as a dict."""
    return product().model_dump(mode="json")


class TestShippedFiles:
    def test_all_four_products_validate(self) -> None:
        names, errors = check_dir(DIR)
        assert errors == []
        assert sorted(names) == sorted(PRODUCTS)

    def test_on_is_not_read_as_a_boolean(self) -> None:
        trigger = product("live_assist.interview").trigger
        assert trigger is not None and trigger.on == "final"

    def test_interview_keeps_expressed_signals_off(self) -> None:
        interview = product("live_assist.interview")
        assert interview.reasoner is not None
        assert interview.reasoner.expressed_signals is False
        support = product("live_assist.support")
        assert support.reasoner is not None
        assert support.reasoner.expressed_signals is True

    def test_live_products_store_no_audio(self) -> None:
        for name in PRODUCTS[1:]:
            assert product(name).privacy.store_audio is False

    def test_post_call_is_not_live(self) -> None:
        assert not product("post_call").is_live
        assert product("voicebot").is_live


class TestRules:
    def test_unknown_key_is_rejected(self) -> None:
        data = live()
        data["surprise"] = 1
        with pytest.raises(ValidationError):
            ProductConfig.model_validate(data)

    def test_signals_outside_support_are_rejected(self) -> None:
        data = product("live_assist.interview").model_dump(mode="json")
        data["reasoner"]["expressed_signals"] = True
        with pytest.raises(ValidationError, match="expressed signals"):
            ProductConfig.model_validate(data)

    def test_two_calls_in_flight_are_rejected(self) -> None:
        data = live()
        data["reasoner"]["max_inflight"] = 2
        with pytest.raises(ValidationError, match="one call in flight"):
            ProductConfig.model_validate(data)

    def test_live_needs_a_trigger(self) -> None:
        data = live()
        data["trigger"] = None
        with pytest.raises(ValidationError, match="needs vad"):
            ProductConfig.model_validate(data)

    def test_live_needs_a_stream_asr(self) -> None:
        data = live()
        data["asr"]["kind"] = "batch"
        with pytest.raises(ValidationError, match="streaming"):
            ProductConfig.model_validate(data)

    def test_live_keeps_no_audio(self) -> None:
        data = live()
        data["privacy"]["store_audio"] = True
        with pytest.raises(ValidationError, match="no audio"):
            ProductConfig.model_validate(data)

    def test_post_call_has_no_trigger(self) -> None:
        data = product("post_call").model_dump(mode="json")
        data["trigger"] = {"on": "final", "speaker_roles": ["agent"]}
        with pytest.raises(ValidationError, match="no live trigger"):
            ProductConfig.model_validate(data)

    def test_role_must_match_product(self) -> None:
        data = live()
        data["reasoner"]["role"] = "conversational"
        with pytest.raises(ValidationError, match="suggester role"):
            ProductConfig.model_validate(data)

    def test_live_names_a_template(self) -> None:
        data = live()
        data["reasoner"]["template"] = None
        with pytest.raises(ValidationError, match="template"):
            ProductConfig.model_validate(data)

    def test_only_the_voicebot_speaks(self) -> None:
        data = live()
        data["tts"] = {"kind": "piper", "voice": "es_ES"}
        with pytest.raises(ValidationError, match="only the voicebot"):
            ProductConfig.model_validate(data)

    def test_pipewire_needs_channels(self) -> None:
        data = live()
        data["source"]["channels"] = {}
        with pytest.raises(ValidationError, match="channels"):
            ProductConfig.model_validate(data)


class TestFiles:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            load_product(tmp_path / "nope.yaml")

    def test_bad_yaml_and_bad_schema(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("a: [", encoding="utf-8")
        (tmp_path / "b.yaml").write_text("product: x\n", encoding="utf-8")
        for name in ("a.yaml", "b.yaml"):
            with pytest.raises(ConfigError):
                load_product(tmp_path / name)
        names, errors = check_dir(tmp_path)
        assert names == [] and len(errors) == 2

    def test_name_must_match_the_file(self, tmp_path: Path) -> None:
        text = (DIR / "post_call.yaml").read_text(encoding="utf-8")
        (tmp_path / "other.yaml").write_text(text, encoding="utf-8")
        _, errors = check_dir(tmp_path)
        assert "differs" in errors[0]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert check_dir(tmp_path)[1]

    def test_bad_transcript(self, tmp_path: Path) -> None:
        bad = tmp_path / "t.json"
        bad.write_text("{}", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_transcript(bad)
        with pytest.raises(ConfigError):
            load_transcript(tmp_path / "missing.json")


def test_shipped_products_reference_existing_domains() -> None:
    """Every knowledge domain a shipped product names has a manifest."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    missing = []
    for path in sorted((root / "config" / "products").glob("*.yaml")):
        knowledge = yaml.safe_load(path.read_text()).get("knowledge") or {}
        for domain in knowledge.get("domains", []):
            if not (root / "knowledge" / "domains" / domain
                    / "manifest.yaml").is_file():
                missing.append(f"{path.name}: {domain}")
    assert not missing
