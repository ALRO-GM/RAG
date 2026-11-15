"""The act tagger's adapters and command without a real model (#82).

The fine-tuned encoder itself is exercised in ``tests/slow``; here the
parts that need no weights: the compatible-set loss, waiting for GPU
memory, refusing a directory without a card, building taggers from a
card in the container, and ``skills acts`` end to end with the rules.
"""

import json
from pathlib import Path
from typing import List, Optional

import pytest
import torch

from src.adapters.outbound.encoder_act_tagger import (
    CARD,
    EncoderActTagger,
    EncoderCard,
    TrainConfig,
    _batches,
    set_loss,
    wait_for_gpu,
    write_json,
)
from src.application.skills.act_tagger import CombinedTagger, TaggerCard
from src.config import SkillsSettings
from src.domain.errors import InvalidInputError, ModelError
from src.domain.skills.act_calibration import fit_outcomes
from src.domain.skills.act_tagging import Reference, Route
from src.domain.skills.dialogue_acts import CATEGORIES
from src.skills_bootstrap import SkillsContainer
from tests.integration.test_skills_cli import assert_error, skills
from tests.skills_samples import ES, write_rubrics, write_transcripts


def write_card(path: Path, encoder: str = "") -> Path:
    """Write a small tagger card with a two-row rule table."""
    table = fit_outcomes(
        ["pregunta:question_mark", "default:statement"],
        [Reference(categories=["pregunta"]),
         Reference(categories=["informacion"])], smoothing=0.5)
    card = TaggerCard(
        name="combined", rules=table, encoder=encoder, temperature=1.5,
        routes=[Route(tagger="rules", categories=["pregunta"])],
        fallback="encoder",
        base_rates={c: 1 / len(CATEGORIES) for c in CATEGORIES},
        margin=0.1, source="test")
    write_json(str(path), card.model_dump())
    return path


def test_set_loss_is_minus_log_of_the_set_mass() -> None:
    logits = torch.zeros((1, len(CATEGORIES)))
    mask = torch.zeros((1, len(CATEGORIES)))
    mask[0, :2] = 1
    expected = -torch.log(torch.tensor(2 / len(CATEGORIES)))
    assert float(set_loss(logits, mask)) == pytest.approx(float(expected))


def test_wait_for_gpu_waits_then_gives_up() -> None:
    readings: List[Optional[float]] = [0.5, 1.0, 3.0]
    slept: List[float] = []

    def free() -> Optional[float]:
        return readings.pop(0)

    assert wait_for_gpu(2.0, 100, free, slept.append, poll=10) == 3.0
    assert slept == [10, 10]
    with pytest.raises(ModelError, match="after 20 s"):
        wait_for_gpu(9.0, 20, lambda: 1.0, lambda s: None, poll=10)
    with pytest.raises(ModelError, match="No CUDA"):
        wait_for_gpu(1.0, 20, lambda: None, lambda s: None)


def test_encoder_needs_its_card(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="act_tagger_train"):
        EncoderActTagger(str(tmp_path))
    card = EncoderCard(version="v", base_model="b",
                       categories=list(reversed(CATEGORIES)),
                       max_length=64, temperature=1.0)
    (tmp_path / CARD).write_text(card.model_dump_json(), encoding="utf-8")
    with pytest.raises(ModelError, match="categories"):
        EncoderActTagger(str(tmp_path))
    good = card.model_copy(update={"categories": list(CATEGORIES)})
    (tmp_path / CARD).write_text(good.model_dump_json(), encoding="utf-8")
    tagger = EncoderActTagger(str(tmp_path), temperature=2.0)
    assert (tagger.name, tagger.version, tagger.temperature) == (
        "encoder", "v", 2.0)
    assert tagger.logits([]).shape == (0, len(CATEGORIES))
    tagger.release()


def test_batches_and_train_config() -> None:
    assert _batches(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert TrainConfig().freeze_embeddings


def test_container_builds_taggers_from_the_card(tmp_path: Path) -> None:
    encoder = tmp_path / "enc"
    encoder.mkdir()
    (encoder / CARD).write_text(EncoderCard(
        version="v1", base_model="b", categories=list(CATEGORIES),
        max_length=64, temperature=1.0).model_dump_json(), encoding="utf-8")
    card = write_card(tmp_path / "card.json", str(encoder))
    app = SkillsContainer(SkillsSettings(act_card=str(card), device="cpu"))
    combined, note = app.act_tagger()
    assert isinstance(combined, CombinedTagger) and note == ""
    assert combined.version == "encoder=v1 rules=rules-1"
    alone, _ = app.act_tagger("encoder")
    assert isinstance(alone, EncoderActTagger) and alone.temperature == 1.5
    assert app.act_tagger("rules")[0].name == "rules"
    with pytest.raises(InvalidInputError, match="Unknown tagger"):
        app.act_tagger("magic")
    missing = write_card(tmp_path / "missing.json", str(tmp_path / "none"))
    app = SkillsContainer(SkillsSettings(act_card=str(missing)))
    rules, note = app.act_tagger()
    assert rules.name == "rules" and "rules answered" in note


def test_acts_command_writes_labels_next_to_the_transcript(
    tmp_path: Path,
) -> None:
    write_rubrics(tmp_path)
    write_transcripts(tmp_path)
    card = write_card(tmp_path / "card.json", str(tmp_path / "none"))
    result = skills(["--act_card", str(card), "acts", "--transcript", ES],
                    tmp_path)
    assert result.returncode == 0, result.stderr
    assert "rules answered instead of the combined tagger" in result.stdout
    written = tmp_path / Path(ES).parent / "es1.acts.json"
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["schema"] == "acts.v1" and data["tagger"]["name"] == "rules"
    assert data["session"] == {"id": "es1"}
    assert data["tagger"]["version"] == "rules-1"
    assert data["tagger"]["margin"] == 0.1
    first = data["utterances"][0]
    assert first["turn"] == 0
    assert {"id", "speaker", "text", "labels", "rest", "by"} <= set(first)
    assert 1 <= len(first["labels"]) <= 3
    assert sum(lab["weight"] for lab in first["labels"]) + first[
        "rest"] == 100
    out = tmp_path / "x.json"
    again = skills(["--act_card", str(card), "acts", "--transcript", ES,
                    "--tagger", "rules", "--output", str(out)], tmp_path)
    assert again.returncode == 0 and out.is_file()
    assert "rules answered" not in again.stdout


def test_acts_command_refuses_bad_options(tmp_path: Path) -> None:
    write_rubrics(tmp_path)
    write_transcripts(tmp_path)
    card = write_card(tmp_path / "card.json")
    assert_error(skills(["--act_card", str(card), "acts", "--transcript",
                         ES, "--tagger", "magic"], tmp_path),
                 "Unknown tagger")
    assert_error(skills(["--act_card", str(card), "acts", "--transcript",
                         ES, "--context", "-1"], tmp_path), "--context")
