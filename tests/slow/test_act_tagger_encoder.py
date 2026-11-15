"""The act tagger's encoder with the real XLM-R base weights (#82).

Trains for a handful of steps on the CPU from the cached base model,
then checks what the directory holds and what the tagger answers. It
needs ``FacebookAI/xlm-roberta-base`` in the Hugging Face cache.
"""

from pathlib import Path

import pytest

from src.adapters.outbound.encoder_act_tagger import (
    CARD,
    EncoderActTagger,
    EncoderCard,
    TrainConfig,
    train_encoder,
)
from src.domain.skills.act_tagging import Reference, sentence_inputs
from src.domain.skills.dialogue_acts import CATEGORIES

pytestmark = pytest.mark.slow


def test_train_save_and_tag(tmp_path: Path) -> None:
    items = sentence_inputs([
        ("1", "A", "¿Cuándo llega?"), ("2", "B", "El martes."),
        ("3", "A", "Vale."), ("4", "B", "Gracias.")] * 4)
    labels = [["pregunta"], ["respuesta"], ["escucha"], ["relacion"]] * 4
    pairs = [(i, Reference(categories=c)) for i, c in zip(items, labels)]
    config = TrainConfig(device="cpu", batch_size=4, accumulation=1,
                         epochs=1, max_length=32, optimizer_8bit=False,
                         gradient_checkpointing=False, version="tiny")
    report = train_encoder(pairs, pairs[:8], str(tmp_path), config)
    assert report.steps == 4 and report.peak_vram_gb is None
    assert report.temperature > 0
    card = EncoderCard.model_validate_json(
        (tmp_path / CARD).read_text(encoding="utf-8"))
    assert card.version == "tiny" and card.training["steps"] == 4
    tagger = EncoderActTagger(str(tmp_path), "cpu")
    out = tagger.predict(items[:2])
    assert set(out[0]) == set(CATEGORIES)
    assert sum(out[0].values()) == pytest.approx(1.0)
    tagger.release()
