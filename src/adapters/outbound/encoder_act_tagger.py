"""Approach D of the dialogue-act tagger: a fine-tuned encoder (#82).

A multilingual encoder (XLM-R base) with a 13-way classification head
reads the context and the sentence
(:func:`~src.domain.skills.act_tagging.encoder_texts`) and is trained on
English meeting corpora mapped to our categories with the
**compatible-set loss** ``−ln Σ_{c ∈ set} p(c)``: a sentence MRDA calls a
statement teaches "informacion or opinion" without forcing either. The
logits are divided by a temperature fitted on a held-out split
(:func:`~src.domain.skills.act_calibration.fit_temperature`).

The weights and a small ``act_tagger.json`` (temperature, base model,
training figures) are written to a directory under the git-ignored
``data/models/``. Training on the workstation's shared GPU waits for
free memory (bounded) and never touches other processes; word
embeddings may be frozen and the optimiser kept in 8 bits to fit.
"""

import importlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.domain.errors import ModelError
from src.domain.skills.act_calibration import (
    Array,
    as_probabilities,
    fit_temperature,
    set_masks,
    softmax,
)
from src.domain.skills.act_tagging import (
    ActInput,
    Probabilities,
    Reference,
    encoder_texts,
)
from src.domain.skills.dialogue_acts import CATEGORIES

LOG = logging.getLogger(__name__)

CARD = "act_tagger.json"
"""File beside the weights with the temperature and training figures."""


class EncoderCard(BaseModel):
    """What was trained and how it is calibrated."""

    version: str = Field(min_length=1)
    base_model: str
    categories: List[str]
    max_length: int = Field(ge=16)
    temperature: float = Field(gt=0)
    training: Dict[str, Any] = {}


class TrainConfig(BaseModel):
    """Settings of one fine-tuning run (see the training script)."""

    base_model: str = "FacebookAI/xlm-roberta-base"
    max_length: int = Field(default=128, ge=16)
    batch_size: int = Field(default=16, ge=1)
    accumulation: int = Field(default=2, ge=1)
    epochs: float = Field(default=1.0, gt=0)
    learning_rate: float = Field(default=3e-5, gt=0)
    warmup: float = Field(default=0.06, ge=0, lt=1)
    freeze_embeddings: bool = True
    optimizer_8bit: bool = True
    gradient_checkpointing: bool = True
    frozen_bfloat16: bool = True
    max_ooms: int = Field(default=50, ge=0)
    seed: int = 13
    device: str = "cuda"
    need_gb: float = Field(default=1.7, ge=0)
    wait_seconds: float = Field(default=3600.0, ge=0)
    version: str = "xlmr-base-ami-mrda"


def free_gpu_gb() -> Optional[float]:
    """Return the free memory of GPU 0 in GB (``None`` without CUDA)."""
    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info(0)
    return float(free) / 1e9


def wait_for_gpu(
    need_gb: float, wait_seconds: float,
    free: Callable[[], Optional[float]] = free_gpu_gb,
    sleep: Callable[[float], None] = time.sleep,
    poll: float = 30.0,
) -> float:
    """Wait until ``need_gb`` are free on the GPU; return the free GB.

    Raises :class:`~src.domain.errors.ModelError` when there is no GPU
    or the memory is still short after ``wait_seconds``. Other
    processes are never touched.
    """
    waited = 0.0
    while True:
        now = free()
        if now is None:
            raise ModelError("No CUDA GPU for the encoder.")
        if now >= need_gb:
            return now
        if waited >= wait_seconds:
            raise ModelError(
                f"Only {now:.1f} GB free on the GPU after {waited:.0f} s; "
                f"{need_gb:.1f} GB needed.")
        LOG.info("GPU: %.1f GB free, %.1f GB needed; waiting", now, need_gb)
        sleep(poll)
        waited += poll


def _batches(n: int, size: int) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` of consecutive batches covering ``n``."""
    return [(i, min(i + size, n)) for i in range(0, n, size)]


class EncoderActTagger:
    """The fine-tuned encoder behind the ``ActTagger`` port."""

    def __init__(self, directory: str, device: str = "cpu",
                 batch_size: int = 64,
                 temperature: Optional[float] = None):
        """Load the card from ``directory``; the weights load lazily.

        ``temperature`` replaces the one fitted at training time (the
        default tagger uses one fitted on the Spanish set).
        """
        self.directory = Path(directory)
        self.device = device
        self.batch_size = batch_size
        path = self.directory / CARD
        if not path.is_file():
            raise ModelError(
                f"No fine-tuned act tagger in {directory} (missing {CARD}); "
                "train it with experiments/skills/act_tagger_train.py.")
        self.card = EncoderCard.model_validate_json(
            path.read_text(encoding="utf-8"))
        if tuple(self.card.categories) != CATEGORIES:
            raise ModelError(f"{path}: categories differ from ours.")
        self._model: Any = None
        self._tokenizer: Any = None
        self.temperature = temperature or self.card.temperature

    @property
    def name(self) -> str:
        """Return ``encoder``."""
        return "encoder"

    @property
    def version(self) -> str:
        """Return the trained model's version."""
        return self.card.version

    def _load(self) -> None:
        """Load tokenizer and model on first use."""
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.directory))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.directory))
            self._model = model.to(self.device).eval()

    def logits(self, items: Sequence[ActInput]) -> Array:
        """Return the raw logits (sentence × category)."""
        self._load()
        out: List[Array] = []
        with torch.inference_mode():
            for a, b in _batches(len(items), self.batch_size):
                batch = encode(self._tokenizer, items[a:b],
                               self.card.max_length).to(self.device)
                with torch.autocast(device_type=self.device.split(":")[0],
                                    enabled=self.device.startswith("cuda")):
                    logits = self._model(**batch).logits
                out.append(logits.float().cpu().numpy())
        if not out:
            return np.zeros((0, len(CATEGORIES)))
        return np.concatenate(out).astype(np.float64)

    def predict(self, items: Sequence[ActInput]) -> List[Probabilities]:
        """Return the temperature-scaled distribution of each item."""
        return as_probabilities(softmax(self.logits(items),
                                        self.temperature))

    def release(self) -> None:
        """Free the model's memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode(tokenizer: Any, items: Sequence[ActInput],
           max_length: int) -> Any:
    """Tokenise context/sentence pairs (context cut first)."""
    pairs = [encoder_texts(item) for item in items]
    return tokenizer([c for c, _ in pairs], [t for _, t in pairs],
                     truncation="longest_first", max_length=max_length,
                     padding=True, return_tensors="pt")


def set_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean ``−ln Σ_{c ∈ set} p(c)`` of a batch."""
    log_p = torch.log_softmax(logits.float(), dim=-1)
    masked = log_p.masked_fill(mask <= 0, float("-inf"))
    return -torch.logsumexp(masked, dim=-1).mean()


class TrainReport(BaseModel):
    """Figures of one training run (written into the card)."""

    sentences: int
    steps: int
    seconds: float
    peak_vram_gb: Optional[float]
    free_gb_at_start: Optional[float]
    oom_batches_dropped: int = 0
    dev_sentences: int
    dev_log_loss_raw: float
    dev_log_loss_calibrated: float
    temperature: float
    config: Dict[str, Any]


def _optimizer(model: Any, config: TrainConfig) -> Any:
    """Return AdamW (8-bit when asked and available)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer_8bit and config.device.startswith("cuda"):
        try:
            optim: Any = importlib.import_module("bitsandbytes.optim")
            return optim.AdamW8bit(params, lr=config.learning_rate,
                                   weight_decay=0.01)
        except ImportError:
            LOG.warning("bitsandbytes missing: 32-bit AdamW")
    return torch.optim.AdamW(params, lr=config.learning_rate,
                             weight_decay=0.01)


def train_encoder(
    train: Sequence[Tuple[ActInput, Reference]],
    dev: Sequence[Tuple[ActInput, Reference]],
    output: str,
    config: TrainConfig,
    log_every: int = 200,
    init: str = "",
) -> TrainReport:
    """Fine-tune the encoder, fit its temperature on ``dev``, save both.

    ``init`` starts from an already fine-tuned directory instead of the
    base model (adding Spanish folds to the English model).
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    free_at_start = None
    if config.device.startswith("cuda"):
        free_at_start = wait_for_gpu(config.need_gb, config.wait_seconds)
        torch.cuda.reset_peak_memory_stats()
    source = init or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(CATEGORIES),
        id2label=dict(enumerate(CATEGORIES)),
        label2id={c: i for i, c in enumerate(CATEGORIES)})
    if config.freeze_embeddings:
        words = model.base_model.embeddings.word_embeddings
        words.weight.requires_grad_(False)
        if config.frozen_bfloat16:
            # 250k × 768 frozen vectors: half the memory in bf16.
            words.to(torch.bfloat16)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(config.device).train()
    optimizer = _optimizer(model, config)
    order = list(range(len(train)))
    per_epoch = math.ceil(len(train) / config.batch_size)
    total_batches = max(1, int(per_epoch * config.epochs))
    steps = max(1, total_batches // config.accumulation)
    warm = int(steps * config.warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(warm, 1),
                                 max(0.0, (steps - s) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=config.device.startswith("cuda"))
    start = time.perf_counter()
    seen = 0
    ooms = 0
    running = 0.0
    batch_no = 0
    while batch_no < total_batches:
        random.shuffle(order)
        for a, b in _batches(len(order), config.batch_size):
            if batch_no >= total_batches:
                break
            picked = [train[i] for i in order[a:b]]
            batch = encode(tokenizer, [x for x, _ in picked],
                           config.max_length).to(config.device)
            mask = torch.tensor(set_masks([r for _, r in picked]),
                                device=config.device)
            try:
                with torch.autocast(
                        device_type=config.device.split(":")[0],
                        enabled=config.device.startswith("cuda")):
                    logits = model(**batch).logits
                loss = set_loss(logits, mask) / config.accumulation
                scaled: Any = scaler.scale(loss)
                scaled.backward()
            except torch.OutOfMemoryError:
                # Another process took the memory: drop this batch and
                # the partial accumulation, wait, go on.
                ooms += 1
                if ooms > config.max_ooms:
                    raise
                del batch, mask
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                LOG.warning("out of memory (%d); waiting", ooms)
                wait_for_gpu(0.5, config.wait_seconds)
                time.sleep(15)
                continue
            running += float(loss.detach()) * config.accumulation
            batch_no += 1
            seen += len(picked)
            if batch_no % config.accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if batch_no % log_every == 0:
                LOG.info("batch %d/%d loss %.4f (%.0f s)", batch_no,
                         total_batches, running / log_every,
                         time.perf_counter() - start)
                running = 0.0
    seconds = time.perf_counter() - start
    peak = (torch.cuda.max_memory_allocated() / 1e9
            if config.device.startswith("cuda") else None)
    model.eval()
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    card = EncoderCard(version=config.version, base_model=config.base_model,
                       categories=list(CATEGORIES),
                       max_length=config.max_length, temperature=1.0)
    (target / CARD).write_text(card.model_dump_json(indent=1),
                               encoding="utf-8")
    del optimizer
    tagger = EncoderActTagger(str(target), config.device)
    tagger._model, tagger._tokenizer = model, tokenizer
    dev_items = [x for x, _ in dev]
    dev_refs = [r for _, r in dev]
    logits_dev = tagger.logits(dev_items)
    temperature = fit_temperature(logits_dev, dev_refs)
    dev_mask = set_masks(dev_refs)

    def nll(t: float) -> float:
        """Return the dev log-loss at temperature ``t``."""
        p = softmax(logits_dev, t)
        return float(
            -np.log(np.clip((p * dev_mask).sum(1), 1e-15, 1)).mean())

    report = TrainReport(
        sentences=seen, steps=steps, seconds=round(seconds, 1),
        oom_batches_dropped=ooms,
        peak_vram_gb=None if peak is None else round(peak, 2),
        free_gb_at_start=(None if free_at_start is None
                          else round(free_at_start, 2)),
        dev_sentences=len(dev), dev_log_loss_raw=round(nll(1.0), 4),
        dev_log_loss_calibrated=round(nll(temperature), 4),
        temperature=round(temperature, 4),
        config=config.model_dump())
    card = card.model_copy(update={"temperature": temperature,
                                   "training": report.model_dump()})
    (target / CARD).write_text(card.model_dump_json(indent=1),
                               encoding="utf-8")
    tagger.release()
    del model
    return report


def write_json(path: str, data: Any) -> None:
    """Write ``data`` as indented UTF-8 JSON, creating the directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
