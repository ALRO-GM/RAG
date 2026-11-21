"""Apply the «Ajustes» page's live choices to a product (``settings.v1``).

The product file says which reasoner model, deadline and attribution a
live session uses; the pipeline settings saved from the web may replace
them when a session starts: ``live.suggester_model`` (haiku, sonnet,
opus), ``live.deadline`` (``product`` keeps the file's, or a number of
milliseconds) and ``live.diarization`` (``off`` hears a one-microphone
session as a single person instead of telling the voices apart). Every
value is matched against an allow-list; anything else keeps the
product's value. Plain data: no I/O.
"""

from typing import Any, Dict, Mapping, Optional

from orchestrator.domain.config import ProductConfig

MODELS = ("haiku", "sonnet", "opus")
DEADLINES = {"1800": 1800, "2800": 2800}


def apply_live_choices(
    config: ProductConfig, choices: Optional[Mapping[str, str]]
) -> ProductConfig:
    """Return ``config`` with the saved live choices applied."""
    if not choices or config.reasoner is None:
        return config
    reasoner: Dict[str, Any] = {}
    model = choices.get("live.suggester_model")
    if model in MODELS:
        reasoner["model"] = model
    deadline = DEADLINES.get(choices.get("live.deadline", ""))
    if deadline is not None:
        reasoner["deadline_ms"] = deadline
    update: Dict[str, Any] = {}
    if reasoner:
        update["reasoner"] = config.reasoner.model_copy(update=reasoner)
    if (choices.get("live.diarization") == "off"
            and config.attribution.kind == "sortformer_stream"):
        update["attribution"] = config.attribution.model_copy(
            update={"kind": "single_speaker"}
        )
    return config.model_copy(update=update) if update else config
