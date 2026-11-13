"""Chat model adapter: Claude through the Anthropic API.

Transcripts are sent to an external service. Use it only with
recordings whose participants agreed to that; the local adapter
(:mod:`transformers_chat_model`) keeps everything on the machine.
"""

import importlib
from typing import Any, Optional

from src.domain.errors import ModelError

DEFAULT_API_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicChatModel:
    """Answer one system + user message with a Claude model.

    Credentials come from the environment (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile). ``max_tokens`` covers the model's
    thinking as well as its reply, so the floor ``min_max_tokens``
    keeps short replies from being cut. If a safety classifier
    declines a request, the API retries it on its default fallback
    model (``fallbacks="default"``).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_API_MODEL,
        effort: str = "medium",
        min_max_tokens: int = 16000,
        client: Optional[Any] = None,
    ):
        """Configure the model; the client is created on first use."""
        self._model_name = model_name
        self.effort = effort
        self.min_max_tokens = min_max_tokens
        self._client = client
        self._sdk: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return the identifier of the language model."""
        return self._model_name

    def _load(self) -> Any:
        """Import the SDK and create the client."""
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("anthropic")
            except ImportError:
                raise ModelError(
                    "The Anthropic SDK is not installed; run "
                    "`uv sync --group api`."
                )
        if self._client is None:
            self._client = self._sdk.Anthropic()
        return self._client

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Return the model's reply to ``user`` under ``system``."""
        client = self._load()
        sdk: Any = self._sdk
        try:
            response = client.beta.messages.create(
                model=self._model_name,
                max_tokens=max(max_tokens, self.min_max_tokens),
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": self.effort},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except sdk.AuthenticationError:
            raise ModelError(
                "Anthropic credentials are missing or invalid; set "
                "ANTHROPIC_API_KEY or run `ant auth login`."
            )
        except sdk.RateLimitError:
            raise ModelError("Anthropic API rate limit reached; retry later.")
        except sdk.APIStatusError as exc:
            raise ModelError(
                f"Anthropic API error {exc.status_code}: {exc.message}"
            )
        except sdk.APIConnectionError:
            raise ModelError("Cannot reach the Anthropic API.")
        if response.stop_reason == "refusal":
            raise ModelError("The model declined to assess this excerpt.")
        if response.stop_reason == "max_tokens":
            raise ModelError("The model's reply was cut; raise max_tokens.")
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
