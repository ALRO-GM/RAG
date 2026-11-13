"""Chat model adapters with fake clients: no network, no weights."""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.adapters.outbound.anthropic_chat_model import (
    DEFAULT_API_MODEL,
    FALLBACK_BETA,
    AnthropicChatModel,
)
from src.adapters.outbound.transformers_chat_model import (
    TransformersChatModel,
)
from src.domain.errors import ModelError

anthropic: Any = pytest.importorskip("anthropic")
httpx2: Any = pytest.importorskip("httpx2")


def text(value: str) -> SimpleNamespace:
    """Return a text content block."""
    return SimpleNamespace(type="text", text=value)


def response(
    *blocks: SimpleNamespace, stop_reason: str = "end_turn"
) -> SimpleNamespace:
    """Return a message-like response."""
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class FakeClient:
    """Stand in for ``anthropic.Anthropic``: record requests, reply."""

    def __init__(
        self,
        reply: Optional[SimpleNamespace] = None,
        error: Optional[Exception] = None,
    ):
        """Answer ``reply`` or raise ``error`` on every request."""
        self.reply = reply or response(text("ok"))
        self.error = error
        self.requests: List[Dict[str, Any]] = []
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=self._create)
        )

    def _create(self, **request: Any) -> SimpleNamespace:
        """Record the request and answer it."""
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.reply


def status_error(kind: str, status: int, message: str) -> Exception:
    """Build a real SDK status error."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    reply = httpx2.Response(status, request=request)
    error: Exception = getattr(anthropic, kind)(
        message, response=reply, body=None
    )
    return error


def connection_error(kind: str = "APIConnectionError") -> Exception:
    """Build a real SDK connection error."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error: Exception = getattr(anthropic, kind)(request=request)
    return error


# ------------------------------------------------------------ Anthropic


def test_anthropic_request_shape() -> None:
    """Model, fallback beta, effort, system and one user message."""
    client = FakeClient(response(text("  the reply \n")))
    model = AnthropicChatModel(client=client)
    assert model.complete("SYSTEM", "USER", 700) == "the reply"
    assert client.requests == [{
        "model": DEFAULT_API_MODEL,
        "max_tokens": 16000,
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
        "output_config": {"effort": "medium"},
        "system": "SYSTEM",
        "messages": [{"role": "user", "content": "USER"}],
    }]
    assert FALLBACK_BETA == "server-side-fallback-2026-07-01"
    assert model.model_name == DEFAULT_API_MODEL == "claude-opus-5-5"


def test_anthropic_model_effort_and_token_floor_are_configurable() -> None:
    """``max_tokens`` is the larger of the request and the floor."""
    client = FakeClient()
    model = AnthropicChatModel(
        "claude-haiku-5-5", effort="low", min_max_tokens=1000, client=client
    )
    model.complete("s", "u", 200)
    model.complete("s", "u", 5000)
    assert model.model_name == "claude-haiku-5-5"
    assert [r["max_tokens"] for r in client.requests] == [1000, 5000]
    assert client.requests[0]["model"] == "claude-haiku-5-5"
    assert client.requests[0]["output_config"] == {"effort": "low"}


def test_anthropic_text_extraction_skips_non_text_blocks() -> None:
    """Thinking and fallback blocks are ignored; text blocks are joined."""
    client = FakeClient(response(
        SimpleNamespace(type="thinking", thinking=""),
        text('{"rating": '),
        SimpleNamespace(type="fallback"),
        text("4}"),
    ))
    assert AnthropicChatModel(client=client).complete("s", "u", 10) == (
        '{"rating": 4}'
    )


def test_anthropic_reply_without_text_is_empty() -> None:
    """No text block gives an empty reply, not a crash."""
    client = FakeClient(response(SimpleNamespace(type="thinking")))
    assert AnthropicChatModel(client=client).complete("s", "u", 10) == ""
    assert AnthropicChatModel(
        client=FakeClient(response())
    ).complete("s", "u", 10) == ""


@pytest.mark.parametrize("stop_reason, message", [
    ("refusal", "declined to assess"),
    ("max_tokens", "reply was cut"),
])
def test_anthropic_stop_reasons_are_model_errors(
    stop_reason: str, message: str
) -> None:
    """A refusal or a truncated reply never reaches the parser."""
    cut = response(text('{"rating": 5'), stop_reason=stop_reason)
    client = FakeClient(cut)
    with pytest.raises(ModelError, match=message):
        AnthropicChatModel(client=client).complete("s", "u", 10)


@pytest.mark.parametrize("error, message", [
    (status_error("AuthenticationError", 401, "bad key"),
     "credentials are missing or invalid"),
    (status_error("RateLimitError", 429, "slow down"), "rate limit reached"),
    (status_error("BadRequestError", 400, "bad beta"),
     "Anthropic API error 400: bad beta"),
    (status_error("InternalServerError", 500, "boom"),
     "Anthropic API error 500: boom"),
    (status_error("APIStatusError", 529, "overloaded"),
     "Anthropic API error 529: overloaded"),
    (connection_error(), "Cannot reach the Anthropic API"),
    (connection_error("APITimeoutError"), "Cannot reach the Anthropic API"),
])
def test_anthropic_sdk_errors_become_model_errors(
    error: Exception, message: str
) -> None:
    """Every SDK failure is mapped to a one-line ``ModelError``."""
    model = AnthropicChatModel(client=FakeClient(error=error))
    with pytest.raises(ModelError, match=message) as caught:
        model.complete("s", "u", 10)
    assert "\n" not in str(caught.value)


def test_anthropic_missing_sdk_is_a_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the package the error says how to install it."""
    def fail(name: str) -> Any:
        """Pretend the SDK is not installed."""
        raise ImportError(name)

    monkeypatch.setattr(
        "src.adapters.outbound.anthropic_chat_model.importlib.import_module",
        fail,
    )
    with pytest.raises(ModelError, match="uv sync --group api"):
        AnthropicChatModel(client=FakeClient()).complete("s", "u", 10)


def test_anthropic_client_is_created_lazily_and_once() -> None:
    """Constructing the adapter touches no SDK; the client is reused."""
    created: List[FakeClient] = []

    def factory() -> FakeClient:
        """Count client constructions."""
        created.append(FakeClient())
        return created[-1]

    model = AnthropicChatModel()
    assert model._client is None and model._sdk is None
    model._sdk = SimpleNamespace(Anthropic=factory)
    model.complete("s", "u", 10)
    model.complete("s", "u", 10)
    assert len(created) == 1 and len(created[0].requests) == 2


# --------------------------------------------------------- Transformers


class FakeTokenizer:
    """A tokenizer whose 'tokens' are whitespace-separated words."""

    eos_token_id = 0

    def __init__(self, accepts_thinking_flag: bool = True):
        """Optionally reject the ``enable_thinking`` template option."""
        self.accepts_thinking_flag = accepts_thinking_flag
        self.templates: List[Dict[str, Any]] = []

    def apply_chat_template(
        self, messages: List[Dict[str, str]], **options: Any
    ) -> str:
        """Join the messages; record the template options."""
        if "enable_thinking" in options and not self.accepts_thinking_flag:
            raise TypeError("unexpected keyword 'enable_thinking'")
        self.templates.append(options)
        return " ".join(m["content"] for m in messages)

    def __call__(self, prompt: str, return_tensors: str) -> Dict[str, Any]:
        """Return fake input ids, one per word."""
        import torch

        return {"input_ids": torch.zeros((1, len(prompt.split())))}

    def decode(self, tokens: Any, skip_special_tokens: bool) -> str:
        """Return the scripted generation."""
        return str(tokens)


class FakeCausalModel:
    """A model 'generating' a fixed text after the prompt."""

    def __init__(self, output: str):
        """Script the generated text."""
        self.output = output
        self.options: List[Dict[str, Any]] = []

    def generate(self, input_ids: Any, **options: Any) -> List[Any]:
        """Return the prompt followed by the scripted text."""
        self.options.append(options)

        class Row:
            """One generated sequence, sliceable past the prompt."""

            def __getitem__(row, piece: slice) -> str:
                """Return the text generated after the prompt."""
                assert piece.start == input_ids.shape[1]
                return self.output

        return [Row()]


def local_model(
    output: str = "ok", max_input_tokens: int = 50,
    tokenizer: Optional[FakeTokenizer] = None,
) -> TransformersChatModel:
    """Return the adapter with a fake tokenizer and model injected."""
    model = TransformersChatModel("tiny-test-model", max_input_tokens)
    model._tokenizer = tokenizer or FakeTokenizer()
    model._model = FakeCausalModel(output)
    return model


def test_transformers_constructor_loads_nothing() -> None:
    """The weights are not touched until the first completion."""
    model = TransformersChatModel("some/model")
    assert model.model_name == "some/model"
    assert model._model is None and model._tokenizer is None


def test_transformers_complete_is_greedy_and_strips_thinking() -> None:
    """Greedy decoding; ``<think>`` blocks are removed from the reply."""
    model = local_model('<think>hmm\nlet me see</think>\n {"rating": 3} ')
    assert model.complete("system words", "user words", 77) == (
        '{"rating": 3}'
    )
    options = model._model.options[0]  # type: ignore[union-attr]
    assert options["max_new_tokens"] == 77
    assert options["do_sample"] is False
    assert model._tokenizer.templates == [  # type: ignore[union-attr]
        {"tokenize": False, "add_generation_prompt": True,
         "enable_thinking": False},
    ]


def test_transformers_template_without_thinking_flag() -> None:
    """Tokenizers that reject ``enable_thinking`` are called without it."""
    tokenizer = FakeTokenizer(accepts_thinking_flag=False)
    model = local_model("fine", tokenizer=tokenizer)
    assert model.complete("s", "u", 5) == "fine"
    assert tokenizer.templates == [
        {"tokenize": False, "add_generation_prompt": True},
    ]


def test_transformers_prompt_too_long_is_a_model_error() -> None:
    """A prompt over the input budget is refused with advice."""
    model = local_model(max_input_tokens=5)
    with pytest.raises(ModelError, match="lower the number of candidate"):
        model.complete("one two three", "four five six", 10)
    assert model.complete("one two", "three four five", 10) == "ok"


def test_transformers_unknown_model_is_a_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot be loaded gives a clean error."""
    import transformers

    def fail(name: str, **options: Any) -> Any:
        """Pretend the model is not available."""
        raise OSError(f"{name} is not a local folder")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fail)
    with pytest.raises(ModelError, match="Cannot load model no/such-model"):
        TransformersChatModel("no/such-model").complete("s", "u", 5)
