"""Use case: the reasoner, one port for the judge, suggester and bot roles.

:class:`Reasoner` wraps the :class:`~src.application.skills.ports.ChatModel`
of the configured backend. A request names a role, a model alias and a
prompt template; the answer is a ``reasoner.response.v1`` that never
raises for a model failure: quota, sign-in, time-outs and unreadable
output become a ``status`` and an ``error``, so a live caller can drop
the suggestion and go on. The ``judge`` role is the analyzer's existing
confirmation (:class:`~src.application.skills.tagging.TagConfirmer`),
unchanged. Tokens are counted per role in the
:class:`~src.domain.skills.usage.UsageLedger`.

Transcript text is never logged here.
"""

import time
from typing import Callable, Dict, List, Optional

from src.application.ports import Cache, Progress
from src.application.skills.judges import last_usage
from src.application.skills.ports import ChatModel
from src.application.skills.tagging import TaggedSentence, TagConfirmer
from src.domain.errors import (
    InvalidInputError,
    ModelAuthError,
    ModelError,
    ModelTimeoutError,
    QuotaExhaustedError,
)
from src.domain.reasoning.roles import (
    Backend,
    ErrorInfo,
    Latency,
    ReasonRequest,
    ReasonResponse,
    Status,
    Usage,
)
from src.domain.reasoning.templates import get_template
from src.domain.skills.rubric import RubricCatalogue
from src.domain.skills.usage import UsageLedger

ModelFactory = Callable[[str], ChatModel]
"""Alias (``haiku``, ``opus``, ``""`` for the default) → chat model."""


def backend_of(model_name: str) -> Backend:
    """Return the backend a chat model's identifier belongs to."""
    if model_name.startswith("claude-code"):
        return "claude-code"
    if model_name.startswith("anthropic"):
        return "anthropic-api"
    if model_name.startswith("opencode"):
        return "opencode"
    return "local"


class MeteredModel:
    """A chat model that counts its calls per role in a ledger.

    It also keeps the usage and latency of the last call, so a caller
    can put them in the response.
    """

    def __init__(
        self, model: ChatModel, role: str, ledger: Optional[UsageLedger]
    ):
        """Wrap ``model``; count calls under ``<name>#<role>``."""
        self.model = model
        self.role = role
        self.ledger = ledger
        self.calls = 0

    @property
    def model_name(self) -> str:
        """Return the wrapped model's identifier."""
        return self.model.model_name

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Ask the model and record the call in the ledger."""
        self.calls += 1
        try:
            return self.model.complete(system, user, max_tokens)
        finally:
            if self.ledger is not None:
                self.ledger.record(
                    f"{self.model.model_name}#{self.role}",
                    last_usage(self.model),
                )


class Reasoner:
    """Answer reasoner requests with a chat model per alias.

    Args:
        models: Builds the chat model of an alias (cached by the caller).
        ledger: Where tokens are counted per role, if anywhere.
        clock: Monotonic seconds (replaced in tests).
    """

    def __init__(
        self,
        models: ModelFactory,
        ledger: Optional[UsageLedger] = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        """Wire the model factory and the ledger."""
        self.models = models
        self.ledger = ledger
        self.clock = clock

    def confirm(
        self,
        tagged: List[TaggedSentence],
        rubric: RubricCatalogue,
        language: Optional[str] = None,
        model: str = "",
        cache: Optional[Cache] = None,
        progress: Optional[Progress] = None,
        context: int = 3,
    ) -> List[TaggedSentence]:
        """Judge role: confirm the candidate tags, as the analyzer does.

        Delegates to :class:`TagConfirmer` with the model of ``model``;
        the output is the confirmer's own, unchanged.
        """
        judge = MeteredModel(self.models(model), "judge", self.ledger)
        return TagConfirmer(
            judge, cache=cache, progress=progress, context=context
        ).confirm(tagged, rubric, language)

    def reason(self, request: ReasonRequest) -> ReasonResponse:
        """Return the answer to ``request`` (never raises a model error).

        An unknown template or a role that does not match it raises
        :class:`~src.domain.errors.InvalidInputError`: the caller's
        mistake, not the model's.
        """
        if request.role == "judge":
            template = None
        else:
            template = get_template(request.template, request.role)
        started = self.clock()
        chat = MeteredModel(self.models(request.model), request.role,
                            self.ledger)
        usage = Usage()
        error: Optional[ErrorInfo] = None
        status: Status = "ok"
        response = ReasonResponse(
            request_id=request.request_id, role=request.role, status="ok",
            model=chat.model_name, backend=backend_of(chat.model_name),
            latency=Latency(),
        )
        try:
            if template is None:
                response.verdict = self._judge(chat, request)
            else:
                cap = request.constraints.max_output_tokens or (
                    template.max_output_tokens
                )
                text = chat.complete(
                    template.system(request), template.user(request), cap
                )
                parsed = template.parse(text, request)
                limit = request.constraints.max_suggestions
                suggestions = list(parsed.suggestions)
                if limit is not None:
                    suggestions = suggestions[:limit]
                response.suggestions = suggestions
                response.reply = parsed.reply
                if not suggestions and parsed.reply is None:
                    status = "no_suggestion"
        except QuotaExhaustedError as exc:
            status = "quota"
            error = ErrorInfo(
                kind="quota", message=str(exc),
                resets_at=(
                    exc.resets_at.isoformat() if exc.resets_at else None
                ),
            )
        except ModelAuthError as exc:
            status, error = "auth", ErrorInfo(kind="auth", message=str(exc))
        except ModelTimeoutError as exc:
            status = "timeout"
            error = ErrorInfo(kind="timeout", message=str(exc))
        except ModelError as exc:
            status = "error"
            error = ErrorInfo(kind="transient", message=str(exc))
        total_ms = (self.clock() - started) * 1000.0
        measured = last_usage(chat.model)
        if measured is not None:
            usage = Usage(
                input_tokens=measured.input_tokens,
                output_tokens=measured.output_tokens,
                cache_creation_input_tokens=(
                    measured.cache_creation_input_tokens
                ),
                cache_read_input_tokens=measured.cache_read_input_tokens,
                cost_usd_equivalent=measured.cost_usd,
            )
        response.status = status
        response.error = error
        response.usage = usage
        response.latency = _latency(chat.model, total_ms)
        return response

    def _judge(
        self, chat: ChatModel, request: ReasonRequest
    ) -> Dict[str, str]:
        """Run a raw judge call (``payload``: ``system`` and ``user``)."""
        system = request.payload.get("system")
        user = request.payload.get("user")
        if not isinstance(system, str) or not isinstance(user, str):
            raise InvalidInputError(
                "A judge request needs payload.system and payload.user "
                "(or use Reasoner.confirm)."
            )
        cap = request.constraints.max_output_tokens or 120
        return {"reply": chat.complete(system, user, cap)}


def _latency(model: object, total_ms: float) -> Latency:
    """Return the backend's own latency breakdown, or just the total.

    A backend that measures itself exposes ``last_latency``; its
    ``total_ms`` is kept, the use case's wall clock is the fallback.
    """
    own = getattr(model, "last_latency", None)
    if isinstance(own, Latency):
        return own.model_copy(
            update={"total_ms": max(own.total_ms, total_ms)}
        )
    return Latency(total_ms=round(total_ms, 1))
