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

import threading
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
    OutputTooLongError,
    QuotaExhaustedError,
)
from src.domain.reasoning.roles import (
    Backend,
    Constraints,
    Context,
    ErrorInfo,
    Latency,
    ReasonRequest,
    ReasonResponse,
    Role,
    Status,
    Usage,
)
from src.domain.reasoning.templates import (
    SLOW_MAX_S,
    get_template,
    template_role,
)
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

    def complete(
        self, system: str, user: str, max_tokens: int,
        deadline_s: Optional[float] = None,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Ask the model and record the call in the ledger.

        With ``deadline_s`` a backend that can cancel
        (``complete_within``) gives up then; any other backend is let
        finish and a late answer is dropped as a time-out. With
        ``on_text`` a backend that streams (``complete_streaming``)
        passes each text piece as it comes; any other backend passes
        the whole reply once, at the end.
        """
        self.calls += 1
        started = time.perf_counter()
        try:
            streaming = getattr(self.model, "complete_streaming", None)
            if on_text is not None and callable(streaming):
                return str(streaming(system, user, max_tokens, deadline_s,
                                     on_text))
            text = self._complete(system, user, max_tokens, deadline_s,
                                  started)
            if on_text is not None:
                on_text(text)
            return text
        finally:
            if self.ledger is not None:
                self.ledger.record(
                    f"{self.model.model_name}#{self.role}",
                    last_usage(self.model),
                )

    def _complete(
        self, system: str, user: str, max_tokens: int,
        deadline_s: Optional[float], started: float,
    ) -> str:
        """Ask a backend that does not stream (see :meth:`complete`)."""
        within = getattr(self.model, "complete_within", None)
        if deadline_s is not None and callable(within):
            return str(within(system, user, max_tokens, deadline_s))
        text = self.model.complete(system, user, max_tokens)
        if (deadline_s is not None
                and time.perf_counter() - started > deadline_s):
            raise ModelTimeoutError(
                f"The answer took longer than {deadline_s:g} s."
            )
        return text


SLOW_CALLS = 3
"""Slow template calls (planner, report) running at once on the cold
backend (#267); more wait, within their own deadline."""
YIELDING_CALLS = 1
"""Of those, most that may be ``yields`` calls (#268: the live concept
ticks), which never wait for a slot."""


class Reasoner:
    """Answer reasoner requests with a chat model per alias.

    Args:
        models: Builds the chat model of an alias (cached by the caller).
        ledger: Where tokens are counted per role, if anywhere.
        clock: Monotonic seconds (replaced in tests).
        slow_calls: Most ``slow`` template calls at once; another waits
            for a free one, and its wait counts against its deadline.
        live_models: Builds the (warm) model of the live roles
            (``suggester``, ``conversational`` and the quick
            ``structured`` slot filler); ``models`` serves them too when
            it is not given, and always the judge and the ``slow``
            templates (the interview's planner and report, #267), with
            their own deadline up to ``SLOW_MAX_S``.
    """

    def __init__(
        self,
        models: ModelFactory,
        ledger: Optional[UsageLedger] = None,
        clock: Callable[[], float] = time.perf_counter,
        live_models: Optional[ModelFactory] = None,
        slow_calls: int = SLOW_CALLS,
    ):
        """Wire the model factories and the ledger."""
        self.models = models
        self.live_models = live_models or models
        self.ledger = ledger
        self.clock = clock
        self._slow = threading.BoundedSemaphore(max(slow_calls, 1))
        self._yielding = threading.BoundedSemaphore(YIELDING_CALLS)

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

    def warm_up(
        self, template_id: str = "suggester.support@2", model: str = "haiku",
        language: str = "es", max_suggestions: int = 3,
    ) -> bool:
        """Start the warm backend for a live template, off the request path.

        The process is keyed by the system prompt and the output cap, so
        the arguments must match the requests the product will send
        (the shipped ``live_assist`` products: Spanish, 3 suggestions).
        Returns whether the backend can be warmed (a cold or unknown
        backend is left alone). ``conversational`` templates (the voice
        bot, the interviewer) and ``structured`` ones (the interview's
        slot filler) are warmed the same way.
        """
        role: Role = template_role(template_id)
        template = get_template(template_id, role)
        chat = self.live_models(model)
        warm = getattr(chat, "warm", None)
        if not callable(warm):
            return False
        request = ReasonRequest(
            request_id="warm-up", role=role, model=model,
            template=template_id, language=language,
            context=Context(window=[]),
            constraints=Constraints(max_suggestions=max_suggestions),
        )
        warm(template.system(request), template.max_output_tokens)
        return True

    def reason(
        self, request: ReasonRequest,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> ReasonResponse:
        """Return the answer to ``request`` (never raises a model error).

        ``on_text`` receives the model's raw output piece by piece while
        it is written (the voice bot speaks the first sentence of a
        conversational reply before the reply ends); a backend that
        does not stream passes the whole output once. An unknown
        template or a role that does not match it raises
        :class:`~src.domain.errors.InvalidInputError`: the caller's
        mistake, not the model's.
        """
        if request.role == "judge":
            template = None
        else:
            template = get_template(request.template, request.role)
        started = self.clock()
        # The judge and the slow templates (planner, report: #267) run on
        # the cold per-call backend with their own deadline; only the live
        # turns use the warm processes and their short time-out, so a long
        # call never holds a warm process a live call is waiting for.
        slow = template is None or template.slow
        factory = self.models if slow else self.live_models
        chat = MeteredModel(factory(request.model), request.role,
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
                asked = request.constraints.max_output_tokens
                cap = min(asked or template.max_output_tokens,
                          template.max_output_tokens)
                deadline = _deadline_s(request)
                if template.slow:
                    deadline = min(deadline or SLOW_MAX_S, SLOW_MAX_S)
                    text = self._slow_call(
                        chat, template.system(request),
                        template.user(request), cap, deadline, on_text,
                        template.yields)
                else:
                    text = chat.complete(
                        template.system(request), template.user(request),
                        cap, deadline, on_text,
                    )
                parsed = template.parse(text, request)
                limit = request.constraints.max_suggestions
                suggestions = list(parsed.suggestions)
                if limit is not None:
                    suggestions = suggestions[:limit]
                response.suggestions = suggestions
                response.reply = parsed.reply
                response.data = parsed.data
                if (not suggestions and parsed.reply is None
                        and parsed.data is None):
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
        except OutputTooLongError as exc:
            status = "error"
            error = ErrorInfo(kind="invalid_output", message=str(exc))
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

    def _slow_call(
        self, chat: MeteredModel, system: str, user: str, cap: int,
        deadline: float, on_text: Optional[Callable[[str], None]],
        yields: bool = False,
    ) -> str:
        """A slow template's call, at most :data:`SLOW_CALLS` at once.

        The wait for a free slot counts against the deadline; a call that
        cannot get one in time is a time-out, as if the model were late.
        A ``yields`` call never waits: it runs only when a slot and the
        :data:`YIELDING_CALLS` allowance are free at once, else it is a
        time-out straight away.
        """
        if yields:
            if not self._yielding.acquire(blocking=False):
                raise ModelTimeoutError("A yielding call is already running.")
            try:
                if not self._slow.acquire(blocking=False):
                    raise ModelTimeoutError("No free slot: the call yields.")
                try:
                    return chat.complete(system, user, cap, deadline,
                                         on_text)
                finally:
                    self._slow.release()
            finally:
                self._yielding.release()
        began = self.clock()
        if not self._slow.acquire(timeout=max(deadline, 0.0)):
            raise ModelTimeoutError(
                f"No free slot for a slow call within {deadline:g} s.")
        try:
            left = deadline - (self.clock() - began)
            if left <= 0:
                raise ModelTimeoutError(
                    f"No free slot for a slow call within {deadline:g} s.")
            return chat.complete(system, user, cap, left, on_text)
        finally:
            self._slow.release()

    def _judge(
        self, chat: MeteredModel, request: ReasonRequest
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
        return {"reply": chat.complete(system, user, cap,
                                       _deadline_s(request))}


def _deadline_s(request: ReasonRequest) -> Optional[float]:
    """Return the request's ``deadline_ms`` in seconds, if it has one."""
    ms = request.constraints.deadline_ms
    return None if ms is None else ms / 1000.0


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
