"""Use case: rebuild an ended live round from what was kept (#264).

A live round's trackers live in the orchestrator's memory: a restart of
the orchestrator, a failed judgement or a session recovered by the
transcriber after its own restart leaves them empty, while the
conversation itself is kept (the transcriber's ``transcript.v1``, or its
``events.jsonl``). :class:`RoundRebuild` replays those events at full
speed through the same trackers a live round uses, without asking for
any suggestion:

* the topics and concepts of the role's plan (:class:`TopicTracker`);
* the product's mandatory items (:class:`MandatoryTracker`), when a
  port for them is given and the product has a knowledge domain;
* the lines better avoided in an interview (:class:`AvoidTracker`),
  when a port for them is given and the employee is the interviewer;

then, with an :class:`~orchestrator.application.interview.
InterviewWiring`, judges the candidate's answers and stores the report
with the plan's results and in the candidate's file, as at the end of
the round (:class:`LiveJudge`; the ``interview rejudge`` command, #262,
is this with only the topics and the judgement). Each part is optional
and independent: the service is the seam other round-level work (#268)
plugs into. Nothing here logs the conversation's words.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from orchestrator.application.avoid import AvoidTracker
from orchestrator.application.interview import InterviewWiring
from orchestrator.application.live_judgement import LiveJudge
from orchestrator.application.mandatory import MandatoryTracker
from orchestrator.application.next_round import TopicTracker
from orchestrator.application.ports import (
    AvoidPort,
    Document,
    Executor,
    MandatoryPort,
)
from orchestrator.application.replay import transcript_to_events
from orchestrator.domain.config import ProductConfig
from orchestrator.domain.events import LiveEvent, Transcript
from orchestrator.domain.interview import InterviewPlan, TopicCoverage
from orchestrator.domain.interview_coverage import without_words
from orchestrator.domain.candidate import Round
from orchestrator.domain.next_round import topic_states

Publish = Callable[[Document], object]


@dataclass
class RebuiltRound:
    """What a rebuilt round says, in the shapes the live round uses.

    ``events`` and ``finals`` count what was replayed; ``covered`` and
    ``revisit`` are the plan's slots (ids); ``mandatory`` is the
    hand-off's account of the checklist; ``avoid`` the flags;
    ``accuracy`` the judgement (``None`` when not asked or nothing was
    answered); ``attribution`` who attributed the answers to the points
    (``after_session``: the model reading the whole round, #268; or
    ``rules``) and ``attribution_error`` why not the model. Each is
    ``None`` when that part did not run.
    """

    session_id: str
    events: int = 0
    finals: int = 0
    covered: Optional[List[str]] = None
    revisit: Optional[List[str]] = None
    mandatory: Optional[Document] = None
    avoid: Optional[List[Document]] = None
    accuracy: Optional[Document] = None
    attribution: Optional[str] = None
    attribution_error: Optional[str] = None
    coverage: Optional[List[Document]] = None
    """The topics' coverage without words (#286), as the hand-off keeps
    it: the stored report's when judged, else the replay's."""
    parts: List[str] = field(default_factory=list)

    def summary(self) -> Document:
        """Counts and states only (no words), for a log or a command."""
        return {
            "session_id": self.session_id, "events": self.events,
            "finals": self.finals, "parts": list(self.parts),
            "covered": len(self.covered or []),
            "revisit": len(self.revisit or []),
            "mandatory_missing": len((self.mandatory or {}).get("missing")
                                     or []),
            "avoid_flags": len(self.avoid or []),
            "accuracy": (self.accuracy or {}).get("state"),
            "saved": bool((self.accuracy or {}).get("saved")),
            "attribution": self.attribution,
            **({"attribution_error": self.attribution_error}
               if self.attribution_error else {}),
        }


class RoundRebuild:
    """Replay a kept round through the trackers and judge it again.

    Args:
        executor: Runs the trackers' checks (inline is fine offline).
        mandatory: The RAG's mandatory items, if they are to be checked.
        avoid: The RAG's check of lines to avoid, if it is to run.
        wiring: The interviews' ports, if the answers are to be judged.
    """

    def __init__(
        self,
        executor: Executor,
        mandatory: Optional[MandatoryPort] = None,
        avoid: Optional[AvoidPort] = None,
        wiring: Optional[InterviewWiring] = None,
    ) -> None:
        """Keep the ports; each part runs only with its port."""
        self.executor = executor
        self.mandatory = mandatory
        self.avoid = avoid
        self.wiring = wiring

    def rebuild(
        self,
        session_id: str,
        events: Sequence[LiveEvent],
        roles: Mapping[str, str],
        product: Optional[ProductConfig] = None,
        plan: Optional[InterviewPlan] = None,
        rounds: Sequence[Round] = (),
        candidate: Optional[str] = None,
        candidate_id: Optional[str] = None,
        model: str = "haiku",
        judge: bool = True,
        publish: Publish = lambda message: None,
        attribute: bool = True,
        earlier: Optional[Mapping[str, TopicCoverage]] = None,
    ) -> RebuiltRound:
        """Replay ``events`` (in order) and return what the round says.

        ``roles``: speaker id → role (``interviewer``, ``candidate``…).
        ``product`` enables the mandatory items (its knowledge domain)
        and the avoid check (an interviewer's product); ``plan`` the
        topics and, with ``judge`` and the wiring, the judgement, which
        stores the report for ``candidate_id``. ``rounds``: the
        candidate's earlier rounds (the order of the topics).
        ``attribute``: the model attributes the round's answers before
        the judgement, as at the end of a live round (#268); ``False``
        keeps the rules' attribution of the replay. ``earlier``: the
        round's coverage kept in its hand-off (#286), used when no report
        of the round is stored.
        """
        result = RebuiltRound(session_id)
        roles_map = dict(roles)
        trackers: List[Any] = []
        topics: Optional[TopicTracker] = None
        if plan is not None:
            topics = TopicTracker(
                plan, topic_states(plan, list(rounds)), roles_map, publish,
                product.language if product is not None else "es",
            )
            trackers.append(topics)
            result.parts.append("topics")
        checklist: Optional[MandatoryTracker] = None
        knowledge = product.knowledge if product is not None else None
        if self.mandatory is not None and knowledge is not None:
            checklist = MandatoryTracker(
                self.mandatory, knowledge.domains[0], roles_map, publish,
                self.executor,
            )
            checklist.start()
            if checklist.active:
                trackers.append(checklist)
                result.parts.append("mandatory")
            else:
                checklist = None
        flags: Optional[AvoidTracker] = None
        if self.avoid is not None and "interviewer" in roles_map.values():
            flags = AvoidTracker(
                self.avoid, roles_map, publish, self.executor,
                product.language if product is not None else "es",
            )
            trackers.append(flags)
            result.parts.append("avoid")
        for event in events:
            result.events += 1
            if event.type == "final":
                result.finals += 1
            for tracker in trackers:
                tracker.on_event(event)
        if checklist is not None:
            checklist.finish()
            result.mandatory = checklist.summary()
        if flags is not None:
            flags.finish()
            result.avoid = flags.view()
        if topics is not None:
            result.covered, result.revisit = topics.result()
            if judge and self.wiring is not None and plan is not None:
                told: List[Document] = []

                def tell(message: Document) -> None:
                    told.append(message)
                    publish(message)

                result.accuracy = LiveJudge(
                    plan, self.wiring, session_id, tell, model,
                    candidate=candidate, candidate_id=candidate_id,
                    attribute=attribute, earlier=earlier,
                ).judge(topics)
                result.parts.append("judgement")
                found = result.accuracy or next(
                    (m for m in reversed(told) if m.get("type") ==
                     "accuracy" and m.get("state") == "empty"), {})
                result.attribution = found.get("attribution")
                result.attribution_error = found.get("attribution_error")
            judged = (result.accuracy or {}).get("coverage")
            result.coverage = without_words(
                [TopicCoverage.model_validate(c) for c in judged]
                if isinstance(judged, list) else topics.coverage())
        return result


def events_of_transcript(
    transcript: Transcript,
) -> Tuple[List[LiveEvent], Dict[str, str]]:
    """The events and roles of a stored ``transcript.v1``."""
    roles = {s.id: s.role or "unknown" for s in transcript.speakers}
    return list(transcript_to_events(transcript, roles)), roles


def events_of_log(
    lines: Iterable[Mapping[str, Any]],
) -> Tuple[List[LiveEvent], Dict[str, str]]:
    """The events and roles of a live log (``events.jsonl``, #264).

    Events that do not read as ``live.event.v1`` are skipped (a line cut
    by a crash); the roles are the speakers' of ``session_started``.
    """
    events: List[LiveEvent] = []
    roles: Dict[str, str] = {}
    for line in lines:
        try:
            event = LiveEvent.model_validate(line)
        except ValueError:
            continue
        if event.type == "session_started":
            for speaker in event.speakers or []:
                roles[speaker.id] = speaker.role
        events.append(event)
    return events, roles
