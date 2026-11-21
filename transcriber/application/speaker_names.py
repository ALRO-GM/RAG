"""Name the voices of a transcript from what people say (TR-20, #102).

:class:`FindSpeakerNames` reads a finished transcript, picks the
sentences that may hold a name (``domain/speaker_names.py``), asks the
local :class:`~transcriber.application.ports.ChatModel` about a bounded
number of them (self-introductions first, then a few of every name; the
other mentions of a name it confirmed count as the rules saw them),
spreads the evidence over the voices of the whole session (#116), gives
a name automatically only where the precision-first policy allows it
and keeps the rest as suggestions (:class:`SpeakerNameStore`).
Names are given through :class:`~transcriber.application.review.
RenameSpeaker`, the path a person's rename takes, and an automatic name
the search no longer supports is taken away with :class:`~transcriber.
application.review.UnnameSpeaker`.

It runs as the last step of a transcription job
(:meth:`FindSpeakerNames.after_transcription`, which never fails the
job) and again on request (:class:`QueueNameSearch`). When the model
cannot run (no GPU memory after waiting, missing weights) the rules'
findings are kept as suggestions only and the record says why.
:class:`ShowSpeakerNames` returns what still applies and
:class:`UndoAutoName` turns an automatic name back into ``S#``.
"""

import logging
import time
from typing import Callable, List, Optional, Tuple

from transcriber.application.common import (
    check_speaker_id,
    job_active,
    require_session,
    require_transcript,
)
from transcriber.application.gpu import drop_frames, retry_on_gpu_memory
from transcriber.application.ports import (
    ChatModel,
    Clock,
    GpuMemory,
    JobRunner,
    Progress,
    SessionLocks,
    SessionRepository,
    SpeakerNameStore,
)
from transcriber.application.review import RenameSpeaker, UnnameSpeaker
from transcriber.domain.errors import InvalidInputError, NotFoundError
from transcriber.domain.models import SessionState, Transcript
from transcriber.domain.sessions import require_state
from transcriber.domain.speaker_names import (
    SYSTEM_PROMPT,
    AutoName,
    NameCandidate,
    NameEvidence,
    NameSearchParameters,
    NameSuggestion,
    SpeakerNames,
    UndoneName,
    attribute,
    decide,
    find_candidates,
    gated_evidence,
    kept_undone,
    model_prompt,
    open_suggestions,
    parse_reply,
    pick_for_model,
    rule_evidence,
    valid_auto,
)

LOG = logging.getLogger(__name__)

NAMES_STAGE = "names"
"""The stage reported while names are searched."""

NO_MODEL_NOTE = (
    "No local model is configured for names: they are only suggested"
    " from the rules."
)
MODEL_FAILED_NOTE = (
    "The local model for names could not run ({reason}): they are only"
    " suggested from the rules."
)
CHANGED_NOTE = (
    "The transcript changed while names were searched; nothing was"
    " named. Search again."
)
SEARCH_FAILED_NOTE = "Names were not searched: {reason}"


def _fingerprint(transcript: Transcript) -> List[Tuple[str, str, str]]:
    """Return what evidence depends on: every sentence and its speaker."""
    return [(u.id, u.speaker, u.text) for u in transcript.utterances]


class FindSpeakerNames:
    """Search a finished transcript for its speakers' names.

    Args:
        sessions: The sessions and their transcripts.
        store: Where the outcome is kept.
        locks: The per-session locks.
        rename: Gives a voice its name (the review path).
        unname: Takes an automatic name away.
        clock: Dates the outcome.
        model: The local chat model; ``None``: rules only, suggestions
            only.
        parameters: The thresholds of the search.
        free_models: Called before the model loads, so the speech and
            voice models leave the GPU first.
        timer: Monotonic seconds, for how long a search took.
        gpu: The GPU memory: a reading that ran out of it frees every
            model and is tried once more (#117); ``None``: no retry.
    """

    def __init__(
        self,
        sessions: SessionRepository,
        store: SpeakerNameStore,
        locks: SessionLocks,
        rename: RenameSpeaker,
        unname: UnnameSpeaker,
        clock: Clock,
        model: Optional[ChatModel] = None,
        parameters: NameSearchParameters = NameSearchParameters(),
        free_models: Callable[[], None] = lambda: None,
        timer: Callable[[], float] = time.monotonic,
        gpu: Optional[GpuMemory] = None,
        choose_model: Optional[Callable[[], Optional[ChatModel]]] = None,
    ) -> None:
        """Wire the ports; nothing is loaded yet.

        ``choose_model``, when given, replaces ``model`` and is asked at
        every search (the model chosen on «Ajustes»; ``None``: rules).
        """
        self._choose_model = choose_model
        self._sessions = sessions
        self._store = store
        self._locks = locks
        self._rename = rename
        self._unname = unname
        self._clock = clock
        self._model = model
        self._parameters = parameters
        self._free_models = free_models
        self._timer = timer
        self._gpu = gpu

    def __call__(self, session_id: str) -> SpeakerNames:
        """Search session ``session_id`` (``done``) and apply the names.

        Returns the stored record. The transcript is read under the
        session's lock, the model runs without it, and the names are
        applied under it after checking the sentences did not change
        meanwhile (if they did, nothing is named and the record says
        so).
        """
        began = self._timer()
        with self._locks.hold(session_id):
            transcript = self._done_transcript(session_id)
        p = self._parameters
        candidates = find_candidates(transcript, p.context, limit=None)
        evidence, model_name, note = self._evidence(transcript, candidates)
        with self._locks.hold(session_id):
            current = self._done_transcript(session_id)
            previous = self._store.names(session_id)
            earlier = valid_auto(
                current.speakers, previous.auto if previous else []
            )
            undone = self._undone(session_id, previous)
            if _fingerprint(current) != _fingerprint(transcript):
                record = SpeakerNames(
                    session=session_id, searched_at=self._clock.now(),
                    model=model_name, candidates=len(candidates),
                    auto=earlier, note=CHANGED_NOTE,
                    seconds=round(self._timer() - began, 3),
                    undone=undone,
                )
                self._store.save_names(record)
                return record
            decision = decide(
                current.speakers, attribute(current, evidence), earlier,
                p.auto_confidence, allow_auto=model_name is not None,
                undone=undone, min_votes=p.min_votes, min_margin=p.margin,
            )
            for label in decision.clear:
                self._unname(session_id, label)
            for name in decision.auto:
                self._rename(session_id, name.speaker, name=name.name)
            record = SpeakerNames(
                session=session_id, searched_at=self._clock.now(),
                model=model_name, candidates=len(candidates),
                auto=decision.auto if model_name is not None else earlier,
                suggestions=decision.suggestions,
                note=note, seconds=round(self._timer() - began, 3),
                undone=undone,
            )
            self._store.save_names(record)
            return record

    def after_transcription(self, session_id: str, progress: Progress) -> None:
        """Search names as the last step of a transcription job.

        Never raises: a failure is logged and told as a note of the
        job, the record is emptied (it may belong to an earlier
        transcript) and the transcript stays as it was saved.
        """
        progress.report(NAMES_STAGE, 1.0)
        try:
            record = self(session_id)
        except Exception as exc:
            drop_frames(exc)
            LOG.warning("Names of %s were not searched: %s", session_id, exc)
            reason = str(exc).strip() or type(exc).__name__
            note = SEARCH_FAILED_NOTE.format(reason=reason)
            progress.note(note)
            self._forget(session_id, note)
            return
        if record.note:
            progress.note(record.note)

    def _forget(self, session_id: str, note: str) -> None:
        """Replace the record by an empty one telling ``note``.

        So suggestions of an earlier transcript are not shown on a new
        one; the names people undid in this transcript are kept. Best
        effort: a failure here is only logged.
        """
        try:
            with self._locks.hold(session_id):
                undone = self._undone(
                    session_id, self._store.names(session_id)
                )
                self._store.save_names(SpeakerNames(
                    session=session_id, searched_at=self._clock.now(),
                    note=note[:500], undone=undone,
                ))
        except Exception as exc:
            LOG.warning("Cannot store the names of %s: %s", session_id, exc)

    def _undone(
        self, session_id: str, previous: Optional[SpeakerNames]
    ) -> List[UndoneName]:
        """Return the names undone in the session's current transcript."""
        if previous is None:
            return []
        run = require_session(self._sessions, session_id).run_id
        return kept_undone(previous.undone, run)

    def _done_transcript(self, session_id: str) -> Transcript:
        """Return the transcript of a ``done`` session."""
        session = require_session(self._sessions, session_id)
        require_state(session, SessionState.DONE)
        return require_transcript(self._sessions, session_id)

    def _evidence(
        self, transcript: Transcript, candidates: List[NameCandidate]
    ) -> Tuple[List[NameEvidence], Optional[str], Optional[str]]:
        """Return the evidence, the model that gave it and a note.

        The model reads at most ``max_candidates`` candidates
        (:func:`pick_for_model`); the others count when the model
        confirmed their name (:func:`gated_evidence`). Without
        candidates it is not loaded. When there is no model, or it
        fails, the rules' own evidence is returned with no model name
        (suggestions only).
        """
        rules = [
            e for e in (rule_evidence(transcript, c) for c in candidates)
            if e is not None
        ]
        model = (
            self._choose_model() if self._choose_model is not None
            else self._model
        )
        if model is None:
            return rules, None, NO_MODEL_NOTE if candidates else None
        if not candidates:
            return [], None, None
        notes: List[str] = []
        try:
            self._free_models()
            self._log_gpu("the model for names loads")
            read: List[Tuple[NameCandidate, Optional[NameEvidence]]] = []
            for candidate in pick_for_model(
                candidates, self._parameters.max_candidates
            ):
                prompt = model_prompt(candidate)
                reply = retry_on_gpu_memory(
                    self._gpu, NAMES_STAGE,
                    lambda prompt=prompt: model.complete(  # type: ignore[misc]
                        SYSTEM_PROMPT, prompt, self._parameters.max_tokens
                    ),
                    notes.append,
                )
                read.append(
                    (candidate, parse_reply(transcript, candidate, reply))
                )
            found = gated_evidence(transcript, candidates, read)
            return found, model.model_name, notes[-1] if notes else None
        except Exception as exc:  # any model failure: rules, no auto
            drop_frames(exc)
            LOG.warning("The model for names failed: %s", exc)
            reason = str(exc).strip() or type(exc).__name__
            return rules, None, MODEL_FAILED_NOTE.format(reason=reason)
        finally:
            model.release()
            self._log_gpu("the model for names was released")

    def _log_gpu(self, moment: str) -> None:
        """Log the GPU memory at debug level, when there is a GPU."""
        if self._gpu is not None and LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("GPU memory when %s: %s", moment, self._gpu.usage())


class QueueNameSearch:
    """Search a finished session for names again, as a background job."""

    def __init__(
        self,
        sessions: SessionRepository,
        runner: JobRunner,
        find: FindSpeakerNames,
        locks: SessionLocks,
    ) -> None:
        """Wire the ports and the search the job runs."""
        self._sessions = sessions
        self._runner = runner
        self._find = find
        self._locks = locks

    def __call__(self, session_id: str) -> bool:
        """Queue the search of ``session_id``, which must be ``done``.

        Returns whether a job was queued: while a job of the session (a
        search, or the tail of its transcription, which searches too)
        is queued or running, no other is added. The job reports the
        ``names`` stage and its note (model not available, transcript
        changed) on the session's job status.
        """
        with self._locks.hold(session_id):
            session = require_session(self._sessions, session_id)
            require_state(session, SessionState.DONE)
            if job_active(self._runner, session_id):
                return False

            def job(progress: Progress) -> None:
                """Search this session's names."""
                self._find.after_transcription(session_id, progress)

            self._runner.submit(session_id, job)
            return True


class ShowSpeakerNames:
    """Return the names a search gave and suggested that still apply."""

    def __init__(
        self,
        sessions: SessionRepository,
        store: SpeakerNameStore,
        runner: JobRunner,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._store = store
        self._runner = runner

    def __call__(self, session_id: str) -> Tuple[Optional[SpeakerNames], bool]:
        """Return the record of ``session_id`` and whether a job runs.

        The record keeps only the automatic names the speakers still
        carry and the suggestions for voices still unnamed; ``None``
        when names were never searched. The flag is true while a job of
        the finished session (a search, or the tail of its
        transcription) is queued or running.
        """
        session = require_session(self._sessions, session_id)
        running = (
            session.state is SessionState.DONE
            and job_active(self._runner, session_id)
        )
        record = self._store.names(session_id)
        transcript = self._sessions.transcript(session_id)
        if record is None or transcript is None:
            return None, running
        return record.model_copy(update={
            "auto": valid_auto(transcript.speakers, record.auto),
            "suggestions": open_suggestions(
                transcript.speakers, record.suggestions
            ),
        }), running


class UndoAutoName:
    """Turn a name given automatically back into the voice's label."""

    def __init__(
        self,
        sessions: SessionRepository,
        store: SpeakerNameStore,
        locks: SessionLocks,
        unname: UnnameSpeaker,
    ) -> None:
        """Wire the ports and the review path that unnames."""
        self._sessions = sessions
        self._store = store
        self._locks = locks
        self._unname = unname

    def __call__(self, session_id: str, label: str) -> Transcript:
        """Remove the automatic name of ``label``; return the transcript.

        Only a name the search gave and the voice still carries can be
        undone (:class:`InvalidInputError` otherwise). The automatic
        name leaves the record; it comes back as a suggestion, and later
        searches of this transcript never give it to ``label``
        automatically again.
        """
        check_speaker_id(label)
        with self._locks.hold(session_id):
            transcript = require_transcript(self._sessions, session_id)
            record = self._store.names(session_id)
            if record is None:
                raise NotFoundError(f"No names were searched in {session_id}.")
            auto = valid_auto(transcript.speakers, record.auto)
            undone = next((a for a in auto if a.speaker == label), None)
            if undone is None:
                raise InvalidInputError(
                    f"{label} has no name given automatically."
                )
            edited = self._unname(session_id, label)
            run = require_session(self._sessions, session_id).run_id
            self._store.save_names(record.model_copy(update={
                "undone": kept_undone(record.undone, run) + [UndoneName(
                    speaker=label, name=undone.name, run=run,
                )],
                "auto": [a for a in record.auto if a.speaker != label],
                "suggestions": list(record.suggestions) + [
                    _as_suggestion(undone)
                ],
            }))
            return edited


def _as_suggestion(name: AutoName) -> NameSuggestion:
    """Return an undone automatic name as a suggestion."""
    return NameSuggestion(
        speaker=name.speaker, name=name.name, kind="self",
        quote=name.quote, utterance=name.utterance, start=name.start,
        confidence=name.confidence, reason="undone",
    )
