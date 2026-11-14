"""Review: correct who said what in a finished transcript.

:class:`RenameSpeaker` gives an anonymous voice a display name or
merges it into a known speaker; :class:`ReassignUtterance` moves one
sentence to another speaker. Both recompute the turns and the previous
speaker of the whole transcript and keep the stored words in step.
Neither creates a voiceprint: a voice is only enrolled with consent.
Both read and write the transcript while holding the session's lock.
"""

from typing import Dict, List, Optional, Sequence

from transcriber.application.common import (
    check_speaker_id,
    require_session,
    require_transcript,
)
from transcriber.application.ports import (
    SessionLocks,
    SessionRepository,
    VoiceprintRepository,
)
from transcriber.domain.errors import InvalidInputError, NotFoundError
from transcriber.domain.models import (
    SessionState,
    Speaker,
    Transcript,
    Utterance,
    Word,
)
from transcriber.domain.sentences import regroup_utterances
from transcriber.domain.sessions import require_state


class _Review:
    """What both review use cases share: lookups and saving an edit."""

    def __init__(
        self,
        sessions: SessionRepository,
        voiceprints: VoiceprintRepository,
        locks: SessionLocks,
    ) -> None:
        """Wire the ports."""
        self._sessions = sessions
        self._voiceprints = voiceprints
        self._locks = locks

    def _transcript(self, session_id: str) -> Transcript:
        """Return the transcript of a ``done`` session."""
        session = require_session(self._sessions, session_id)
        require_state(session, SessionState.DONE)
        return require_transcript(self._sessions, session_id)

    def _target(self, transcript: Transcript, speaker_id: str) -> Speaker:
        """Return the speaker ``speaker_id`` an edit may move speech to.

        A speaker of the transcript, or an enrolled person (named by
        their consent record) who was not recognised in it.
        """
        check_speaker_id(speaker_id)
        for speaker in transcript.speakers:
            if speaker.id == speaker_id:
                return speaker
        consent = self._voiceprints.consent(speaker_id)
        if consent is None:
            raise NotFoundError(
                f"{speaker_id} is neither a speaker of this session nor"
                " an enrolled person."
            )
        return Speaker(id=speaker_id, name=consent.name, enrolled=True)

    def _save(
        self,
        transcript: Transcript,
        utterances: Sequence[Utterance],
        known: Sequence[Speaker],
        moved: Dict[tuple[str, float, float], str],
        relabel: Optional[Dict[str, str]] = None,
    ) -> Transcript:
        """Store the edited transcript and the words in step with it.

        ``utterances`` are in transcript order with their new speakers;
        turns and previous speakers are recomputed. Speakers are listed
        in order of first appearance, from ``known``; one left without
        speech is dropped. Stored words named in ``moved`` (by text,
        start and end) or whose speaker is in ``relabel`` change
        speaker.
        """
        regrouped = regroup_utterances(utterances)
        by_id = {s.id: s for s in known}
        speakers = [by_id[i] for i in dict.fromkeys(
            u.speaker for u in regrouped
        )]
        edited = Transcript(
            session=transcript.session, speakers=speakers,
            utterances=regrouped, pipeline=transcript.pipeline,
        )
        session_id = transcript.session.id
        words: List[Word] = []
        for word in self._sessions.words(session_id):
            key = (word.text, word.start, word.end)
            target = moved.get(key) or (relabel or {}).get(word.speaker or "")
            words.append(
                word.model_copy(update={"speaker": target}) if target
                else word
            )
        self._sessions.save_transcript(edited)
        self._sessions.save_words(session_id, words)
        return edited


class RenameSpeaker(_Review):
    """Name an anonymous voice, or merge it into a known speaker."""

    def __call__(
        self,
        session_id: str,
        label: str,
        name: Optional[str] = None,
        merge_into: Optional[str] = None,
    ) -> Transcript:
        """Give anonymous ``label`` a display ``name`` or ``merge_into``.

        Exactly one of the two is given. ``name`` only sets the display
        name: the id stays the anonymous label. ``merge_into`` is a
        speaker of the transcript or an enrolled person; every sentence
        of ``label`` moves to them, with no speaker confidence (the
        attribution is by hand). Enrolled ids cannot be renamed here.
        """
        if (name is None) == (merge_into is None):
            raise InvalidInputError("Give either a name or a speaker id.")
        check_speaker_id(label)
        with self._locks.hold(session_id):
            return self._rename(session_id, label, name, merge_into)

    def _rename(
        self,
        session_id: str,
        label: str,
        name: Optional[str],
        merge_into: Optional[str],
    ) -> Transcript:
        """Apply the rename or merge; the caller holds the lock."""
        transcript = self._transcript(session_id)
        source = next(
            (s for s in transcript.speakers if s.id == label), None
        )
        if source is None:
            raise NotFoundError(f"No speaker {label} in {session_id}.")
        if source.enrolled:
            raise InvalidInputError(
                f"{label} is an enrolled person; only anonymous voices"
                " are renamed."
            )
        known = list(transcript.speakers)
        if merge_into is None:
            clean = " ".join((name or "").split())
            if not clean:
                raise InvalidInputError("The name cannot be empty.")
            known = [
                s.model_copy(update={"name": clean}) if s.id == label else s
                for s in known
            ]
            return self._save(transcript, transcript.utterances, known, {})
        if merge_into == label:
            raise InvalidInputError("A speaker cannot merge into itself.")
        target = self._target(transcript, merge_into)
        utterances = [
            u.model_copy(update={
                "speaker": target.id, "speaker_confidence": None,
            }) if u.speaker == label else u
            for u in transcript.utterances
        ]
        return self._save(
            transcript, utterances, known + [target], {},
            relabel={label: target.id},
        )


class ReassignUtterance(_Review):
    """Move one sentence to another speaker."""

    def __call__(
        self, session_id: str, utterance_id: str, speaker_id: str
    ) -> Transcript:
        """Give utterance ``utterance_id`` to ``speaker_id``.

        The speaker is one of the transcript or an enrolled person. The
        sentence loses its speaker confidence (the attribution is by
        hand); its turn splits or joins its neighbours as needed, so
        utterance ids after it may change.
        """
        check_speaker_id(speaker_id)
        with self._locks.hold(session_id):
            return self._reassign(session_id, utterance_id, speaker_id)

    def _reassign(
        self, session_id: str, utterance_id: str, speaker_id: str
    ) -> Transcript:
        """Move the utterance; the caller holds the lock."""
        transcript = self._transcript(session_id)
        current = next(
            (u for u in transcript.utterances if u.id == utterance_id), None
        )
        if current is None:
            raise NotFoundError(
                f"No utterance {utterance_id} in {session_id}."
            )
        if current.speaker == speaker_id:
            return transcript
        target = self._target(transcript, speaker_id)
        utterances = [
            u.model_copy(update={
                "speaker": target.id, "speaker_confidence": None,
            }) if u.id == utterance_id else u
            for u in transcript.utterances
        ]
        moved = {(w.text, w.start, w.end): target.id for w in current.words}
        return self._save(
            transcript, utterances, list(transcript.speakers) + [target],
            moved,
        )
