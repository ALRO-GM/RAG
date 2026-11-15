"""Use cases of the search for speakers' names (#102), with fakes."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from tests.unit.transcriber.fakes import (
    FakeJobRunner,
    FakeSessionLocks,
    FakeSessionRepository,
    FakeVoiceprintRepository,
    FixedClock,
    Line,
    RecordingProgress,
    Scene,
    voice,
)
from tests.unit.transcriber.names_fakes import (
    NOW,
    FakeChatModel,
    FakeNameStore,
    conversation,
    judged,
    marked,
    no_room,
    reply,
)
from tests.unit.transcriber.test_transcribe import world
from transcriber.application.review import (
    RenameSpeaker,
    ReassignUtterance,
    UnnameSpeaker,
)
from transcriber.application.speaker_names import (
    CHANGED_NOTE,
    NAMES_STAGE,
    NO_MODEL_NOTE,
    FindSpeakerNames,
    QueueNameSearch,
    ShowSpeakerNames,
    UndoAutoName,
)
from transcriber.domain.errors import (
    InvalidInputError,
    NotFoundError,
    StateError,
)
from transcriber.domain.models import Session, SessionState, Transcript

INTRODUCTIONS = [
    ("S1", "Buenos días, empezamos."),
    ("S2", "Hola, soy Ana, de compras."),
    ("S1", "Gracias, Ana."),
    ("S3", "Y yo me llamo Pedro."),
    ("S1", "Soy de Madrid, por cierto."),
]


def understands(user: str) -> str:
    """A model that gets introductions right and refuses look-alikes."""
    text, word = marked(user), judged(user)
    for trigger in ("soy", "me llamo"):
        where = text.lower().find(f"{trigger} {word.lower()}")
        if where >= 0:
            quote = text[where:where + len(trigger) + len(word) + 1]
            return reply(word, "self", 0.95, quote)
    if word in text:
        return reply(word, "addressed", 0.7)
    return reply(None)


class Kit:
    """A finished session ``s1`` and the use cases around it."""

    def __init__(
        self,
        lines: Sequence[Tuple[str, str]] = INTRODUCTIONS,
        model: Optional[FakeChatModel] = None,
        names: Optional[Dict[str, str]] = None,
        enrolled: Sequence[str] = (),
        no_model: bool = False,
    ) -> None:
        """Store the transcript of ``lines`` and wire the use cases."""
        self.locks = FakeSessionLocks()
        self.sessions = FakeSessionRepository(locks=self.locks)
        transcript = conversation(lines, names, enrolled)
        self.sessions.sessions["s1"] = transcript.session
        self.sessions.transcripts["s1"] = transcript
        self.voiceprints = FakeVoiceprintRepository()
        self.store = FakeNameStore()
        self.runner = FakeJobRunner()
        self.model = None if no_model else (
            model or FakeChatModel(understands)
        )
        self.rename = RenameSpeaker(
            self.sessions, self.voiceprints, self.locks
        )
        self.unname = UnnameSpeaker(
            self.sessions, self.voiceprints, self.locks
        )
        self.freed: List[str] = []
        self.ticks = iter(range(0, 1000, 2))
        self.find = FindSpeakerNames(
            self.sessions, self.store, self.locks, self.rename, self.unname,
            FixedClock(NOW), self.model,
            free_models=lambda: self.freed.append("speech"),
            timer=lambda: float(next(self.ticks)),
        )

    def speakers(self) -> List[Tuple[str, Optional[str]]]:
        """``(id, name)`` of the stored transcript's speakers."""
        return [(s.id, s.name) for s in self.transcript().speakers]

    def transcript(self) -> Transcript:
        """The stored transcript."""
        stored = self.sessions.transcript("s1")
        assert stored is not None
        return stored


class TestFindSpeakerNames:
    def test_clear_introductions_name_the_voices(self) -> None:
        kit = Kit()
        record = kit.find("s1")
        assert kit.speakers() == [
            ("S1", None), ("S2", "Ana"), ("S3", "Pedro"),
        ]
        assert [(a.speaker, a.name, a.quote) for a in record.auto] == [
            ("S2", "Ana", "soy Ana"), ("S3", "Pedro", "me llamo Pedro"),
        ]
        assert record.suggestions == []
        assert (record.model, record.candidates, record.seconds) == (
            "fake-qwen", 3, 2.0
        )
        assert record.searched_at == NOW and record.note is None
        assert kit.store.names("s1") == record
        assert kit.freed == ["speech"] and kit.model is not None
        assert kit.model.released == 1
        assert kit.sessions.unlocked_writes == []

    def test_the_model_reads_only_the_candidates(self) -> None:
        kit = Kit()
        kit.find("s1")
        assert kit.model is not None
        assert len(kit.model.prompts) == 3
        assert all("Soy de Madrid" not in marked(p) for p in kit.model.prompts)

    def test_without_candidates_the_model_is_not_loaded(self) -> None:
        kit = Kit(lines=[("S1", "Hola."), ("S2", "Adiós.")])
        record = kit.find("s1")
        assert kit.model is not None and kit.model.prompts == []
        assert kit.freed == [] and record.candidates == 0
        assert (record.model, record.auto, record.note) == (None, [], None)

    def test_a_model_that_cannot_run_leaves_suggestions_only(self) -> None:
        kit = Kit(model=FakeChatModel(fail=no_room()))
        record = kit.find("s1")
        assert kit.speakers() == [("S1", None), ("S2", None), ("S3", None)]
        assert record.model is None and record.auto == []
        assert record.note is not None and "could not run" in record.note
        assert "6.7 GB" in record.note
        assert {s.reason for s in record.suggestions} >= {"rules_only"}
        assert kit.model is not None and kit.model.released == 1

    def test_no_model_configured(self) -> None:
        kit = Kit(no_model=True)
        record = kit.find("s1")
        assert record.note == NO_MODEL_NOTE and record.auto == []
        assert {(s.speaker, s.name) for s in record.suggestions} >= {
            ("S2", "Ana"), ("S3", "Pedro"),
        }

    def test_hand_names_and_enrolled_people_are_never_overwritten(
        self,
    ) -> None:
        lines = [("S1", "Soy Ana."), ("luis", "Soy Pedro."),
                 ("S2", "Me llamo Marta.")]
        kit = Kit(lines=lines, names={"S1": "Anita", "luis": "Luis"},
                  enrolled=["luis"])
        record = kit.find("s1")
        assert kit.speakers() == [
            ("S1", "Anita"), ("luis", "Luis"), ("S2", "Marta"),
        ]
        assert [a.speaker for a in record.auto] == ["S2"]

    def test_two_people_with_the_same_name_are_only_suggested(self) -> None:
        kit = Kit(lines=[("S1", "Soy Ana."), ("S2", "Yo también soy Ana.")])
        record = kit.find("s1")
        assert kit.speakers() == [("S1", None), ("S2", None)]
        assert sorted((s.speaker, s.reason) for s in record.suggestions) == [
            ("S1", "same_name"), ("S2", "same_name"),
        ]

    def test_a_rerun_keeps_or_clears_its_own_names(self) -> None:
        answers = {"on": True}

        def flaky(user: str) -> str:
            return understands(user) if answers["on"] else reply(None)

        kit = Kit(model=FakeChatModel(flaky))
        kit.find("s1")
        again = kit.find("s1")
        assert [a.speaker for a in again.auto] == ["S2", "S3"]
        answers["on"] = False
        kit.find("s1")
        assert kit.speakers() == [("S1", None), ("S2", None), ("S3", None)]

    def test_a_rerun_without_the_model_keeps_earlier_names(self) -> None:
        kit = Kit()
        kit.find("s1")
        kit.find._model = FakeChatModel(fail=no_room())
        record = kit.find("s1")
        assert [a.speaker for a in record.auto] == ["S2", "S3"]
        assert kit.speakers()[1:] == [("S2", "Ana"), ("S3", "Pedro")]

    def test_a_transcript_changed_meanwhile_is_not_named(self) -> None:
        kit = Kit()
        reassign = ReassignUtterance(kit.sessions, kit.voiceprints, kit.locks)

        def meddling(user: str) -> str:
            if not kit.locks.held and kit.transcript().utterances[0].speaker \
                    == "S1":
                reassign("s1", kit.transcript().utterances[0].id, "S2")
            return understands(user)

        kit.model = FakeChatModel(meddling)
        kit.find._model = kit.model
        record = kit.find("s1")
        assert record.note == CHANGED_NOTE and record.auto == []
        assert all(name is None for _, name in kit.speakers())

    def test_the_model_runs_without_the_session_lock(self) -> None:
        kit = Kit()
        held: List[bool] = []

        def watch(user: str) -> str:
            held.append(kit.locks.holds("s1"))
            return understands(user)

        kit.find._model = FakeChatModel(watch)
        kit.find("s1")
        assert held and not any(held)

    def test_only_finished_sessions(self) -> None:
        kit = Kit()
        session = kit.sessions.sessions["s1"]
        kit.sessions.sessions["s1"] = session.model_copy(
            update={"state": SessionState.RUNNING}
        )
        with pytest.raises(StateError):
            kit.find("s1")
        with pytest.raises(NotFoundError):
            kit.find("nope")


class TestAfterTranscription:
    def test_reports_the_stage_and_the_note(self) -> None:
        kit = Kit(model=FakeChatModel(fail=no_room()))
        progress = RecordingProgress()
        kit.find.after_transcription("s1", progress)
        assert progress.reports == [(NAMES_STAGE, 1.0)]
        assert len(progress.notes) == 1
        assert "could not run" in progress.notes[0]

    def test_never_raises_and_empties_the_record(self) -> None:
        kit = Kit()
        kit.find("s1")
        kit.sessions.transcripts.pop("s1")
        progress = RecordingProgress()
        kit.find.after_transcription("s1", progress)
        assert progress.notes[0].startswith("Names were not searched:")
        record = kit.store.names("s1")
        assert record is not None and record.auto == []
        assert record.note == progress.notes[0]

    def test_a_store_that_fails_too_is_only_logged(self) -> None:
        kit = Kit()

        def broken(names: object) -> None:
            raise OSError("disk full")

        kit.store.save_names = broken  # type: ignore[method-assign]
        progress = RecordingProgress()
        kit.find.after_transcription("s1", progress)
        assert "disk full" in progress.notes[0]

    def test_a_clean_run_adds_no_note(self) -> None:
        kit = Kit()
        progress = RecordingProgress()
        kit.find.after_transcription("s1", progress)
        assert progress.notes == []
        assert kit.speakers()[1] == ("S2", "Ana")


class TestTranscriptionJob:
    def scene(self) -> Scene:
        """Two people who introduce themselves."""
        return Scene(
            lines=[
                Line("ana", 0.0, 6.0, "Hola a todos. Soy Ana y llevo"
                     " el proyecto."),
                Line("luis", 7.0, 13.0, "Encantado. Me llamo Luis y hago"
                     " las pruebas."),
            ],
            voices={"ana": voice(0), "luis": voice(1)},
        )

    def test_the_job_ends_with_the_search(self) -> None:
        w = world(self.scene())
        store = FakeNameStore()
        find = FindSpeakerNames(
            w.sessions, store, w.locks,
            RenameSpeaker(w.sessions, w.voiceprints, w.locks),
            UnnameSpeaker(w.sessions, w.voiceprints, w.locks),
            FixedClock(NOW), FakeChatModel(understands),
        )
        w.queue._after = find.after_transcription
        transcript = w.run()
        names = {s.name for s in transcript.speakers}
        assert names == {"Ana", "Luis"}
        assert w.runner.statuses["s1"].state is SessionState.DONE
        record = store.names("s1")
        assert record is not None and len(record.auto) == 2

    def test_a_failed_transcription_does_not_search(self) -> None:
        w = world(self.scene())
        calls: List[str] = []
        w.queue._after = lambda session_id, progress: calls.append(
            session_id
        )
        failing: Any = _boom
        w.recogniser.recognise = failing  # type: ignore[method-assign]
        w.upload()
        w.queue("s1")
        w.runner.run_all()
        assert calls == []


def _boom(*args: object, **kwargs: object) -> None:
    """A recogniser that fails."""
    raise RuntimeError("no speech model")


class TestQueueAndShow:
    def test_queue_runs_the_search_as_a_job(self) -> None:
        kit = Kit()
        queue = QueueNameSearch(kit.sessions, kit.runner, kit.find)
        show = ShowSpeakerNames(kit.sessions, kit.store, kit.runner)
        queue("s1")
        record, running = show("s1")
        assert record is None and running
        kit.runner.run_all()
        record, running = show("s1")
        assert record is not None and not running
        assert [a.name for a in record.auto] == ["Ana", "Pedro"]

    def test_only_finished_sessions_are_queued(self) -> None:
        kit = Kit()
        kit.sessions.sessions["s1"] = kit.sessions.sessions["s1"].model_copy(
            update={"state": SessionState.QUEUED}
        )
        with pytest.raises(StateError):
            QueueNameSearch(kit.sessions, kit.runner, kit.find)("s1")

    def test_show_drops_what_no_longer_applies(self) -> None:
        kit = Kit(lines=INTRODUCTIONS + [("S4", "Gracias, Marta.")])
        kit.find("s1")
        show = ShowSpeakerNames(kit.sessions, kit.store, kit.runner)
        record, _ = show("s1")
        assert record is not None
        assert [s.speaker for s in record.suggestions] == ["S1"]
        kit.rename("s1", "S2", name="Anita")
        kit.rename("s1", "S3", name="Pedro")
        record, _ = show("s1")
        assert record is not None
        assert [a.speaker for a in record.auto] == ["S3"]

    def test_show_without_a_transcript_or_a_search(self) -> None:
        kit = Kit()
        show = ShowSpeakerNames(kit.sessions, kit.store, kit.runner)
        assert show("s1") == (None, False)
        kit.sessions.sessions["s2"] = Session(id="s2", language="es")
        assert show("s2") == (None, False)


class TestUndo:
    def test_undo_gives_the_label_back_and_keeps_a_suggestion(self) -> None:
        kit = Kit()
        kit.find("s1")
        undo = UndoAutoName(kit.sessions, kit.store, kit.locks, kit.unname)
        out = undo("s1", "S2")
        assert [(s.id, s.name) for s in out.speakers][1] == ("S2", None)
        record = kit.store.names("s1")
        assert record is not None
        assert [a.speaker for a in record.auto] == ["S3"]
        assert [(s.speaker, s.reason) for s in record.suggestions] == [
            ("S2", "undone")
        ]
        assert kit.sessions.unlocked_writes == []

    def test_only_automatic_names_can_be_undone(self) -> None:
        kit = Kit()
        undo = UndoAutoName(kit.sessions, kit.store, kit.locks, kit.unname)
        with pytest.raises(NotFoundError):
            undo("s1", "S2")
        kit.find("s1")
        with pytest.raises(InvalidInputError):
            undo("s1", "S1")
        kit.rename("s1", "S2", name="Anita")
        with pytest.raises(InvalidInputError):
            undo("s1", "S2")
        with pytest.raises(InvalidInputError):
            undo("s1", "../x")


class TestUnnameSpeaker:
    def test_unname(self) -> None:
        kit = Kit(names={"S2": "Ana"}, lines=INTRODUCTIONS[:3])
        out = kit.unname("s1", "S2")
        assert [(s.id, s.name) for s in out.speakers] == [
            ("S1", None), ("S2", None)
        ]
        assert out.utterances == conversation(INTRODUCTIONS[:3]).utterances

    def test_unnamed_and_enrolled_and_unknown(self) -> None:
        kit = Kit(lines=[("S1", "Hola."), ("ana", "Hola.")], enrolled=["ana"],
                  names={"ana": "Ana"})
        before = kit.transcript()
        assert kit.unname("s1", "S1") == before
        with pytest.raises(InvalidInputError):
            kit.unname("s1", "ana")
        with pytest.raises(NotFoundError):
            kit.unname("s1", "S7")
