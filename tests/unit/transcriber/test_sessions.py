"""Transcriber sessions: the state table, chunks, regrouping, voices."""

from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from transcriber.domain.errors import InvalidInputError, StateError
from transcriber.domain.models import (
    Session,
    SessionState,
    Turn,
    Utterance,
    Word,
)
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.sentences import build_utterances, regroup_utterances
from transcriber.domain.sessions import (
    INTERRUPTED,
    TRANSITIONS,
    ChunkAction,
    accept_chunk,
    can_delete,
    can_move,
    interrupt,
    move,
    require_state,
    with_chunk,
)
from transcriber.domain.voices import (
    FloatArray,
    centre_on_session,
    group_voices,
    is_single_voice,
    unit_vectors,
)

S = SessionState
DIGEST = "a" * 64
OTHER = "b" * 64


def session(state: SessionState = S.RECORDING, chunks: int = 0) -> Session:
    """A session in ``state`` with ``chunks`` accepted."""
    return Session(id="s1", language="es", state=state, chunks=chunks,
                   last_chunk_sha256=DIGEST if chunks else None)


class TestTransitions:
    def test_the_ticket_order_and_the_retry(self) -> None:
        path = [S.RECORDING, S.UPLOADED, S.QUEUED, S.RUNNING, S.DONE]
        for current, target in zip(path, path[1:]):
            assert can_move(current, target)
        assert can_move(S.RUNNING, S.FAILED)
        assert can_move(S.QUEUED, S.FAILED)
        assert can_move(S.FAILED, S.QUEUED)
        assert can_move(S.DONE, S.QUEUED)

    def test_the_table_is_complete_and_nothing_else_moves(self) -> None:
        assert set(TRANSITIONS) == set(SessionState)
        allowed = {(a, b) for a, targets in TRANSITIONS.items()
                   for b in targets}
        assert allowed == {
            (S.RECORDING, S.UPLOADED), (S.UPLOADED, S.QUEUED),
            (S.QUEUED, S.RUNNING), (S.QUEUED, S.FAILED),
            (S.RUNNING, S.DONE), (S.RUNNING, S.FAILED),
            (S.FAILED, S.QUEUED), (S.DONE, S.QUEUED),
        }

    @pytest.mark.parametrize("current, target", [
        (S.RECORDING, S.QUEUED), (S.UPLOADED, S.RUNNING),
        (S.DONE, S.FAILED), (S.DONE, S.RUNNING), (S.FAILED, S.RUNNING),
        (S.RUNNING, S.RUNNING), (S.RECORDING, S.DONE),
    ])
    def test_invalid_moves_are_errors(
        self, current: SessionState, target: SessionState
    ) -> None:
        with pytest.raises(StateError, match=target.value):
            move(session(current), target)

    def test_a_finished_session_can_be_queued_again(self) -> None:
        again = move(session(S.DONE), S.QUEUED)
        assert (again.state, again.failure) == (S.QUEUED, None)

    @pytest.mark.parametrize("state", [S.QUEUED, S.RUNNING])
    def test_an_interrupted_job_fails_its_session(
        self, state: SessionState
    ) -> None:
        failed = interrupt(session(state))
        assert (failed.state, failed.failure) == (S.FAILED, INTERRUPTED)
        assert can_move(failed.state, S.QUEUED)

    @pytest.mark.parametrize("state", [
        S.RECORDING, S.UPLOADED, S.DONE, S.FAILED,
    ])
    def test_interrupt_leaves_other_states_alone(
        self, state: SessionState
    ) -> None:
        assert interrupt(session(state)) == session(state)

    @pytest.mark.parametrize("state", list(SessionState))
    def test_every_state_but_running_can_be_deleted(
        self, state: SessionState
    ) -> None:
        assert can_delete(session(state)) is (state is not S.RUNNING)

    def test_failure_reason_rules(self) -> None:
        failed = move(session(S.RUNNING), S.FAILED, "recognise: boom")
        assert failed.state is S.FAILED
        assert failed.failure == "recognise: boom"
        assert move(failed, S.QUEUED).failure is None
        with pytest.raises(InvalidInputError, match="reason"):
            move(session(S.RUNNING), S.FAILED)
        with pytest.raises(InvalidInputError, match="reason"):
            move(session(S.RUNNING), S.FAILED, "  ")
        with pytest.raises(InvalidInputError, match="Only a failed"):
            move(session(S.RUNNING), S.DONE, "why")

    def test_require_state(self) -> None:
        require_state(session(S.DONE), S.DONE)
        with pytest.raises(StateError, match="uploaded or failed"):
            require_state(session(S.DONE), S.UPLOADED, S.FAILED)

    def test_state_names_are_the_tickets(self) -> None:
        assert [s.value for s in SessionState] == [
            "recording", "uploaded", "queued", "running", "done", "failed",
        ]

    def test_session_chunk_fields_are_validated(self) -> None:
        with pytest.raises(ValidationError):
            Session(id="s", language="es", chunks=-1)
        with pytest.raises(ValidationError):
            Session(id="s", language="es", last_chunk_sha256="nope")


class TestChunks:
    def test_next_chunk_is_appended(self) -> None:
        assert accept_chunk(session(), 0, DIGEST) is ChunkAction.APPEND
        after = with_chunk(session(), DIGEST)
        assert (after.chunks, after.last_chunk_sha256) == (1, DIGEST)
        assert accept_chunk(after, 1, OTHER) is ChunkAction.APPEND

    def test_the_last_chunk_repeated_is_a_duplicate(self) -> None:
        assert accept_chunk(session(chunks=3), 2, DIGEST) is (
            ChunkAction.DUPLICATE
        )

    def test_the_last_number_with_other_bytes_is_an_error(self) -> None:
        with pytest.raises(InvalidInputError, match="other content"):
            accept_chunk(session(chunks=3), 2, OTHER)

    def test_an_earlier_chunk_is_an_error(self) -> None:
        with pytest.raises(InvalidInputError, match="next chunk is 3"):
            accept_chunk(session(chunks=3), 1, DIGEST)

    def test_a_gap_is_an_error(self) -> None:
        with pytest.raises(InvalidInputError, match="gap"):
            accept_chunk(session(chunks=3), 4, DIGEST)
        with pytest.raises(InvalidInputError, match="gap"):
            accept_chunk(session(), 1, DIGEST)

    def test_numbers_start_at_zero(self) -> None:
        with pytest.raises(InvalidInputError, match="from 0"):
            accept_chunk(session(), -1, DIGEST)

    @pytest.mark.parametrize("state", [
        S.UPLOADED, S.QUEUED, S.RUNNING, S.DONE, S.FAILED,
    ])
    def test_only_while_recording(self, state: SessionState) -> None:
        with pytest.raises(StateError, match="recording"):
            accept_chunk(session(state, chunks=1), 1, DIGEST)


def turn(speaker: str, text: str, start: float, end: float) -> Turn:
    """A turn with one word per token, spread over its time."""
    tokens = text.split()
    size = (end - start) / len(tokens)
    return Turn(speaker=speaker, text=text, start=start, end=end, words=[
        Word(text=t, start=start + n * size, end=start + (n + 1) * size,
             speaker=speaker)
        for n, t in enumerate(tokens)
    ], speaker_confidence=0.8)


def three_turns() -> List[Utterance]:
    """a: two sentences, b: one, a: one."""
    return build_utterances([
        turn("a", "Uno. Dos.", 0.0, 2.0), turn("b", "Tres.", 2.0, 3.0),
        turn("a", "Cuatro.", 3.0, 4.0),
    ])


class TestRegroupUtterances:
    def test_unchanged_speakers_give_the_same_utterances(self) -> None:
        original = three_turns()
        assert regroup_utterances(original) == original

    def test_moving_the_middle_turn_joins_three_turns(self) -> None:
        moved = [
            u.model_copy(update={"speaker": "a"}) if u.speaker == "b" else u
            for u in three_turns()
        ]
        out = regroup_utterances(moved)
        assert [u.id for u in out] == ["t0.s0", "t0.s1", "t0.s2", "t0.s3"]
        assert {u.previous_speaker for u in out} == {None}
        assert [w.speaker for w in out[2].words] == ["a"]
        assert [u.text for u in out] == ["Uno.", "Dos.", "Tres.", "Cuatro."]
        assert out[2].start == 2.0 and out[2].end == 3.0

    def test_moving_one_sentence_splits_its_turn(self) -> None:
        moved = [
            u.model_copy(update={"speaker": "c"}) if u.id == "t0.s1" else u
            for u in three_turns()
        ]
        out = regroup_utterances(moved)
        assert [(u.id, u.speaker, u.previous_speaker, u.previous_utterance)
                for u in out] == [
            ("t0.s0", "a", None, None),
            ("t1.s0", "c", "a", "t0.s0"),
            ("t2.s0", "b", "c", "t1.s0"),
            ("t3.s0", "a", "b", "t2.s0"),
        ]

    def test_empty(self) -> None:
        assert regroup_utterances([]) == []


class TestGroupVoices:
    def two_voices(self) -> FloatArray:
        """Four long windows: a, b, a, b around a shared component."""
        a = np.array([10.0, 1.0, 0.0], dtype=np.float32)
        b = np.array([10.0, -1.0, 0.0], dtype=np.float32)
        return np.stack([a, b, a, b]).astype(np.float32)

    def test_two_passes(self) -> None:
        coarse = self.two_voices()
        fine = np.stack([coarse[1], coarse[0], coarse[1]]).astype(np.float32)
        groups = group_voices(coarse, fine)
        assert groups.coarse == [0, 1, 0, 1]
        assert groups.fine == [1, 0, 1]
        assert groups.vectors.shape == (2, 3)
        # Group vectors are raw unit means, not centred ones.
        unit = coarse[0] / np.linalg.norm(coarse[0])
        assert np.allclose(groups.vectors[0], unit, atol=1e-6)

    def test_without_centring_the_shared_part_merges_them(self) -> None:
        groups = group_voices(self.two_voices(),
                              np.zeros((0, 3), dtype=np.float32),
                              centre_windows=False)
        assert groups.coarse == [0, 0, 0, 0]
        assert groups.fine == []

    def test_known_number_of_speakers(self) -> None:
        groups = group_voices(self.two_voices(),
                              np.zeros((0, 0), dtype=np.float32),
                              num_speakers=1)
        assert groups.coarse == [0, 0, 0, 0]

    def one_voice(self) -> FloatArray:
        """Six long windows of one voice, each a little different."""
        rng = np.random.default_rng(3)
        base = np.array([1.0, 0.5, 0.2, 0.0], dtype=np.float32)
        noisy: FloatArray = (
            base + rng.normal(0.0, 0.03, (6, 4))
        ).astype(np.float32)
        return noisy

    def test_centring_one_voice_splits_it(self) -> None:
        """Why the single-voice test exists: the bug it prevents."""
        groups = group_voices(self.one_voice(),
                              np.zeros((0, 4), dtype=np.float32))
        assert len(set(groups.coarse)) > 1

    def test_one_voice_is_one_group(self) -> None:
        coarse = self.one_voice()
        fine = coarse[:3].copy()
        groups = group_voices(coarse, fine, single_voice_threshold=0.9)
        assert groups.coarse == [0] * 6
        assert groups.fine == [0] * 3
        assert groups.vectors.shape == (1, 4)
        assert np.allclose(np.linalg.norm(groups.vectors[0]), 1.0)

    def test_the_single_voice_test_needs_every_window(self) -> None:
        # Each window has cosine 10 / sqrt(101), about 0.995, with the
        # mean direction: one voice at 0.99, two voices at 0.999.
        two = self.two_voices()
        assert is_single_voice(two, 0.99)
        assert not is_single_voice(two, 0.999)
        groups = group_voices(two, np.zeros((0, 3), dtype=np.float32),
                              single_voice_threshold=0.999)
        assert groups.coarse == [0, 1, 0, 1]

    def test_a_known_number_overrides_the_single_voice_test(self) -> None:
        groups = group_voices(self.one_voice(),
                              np.zeros((0, 4), dtype=np.float32),
                              num_speakers=2, single_voice_threshold=0.9)
        assert sorted(set(groups.coarse)) == [0, 1]

    def test_one_known_speaker_is_one_group_without_clustering(
        self,
    ) -> None:
        groups = group_voices(self.two_voices(),
                              self.two_voices()[:2],
                              num_speakers=1)
        assert groups.coarse == [0] * 4 and groups.fine == [0, 0]
        with pytest.raises(InvalidInputError, match="at least 1"):
            group_voices(self.two_voices(), self.two_voices(),
                         num_speakers=0)

    def test_single_voice_degenerate_inputs(self) -> None:
        assert not is_single_voice(np.zeros((0, 3), np.float32), 0.0)
        opposite = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        assert not is_single_voice(opposite, -1.0)

    def test_no_windows(self) -> None:
        none = np.zeros((0, 0), dtype=np.float32)
        groups = group_voices(none, none)
        assert groups.coarse == [] and groups.fine == []
        assert groups.vectors.shape[0] == 0

    def test_invalid_shapes(self) -> None:
        with pytest.raises(InvalidInputError, match="2-D"):
            group_voices(np.zeros(3, dtype=np.float32),
                         np.zeros((0, 3), dtype=np.float32))
        with pytest.raises(InvalidInputError, match="differ"):
            group_voices(self.two_voices(), np.zeros((1, 2), np.float32))


class TestTranscriptionParameters:
    def test_defaults_and_choices(self) -> None:
        params = TranscriptionParameters()
        assert params.coarse() == (5.0, 2.5, 0.4)
        assert params.fine() == (1.5, 0.75, 0.4)
        assert params.normaliser() is unit_vectors
        assert params.single_voice_threshold == 0.9
        assert (params.activity_frame, params.activity_threshold) == (
            0.025, 0.003,
        )
        centred = TranscriptionParameters(normalisation="centre_on_session")
        assert centred.normaliser() is centre_on_session

    def test_invalid_values(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptionParameters(window=0.0)
        with pytest.raises(ValidationError, match="window_minimum"):
            TranscriptionParameters(fine_window=0.3, window_minimum=0.4)
        with pytest.raises(ValidationError):
            TranscriptionParameters(normalisation="other")  # type: ignore
        with pytest.raises(ValidationError):
            TranscriptionParameters(single_voice_threshold=1.5)
        with pytest.raises(ValidationError):
            TranscriptionParameters(activity_frame=0.0)
        with pytest.raises(ValidationError):
            TranscriptionParameters(activity_threshold=-0.1)
