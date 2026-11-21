"""Mono live diarization rules: frames, picks, revisions, naming (LV-6)."""

from datetime import datetime, timezone
from typing import List

import numpy as np
import pytest

from transcriber.domain.errors import InvalidInputError
from transcriber.domain.live import LiveUtterance
from transcriber.domain.live_diarization import (
    MonoDiarizationSettings,
    MonoSpeakers,
    NamingPlan,
    SlotPick,
    SpeakerFrames,
    name_slots,
    pick_slot,
    slot_labels,
    unit_sum,
)
from transcriber.domain.models import Voiceprint

A = [0.9, 0.0, 0.0, 0.0]
B = [0.0, 0.9, 0.0, 0.0]
AB = [0.9, 0.8, 0.0, 0.0]
QUIET = [0.0, 0.0, 0.0, 0.0]


def frames(*rows: List[float]) -> SpeakerFrames:
    f = SpeakerFrames(0.08)
    f.extend(0, list(rows))
    return f


def sentence(ident: str, start: float, end: float,
             speaker: str = "S1") -> LiveUtterance:
    return LiveUtterance(
        id=ident, turn=0, index=int(ident[-1]), speaker=speaker,
        start=start, end=end, text="hola.", revision=3,
    )


def voiceprint(ident: str, vector: List[float]) -> Voiceprint:
    return Voiceprint(
        speaker_id=ident, model="m", vector=vector, consent_version="c1",
        enrolled_at=datetime(2026, 11, 21, tzinfo=timezone.utc),
    )


# ------------------------------------------------------------ frames

def test_frames_grow_in_order_and_know_their_coverage() -> None:
    f = SpeakerFrames(0.08)
    assert len(f) == 0 and f.covered == 0.0
    f.extend(0, [A, A])
    f.extend(2, [B])
    assert len(f) == 3 and f.covered == pytest.approx(0.24)
    assert np.allclose(f.rows(1, 10), [A, B])
    assert f.rows(5, 9).shape == (0, 4)


def test_frames_refuse_gaps_wrong_widths_and_bad_shapes() -> None:
    f = SpeakerFrames(0.08)
    with pytest.raises(InvalidInputError):
        f.extend(1, [A])
    with pytest.raises(InvalidInputError):
        f.extend(0, [[0.5, 0.5]])
    with pytest.raises(InvalidInputError):
        SpeakerFrames(0.0)
    with pytest.raises(InvalidInputError):
        SpeakerFrames(0.08, slots=5)


def test_window_takes_the_frames_whose_centres_are_inside() -> None:
    f = SpeakerFrames(0.08)
    assert f.window(0.0, 0.24) == (0, 3)
    assert f.window(0.05, 0.24) == (1, 3)
    # Shorter than a frame: the frame holding its middle.
    assert f.window(0.17, 0.19) == (2, 3)


# ------------------------------------------------------------ picks

def test_pick_takes_the_slot_that_talks_most() -> None:
    f = frames(A, A, B, A, QUIET)
    pick = pick_slot(f, 0.0, 0.4, 0.5)
    assert (pick.slot, pick.coverage) == (0, 1.0)
    assert pick.purity == pytest.approx(0.75)
    assert pick.complete


def test_pick_reports_missing_frames_and_overlap() -> None:
    f = frames(AB, AB, B)
    pick = pick_slot(f, 0.0, 0.48, 0.5)
    assert pick.slot == 1
    assert pick.coverage == pytest.approx(0.5) and not pick.complete
    assert pick.purity == pytest.approx(1 / 3)


def test_pick_without_speech_has_no_slot() -> None:
    assert pick_slot(frames(QUIET, QUIET), 0.0, 0.16, 0.5).slot is None
    empty = pick_slot(SpeakerFrames(0.08), 0.0, 0.16, 0.5)
    assert empty == SlotPick(None, 0.0, 0.0)


def test_slot_labels_use_given_ids_then_free_anonymous_ones() -> None:
    assert slot_labels([]) == ["S1", "S2", "S3", "S4"]
    assert slot_labels(["agente", "S2"], 4, {"S1"}) == [
        "agente", "S2", "S3", "S4",
    ]
    assert slot_labels(["a", "b", "c", "d", "e"]) == ["a", "b", "c", "d"]


# ------------------------------------------------------------ revisions

def test_a_provisional_final_is_revised_when_its_frames_arrive() -> None:
    speakers = MonoSpeakers(slot_labels([]))
    f = frames(A, A)
    u = sentence("t0.s0", 0.0, 0.48)
    pick = pick_slot(f, u.start, u.end, 0.5)
    speaker, provisional = speakers.speaker(pick, "S1")
    assert (speaker, provisional) == ("S1", True)
    sent = u.model_copy(update={"speaker": speaker,
                                "speaker_provisional": provisional})
    speakers.sent(sent, pick)
    assert speakers.last == "S1" and speakers.pending() == [sent]
    assert speakers.revise(f) == []
    f.extend(2, [B, B, B, B])
    revised = speakers.revise(f)
    assert [(r.speaker, r.speaker_provisional, r.revision)
            for r in revised] == [("S2", False, 4)]
    assert speakers.pending() == []
    assert speakers.sent_utterances() == revised


def test_completion_alone_is_a_revision() -> None:
    speakers = MonoSpeakers(slot_labels([]))
    f = frames(A)
    u = sentence("t0.s0", 0.0, 0.16)
    pick = pick_slot(f, u.start, u.end, 0.5)
    speakers.sent(u.model_copy(update={"speaker_provisional": True}), pick)
    f.extend(1, [A])
    [revised] = speakers.revise(f)
    assert revised.speaker == "S1" and not revised.speaker_provisional


def test_without_speech_the_fallback_speaker_stays() -> None:
    speakers = MonoSpeakers(["S1", "S2"])
    pick = pick_slot(frames(QUIET, QUIET), 0.0, 0.16, 0.5)
    assert speakers.speaker(pick, "S2") == ("S2", False)


def test_rename_revises_every_final_of_the_slot_and_back() -> None:
    speakers = MonoSpeakers(slot_labels([]))
    f = frames(A, A, B, B)
    first = sentence("t0.s0", 0.0, 0.16)
    second = sentence("t1.s0", 0.16, 0.32, "S2")
    for u in (first, second):
        speakers.sent(u, pick_slot(f, u.start, u.end, 0.5))
    speakers.sent(sentence("t1.s1", 0.0, 0.0),
                  SlotPick(None, 1.0, 0.0))
    named = speakers.rename({1: "lucia", 7: "nadie"})
    assert [(u.id, u.speaker, u.revision) for u in named] == [
        ("t1.s0", "lucia", 4),
    ]
    assert speakers.names == {1: "lucia"} and speakers.label(1) == "lucia"
    assert speakers.rename({1: "lucia"}) == []
    back = speakers.rename({})
    assert [(u.speaker, u.revision) for u in back] == [("S2", 5)]


def test_mono_speakers_need_distinct_labels() -> None:
    with pytest.raises(InvalidInputError):
        MonoSpeakers(["S1", "S1"])
    with pytest.raises(InvalidInputError):
        MonoSpeakers([])


# ------------------------------------------------------------ naming

def test_naming_plan_is_due_at_growing_thresholds() -> None:
    plan = NamingPlan(MonoDiarizationSettings(
        naming_seconds=4.0, naming_growth=2.0, naming_max_seconds=10.0,
    ))
    plan.add(0, 3.0)
    assert plan.due() == []
    plan.add(0, 1.5)
    plan.add(1, 5.0)
    assert plan.due() == [0, 1]
    plan.add(0, 2.0)
    assert plan.due() == []
    plan.add(0, 4.0)
    assert plan.due() == [0]
    assert not plan.wants(0) and plan.wants(1) and plan.wants(2)


def test_unit_sum_adds_the_voiced_rows_made_unit() -> None:
    rows = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    summed = unit_sum(rows)
    assert summed is not None
    total, count = summed
    assert count == 2 and np.allclose(total, [0.6, 1.8])
    assert unit_sum(np.zeros((2, 3), dtype=np.float32)) is None
    assert unit_sum(np.zeros((0, 3), dtype=np.float32)) is None


def test_name_slots_gives_each_person_to_one_slot() -> None:
    prints = [voiceprint("ana", [1.0, 0.0]), voiceprint("luis", [0.0, 1.0])]
    voices = {
        0: np.array([0.99, 0.14], dtype=np.float32),
        2: np.array([0.7, 0.7], dtype=np.float32),
    }
    names, nearest = name_slots(voices, prints, 0.95, model="m")
    assert names == {0: "ana"}
    assert nearest[0][0] == "ana" and nearest[2][1] is not None
    assert name_slots({}, prints, 0.95) == ({}, {})
    empty = {0: (None, None), 2: (None, None)}
    assert name_slots(voices, [], 0.95) == ({}, empty)
