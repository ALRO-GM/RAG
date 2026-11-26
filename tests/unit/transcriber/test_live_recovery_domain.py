"""The pure parts of recovering a live session's lost audio (#265).

Spans of samples, which parts of the uploaded audio the live transcript
lacks, the speaker mapping in mono and stereo, the deterministic merge,
and the job's state machine. Invented sentences only.
"""

from datetime import timedelta

import pytest

from tests.unit.transcriber.recovery_fakes import T0, transcript
from transcriber.domain.live import LiveSpeaker
from transcriber.domain.live_recovery import (
    END,
    group_spans,
    MAX_TRIES,
    RecoveredPart,
    RecoveryJob,
    intersect,
    merge_recovered,
    merge_spans,
    recovered_spans,
    seconds_of,
    speaker_map,
)
from transcriber.domain.models import Speaker

S = 16000


def test_close_spans_share_a_part_and_far_ones_do_not() -> None:
    assert group_spans([(0, S), (10 * S, 11 * S), (100 * S, 101 * S)]) == [
        (0, 11 * S), (100 * S, 101 * S)]
    assert group_spans([(0, S), (2 * S, 3 * S)], longest=2 * S) == [
        (0, S), (2 * S, 3 * S)]


def test_spans_are_sorted_joined_and_intersected() -> None:
    assert merge_spans([(5, 9), (0, 3), (3, 4), (8, 12), (7, 7)]) == [
        (0, 4), (5, 12)]
    assert intersect([(0, 10), (20, 30)], [(5, 25)]) == [(5, 10), (20, 25)]
    assert intersect([], [(0, 5)]) == []
    assert seconds_of([(0, S), (S, 3 * S)]) == 3.0


def test_only_the_silence_and_the_tail_are_taken_from_the_upload() -> None:
    """The server heard 0–10 s and 40–50 s itself, put silence in for
    10–40 s and ended at 50 s; the page uploads 8–60 s: only 10–40 s and
    50–60 s are recovered, what the server heard counts once."""
    spans = recovered_spans([(8 * S, 60 * S)], [(10 * S, 40 * S)], 50 * S)
    assert spans == [(10 * S, 40 * S), (50 * S, 60 * S)]
    assert recovered_spans([(0, 5 * S)], [], 10 * S) == []
    assert recovered_spans([(12 * S, 20 * S)], [], 10 * S) == [
        (12 * S, 20 * S)]


def test_a_stereo_part_takes_the_speaker_of_its_channel() -> None:
    batch = transcript([("S1", 0, 1, "uno"), ("S2", 1, 2, "dos")])
    live = [LiveSpeaker("ana", 0, "interviewer"),
            LiveSpeaker("cand", 1, "candidate")]
    assert speaker_map(batch.utterances, batch.speakers, live, 2, False,
                       channel=1) == {"S1": "cand", "S2": "cand"}
    assert speaker_map(batch.utterances, batch.speakers, [], 2, False,
                       channel=0) == {"S1": "S1", "S2": "S1"}


def test_a_mono_session_without_diarization_has_one_voice() -> None:
    batch = transcript([("S1", 0, 1, "uno"), ("S2", 1, 2, "dos")])
    live = [LiveSpeaker("mix", 0, "unknown")]
    assert speaker_map(batch.utterances, batch.speakers, live, 1,
                       False) == {"S1": "mix", "S2": "mix"}
    assert set(speaker_map(batch.utterances, batch.speakers, [], 1,
                           False).values()) == {"S1"}


def test_two_parties_once_the_employee_is_recognised() -> None:
    """The batch pipeline recognised the employee's voiceprint: every
    other voice is the other party; not recognised: never guessed."""
    said = [("S2", 0, 1, "hola"), ("emp", 1, 2, "buenos días"),
            ("S3", 2, 3, "vale")]
    found = transcript(said, [Speaker(id="emp", enrolled=True)])
    live = [LiveSpeaker("emp", 0, "interviewer", enrolled=True),
            LiveSpeaker("cand", 0, "candidate")]
    pair = ("emp", "cand")
    assert speaker_map(found.utterances, found.speakers, live, 1, True,
                       parties=pair) == {"S2": "cand", "emp": "emp",
                                         "S3": "cand"}
    unknown = transcript([("S2", 0, 1, "hola"), ("S1", 1, 2, "sí")])
    assert speaker_map(unknown.utterances, unknown.speakers, live, 1, True,
                       parties=pair) == {"S2": "R1", "S1": "R2"}
    assert speaker_map(found.utterances, found.speakers, live, 1,
                       True) == {"S2": "R1", "emp": "emp", "S3": "R2"}


def _live_with_gap() -> RecoveredPart:
    return RecoveredPart(transcript([]), 0, {})


def test_the_merge_puts_the_recovered_sentences_in_the_gap() -> None:
    """Live heard 0–10 s and 40–50 s; 10–40 s was silence (a sentence the
    recogniser made up over it is dropped). The recovered audio starts
    at 10 s; its sentences go in place of the gap, in time order, and
    one that falls outside the recovered spans (what the server heard)
    is not taken twice."""
    live = transcript([
        ("ana", 1, 4, "primera pregunta"), ("cand", 5, 9, "respuesta"),
        ("cand", 20, 21, "ruido"), ("ana", 41, 45, "siguiente"),
    ], [Speaker(id="ana", enrolled=False, role="interviewer"),
        Speaker(id="cand", enrolled=False, role="candidate")],
        duration=50)
    batch = transcript([
        ("S1", 0.5, 3.0, "segunda pregunta"),
        ("S2", 4.0, 12.0, "respuesta larga"),
        ("S1", 31.0, 36.0, "esto ya lo oyó el servidor"),
        ("S2", 41.0, 45.0, "cola tras el final"),
    ], duration=46)
    spans = [(10 * S, 40 * S), (50 * S, 56 * S)]
    names = {"S1": "ana", "S2": "cand"}
    merged = merge_recovered(live, [RecoveredPart(batch, 10 * S, names)],
                             spans, {"ana": "interviewer",
                                     "cand": "candidate"})
    got = [(u.speaker, u.start, u.text) for u in merged.transcript.utterances]
    assert got == [
        ("ana", 1, "primera pregunta"), ("cand", 5, "respuesta"),
        ("ana", 10.5, "segunda pregunta"), ("cand", 14.0, "respuesta larga"),
        ("ana", 41, "siguiente"), ("cand", 51.0, "cola tras el final"),
    ]
    assert [u.id for u in merged.transcript.utterances] == [
        "t0.s0", "t1.s0", "t2.s0", "t3.s0", "t4.s0", "t5.s0"]
    assert merged.added == 3 and merged.dropped == 1
    assert merged.seconds == 14.5, "the speech added, not the gap"
    assert merged.transcript.session.duration == 55.0
    # Run twice, the same result: the merge is deterministic.
    again = merge_recovered(live, [RecoveredPart(batch, 10 * S, names)],
                            spans, {})
    assert again.transcript == merged.transcript
    contract = merged.transcript.to_contract()
    assert contract["schema"] == "transcript.v1"


def test_unknown_recovered_voices_are_listed_with_their_role() -> None:
    live = transcript([("ana", 0, 1, "hola")],
                      [Speaker(id="ana", enrolled=False,
                               role="interviewer")])
    batch = transcript([("S1", 0, 1, "uno"), ("S2", 1, 2, "dos")])
    merged = merge_recovered(
        live, [RecoveredPart(batch, 5 * S, {"S1": "cand", "S2": "R1"})],
        [(5 * S, END)], {"cand": "candidate"})
    roles = {s.id: s.role for s in merged.transcript.speakers}
    assert roles == {"ana": "interviewer", "cand": "candidate", "R1": None}
    assert merged.seconds == 2.0
    assert merged.unidentified, "R1 feeds no indicator: said so"


def test_ties_keep_the_live_sentence_first() -> None:
    live = transcript([("ana", 2, 3, "vivo")])
    batch = transcript([("S1", 0, 1, "recuperado")])
    merged = merge_recovered(live, [RecoveredPart(batch, 2 * S,
                                                  {"S1": "cand"})],
                             [(4 * S, 6 * S), (2 * S, 3 * S - 1)], {})
    assert [u.text for u in merged.transcript.utterances] == ["recuperado"]
    merged = merge_recovered(live, [_live_with_gap()], [], {})
    assert [u.text for u in merged.transcript.utterances] == ["vivo"]


def _job(state: str = "queued", **more: object) -> RecoveryJob:
    return RecoveryJob("iv-1", state, T0.isoformat(),
                       (T0 + timedelta(hours=24)).isoformat(),
                       **more)  # type: ignore[arg-type]


def test_a_failed_step_waits_longer_each_time_then_fails() -> None:
    job = _job()
    assert job.due(T0)
    assert not job.failed_try("PublishError", T0)
    assert not job.due(T0 + timedelta(seconds=29))
    assert job.due(T0 + timedelta(seconds=30))
    assert not job.failed_try("PublishError", T0)
    assert job.next_try == (T0 + timedelta(seconds=120)).isoformat()
    assert job.failed_try("PublishError", T0) and job.tries == MAX_TRIES
    assert job.state == "failed" and job.failed_step == "queued"
    assert job.final and not job.due(T0)


def test_a_job_reads_back_and_knows_its_expiry_and_uploads() -> None:
    job = _job(chunks={"0": [0, 100], "1": [100, 50], "3": [400, 10]})
    copy = RecoveryJob.from_dict({**job.to_dict(), "extra": 1})
    assert copy.uploaded() == [(0, 150), (400, 410)]
    assert not copy.expired(T0) and copy.expired(T0 + timedelta(hours=24))
    copy.move("merging", T0)
    assert copy.state == "merging" and copy.tries == 0
    with pytest.raises(ValueError):
        copy.move("nowhere", T0)
    with pytest.raises(ValueError):
        RecoveryJob.from_dict({**job.to_dict(), "state": "bad"})
    view = copy.view()
    assert view["chunks"] == [0, 1, 3] and "meta" not in view
    assert not _job("uploading").due(T0)
