"""Transcript models and the text / JSON transcript parsers."""

import json
from typing import Any, Dict, List

import pytest

from src.domain.errors import CorruptedDataError, InvalidInputError
from src.domain.skills.transcript import (
    Transcript,
    Turn,
    build_transcript,
    parse_clock,
    parse_json,
    parse_text,
    parse_transcript,
)

Segment = Dict[str, Any]


def seg(speaker: str, text: str, *times: float) -> Segment:
    """Return a diarizer segment, with ``start, end`` when given."""
    segment: Segment = {"speaker": speaker, "text": text}
    if times:
        segment.update(start=times[0], end=times[1])
    return segment


def spoken(transcript: Transcript) -> List[tuple[str, str]]:
    """Return the (speaker, text) pairs of a transcript."""
    return [(t.speaker, t.text) for t in transcript.turns]


# ------------------------------------------------------------- models


def test_turn_counts_unicode_words() -> None:
    """Accented words and numbers count as one word each."""
    turn = Turn(index=0, speaker="Ana", text="¿Qué pasó, señor? 3 veces")
    assert turn.n_words() == 5


def test_transcript_speakers_in_order_of_first_appearance() -> None:
    """Speakers are listed once, in speaking order."""
    transcript = build_transcript("s", [
        seg("B", "one"), seg("A", "two"), seg("B", "three"),
    ])
    assert transcript.speakers() == ["B", "A"]
    assert transcript.text() == "B: one\nA: two\nB: three"


def test_has_timestamps_needs_every_turn_timed() -> None:
    """One untimed turn makes the whole transcript untimed."""
    timed = build_transcript("s", [seg("A", "x", 0, 1), seg("B", "y", 1, 2)])
    partial = build_transcript("s", [seg("A", "x", 0, 1), seg("B", "y")])
    assert timed.has_timestamps()
    assert not partial.has_timestamps()
    assert not Transcript(session_id="s", turns=[]).has_timestamps()


@pytest.mark.parametrize("clock, seconds", [
    ("01:05", 65.0),
    ("1:05", 65.0),
    ("01:02:03", 3723.0),
    ("00:00:07.250", 7.25),
    ("00:07,5", 7.5),
])
def test_parse_clock(clock: str, seconds: float) -> None:
    """``mm:ss`` and ``hh:mm:ss`` with decimals convert to seconds."""
    assert parse_clock(clock) == pytest.approx(seconds)


# ------------------------------------------------------ build_transcript


def test_build_merges_consecutive_untimed_segments_of_a_speaker() -> None:
    """Diarizer fragments of one contribution become one turn."""
    transcript = build_transcript("s", [
        seg("A", "first"), seg("A", "second"), seg("B", "reply"),
        seg("A", "third"),
    ])
    assert spoken(transcript) == [
        ("A", "first second"), ("B", "reply"), ("A", "third"),
    ]
    assert [t.index for t in transcript.turns] == [0, 1, 2]


def test_build_merges_within_the_gap_and_splits_beyond() -> None:
    """Segments at most ``merge_gap`` apart merge; later ones do not."""
    transcript = build_transcript("s", [
        seg("A", "x", 0.0, 1.0), seg("A", "y", 2.0, 3.0),
        seg("A", "z", 4.5, 5.0),
    ], merge_gap=1.0)
    assert spoken(transcript) == [("A", "x y"), ("A", "z")]
    assert (transcript.turns[0].start, transcript.turns[0].end) == (0.0, 3.0)
    assert (transcript.turns[1].start, transcript.turns[1].end) == (4.5, 5.0)


def test_build_merge_gap_is_configurable() -> None:
    """A larger gap merges more, a zero gap only touching segments."""
    segments = [seg("A", "x", 0.0, 1.0), seg("A", "y", 3.0, 4.0)]
    assert len(build_transcript("s", segments, merge_gap=2.0).turns) == 1
    assert len(build_transcript("s", segments, merge_gap=0.0).turns) == 2


def test_build_merged_turn_keeps_its_end_when_next_has_none() -> None:
    """Merging an untimed fragment does not erase the end time."""
    transcript = build_transcript("s", [
        seg("A", "x", 0.0, 1.0), seg("A", "y"),
    ])
    assert transcript.turns[0].end == 1.0


def test_build_skips_segments_without_text_or_speaker() -> None:
    """Empty, missing or null fields drop the segment silently."""
    transcript = build_transcript("s", [
        {"speaker": "A", "text": "   "},
        {"speaker": "", "text": "nobody"},
        {"text": "no speaker"},
        {"speaker": "A"},
        {"speaker": None, "text": None},
        seg("B", "kept"),
    ])
    assert spoken(transcript) == [("B", "kept")]


def test_build_normalises_whitespace_and_coerces_types() -> None:
    """Inner whitespace collapses; numeric strings become floats."""
    transcript = build_transcript("s", [
        {"speaker": " A ", "text": " a\n b\t c ", "start": "1.5", "end": 2},
    ])
    turn = transcript.turns[0]
    assert (turn.speaker, turn.text) == ("A", "a b c")
    assert (turn.start, turn.end) == (1.5, 2.0)


def test_build_without_any_turn_is_an_error() -> None:
    """A transcript where nobody speaks is rejected."""
    with pytest.raises(InvalidInputError, match="no speaker turn"):
        build_transcript("empty", [])
    with pytest.raises(InvalidInputError, match="'empty'"):
        build_transcript("empty", [{"speaker": "A", "text": ""}])


def test_build_keeps_the_language() -> None:
    """The declared language is stored on the transcript."""
    assert build_transcript("s", [seg("A", "x")], "es").language == "es"


# ----------------------------------------------------------- parse_text


def test_parse_text_plain_speaker_lines() -> None:
    """``Speaker: words`` lines become untimed turns."""
    transcript = parse_text("s", "Ana: Hola a todos.\nLuis: Buenas.\n")
    assert spoken(transcript) == [
        ("Ana", "Hola a todos."), ("Luis", "Buenas."),
    ]
    assert not transcript.has_timestamps()
    assert transcript.session_id == "s"


@pytest.mark.parametrize("stamp, seconds", [
    ("[00:05]", 5.0),
    ("00:05", 5.0),
    ("[00:00:05]", 5.0),
    ("00:00:05", 5.0),
    ("[00:00:05.500]", 5.5),
    ("00:00:05,500", 5.5),
    ("[1:02:03]", 3723.0),
    ("00:05 -", 5.0),
    ("[00:05] –", 5.0),
])
def test_parse_text_timestamp_formats(stamp: str, seconds: float) -> None:
    """Clocks with or without brackets, hours and decimals are read."""
    transcript = parse_text("s", f"{stamp} Ana: hola\n")
    turn = transcript.turns[0]
    assert (turn.speaker, turn.text) == ("Ana", "hola")
    assert turn.start == pytest.approx(seconds)


def test_parse_text_turn_ends_where_the_next_starts() -> None:
    """A turn ends at the next timestamp; the last ends where it starts."""
    transcript = parse_text(
        "s", "[00:01] Ana: uno\n[00:05] Luis: dos\n[00:09] Ana: tres\n"
    )
    assert [(t.start, t.end) for t in transcript.turns] == [
        (1.0, 5.0), (5.0, 9.0), (9.0, 9.0),
    ]
    assert transcript.has_timestamps()


def test_parse_text_continuation_lines_extend_the_turn() -> None:
    """A line without a speaker prefix continues the previous turn."""
    transcript = parse_text(
        "s", "Ana: primera línea\n  y sigue aquí\n\nLuis: vale\n"
    )
    assert spoken(transcript) == [
        ("Ana", "primera línea y sigue aquí"), ("Luis", "vale"),
    ]


def test_parse_text_ignores_text_before_the_first_speaker() -> None:
    """A title or note before the first turn is not a turn."""
    transcript = parse_text("s", "Meeting notes\n\nAna: hola\n")
    assert spoken(transcript) == [("Ana", "hola")]


def test_parse_text_keeps_colons_inside_the_words() -> None:
    """Only the first ``Speaker:`` splits; later colons are text."""
    transcript = parse_text(
        "s", "Ana: quedamos a las 10:30: en la sala\nLuis: nota: vale\n"
    )
    assert spoken(transcript) == [
        ("Ana", "quedamos a las 10:30: en la sala"), ("Luis", "nota: vale"),
    ]


def test_parse_text_urls_are_not_speakers() -> None:
    """A continuation line with a URL stays in the current turn."""
    transcript = parse_text(
        "s",
        "Ana: mirad esto\nhttps://example.com/a:b está aquí\n"
        "ver http://example.com también\nLuis: Mira https://x.y: ok\n",
    )
    assert spoken(transcript) == [
        ("Ana", "mirad esto https://example.com/a:b está aquí "
                "ver http://example.com también"),
        ("Luis", "Mira https://x.y: ok"),
    ]


def test_parse_text_speaker_names_with_spaces_and_accents() -> None:
    """Full names, accents, digits and diarizer labels are speakers."""
    transcript = parse_text(
        "s",
        "María José Núñez: hola\nSPEAKER_01: hi\n[00:03] Dr. O'Neil-2: yes\n",
    )
    assert transcript.speakers() == [
        "María José Núñez", "SPEAKER_01", "Dr. O'Neil-2",
    ]


def test_parse_text_a_name_over_forty_characters_is_not_a_speaker() -> None:
    """A long sentence ending in a colon is a continuation, not a name."""
    long_line = "x" * 41 + ": not a speaker"
    transcript = parse_text("s", f"Ana: hola\n{long_line}\n")
    assert transcript.speakers() == ["Ana"]
    assert transcript.turns[0].text.endswith("not a speaker")


def test_parse_text_consecutive_lines_of_a_speaker_merge() -> None:
    """Two lines in a row by one speaker are one turn, timed or not."""
    untimed = parse_text("s", "Ana: uno\nAna: dos\nLuis: tres\n")
    timed = parse_text(
        "s", "[00:01] Ana: uno\n[00:30] Ana: dos\n[00:40] Luis: tres\n"
    )
    assert spoken(untimed) == [("Ana", "uno dos"), ("Luis", "tres")]
    assert spoken(timed) == [("Ana", "uno dos"), ("Luis", "tres")]
    assert (timed.turns[0].start, timed.turns[0].end) == (1.0, 40.0)


def test_parse_text_speaker_without_words_is_skipped() -> None:
    """``Ana:`` followed by nothing produces no empty turn."""
    transcript = parse_text("s", "Ana: \nLuis: ok\n")
    assert spoken(transcript) == [("Luis", "ok")]


def test_parse_text_windows_newlines() -> None:
    """CRLF line ends leave no carriage return in the text."""
    transcript = parse_text("s", "Ana: hola\r\nLuis: adiós\r\n")
    assert spoken(transcript) == [("Ana", "hola"), ("Luis", "adiós")]


def test_parse_text_without_any_speaker_line_is_an_error() -> None:
    """Free text with no ``Speaker:`` line is not a transcript."""
    with pytest.raises(InvalidInputError, match="no speaker turn"):
        parse_text("s", "just some notes\nwithout speakers\n")


# ----------------------------------------------------------- parse_json


def test_parse_json_own_shape_with_language_and_session() -> None:
    """``{"turns": [...]}`` carries its language and session id."""
    document = {
        "session_id": "named", "language": "es",
        "turns": [seg("Ana", "hola", 0.0, 1.0), seg("Luis", "qué tal")],
    }
    transcript = parse_json("file", json.dumps(document))
    assert transcript.session_id == "named"
    assert transcript.language == "es"
    assert spoken(transcript) == [("Ana", "hola"), ("Luis", "qué tal")]
    assert transcript.turns[0].end == 1.0


def test_parse_json_whisperx_segments_are_merged() -> None:
    """A diarizer output merges close segments of a speaker (gap 1 s)."""
    document = {"language": "en", "segments": [
        {"start": 0.0, "end": 1.2, "text": " Hello there.",
         "speaker": "SPEAKER_00", "words": [{"word": "Hello"}]},
        {"start": 1.5, "end": 2.0, "text": " How are you?",
         "speaker": "SPEAKER_00"},
        {"start": 2.1, "end": 3.0, "text": " Fine.", "speaker": "SPEAKER_01"},
        {"start": 9.0, "end": 9.5, "text": " Good.", "speaker": "SPEAKER_01"},
        {"start": 9.6, "end": 9.9, "text": " no speaker here"},
    ]}
    transcript = parse_json("wx", json.dumps(document))
    assert spoken(transcript) == [
        ("SPEAKER_00", "Hello there. How are you?"),
        ("SPEAKER_01", "Fine."),
        ("SPEAKER_01", "Good."),
    ]
    assert transcript.session_id == "wx"
    assert transcript.language == "en"
    assert transcript.has_timestamps()


def test_parse_json_bare_list_of_segments() -> None:
    """A top-level list of segments is accepted."""
    transcript = parse_json("s", json.dumps([seg("A", "x"), seg("B", "y")]))
    assert spoken(transcript) == [("A", "x"), ("B", "y")]
    assert transcript.language is None


def test_parse_json_round_trips_a_dumped_transcript() -> None:
    """Our own ``model_dump_json`` output parses back identically."""
    original = build_transcript(
        "s", [seg("A", "x", 0.0, 1.0), seg("B", "y", 1.0, 2.5)], "en"
    )
    assert parse_json("other", original.model_dump_json()) == original


@pytest.mark.parametrize("text", [
    "{not json", "[1, 2", '{"turns": [}',
])
def test_parse_json_invalid_json_is_corrupted_data(text: str) -> None:
    """Malformed JSON is reported with the session name."""
    with pytest.raises(CorruptedDataError, match="'s' is not valid JSON"):
        parse_json("s", text)


@pytest.mark.parametrize("document", [
    {}, {"turns": None}, {"turns": "text"}, {"turns": [1, 2]},
    {"segments": [["A", "x"]]}, {"other": []}, 42, "text", None,
    [1], ["Ana: hola"],
])
def test_parse_json_without_a_list_of_objects_is_corrupted(
    document: object,
) -> None:
    """Anything but a list of segment objects is rejected."""
    with pytest.raises(CorruptedDataError, match="list of turns"):
        parse_json("s", json.dumps(document))


@pytest.mark.parametrize("fields", [
    {"start": "soon"}, {"end": "later"}, {"start": [1]}, {"end": {}},
])
def test_parse_json_invalid_times_are_corrupted_data(
    fields: Dict[str, Any],
) -> None:
    """A non-numeric time names the invalid segment, without a crash."""
    document = [{"speaker": "A", "text": "x", **fields},
                {"speaker": "A", "text": "y", **fields}]
    with pytest.raises(CorruptedDataError, match="invalid segment"):
        parse_json("s", json.dumps(document))


def test_parse_json_non_string_language_is_corrupted_data() -> None:
    """A language that is not text is a format error, not a crash."""
    document = {"language": 5, "turns": [seg("A", "x")]}
    with pytest.raises(CorruptedDataError):
        parse_json("s", json.dumps(document))


def test_parse_json_empty_list_has_no_turn() -> None:
    """A valid but empty document is an input error."""
    with pytest.raises(InvalidInputError, match="no speaker turn"):
        parse_json("s", "[]")
    with pytest.raises(InvalidInputError, match="no speaker turn"):
        parse_json("s", '{"segments": []}')


# ------------------------------------------------------ parse_transcript


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_parse_transcript_empty_input(text: str) -> None:
    """Blank input is reported as an empty transcript."""
    with pytest.raises(InvalidInputError, match="'s' is empty"):
        parse_transcript("s", text)


def test_parse_transcript_bracketed_timestamp_is_text_not_json() -> None:
    """``[00:01] Ana: ...`` starts with a bracket but is a text line."""
    transcript = parse_transcript("s", "[00:01] Ana: hola\n[00:02] Luis: ey")
    assert spoken(transcript) == [("Ana", "hola"), ("Luis", "ey")]
    assert transcript.turns[0].start == 1.0


def test_parse_transcript_json_array_first_character() -> None:
    """A JSON array, compact or indented, is parsed as JSON."""
    segments = [seg("Ana", "hola"), seg("Luis", "ey")]
    for text in (json.dumps(segments), json.dumps(segments, indent=2),
                 "\n  " + json.dumps(segments)):
        assert spoken(parse_transcript("s", text)) == [
            ("Ana", "hola"), ("Luis", "ey"),
        ]


def test_parse_transcript_indented_json_object() -> None:
    """A pretty-printed JSON object is parsed as JSON."""
    text = json.dumps({"segments": [seg("Ana", "hola")]}, indent=2)
    assert spoken(parse_transcript("s", text)) == [("Ana", "hola")]


def test_parse_transcript_single_line_json_object() -> None:
    """A compact ``{"segments": [...]}`` document is parsed as JSON."""
    text = json.dumps({"segments": [seg("Ana", "hola a todos")]})
    assert spoken(parse_transcript("s", text)) == [("Ana", "hola a todos")]


def test_parse_transcript_blank_line_before_bracketed_timestamp() -> None:
    """Leading blank lines do not turn a text transcript into JSON."""
    transcript = parse_transcript("s", "\n[00:01] Ana: hola\n")
    assert spoken(transcript) == [("Ana", "hola")]


def test_parse_transcript_minutes_past_ninety_nine() -> None:
    """A ``mm:ss`` clock keeps working after 99 minutes."""
    transcript = parse_transcript("s", "[100:05] Ana: hola\n")
    assert transcript.turns[0].start == 6005.0


def test_parse_transcript_text_with_braces_inside_is_text() -> None:
    """A text transcript mentioning JSON is still text."""
    transcript = parse_transcript("s", 'Ana: usa {"a": 1} y [x]\nLuis: ok')
    assert spoken(transcript) == [
        ("Ana", 'usa {"a": 1} y [x]'), ("Luis", "ok"),
    ]


def test_parse_transcript_garbage_starting_with_a_brace() -> None:
    """Broken JSON is corrupted data, not an empty transcript."""
    with pytest.raises(CorruptedDataError):
        parse_transcript("s", "{\n oops")


def test_parse_transcript_brace_first_is_always_json() -> None:
    """Text starting with ``{`` is JSON or corrupted, never a speaker."""
    for text in ('{"hello": [1, 2]}', '{"a": 1}\nAna: hola', "{Ana}: hola"):
        with pytest.raises(CorruptedDataError):
            parse_transcript("s", text)


def test_parse_transcript_bracket_first_falls_back_to_text() -> None:
    """``[`` starts JSON only when the whole text parses as JSON."""
    text = "[inaudible] noise\n[00:02] Ana: hola\n"
    assert spoken(parse_transcript("s", text)) == [("Ana", "hola")]
    with pytest.raises(CorruptedDataError, match="list of turns"):
        parse_transcript("s", "[1, 2, 3]")


def test_parse_text_speakers_cannot_hold_braces_or_start_quoted() -> None:
    """JSON-looking prefixes are continuation text, not speakers."""
    transcript = parse_text(
        "s", 'Ana: mira\n"key": value\nob{j}: x\nLuis: ok\n'
    )
    assert spoken(transcript) == [
        ("Ana", 'mira "key": value ob{j}: x'), ("Luis", "ok"),
    ]


def test_build_keeps_an_integer_speaker_zero() -> None:
    """Diarizers numbering speakers from 0 do not lose the first one."""
    transcript = build_transcript("s", [
        {"speaker": 0, "text": "first"}, {"speaker": 1, "text": "second"},
    ])
    assert transcript.speakers() == ["0", "1"]


def test_build_unusable_timestamps_become_none() -> None:
    """NaN, infinite and boolean times are treated as missing."""
    text = ('[{"speaker": "A", "text": "x", "start": NaN, "end": Infinity},'
            ' {"speaker": "B", "text": "y", "start": true, "end": false}]')
    transcript = parse_json("s", text)
    assert [(t.start, t.end) for t in transcript.turns] == [
        (None, None), (None, None),
    ]
    assert not transcript.has_timestamps()


# -- transcript.v1 from the transcriber ---------------------------------

def test_a_contract_document_keeps_turns_ids_times_and_names() -> None:
    from src.domain.skills.speech import transcript_sentences
    from tests.unit.test_skills_ingest import example

    document = example()
    document["utterances"][0]["confidence"] = 0.8
    document["utterances"][1]["confidence"] = 0.6
    transcript = parse_transcript("file-name", json.dumps(document))
    assert transcript.session_id == "demo-es"
    assert transcript.language == "es"
    assert [(t.index, t.speaker, t.previous_speaker)
            for t in transcript.turns] == [
        (0, "Marta", None), (1, "Luis", "Marta"), (2, "S1", "Luis"),
    ]
    first = transcript.turns[0]
    assert first.text == (
        "Creo que con un solo proceso no cumplimos el requisito. "
        "Pero no estoy segura."
    )
    assert (first.start, first.end, first.confidence) == (0.4, 6.3, 0.7)
    assert transcript.turns[1].confidence is None
    sentences = transcript_sentences(transcript)
    assert [(s.id, s.speaker, s.start, s.end) for s in sentences] == [
        (u["id"], s.speaker, u["start"], u["end"])
        for u, s in zip(document["utterances"], sentences)
    ]
    assert transcript.speakers() == ["Marta", "Luis", "S1"]


def test_speakers_who_share_a_name_stay_apart() -> None:
    from src.domain.skills.transcript import from_contract
    from src.domain.skills.transcript_contract import read_contract
    from tests.unit.test_skills_ingest import example

    document = example()
    document["speakers"][1]["name"] = "Marta"   # luis is also "Marta"
    document["speakers"][2]["name"] = "luis"    # S1 is called like an id
    transcript = from_contract(read_contract(document))
    assert [(t.speaker, t.previous_speaker) for t in transcript.turns] == [
        ("Marta (marta)", None), ("Marta (luis)", "Marta (marta)"),
        ("luis (S1)", "Marta (luis)"),
    ]


def test_a_contract_without_utterances_has_no_turn() -> None:
    from tests.unit.test_skills_ingest import example

    document = example()
    document["utterances"] = []
    with pytest.raises(InvalidInputError, match="no speaker turn"):
        parse_transcript("x", json.dumps(document))


@pytest.mark.parametrize("change", [
    {"schema": "transcript.v2"},
    {"utterances": [{"id": "t0.s0"}]},
])
def test_a_broken_contract_document_is_corrupted(
    change: Dict[str, Any]
) -> None:
    from tests.unit.test_skills_ingest import example

    document = {**example(), **change}
    with pytest.raises(CorruptedDataError, match="Transcript 'x'"):
        parse_transcript("x", json.dumps(document))
