"""Speech domain: sentence cutting, the spoken text, TTS requests."""

import json

import jsonschema
import pytest

from orchestrator.domain.speech import (
    ReplyText,
    SentenceChunker,
    SpeechChunk,
    split_sentences,
    tts_request,
)
from tests.unit.orchestrator.fixtures import REPO

SCHEMA = json.loads(
    (REPO / "contracts/tts.request.v1.schema.json").read_text("utf-8")
)


def test_sentences_come_out_as_soon_as_they_end() -> None:
    chunker = SentenceChunker()
    assert chunker.feed("Las devoluciones se aceptan") == []
    assert chunker.feed(" durante 30 días. Para empezar") == [
        "Las devoluciones se aceptan durante 30 días."
    ]
    assert chunker.feed(" necesito tu pedido.") == []
    assert chunker.flush() == ["Para empezar necesito tu pedido."]
    assert chunker.flush() == []


def test_a_short_sentence_waits_for_the_next() -> None:
    assert split_sentences("Sí. Te lo explico ahora mismo. ¿Vale?") == [
        "Sí. Te lo explico ahora mismo.", "¿Vale?",
    ]


def test_questions_exclamations_and_quotes_end_sentences() -> None:
    text = "¡Claro que sí, sin problema! ¿Tienes el número «A-12»? Bien."
    assert split_sentences(text) == [
        "¡Claro que sí, sin problema!", "¿Tienes el número «A-12»?",
        "Bien.",
    ]


def test_decimal_commas_and_numbers_do_not_cut() -> None:
    assert split_sentences("Cuesta 4,90 euros y tarda 5 a 7 días.") == [
        "Cuesta 4,90 euros y tarda 5 a 7 días.",
    ]


def test_a_long_sentence_is_cut_at_a_comma() -> None:
    chunker = SentenceChunker(min_chars=5, max_chars=40)
    out = chunker.feed("Primero revisamos el pedido, luego la factura y "
                       "después todo lo demás que haga falta")
    assert out[0] == "Primero revisamos el pedido,"
    assert all(len(s) <= 40 for s in out)


def test_a_long_text_without_commas_is_cut_at_a_space() -> None:
    out = split_sentences("palabra " * 60, min_chars=12)
    assert len(out) > 1 and all(len(s) <= 220 for s in out)
    chunker = SentenceChunker(min_chars=5, max_chars=10)
    assert chunker.feed("x" * 25) == ["x" * 10, "x" * 10]


def test_chunker_limits_are_checked() -> None:
    with pytest.raises(ValueError):
        SentenceChunker(min_chars=10, max_chars=10)


def stream(pieces: list[str]) -> str:
    reply = ReplyText()
    return "".join(reply.feed(p) for p in pieces)


def test_reply_text_reads_the_json_value_piece_by_piece() -> None:
    reply = ReplyText()
    assert reply.feed('{"te') == ""
    assert reply.feed('xt": "Hola, ') == "Hola, "
    assert reply.feed('qué tal.", "cites": ["c1"]}') == "qué tal."
    assert reply.done
    assert reply.feed("más") == ""


def test_reply_text_decodes_escapes_split_across_pieces() -> None:
    assert stream(['{"text": "Di \\', '"hola\\"', ' y \\u00', 'e1', 'no\\n'
                   'fin"}']) == 'Di "hola" y áno fin'


def test_reply_text_skips_a_code_fence_and_takes_plain_text() -> None:
    assert stream(["```json\n", '{"text": "Vale."}', "\n```"]) == "Vale."
    assert stream(["  ", "Hola sin", " JSON."]) == "Hola sin JSON."


def test_reply_text_bad_unicode_escape_is_dropped() -> None:
    assert stream(['{"text": "a\\uZZZZb"}']) == "ab"


def test_tts_request_follows_the_contract() -> None:
    doc = tts_request("t-1-s0", "  Hola,\n ¿qué tal? ", "es_ES-davefx-medium",
                      "es", interrupt_id="t-0-s2")
    jsonschema.validate(doc, SCHEMA)
    assert doc["text"] == "Hola, ¿qué tal?"
    assert doc["format"] == "wav" and doc["interrupt_id"] == "t-0-s2"
    example = json.loads((REPO / "contracts/examples/tts.request.v1.json")
                         .read_text("utf-8"))
    jsonschema.validate(example, SCHEMA)


@pytest.mark.parametrize("kwargs", [
    {"text": "  "}, {"request_id": ""}, {"request_id": "x" * 65},
    {"voice": "../etc/passwd"}, {"audio_format": "mp3"},
    {"length_scale": 3.0},
])
def test_tts_request_refuses_what_the_contract_refuses(
    kwargs: dict[str, object]
) -> None:
    args: dict[str, object] = {"request_id": "r", "text": "Hola.",
                               "voice": "es_ES-davefx-medium"}
    args.update(kwargs)
    with pytest.raises(ValueError):
        tts_request(**args)  # type: ignore[arg-type]


def test_long_text_is_cut_to_the_contract_limit() -> None:
    doc = tts_request("r", "a " * 900, "v")
    assert len(doc["text"]) == 1000


def test_chunk_duration_from_its_samples() -> None:
    pcm = SpeechChunk("r", 0, "x", b"\0\0" * 22050, "pcm_s16le", 22050, 1.0)
    wav = SpeechChunk("r", 0, "x", b"\0" * 44 + b"\0\0" * 11025, "wav",
                      22050, 1.0)
    assert pcm.duration_ms == 1000.0 and wav.duration_ms == 500.0
