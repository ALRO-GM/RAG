"""Speakers' names from what they say: rules, model answers, decision."""

from typing import Dict, List, Optional, Tuple

import pytest

from tests.unit.transcriber.names_fakes import conversation, reply
from transcriber.domain.models import Speaker
from transcriber.domain.speaker_names import (
    AutoName,
    NameEvidence,
    NameSuggestion,
    UndoneName,
    clean_name,
    contains_name,
    decide,
    find_candidates,
    kept_undone,
    model_prompt,
    open_suggestions,
    parse_reply,
    rule_evidence,
    same_name,
    target_speaker,
    valid_auto,
)


def found(text: str) -> List[Tuple[str, str]]:
    """``(kind, name)`` of the candidates of a one-line conversation."""
    transcript = conversation([("S1", text)])
    return [(c.kind, c.name) for c in find_candidates(transcript)]


def evidence(
    speaker: str, name: str, kind: str = "self", confidence: float = 0.9,
    start: float = 0.0,
) -> NameEvidence:
    """One piece of evidence from the model."""
    return NameEvidence(
        speaker=speaker, name=name, kind=kind,  # type: ignore[arg-type]
        quote=f"soy {name}", utterance="t0.s0", start=start,
        confidence=confidence, source="model",
    )


def anonymous(
    *ids: str, named: Optional[Dict[str, str]] = None
) -> List[Speaker]:
    """Anonymous speakers, some with a name."""
    return [
        Speaker(id=i, name=(named or {}).get(i), enrolled=False) for i in ids
    ]


class TestRules:
    @pytest.mark.parametrize("text, expected", [
        ("Hola, soy Ana y llevo el proyecto.", [("self", "Ana")]),
        ("Me llamo Pedro García, encantado.", [("self", "Pedro García")]),
        ("Mi nombre es Lucía.", [("self", "Lucía")]),
        ("Hola, aquí Luis de ventas.", [("self", "Luis")]),
        ("Soy yo, Carmen.", [("self", "Carmen")]),
        ("Hi, I'm Kate, from finance.", [("self", "Kate")]),
        ("Gracias, Marta.", [("addressed", "Marta")]),
        ("Os presento a Ana, que viene de compras.", [("addressed", "Ana")]),
        ("Nerea, ¿tienes los números?", [("addressed", "Nerea")]),
        ("¿Y tú, Marta, qué opinas?", [("addressed", "Marta")]),
        ("Lo vemos, Andrés.", [("addressed", "Andrés")]),
    ])
    def test_introductions_and_names_said_to_others(
        self, text: str, expected: List[Tuple[str, str]]
    ) -> None:
        assert found(text) == expected

    @pytest.mark.parametrize("text", [
        "Soy de Madrid.",
        "Eso me llamo la atención.",
        "Soy consciente de eso.",
        "Doce, en realidad.",
        "Bueno, vale.",
        "Buenos días a todos.",
        "Precisamente, eso digo.",
        "soy ana",
    ])
    def test_look_alikes_are_not_candidates(self, text: str) -> None:
        assert found(text) == []

    def test_one_candidate_per_sentence_and_name_self_first(self) -> None:
        transcript = conversation([("S1", "Soy yo, Carmen.")])
        (only,) = find_candidates(transcript)
        assert (only.kind, only.towards) == ("self", "self")

    def test_context_and_limit(self) -> None:
        transcript = conversation([
            ("S1", "Buenos días."), ("S2", "Gracias, Ana."),
            ("S1", "Soy Ana."), ("S2", "Vale."),
        ])
        cands = find_candidates(transcript, context=1, limit=1)
        assert [c.kind for c in cands] == ["self"]
        assert [u.text for u in cands[0].before] == ["Gracias, Ana."]
        assert [u.text for u in cands[0].after] == ["Vale."]
        both = find_candidates(transcript, context=0)
        assert [c.kind for c in both] == ["addressed", "self"]
        assert both[0].before == () and both[0].after == ()

    def test_clean_name(self) -> None:
        assert clean_name("Ana y") == "Ana"
        assert clean_name("Pedro García López") == "Pedro García"
        assert clean_name("Ana, que") == "Ana"
        assert clean_name("la atención") is None
        assert clean_name("Doce") is None
        assert clean_name("Básicamente") is None
        assert clean_name("") is None

    def test_same_name_and_contains(self) -> None:
        assert same_name("Ana", "ana garcía")
        assert same_name("Ángela", "angela")
        assert not same_name("Ana García", "Ana López")
        assert not same_name("", "Ana")
        assert contains_name("Hola, soy Ángela.", "Angela")
        assert not contains_name("Hola, soy Anabel.", "Ana")


class TestTargets:
    def test_self_is_the_speaker_addressed_is_a_neighbour(self) -> None:
        t = conversation([
            ("S1", "Doce."), ("S2", "Gracias, Nerea."), ("S3", "Una cosa."),
        ])
        thanks = t.utterances[1]
        assert target_speaker(t, thanks, "self", "self") == "S2"
        assert target_speaker(t, thanks, "addressed", "previous") == "S1"
        assert target_speaker(t, thanks, "addressed", "next") == "S3"

    def test_falls_back_to_the_other_side_or_nobody(self) -> None:
        t = conversation([("S1", "Hola."), ("S2", "Gracias, Ana.")])
        last = t.utterances[1]
        assert target_speaker(t, last, "addressed", "next") == "S1"
        alone = conversation([("S1", "Gracias, Ana.")])
        assert target_speaker(
            alone, alone.utterances[0], "addressed", "previous"
        ) is None

    def test_rule_evidence(self) -> None:
        t = conversation([("S1", "Hola."), ("S2", "Soy Ana, de compras.")])
        (cand,) = find_candidates(t)
        item = rule_evidence(t, cand)
        assert item is not None
        assert (item.speaker, item.name, item.kind, item.source) == (
            "S2", "Ana", "self", "rules"
        )
        assert item.confidence == 0.8
        assert item.start == 3.0
        alone = conversation([("S1", "Gracias, Ana.")])
        (lonely,) = find_candidates(alone)
        assert rule_evidence(alone, lonely) is None

    def test_long_sentences_are_quoted_around_the_name(self) -> None:
        text = ("Bueno, antes de empezar quiero decir una cosa sobre el "
                "calendario de entregas que tenemos este trimestre y "
                "sobre los retrasos, y por cierto me llamo Ricardo y "
                "llevo poco tiempo aquí pero ya he visto muchos de los "
                "problemas que tenemos con los proveedores de siempre.")
        t = conversation([("S1", text)])
        (cand,) = find_candidates(t)
        item = rule_evidence(t, cand)
        assert item is not None
        assert "Ricardo" in item.quote and len(item.quote) <= 160


class TestModelReplies:
    def setup_method(self) -> None:
        self.t = conversation([
            ("S1", "Empezamos."), ("S2", "Hola, soy Ana."),
            ("S1", "Gracias, Ana."), ("S3", "Yo sigo."),
        ])
        self.cands = find_candidates(self.t)

    def parse(self, index: int, text: str) -> Optional[NameEvidence]:
        return parse_reply(self.t, self.cands[index], text)

    def test_the_prompt_marks_the_line_and_names_the_word(self) -> None:
        prompt = model_prompt(self.cands[0])
        assert ">> S2: Hola, soy Ana." in prompt
        assert "   S1: Empezamos." in prompt
        assert prompt.startswith("Lines:\n")
        assert prompt.endswith(
            "Marked line: S2: Hola, soy Ana.\nWord to judge: Ana"
        )

    def test_a_self_introduction(self) -> None:
        item = self.parse(0, "Sure: " + reply("Ana", quote="soy Ana"))
        assert item is not None
        assert (item.speaker, item.name, item.kind) == ("S2", "Ana", "self")
        assert (item.quote, item.confidence, item.source) == (
            "soy Ana", 0.95, "model"
        )

    def test_the_target_comes_from_the_rule_not_the_model(self) -> None:
        item = self.parse(1, reply("Ana", "addressed", 0.7))
        assert item is not None
        assert (item.speaker, item.kind) == ("S2", "addressed")
        switched = self.parse(1, reply("Ana", "self"))
        assert switched is not None and switched.speaker == "S1"
        other_way = self.parse(0, reply("Ana", "addressed"))
        assert other_way is not None and other_way.speaker == "S1"

    @pytest.mark.parametrize("text", [
        "no json here",
        "{not json}",
        "[1, 2]",
        reply(None),
        reply("Pedro"),
        '{"name": "Ana", "kind": "self", "confidence": "high"}',
        '{"name": 3, "kind": "self"}',
    ])
    def test_unusable_answers_are_dropped(self, text: str) -> None:
        assert self.parse(0, text) is None

    def test_a_quote_not_in_the_line_is_replaced(self) -> None:
        item = self.parse(0, reply("Ana", quote="me llamo Ana"))
        assert item is not None and item.quote == "Hola, soy Ana."

    def test_confidence_is_clamped(self) -> None:
        item = self.parse(0, reply("Ana", confidence=7))
        assert item is not None and item.confidence == 1.0

    def test_nobody_to_give_the_name_to(self) -> None:
        alone = conversation([("S1", "Gracias, Ana.")])
        (cand,) = find_candidates(alone)
        assert parse_reply(alone, cand, reply("Ana", "addressed")) is None


class TestDecide:
    def test_a_clear_self_introduction_is_given(self) -> None:
        d = decide(anonymous("S1", "S2"), [evidence("S2", "Ana")])
        assert [(a.speaker, a.name) for a in d.auto] == [("S2", "Ana")]
        assert d.suggestions == [] and d.clear == []

    def test_below_the_confidence_it_is_a_suggestion(self) -> None:
        d = decide(anonymous("S1"), [evidence("S1", "Ana", confidence=0.6)])
        assert d.auto == []
        assert [s.reason for s in d.suggestions] == ["unsure"]

    def test_addressed_names_are_suggestions(self) -> None:
        d = decide(anonymous("S1", "S2"), [
            evidence("S1", "Marta", "addressed", 0.9),
            evidence("S1", "Marta", "addressed", 0.8, start=9.0),
        ])
        assert d.auto == []
        (only,) = d.suggestions
        assert (only.reason, only.count, only.start) == ("addressed", 2, 0.0)

    def test_two_voices_with_the_same_name_get_none(self) -> None:
        d = decide(anonymous("S1", "S2"), [
            evidence("S1", "Ana"), evidence("S2", "Ana", start=3.0),
        ])
        assert d.auto == []
        assert sorted((s.speaker, s.reason) for s in d.suggestions) == [
            ("S1", "same_name"), ("S2", "same_name"),
        ]

    def test_a_name_others_gave_to_another_voice_blocks_it(self) -> None:
        d = decide(anonymous("S1", "S2", "S3"), [
            evidence("S1", "Ana"), evidence("S3", "Ana", "addressed"),
        ])
        assert d.auto == []
        assert sorted((s.speaker, s.reason) for s in d.suggestions) == [
            ("S1", "same_name"), ("S3", "addressed"),
        ]

    def test_a_voice_with_two_names_gets_none(self) -> None:
        d = decide(anonymous("S1"), [
            evidence("S1", "Ana"), evidence("S1", "Pedro"),
        ])
        assert d.auto == []
        assert {s.reason for s in d.suggestions} == {"several_names"}

    def test_a_name_another_speaker_has_is_not_given(self) -> None:
        speakers = anonymous("S1", "S2", named={"S1": "Ana"})
        d = decide(speakers, [evidence("S2", "Ana")])
        assert d.auto == []
        assert [s.reason for s in d.suggestions] == ["name_taken"]
        enrolled = [Speaker(id="ana", name="Ana", enrolled=True)]
        d = decide(enrolled + anonymous("S2"), [evidence("S2", "ana")])
        assert d.auto == []

    def test_called_otherwise_by_others_is_a_conflict(self) -> None:
        d = decide(anonymous("S1", "S2"), [
            evidence("S2", "Ana"), evidence("S2", "Marta", "addressed"),
        ])
        assert d.auto == []
        reasons = {(s.name, s.reason) for s in d.suggestions}
        assert reasons == {("Ana", "conflict"), ("Marta", "addressed")}

    def test_called_the_same_by_others_is_fine(self) -> None:
        d = decide(anonymous("S1", "S2"), [
            evidence("S2", "Ana"), evidence("S2", "Ana", "addressed"),
        ])
        assert [a.name for a in d.auto] == ["Ana"]

    def test_enrolled_and_hand_named_voices_are_left_alone(self) -> None:
        speakers = [
            Speaker(id="luis", name="Luis", enrolled=True),
            Speaker(id="S1", name="Pepe", enrolled=False),
        ]
        d = decide(speakers, [
            evidence("luis", "Ana"), evidence("S1", "Ana"),
            evidence("S9", "Ana"),
        ])
        assert d == ([], [], [])

    def test_an_earlier_automatic_name_can_be_replaced_or_cleared(
        self,
    ) -> None:
        earlier = [AutoName(
            speaker="S1", name="Ana", quote="soy Ana", utterance="t0.s0",
            start=0.0, confidence=0.9,
        )]
        speakers = anonymous("S1", named={"S1": "Ana"})
        again = decide(speakers, [evidence("S1", "Ana")], earlier)
        assert [a.name for a in again.auto] == ["Ana"] and not again.clear
        gone = decide(speakers, [], earlier)
        assert gone.clear == ["S1"]
        kept = decide(speakers, [], earlier, allow_auto=False)
        assert kept.clear == []

    def test_without_the_model_only_suggestions(self) -> None:
        d = decide(anonymous("S1"), [evidence("S1", "Ana")],
                   allow_auto=False)
        assert d.auto == []
        assert [s.reason for s in d.suggestions] == ["rules_only"]

    def test_an_undone_name_is_never_given_again_to_that_voice(
        self,
    ) -> None:
        undone = [UndoneName(speaker="S1", name="Ana")]
        d = decide(anonymous("S1", "S2"), [
            evidence("S1", "Ana García"), evidence("S2", "Luis", start=3.0),
        ], undone=undone)
        assert [(a.speaker, a.name) for a in d.auto] == [("S2", "Luis")]
        assert [(s.speaker, s.name, s.reason) for s in d.suggestions] == [
            ("S1", "Ana García", "undone"),
        ]
        other = decide(anonymous("S1", "S2"), [evidence("S2", "Ana")],
                       undone=undone)
        assert [(a.speaker, a.name) for a in other.auto] == [("S2", "Ana")]
        renamed = decide(anonymous("S1"), [evidence("S1", "Eva")],
                         undone=undone)
        assert [a.name for a in renamed.auto] == ["Eva"]


def test_undone_names_apply_to_their_run_only() -> None:
    run_a, run_b = "a" * 32, "b" * 32
    undone = [
        UndoneName(speaker="S1", name="Ana", run=run_a),
        UndoneName(speaker="S2", name="Luis", run=run_b),
        UndoneName(speaker="S3", name="Eva"),
    ]
    assert [u.speaker for u in kept_undone(undone, run_a)] == ["S1"]
    assert [u.speaker for u in kept_undone(undone, None)] == ["S3"]


def test_what_still_applies() -> None:
    auto = [AutoName(speaker=s, name=n, quote="q", utterance="t0.s0",
                     start=0.0, confidence=0.9)
            for s, n in (("S1", "Ana"), ("S2", "Luis"))]
    speakers = anonymous("S1", "S2", "S3", named={"S1": "Ana", "S2": "Pepe"})
    assert [a.speaker for a in valid_auto(speakers, auto)] == ["S1"]
    suggestions = [
        NameSuggestion(speaker=s, name="X", kind="self", quote="q",
                       utterance="t0.s0", start=0.0, confidence=0.5,
                       reason="unsure")
        for s in ("S1", "S3", "S9")
    ]
    assert [s.speaker for s in open_suggestions(speakers, suggestions)] == [
        "S3"
    ]
