"""Dialogue acts: category mapping of each corpus, records and the importer."""

from typing import Iterable, Iterator, List, Optional

import pytest

from src.application.skills.dialogue_acts import import_corpus
from src.domain.errors import InvalidInputError
from src.domain.skills.dialogue_acts import (
    CATEGORIES,
    MAPPERS,
    ActRecord,
    CorpusInfo,
    SourceDialogue,
    SourceUtterance,
    ami_categories,
    category_key,
    dihana_categories,
    is_category,
    label_distribution,
    mrda_categories,
    mrda_tags,
    own_categories,
    swda_categories,
    swda_cluster,
    to_records,
)


def utt(
    label: str, uid: str = "u1", speaker: str = "A", text: str = "hola",
    responds_to: Optional[str] = None, reply_type: Optional[str] = None,
) -> SourceUtterance:
    """A source segment with the given label."""
    return SourceUtterance(id=uid, speaker=speaker, text=text,
                           source_label=label, responds_to=responds_to,
                           reply_type=reply_type)


INFO = CorpusInfo(name="own", licence="test", non_commercial=False,
                  language="es")


def test_thirteen_categories_with_opinion_split_from_information() -> None:
    assert len(CATEGORIES) == 13 == len(set(CATEGORIES))
    assert {"informacion", "opinion"} <= set(CATEGORIES)
    assert is_category("rechazo") and not is_category("Rechazo")


def test_every_mapping_points_only_at_our_categories() -> None:
    from src.domain.skills import dialogue_acts as module
    tables = [module.AMI_TYPES, module.SWDA_TAGS, module.DIHANA_LEVEL1,
              dict(module._MRDA_GENERAL), dict(module._MRDA_SPECIFIC)]
    for table in tables:
        for categories in table.values():
            assert all(is_category(c) for c in categories), categories
    assert set(MAPPERS) == {"own", "ami", "mrda", "swda", "dihana"}


class TestAmi:
    @pytest.mark.parametrize("label, expected", [
        ("bck", ("escucha",)), ("inf", ("informacion",)),
        ("ass", ("opinion",)), ("sug", ("propuesta", "peticion")),
        ("off", ("compromiso", "propuesta")), ("el.inf", ("pregunta",)),
        ("be.pos", ("relacion",)), ("fra", ("otro",)), ("unknown", ()),
    ])
    def test_types(self, label: str, expected: tuple[str, ...]) -> None:
        assert ami_categories(utt(label), None) == expected

    def test_content_reply_to_an_elicitation_is_an_answer(self) -> None:
        first = utt("el.inf", "q")
        assert ami_categories(utt("inf", responds_to="q"),
                              first) == ("respuesta",)
        assert ami_categories(utt("bck", responds_to="q"),
                              first) == ("escucha",)

    def test_positive_and_negative_replies(self) -> None:
        proposal = utt("sug", "p")
        assert ami_categories(utt("ass", reply_type="POS"),
                              proposal) == ("aceptacion",)
        assert ami_categories(utt("inf", reply_type="NEG"),
                              proposal) == ("rechazo",)
        assert ami_categories(utt("ass", reply_type="PART"),
                              proposal) == ("opinion",)

    def test_positive_reply_to_an_inform_is_not_an_acceptance(self) -> None:
        assert ami_categories(utt("ass", reply_type="POS"),
                              utt("inf", "i")) == ("opinion",)


class TestMrda:
    @pytest.mark.parametrize("label, general, specific", [
        ("s^aa", "s", ["aa"]), ("s.%--", "s", ["%--"]),
        ("qy^d^rt|s", "qy", ["d", "rt"]), ("b", "b", []), ("", "", []),
        ("s^bk:s", "s", ["bk"]),
    ])
    def test_tags(self, label: str, general: str,
                  specific: List[str]) -> None:
        assert mrda_tags(label) == (general, specific)

    @pytest.mark.parametrize("label, expected", [
        ("s", ("informacion", "opinion")), ("s^aa", ("aceptacion",)),
        ("s^ar", ("rechazo",)), ("s^cc", ("compromiso",)),
        ("s^co", ("peticion",)), ("s^cs", ("propuesta",)),
        ("s^ny", ("respuesta",)), ("qw", ("pregunta",)),
        ("b", ("escucha",)), ("s^bk", ("escucha",)), ("fg", ("gestion",)),
        ("s^tc", ("gestion",)), ("s^ft", ("relacion",)),
        ("%--", ("otro",)), ("qh", ()), ("z", ()),
        ("s.%--", ("informacion", "opinion", "otro")),
        ("b.%", ("escucha", "otro")),
    ])
    def test_categories(self, label: str,
                        expected: tuple[str, ...]) -> None:
        assert mrda_categories(utt(label), None) == expected


class TestSwda:
    @pytest.mark.parametrize("raw, cluster", [
        ("sd", "sd"), ("sd^e", "sd"), ("qy^d", "qy^d"), ("qr", "qy"),
        ("fe", "ba"), ("cc", "oo_co_cc"), ("nn^e", "ng"), ("ny^e", "na"),
        ("sd(^q)", "sd"), ("+@", "+"), ("aap", "aap_am"),
        ("sd, sv", "sd"), ("o", "fo_o_fw_\"_by_bc"), ("", ""),
    ])
    def test_cluster(self, raw: str, cluster: str) -> None:
        assert swda_cluster(raw) == cluster

    @pytest.mark.parametrize("raw, expected", [
        ("sd", ("informacion",)), ("sv", ("opinion",)),
        ("b", ("escucha",)), ("aa", ("aceptacion",)),
        ("ad", ("peticion",)), ("+", ()), ("qh", ()),
    ])
    def test_categories(self, raw: str, expected: tuple[str, ...]) -> None:
        assert swda_categories(utt(raw), None) == expected


@pytest.mark.parametrize("label, expected", [
    ("Pregunta:Hora_salida:Destino", ("pregunta",)),
    ("Respuesta:Dia:Dia", ("respuesta",)),
    ("Confirmacion:Destino:Destino", ("pregunta",)),
    ("Negacion:Destino:Nil", ("respuesta", "rechazo")),
    ("No_entendido:Nil:Nil", ("pregunta",)),
    ("Afirmación:Nil:Nil", ("respuesta", "aceptacion")),
    ("Indefinida:Nil:Nil", ()),
])
def test_dihana_level_one(label: str, expected: tuple[str, ...]) -> None:
    assert dihana_categories(utt(label), None) == expected


def test_own_label_is_the_category() -> None:
    assert own_categories(utt("compromiso"), None) == ("compromiso",)
    assert own_categories(utt("nada"), None) == ()


def test_records_keep_context_and_skip_empty_segments() -> None:
    utterances = [
        utt("pregunta", "a", "A", "¿Vale?"),
        utt("otro", "b", "B", "   "),
        utt("respuesta", "c", "B", " Sí,  vale. ", responds_to="a"),
        utt("informacion", "d", "A", "Bien."),
    ]
    dialogue = SourceDialogue(id="d1", session_type="negociacion",
                              utterances=utterances)
    records = to_records(INFO, dialogue, own_categories, context=1)
    assert [r.id for r in records] == ["a", "c", "d"]
    assert [r.order for r in records] == [0, 1, 2]
    assert records[1].text == "Sí, vale."
    assert records[1].responds_to == "a"
    assert [c.text for c in records[2].context] == ["Sí, vale."]
    assert records[0].context == []
    assert records[0].session_type == "negociacion"
    assert to_records(INFO, dialogue, own_categories, 0)[2].context == []
    with pytest.raises(ValueError):
        to_records(INFO, dialogue, own_categories, -1)


def test_category_key_and_distribution() -> None:
    assert category_key(["opinion", "informacion"]) == "informacion|opinion"
    assert category_key([]) == "-"
    dialogue = SourceDialogue(id="d", utterances=[
        utt("pregunta", "a"), utt("pregunta", "b"), utt("x", "c")])
    records = to_records(INFO, dialogue, own_categories)
    assert label_distribution(records) == {"pregunta": 2, "-": 1}


class FakeCorpus:
    """A corpus held in memory."""

    def __init__(self, info: CorpusInfo,
                 dialogues: List[SourceDialogue]) -> None:
        """Serve ``dialogues`` under ``info``."""
        self._info = info
        self._dialogues = dialogues

    @property
    def info(self) -> CorpusInfo:
        """Return the corpus description."""
        return self._info

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield the dialogues."""
        yield from self._dialogues


class ListSink:
    """Keep written records in a list."""

    def __init__(self) -> None:
        """Start empty."""
        self.records: List[ActRecord] = []

    def write(self, records: Iterable[ActRecord]) -> int:
        """Store and count the records."""
        self.records.extend(records)
        return len(self.records)


def corpus(name: str = "mrda") -> FakeCorpus:
    """Two small dialogues of a fake corpus called ``name``."""
    info = CorpusInfo(name=name, licence="CC BY 4.0",
                      non_commercial=name == "swda", language="en")
    return FakeCorpus(info, [
        SourceDialogue(id="m1", utterances=[
            utt("qy", "1"), utt("s^aa", "2", "B"), utt("z", "3")]),
        SourceDialogue(id="m2", utterances=[utt("b", "4")]),
    ])


class TestImportCorpus:
    def test_summary_counts_categories_and_unmapped_labels(self) -> None:
        sink = ListSink()
        summary = import_corpus(corpus(), sink, context=2)
        assert summary.dialogues == 2 and summary.records == 4
        assert summary.mapped == 3
        assert summary.distribution == {
            "-": 1, "aceptacion": 1, "escucha": 1, "pregunta": 1}
        assert summary.unmapped_labels == {"z": 1}
        assert [r.corpus for r in sink.records] == ["mrda"] * 4
        assert sink.records[1].context[0].text == "hola"

    def test_limit_stops_after_that_many_dialogues(self) -> None:
        sink = ListSink()
        summary = import_corpus(corpus(), sink, limit=1)
        assert summary.dialogues == 1 and summary.records == 3

    def test_non_commercial_flag_is_carried(self) -> None:
        sink = ListSink()
        summary = import_corpus(corpus("swda"), sink)
        assert summary.non_commercial
        assert all(r.non_commercial for r in sink.records)

    @pytest.mark.parametrize("kwargs, message", [
        ({"context": -1}, "--context"), ({"limit": 0}, "--limit"),
    ])
    def test_invalid_options(self, kwargs: dict[str, int],
                             message: str) -> None:
        with pytest.raises(InvalidInputError, match=message):
            import_corpus(corpus(), ListSink(), **kwargs)

    def test_unknown_corpus_is_refused(self) -> None:
        with pytest.raises(InvalidInputError, match="No label mapping"):
            import_corpus(corpus("nope"), ListSink())
