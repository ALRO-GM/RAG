"""Corpus readers and the importer CLI over small archives built here.

Each archive copies the layout of the official distribution
(``docs/skills/DIALOGUE_ACT_DATA.md``) with a few invented segments,
so CI never downloads a corpus.
"""

import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List

import pytest

from src.adapters.outbound.dialogue_act_corpora import (
    AmiDialogueActs,
    DihanaDialogueActs,
    IcsiMrda,
    OwnSpanishSet,
    SwitchboardDialogueActs,
    clean_swda,
    href_ids,
    join_tokens,
    parse_dihana,
)
from src.adapters.outbound.jsonl_act_records import JsonlActRecordSink
from src.application.skills.dialogue_acts import import_corpus
from src.domain.errors import CorruptedDataError, NotFoundError
from tests.integration.test_skills_cli import assert_error, skills

NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def write_zip(path: Path, members: Dict[str, str]) -> str:
    """Write ``members`` (name → text) into a zip at ``path``."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text.encode("latin-1"))
    return str(path)


def words(meeting: str, speaker: str, items: List[tuple[str, float]],
          tag: str = "w") -> str:
    """An NXT words file whose ids are ``<meeting>.<speaker>.wN``."""
    body = "".join(
        f'<{tag} nite:id="{meeting}.{speaker}.w{i}" starttime="{t}">'
        f"{text}</{tag}>"
        for i, (text, t) in enumerate(items))
    return f'<nite:root {NS}>{body}<vocalsound nite:id="v1"/></nite:root>'


def ami_archive(tmp_path: Path) -> str:
    """An AMI-like zip: one meeting, two speakers, one adjacency pair."""
    da = (f'<da-type {NS} nite:id="root">'
          '<da-type nite:id="c" name="task">'
          '<da-type nite:id="ami_da_6" name="sug"/>'
          '<da-type nite:id="ami_da_9" name="ass"/>'
          '</da-type></da-type>')
    ap = (f'<ap-type {NS} nite:id="r"><ap-type nite:id="apt_1" name="POS"/>'
          '</ap-type>')

    def acts(speaker: str, items: List[tuple[str, str, str]]) -> str:
        body = "".join(
            f'<dact nite:id="M.{speaker}.da{n}">'
            f'<nite:pointer role="da-aspect" href="da-types.xml#id({t})"/>'
            f'<nite:child href="M.{speaker}.words.xml#{span}"/></dact>'
            for n, t, span in items)
        return f"<nite:root {NS}>{body}</nite:root>"

    pairs = (f'<nite:root {NS}><adjacency-pair nite:id="p1">'
             '<nite:pointer role="type" href="ap-types.xml#id(apt_1)"/>'
             '<nite:pointer role="source" '
             'href="M.A.dialog-act.xml#id(M.A.da1)"/>'
             '<nite:pointer role="target" '
             'href="M.B.dialog-act.xml#id(M.B.da1)"/>'
             '</adjacency-pair></nite:root>')
    return write_zip(tmp_path / "ami.zip", {
        "ontologies/da-types.xml": da, "ontologies/ap-types.xml": ap,
        "words/M.A.words.xml": words("M", "A", [
            ("Let's", 1.0), ("do", 1.2), ("it", 1.3), (".", 1.3)]),
        "words/M.B.words.xml": words("M", "B", [("Great", 2.0)]),
        "dialogueActs/M.A.dialog-act.xml": acts("A", [
            ("1", "ami_da_6", "id(M.A.w0)..id(M.A.w3)")]),
        "dialogueActs/M.B.dialog-act.xml": acts("B", [
            ("1", "ami_da_9", "id(M.B.w0)")]),
        "dialogueActs/M.adjacency-pairs.xml": pairs,
        # A speaker with acts but no words file is skipped.
        "dialogueActs/M.C.dialog-act.xml": acts("C", []),
    })


def test_ami_reader_orders_by_time_and_links_pairs(tmp_path: Path) -> None:
    (meeting,) = AmiDialogueActs(ami_archive(tmp_path)).dialogues()
    assert meeting.id == "M"
    first, second = meeting.utterances
    assert (first.text, first.source_label) == ("Let's do it.", "sug")
    assert second.responds_to == "M.A.da1" and second.reply_type == "POS"
    sink = JsonlActRecordSink(str(tmp_path / "out" / "ami.jsonl"))
    summary = import_corpus(AmiDialogueActs(ami_archive(tmp_path)), sink)
    assert summary.distribution == {"aceptacion": 1, "propuesta|peticion": 1}


def icsi_archive(tmp_path: Path) -> str:
    """An ICSI-like zip: one meeting, adjacency ``1a``/``1b``/``1b+``."""
    def acts(speaker: str, items: List[tuple[str, object, str, str, str]]
             ) -> str:
        body = "".join(
            f'<dialogueact nite:id="{n}" starttime="{t}" type="{label}" '
            f'{adj} participant="p{speaker}">'
            f'<nite:child href="B.{speaker}.words.xml#{span}"/>'
            "</dialogueact>"
            for n, t, label, adj, span in items)
        return f"<nite:root {NS}>{body}</nite:root>"

    return write_zip(tmp_path / "icsi.zip", {
        "ICSI/Words/B.A.words.xml": words("B", "A", [
            ("Shall", 1.0), ("we", 1.1), ("start", 1.2), ("?", 1.2)]),
        "ICSI/Words/B.B.words.xml": words("B", "B", [
            ("Yes", 2.0), ("sure", 3.0)]),
        "ICSI/DialogueActs/B.A.dialogue-acts.xml": acts("A", [
            ("a1", 1.0, "qy", 'adjacency="1a"', "id(B.A.w0)..id(B.A.w3)"),
        ]),
        "ICSI/DialogueActs/B.B.dialogue-acts.xml": acts("B", [
            ("b1", 2.0, "s^aa", 'adjacency="1b"', "id(B.B.w0)"),
            ("b2", 3.0, "s.%--", 'adjacency="1b+"', "id(B.B.w1)"),
            ("b3", "x", "z", "", "id(missing)"),
        ]),
    })


def test_icsi_reader_links_second_parts(tmp_path: Path) -> None:
    (meeting,) = IcsiMrda(icsi_archive(tmp_path)).dialogues()
    texts = [u.text for u in meeting.utterances]
    assert texts == ["Shall we start?", "Yes", "sure", ""]
    assert meeting.utterances[0].speaker == "pA"
    assert meeting.utterances[1].responds_to == "a1"
    assert meeting.utterances[2].responds_to is None


def test_swda_reader_cleans_markup(tmp_path: Path) -> None:
    header = ("swda_filename,ptb_basename,conversation_no,"
              "transcript_index,act_tag,caller,utterance_index,"
              "subutterance_index,text,pos,trees,ptb_treenumbers\n")
    rows = ('f,b,1,0,qy,A,1,1,"{D So, } do you [ like, + like ] it? /",,,\n'
            "f,b,1,1,ny,B,2,1,Yeah. <laughter> /,,,\n")
    archive = write_zip(tmp_path / "swda.zip", {
        "swda/sw00utt/sw_0001_1.utt.csv": header + rows,
        "__MACOSX/swda/sw00utt/._sw_0001_1.utt.csv": "junk",
    })
    (conversation,) = SwitchboardDialogueActs(archive).dialogues()
    assert conversation.id == "sw_0001_1"
    assert [u.text for u in conversation.utterances] == [
        "So, do you like, like it?", "Yeah."]
    assert SwitchboardDialogueActs(archive).info.non_commercial
    broken = write_zip(tmp_path / "bad.zip",
                       {"swda/x.utt.csv": "a,b\n1,2\n"})
    with pytest.raises(CorruptedDataError, match="missing column"):
        list(SwitchboardDialogueActs(broken).dialogues())


DIA = """M1: ¿ En qué puedo ayudarle ? .
¿ En qué puedo ayudarle ? . (M:Apertura:Nil:Nil) pal: 1 - 6


U1: quería ir a Málaga .
quería ir a Málaga . (U:Pregunta:Hora_salida:Destino) pal: 1 - 5
"""


def test_dihana_reader_reads_dia_members(tmp_path: Path) -> None:
    path = tmp_path / "dihana.tgz"
    with tarfile.open(path, "w:gz") as archive:
        for name, text in (("dihana/data/B1/X/B1_X.dia", DIA),
                           ("dihana/data/B1/X/B1_X.ul", "audio")):
            data = text.encode("latin-1")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    (dialogue,) = DihanaDialogueActs(str(path)).dialogues()
    assert dialogue.id == "B1_X"
    assert [(u.speaker, u.text) for u in dialogue.utterances] == [
        ("M", "¿En qué puedo ayudarle?"), ("U", "quería ir a Málaga")]
    assert dialogue.utterances[1].source_label == (
        "Pregunta:Hora_salida:Destino")
    assert parse_dihana("e", "nothing here").utterances == []


def test_helpers() -> None:
    assert join_tokens(["Hi", ",", "I", "", "(", "am", ")", "."]) == (
        "Hi, I (am).")
    assert href_ids("f.xml#id(a)..id(b)") == ["a", "b"]
    assert clean_swda("{F uh, } ((well)) -- yes #") == "uh, well yes"


@pytest.mark.parametrize("reader", [
    AmiDialogueActs, IcsiMrda, SwitchboardDialogueActs, DihanaDialogueActs,
])
def test_missing_and_broken_archives(tmp_path: Path, reader: type) -> None:
    with pytest.raises(NotFoundError):
        list(reader(str(tmp_path / "none.zip")).dialogues())
    broken = tmp_path / "broken"
    broken.write_text("not an archive")
    with pytest.raises(CorruptedDataError):
        list(reader(str(broken)).dialogues())


def test_broken_xml_member(tmp_path: Path) -> None:
    archive = write_zip(tmp_path / "x.zip", {
        "ontologies/da-types.xml": "<broken", "ontologies/ap-types.xml": ""})
    with pytest.raises(CorruptedDataError, match="da-types"):
        list(AmiDialogueActs(archive).dialogues())


def test_own_set_reader(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text(
        "=== d1 | venta | T\nAna: ¿Lo quiere? | pregunta #q\n"
        "Luis: Sí. | respuesta >q\n", encoding="utf-8")
    (dialogue,) = OwnSpanishSet(str(tmp_path)).dialogues()
    assert dialogue.session_type == "venta"
    assert dialogue.utterances[1].responds_to == "t0.s0"
    with pytest.raises(NotFoundError):
        list(OwnSpanishSet(str(tmp_path / "empty")).dialogues())
    (tmp_path / "empty").mkdir()
    with pytest.raises(NotFoundError, match="No"):
        list(OwnSpanishSet(str(tmp_path / "empty")).dialogues())


def test_sink_leaves_no_file_when_writing_fails(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"

    def broken() -> "list[object]":
        raise RuntimeError("boom")

    sink = JsonlActRecordSink(str(target))
    with pytest.raises(RuntimeError):
        sink.write(x for x in broken())  # type: ignore[misc]
    assert list(tmp_path.iterdir()) == []


class TestCli:
    def test_import_prints_aggregates_and_writes_jsonl(
        self, tmp_path: Path
    ) -> None:
        archive = icsi_archive(tmp_path)
        result = skills(["dialogue_acts_import", "--corpus", "mrda",
                         "--source", archive, "--output", "o/m.jsonl"],
                        tmp_path)
        assert result.returncode == 0, result.stderr
        assert "mrda: 1 dialogues, 3 sentences, 3 mapped" in result.stdout
        assert "pregunta: 1" in result.stdout
        lines = (tmp_path / "o" / "m.jsonl").read_text().splitlines()
        assert json.loads(lines[1])["categories"] == ["aceptacion"]

    def test_unmapped_labels_and_non_commercial_flag(
        self, tmp_path: Path
    ) -> None:
        header = ("swda_filename,ptb_basename,conversation_no,"
                  "transcript_index,act_tag,caller,utterance_index,"
                  "subutterance_index,text,pos,trees,ptb_treenumbers\n")
        archive = write_zip(tmp_path / "s.zip", {
            "swda/a.utt.csv": header + "f,b,1,0,x,A,1,1,<laughter> uh,,,\n"})
        result = skills(["dialogue_acts_import", "--corpus", "swda",
                         "--source", archive, "--output", "s.jsonl",
                         "--limit", "1", "--context", "0"], tmp_path)
        assert result.returncode == 0, result.stderr
        assert "non-commercial" in result.stdout
        assert "unmapped source labels: x 1" in result.stdout

    @pytest.mark.parametrize("args, message", [
        (["--corpus", "nope"], "Unknown corpus"),
        (["--corpus", "ami", "--context", "two"], "--context"),
        (["--corpus", "ami", "--limit", "one"], "--limit"),
        (["--corpus", "ami"], "does not exist"),
    ])
    def test_import_errors(self, tmp_path: Path, args: List[str],
                           message: str) -> None:
        result = skills(["dialogue_acts_import", "--source", "none.zip",
                         "--output", "o.jsonl", *args], tmp_path)
        assert_error(result, message)

    def test_set_command(self, tmp_path: Path) -> None:
        (tmp_path / "set").mkdir()
        (tmp_path / "set" / "a.txt").write_text(
            "=== d1 | venta | T\nAna: ¿Lo quiere? | pregunta #q\n"
            "Luis: Sí. | respuesta >q\n", encoding="utf-8")
        result = skills(["dialogue_acts_set", "--source", "set",
                         "--output", "set.json"], tmp_path)
        assert result.returncode == 0, result.stderr
        assert "1 dialogues, 2 sentences" in result.stdout
        assert "types: venta 1" in result.stdout
        document = json.loads((tmp_path / "set.json").read_text())
        assert document[0]["utterances"][1]["act"] == "respuesta"
        assert_error(skills(["dialogue_acts_set", "--source", "nowhere"],
                            tmp_path), "No dialogues")
