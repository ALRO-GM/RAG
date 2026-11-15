"""Readers of the public dialogue-act corpora, straight from their archives.

Each reader takes the archive exactly as its official page distributes
it (nothing is unpacked) and yields
:class:`~src.domain.skills.dialogue_acts.SourceDialogue` objects with
the corpus's own labels; the mapping to our categories is the domain's.
Where to download each archive, and its licence, is in
``docs/skills/DIALOGUE_ACT_DATA.md``:

- AMI: ``ami_public_manual_1.6.2.zip`` (NXT XML; CC BY 4.0).
- ICSI MRDA: ``ICSI_core_NXT.zip`` (NXT XML; CC BY 4.0).
- Switchboard-DA: ``swda.zip`` (CSV; CC BY-NC-SA 3.0: non-commercial).
- DIHANA: ``dihanaCorpus.tgz`` (Latin-1 text; CC BY 4.0).
- Our Spanish set: a directory of ``*.txt`` files
  (:mod:`src.domain.skills.dialogue_act_set`).

The archives are third-party data: XML is parsed with the standard
library parser (no external entities or DTDs are fetched by it), and
only the members a reader needs are opened.
"""

import csv
import io
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from src.application.skills.dialogue_acts import DialogueActCorpus
from src.domain.errors import CorruptedDataError, NotFoundError
from src.domain.skills.dialogue_act_set import parse_set
from src.domain.skills.dialogue_acts import (
    CorpusInfo,
    SourceDialogue,
    SourceUtterance,
)

NITE = "{http://nite.sourceforge.net/}"
_HREF_ID = re.compile(r"id\(([^)]*)\)")
_PUNCT = re.compile(r"^[^\w]+$", re.UNICODE)
_OPENING = "¿¡("


def _existing(path: str) -> Path:
    """Return ``path`` as a Path; refuse a missing file or directory."""
    found = Path(path)
    if not found.exists():
        raise NotFoundError(f"{path} does not exist.")
    return found


def _zip(path: str) -> zipfile.ZipFile:
    """Open a zip archive; refuse a file that is not one."""
    try:
        return zipfile.ZipFile(_existing(path))
    except zipfile.BadZipFile as exc:
        raise CorruptedDataError(f"{path} is not a zip archive: {exc}")


def _xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    """Parse one XML member of ``archive``."""
    try:
        return ET.fromstring(archive.read(name))
    except ET.ParseError as exc:
        raise CorruptedDataError(f"{name}: {exc}")


def join_tokens(tokens: List[str]) -> str:
    """Join word tokens, attaching punctuation to the word before it.

    Opening marks (``¿``, ``¡``, ``(``) attach to the word after them.
    """
    text = ""
    opening = False
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if text and not opening and (
                not _PUNCT.match(token) or token[0] in _OPENING):
            text += " "
        text += token
        opening = token[-1] in _OPENING
    return text


def href_ids(href: str) -> List[str]:
    """Return the one or two ids of an NXT ``href`` (``id(a)..id(b)``)."""
    return _HREF_ID.findall(href)


class _Words:
    """The elements of one NXT words file, in document order."""

    def __init__(self, root: ET.Element) -> None:
        """Index every element that has an id."""
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.starts: List[Optional[float]] = []
        self.position: Dict[str, int] = {}
        for element in root:
            identifier = element.get(f"{NITE}id")
            if identifier is None:
                continue
            self.position[identifier] = len(self.ids)
            self.ids.append(identifier)
            is_word = element.tag == "w"
            self.texts.append((element.text or "") if is_word else "")
            start = element.get("starttime") if is_word else None
            try:
                self.starts.append(float(start) if start else None)
            except ValueError:
                self.starts.append(None)

    def span(self, href: str) -> Tuple[str, Optional[float]]:
        """Return the text and start time of the words ``href`` covers."""
        ids = href_ids(href)
        if not ids or ids[0] not in self.position:
            return "", None
        first = self.position[ids[0]]
        last = self.position.get(ids[-1], first)
        texts = self.texts[first:last + 1]
        starts = [s for s in self.starts[first:last + 1] if s is not None]
        return join_tokens(texts), (starts[0] if starts else None)


def _ordered(
    segments: List[Tuple[float, str, SourceUtterance]]
) -> List[SourceUtterance]:
    """Sort segments by start time, then by speaker and id."""
    return [s for _, _, s in sorted(segments, key=lambda x: (x[0], x[1]))]


class AmiDialogueActs:
    """AMI meetings with their dialogue acts and adjacency pairs."""

    def __init__(self, archive: str) -> None:
        """Read from ``ami_public_manual_1.6.2.zip`` (or a copy of it)."""
        self.archive = archive

    @property
    def info(self) -> CorpusInfo:
        """Return AMI's name, licence and language."""
        return CorpusInfo(
            name="ami", licence="CC BY 4.0", non_commercial=False,
            language="en",
            url="https://groups.inf.ed.ac.uk/ami/download/")

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield one dialogue per meeting that has dialogue acts."""
        with _zip(self.archive) as archive:
            names = set(archive.namelist())
            types = self._names(archive, "ontologies/da-types.xml",
                                "da-type")
            replies = self._names(archive, "ontologies/ap-types.xml",
                                  "ap-type")
            meetings: Dict[str, List[str]] = defaultdict(list)
            for name in sorted(names):
                if (name.startswith("dialogueActs/")
                        and name.endswith(".dialog-act.xml")):
                    meetings[Path(name).name.split(".")[0]].append(name)
            for meeting, files in sorted(meetings.items()):
                yield self._meeting(archive, names, meeting, files, types,
                                    replies)

    @staticmethod
    def _names(
        archive: zipfile.ZipFile, member: str, tag: str
    ) -> Dict[str, str]:
        """Map the ids of an ontology file to their ``name``."""
        root = _xml(archive, member)
        return {
            e.get(f"{NITE}id", ""): e.get("name", "")
            for e in root.iter(tag)
        }

    def _meeting(
        self,
        archive: zipfile.ZipFile,
        names: set[str],
        meeting: str,
        files: List[str],
        types: Dict[str, str],
        replies: Dict[str, str],
    ) -> SourceDialogue:
        """Read the acts of every speaker of ``meeting``."""
        links = self._pairs(archive, names, meeting, replies)
        segments: List[Tuple[float, str, SourceUtterance]] = []
        for name in files:
            speaker = Path(name).name.split(".")[1]
            words_name = f"words/{meeting}.{speaker}.words.xml"
            if words_name not in names:
                continue
            words = _Words(_xml(archive, words_name))
            for act in _xml(archive, name).iter("dact"):
                identifier = act.get(f"{NITE}id", "")
                label = ""
                text, start = "", None
                for child in act:
                    href = child.get("href", "")
                    if child.get("role") == "da-aspect":
                        found = href_ids(href)
                        label = types.get(found[0], "") if found else ""
                    elif child.tag == f"{NITE}child":
                        text, start = words.span(href)
                source, kind = links.get(identifier, (None, None))
                segments.append((
                    start if start is not None else float("inf"), speaker,
                    SourceUtterance(
                        id=identifier, speaker=speaker, text=text,
                        source_label=label, responds_to=source,
                        reply_type=kind)))
        return SourceDialogue(id=meeting, utterances=_ordered(segments))

    @staticmethod
    def _pairs(
        archive: zipfile.ZipFile,
        names: set[str],
        meeting: str,
        replies: Dict[str, str],
    ) -> Dict[str, Tuple[str, str]]:
        """Map each second part to its first part and reply type."""
        member = f"dialogueActs/{meeting}.adjacency-pairs.xml"
        if member not in names:
            return {}
        links: Dict[str, Tuple[str, str]] = {}
        for pair in _xml(archive, member).iter("adjacency-pair"):
            roles: Dict[str, str] = {}
            for pointer in pair:
                found = href_ids(pointer.get("href", ""))
                if found:
                    roles[pointer.get("role", "")] = found[0]
            if "source" in roles and "target" in roles:
                links.setdefault(roles["target"], (
                    roles["source"], replies.get(roles.get("type", ""), "")))
        return links


_ADJACENCY = re.compile(r"(\d+)([a-z])[a-z]?(?:-\d+)?(\+*)")


class IcsiMrda:
    """ICSI meetings with their MRDA dialogue acts and adjacency pairs.

    The ``adjacency`` attribute (``5a``, ``5b-2``, ``5b+``; several
    joined with ``.``) is read as: number = pair, ``a`` = first part,
    any later letter = a second part, ``+`` = continuation of a part
    (not linked). A second part is linked to the nearest earlier first
    part with the same number in the meeting.
    """

    def __init__(self, archive: str) -> None:
        """Read from ``ICSI_core_NXT.zip`` (or a copy of it)."""
        self.archive = archive

    @property
    def info(self) -> CorpusInfo:
        """Return MRDA's name, licence and language."""
        return CorpusInfo(
            name="mrda", licence="CC BY 4.0", non_commercial=False,
            language="en",
            url="https://groups.inf.ed.ac.uk/ami/icsi/download/")

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield one dialogue per meeting that has dialogue acts."""
        with _zip(self.archive) as archive:
            names = set(archive.namelist())
            meetings: Dict[str, List[str]] = defaultdict(list)
            for name in sorted(names):
                if ("/DialogueActs/" in f"/{name}"
                        and name.endswith(".dialogue-acts.xml")):
                    meetings[Path(name).name.split(".")[0]].append(name)
            for meeting, files in sorted(meetings.items()):
                yield self._meeting(archive, names, meeting, files)

    def _meeting(
        self,
        archive: zipfile.ZipFile,
        names: set[str],
        meeting: str,
        files: List[str],
    ) -> SourceDialogue:
        """Read and order the acts of every channel of ``meeting``."""
        rows: List[Tuple[float, str, str, str, str, str]] = []
        for name in files:
            speaker = Path(name).name.split(".")[1]
            words_name = str(Path(name).parent.parent / "Words"
                             / f"{meeting}.{speaker}.words.xml")
            if words_name not in names:
                continue
            words = _Words(_xml(archive, words_name))
            for act in _xml(archive, name).iter("dialogueact"):
                text = ""
                for child in act:
                    if child.tag == f"{NITE}child":
                        text, _ = words.span(child.get("href", ""))
                try:
                    start = float(act.get("starttime") or "inf")
                except ValueError:
                    start = float("inf")
                rows.append((
                    start, act.get("participant") or speaker,
                    act.get(f"{NITE}id", ""), text, act.get("type", ""),
                    act.get("adjacency", "")))
        rows.sort(key=lambda r: (r[0], r[1]))
        open_pairs: Dict[str, str] = {}
        utterances: List[SourceUtterance] = []
        for _, speaker, identifier, text, label, adjacency in rows:
            responds_to = None
            for token in adjacency.split("."):
                match = _ADJACENCY.fullmatch(token)
                if match is None or match.group(3):
                    continue
                number, part = match.group(1), match.group(2)
                if part == "a":
                    open_pairs[number] = identifier
                elif responds_to is None and number in open_pairs:
                    responds_to = open_pairs[number]
            utterances.append(SourceUtterance(
                id=identifier, speaker=speaker, text=text,
                source_label=label, responds_to=responds_to))
        return SourceDialogue(id=meeting, utterances=utterances)


_SWDA_MARKUP = (
    (re.compile(r"\{[A-Z] "), ""), (re.compile(r"[{}\[\]+/#]"), " "),
    (re.compile(r"<+[^<>]*>+"), " "), (re.compile(r"\(\(|\)\)"), " "),
    (re.compile(r"\s--\s"), " "),
)


def clean_swda(text: str) -> str:
    """Remove Switchboard's disfluency and noise markup from ``text``."""
    for pattern, replacement in _SWDA_MARKUP:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    return " ".join(text.split())


class SwitchboardDialogueActs:
    """Switchboard telephone conversations with SWBD-DAMSL tags."""

    def __init__(self, archive: str) -> None:
        """Read from Potts' ``swda.zip`` (or a copy of it)."""
        self.archive = archive

    @property
    def info(self) -> CorpusInfo:
        """Return SwDA's name, licence and language (non-commercial)."""
        return CorpusInfo(
            name="swda", licence="CC BY-NC-SA 3.0", non_commercial=True,
            language="en",
            url="https://github.com/cgpotts/swda")

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield one dialogue per ``*.utt.csv`` conversation file."""
        with _zip(self.archive) as archive:
            members = sorted(
                n for n in archive.namelist()
                if n.endswith(".utt.csv") and "__MACOSX" not in n)
            for member in members:
                text = archive.read(member).decode("utf-8")
                yield self._conversation(member, text)

    @staticmethod
    def _conversation(member: str, text: str) -> SourceDialogue:
        """Read one conversation file."""
        reader = csv.DictReader(io.StringIO(text))
        missing = {"act_tag", "caller", "text", "transcript_index"} - set(
            reader.fieldnames or [])
        if missing:
            raise CorruptedDataError(
                f"{member}: missing column(s) {', '.join(sorted(missing))}.")
        name = Path(member).name.split(".")[0]
        utterances = [
            SourceUtterance(
                id=f"{name}.{row['transcript_index']}",
                speaker=row["caller"], text=clean_swda(row["text"]),
                source_label=row["act_tag"])
            for row in reader
        ]
        return SourceDialogue(id=name, utterances=utterances)


_DIHANA_SEGMENT = re.compile(
    r"^(?P<text>.*?)\s*\((?P<speaker>[A-Z]):(?P<label>[^)]*)\)\s*"
    r"pal:\s*\d+\s*-\s*\d+\s*$")


def parse_dihana(name: str, text: str) -> SourceDialogue:
    """Parse one DIHANA ``.dia`` file (turn headers + labelled segments).

    Segment lines look like ``¿ Quiere viajar a madrid ? .
    (M:Confirmacion:Destino:Destino) pal: 1 - 7``; ``M`` is the system
    (the wizard), ``U`` the caller. Other lines are turn headers. The
    last ``.`` of a segment is the corpus's segment terminator and is
    dropped.
    """
    utterances: List[SourceUtterance] = []
    for line in text.splitlines():
        match = _DIHANA_SEGMENT.match(line.strip())
        if match is None:
            continue
        words = match.group("text").split()
        if words and words[-1] == ".":
            words.pop()
        utterances.append(SourceUtterance(
            id=f"{name}.{len(utterances)}", speaker=match.group("speaker"),
            text=join_tokens(words), source_label=match.group("label")))
    return SourceDialogue(id=name, utterances=utterances)


class DihanaDialogueActs:
    """DIHANA: Spanish telephone dialogues about train information."""

    def __init__(self, archive: str) -> None:
        """Read from ``dihanaCorpus.tgz`` (or a copy of it)."""
        self.archive = archive

    @property
    def info(self) -> CorpusInfo:
        """Return DIHANA's name, licence and language."""
        return CorpusInfo(
            name="dihana", licence="CC BY 4.0", non_commercial=False,
            language="es", url="https://zenodo.org/records/5572470")

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield one dialogue per ``.dia`` file, in archive order."""
        try:
            with tarfile.open(_existing(self.archive), "r:*") as archive:
                for member in archive:
                    if not (member.isfile() and member.name.endswith(".dia")):
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    text = handle.read().decode("latin-1")
                    yield parse_dihana(Path(member.name).stem, text)
        except tarfile.TarError as exc:
            raise CorruptedDataError(f"{self.archive}: {exc}")


class OwnSpanishSet:
    """Our written Spanish set: every ``*.txt`` file of a directory."""

    def __init__(self, directory: str) -> None:
        """Read the set kept in ``directory``."""
        self.directory = directory

    @property
    def info(self) -> CorpusInfo:
        """Return the set's name, licence and language."""
        return CorpusInfo(
            name="own", licence="project (synthetic, written for it)",
            non_commercial=False, language="es",
            url="examples/skills/dialogue_acts/")

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield the dialogues of every file, files in name order."""
        directory = _existing(self.directory)
        files = sorted(directory.glob("*.txt"))
        if not files:
            raise NotFoundError(f"No *.txt dialogue files in {directory}.")
        for path in files:
            for dialogue in parse_set(path.read_text(encoding="utf-8"),
                                      path.name):
                yield SourceDialogue(
                    id=dialogue.session.id,
                    session_type=dialogue.session.type,
                    utterances=[
                        SourceUtterance(
                            id=u.id, speaker=u.speaker, text=u.text,
                            source_label=u.act, responds_to=u.responds_to)
                        for u in dialogue.utterances
                    ])


READERS: Mapping[str, Callable[[str], DialogueActCorpus]] = {
    "ami": AmiDialogueActs,
    "mrda": IcsiMrda,
    "swda": SwitchboardDialogueActs,
    "dihana": DihanaDialogueActs,
    "own": OwnSpanishSet,
}
"""Reader class of each corpus, by the name the CLI takes."""
