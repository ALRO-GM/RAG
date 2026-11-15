"""Dialogue acts: our 13 categories and the records every corpus maps to.

The analyzer classifies each sentence into one first-layer category
(``docs/skills/CONVERSATION.md`` §5, with Información split into
Información and Opinión by the owner's decision on #57). Public
corpora use their own tag sets, so each source label is mapped to the
**set of our categories it is compatible with**: one category when
the source makes the same distinction we do, several when it does not
(MRDA has no fact/opinion split, so a statement is ``informacion`` or
``opinion``), none when the source label has no counterpart (non-speech,
unlabelled). A model's prediction for a sentence counts as right when
it falls inside that set; categories a corpus can never tell apart are
listed in ``docs/skills/DIALOGUE_ACT_DATA.md``.

Everything here is plain data and string handling: the readers of each
corpus's files are adapters.
"""

import re
from collections import Counter
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from pydantic import BaseModel, ConfigDict, Field

CATEGORIES: Tuple[str, ...] = (
    "pregunta", "respuesta", "informacion", "opinion", "propuesta",
    "peticion", "aceptacion", "rechazo", "compromiso", "escucha",
    "gestion", "relacion", "otro",
)
"""The 13 first-layer categories, by id (Spanish, no accents)."""

SECOND_PARTS: Tuple[str, ...] = ("respuesta", "aceptacion", "rechazo")
"""Categories that reply to an earlier sentence and must point at it."""

SESSION_TYPES: Tuple[str, ...] = (
    "reunion_decision", "lluvia_ideas", "resolucion_problemas",
    "negociacion", "entrevista", "feedback", "conflicto", "coaching",
    "venta",
)
"""Declared conversation types (CONVERSATION.md §4.2)."""

Categories = Tuple[str, ...]


class _Frozen(BaseModel):
    """Immutable model that refuses unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CorpusInfo(_Frozen):
    """Where a corpus comes from and what may be done with it.

    ``non_commercial`` marks a licence that allows internal evaluation
    only (Switchboard-DA): such data never trains a shipped model.
    """

    name: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    non_commercial: bool
    language: str = Field(min_length=2)
    url: str = ""


class SourceUtterance(_Frozen):
    """One labelled segment as a corpus stores it.

    ``responds_to`` is the id of the segment it replies to, when the
    corpus links them (adjacency pairs); ``reply_type`` is the kind of
    link the corpus gives (AMI: positive, negative, …).
    """

    id: str = Field(min_length=1)
    speaker: str
    text: str
    source_label: str
    responds_to: Optional[str] = None
    reply_type: Optional[str] = None


class SourceDialogue(_Frozen):
    """The labelled segments of one conversation, in spoken order."""

    id: str = Field(min_length=1)
    utterances: List[SourceUtterance]
    session_type: Optional[str] = None


class ContextLine(_Frozen):
    """A sentence said before the one being classified."""

    speaker: str
    text: str


class ActRecord(_Frozen):
    """One sentence in the common format all corpora are imported to.

    ``categories`` are the categories of ours the source label is
    compatible with (empty: not usable for evaluation);
    ``source_label`` keeps the corpus's own tag so a mapping can be
    revised without re-reading the corpus.
    """

    corpus: str
    licence: str
    non_commercial: bool
    language: str
    dialogue: str
    id: str
    order: int = Field(ge=0)
    speaker: str
    text: str
    context: List[ContextLine]
    source_label: str
    categories: List[str]
    responds_to: Optional[str] = None
    session_type: Optional[str] = None


Mapper = Callable[[SourceUtterance, Optional[SourceUtterance]], Categories]
"""Map a segment (and the segment it replies to) to our categories."""


def is_category(value: str) -> bool:
    """Tell whether ``value`` is one of the 13 category ids."""
    return value in CATEGORIES


def to_records(
    info: CorpusInfo,
    dialogue: SourceDialogue,
    mapper: Mapper,
    context: int = 3,
) -> List[ActRecord]:
    """Turn a corpus dialogue into common records with ``context`` lines.

    Empty segments (no text after stripping) are left out, and so is
    their place in the context.
    """
    if context < 0:
        raise ValueError("context must be >= 0")
    by_id = {u.id: u for u in dialogue.utterances}
    records: List[ActRecord] = []
    previous: List[ContextLine] = []
    for utterance in dialogue.utterances:
        text = " ".join(utterance.text.split())
        if not text:
            continue
        first = (by_id.get(utterance.responds_to)
                 if utterance.responds_to else None)
        records.append(ActRecord(
            corpus=info.name, licence=info.licence,
            non_commercial=info.non_commercial, language=info.language,
            dialogue=dialogue.id, id=utterance.id, order=len(records),
            speaker=utterance.speaker, text=text,
            context=previous[-context:] if context else [],
            source_label=utterance.source_label,
            categories=list(mapper(utterance, first)),
            responds_to=utterance.responds_to,
            session_type=dialogue.session_type,
        ))
        previous.append(ContextLine(speaker=utterance.speaker, text=text))
    return records


def category_key(categories: Sequence[str]) -> str:
    """Return the key a label distribution counts ``categories`` under.

    One category is its id; a compatible set is its ids joined with
    ``|`` in the order of :data:`CATEGORIES`; none is ``-``.
    """
    if not categories:
        return "-"
    return "|".join(c for c in CATEGORIES if c in categories)


def label_distribution(records: Iterable[ActRecord]) -> Dict[str, int]:
    """Count records per :func:`category_key`, most frequent first."""
    counts = Counter(category_key(r.categories) for r in records)
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# --- Our own set: the source label is already our category -------------


def own_categories(
    utterance: SourceUtterance, first: Optional[SourceUtterance]
) -> Categories:
    """Map a sentence of our Spanish set: its label is the category."""
    return (utterance.source_label,) if is_category(
        utterance.source_label) else ()


# --- AMI (15 dialogue-act types + adjacency pairs) ---------------------

AMI_TYPES: Mapping[str, Categories] = {
    "bck": ("escucha",),
    "stl": ("otro",),
    "fra": ("otro",),
    "inf": ("informacion",),
    "el.inf": ("pregunta",),
    "sug": ("propuesta", "peticion"),
    "off": ("compromiso", "propuesta"),
    "el.sug": ("pregunta",),
    "ass": ("opinion",),
    "und": ("escucha",),
    "el.ass": ("pregunta",),
    "el.und": ("pregunta",),
    "be.pos": ("relacion",),
    "be.neg": ("relacion",),
    "oth": ("otro",),
}
"""AMI type (``ontologies/da-types.xml`` name) → our categories.

``sug`` covers suggestions and requests alike, and ``off`` offers with
and without conditions, so each maps to two categories.
"""

_AMI_ELICIT = ("el.inf", "el.ass", "el.und", "el.sug")
_AMI_CONTENT = ("inf", "ass", "sug", "off")


def ami_categories(
    utterance: SourceUtterance, first: Optional[SourceUtterance]
) -> Categories:
    """Map an AMI dialogue act, using its adjacency pair when it has one.

    - A content act (inform, assess, suggest, offer) that replies to an
      elicitation is a ``respuesta``.
    - An assessment or comment about understanding that replies
      positively (adjacency-pair type ``POS``) to a suggestion, offer or
      assessment is an ``aceptacion``.
    - An assessment, comment, inform or other act that replies
      negatively (``NEG``) to a suggestion, offer, assessment or inform
      is a ``rechazo`` (disagreement or correction).
    - Anything else keeps its own type's mapping: partial, uncertain and
      elaborating replies, positive replies to an inform (AMI links
      introductions and acknowledgements that way), backchannels,
      stalls and fragments.
    """
    label = utterance.source_label.lower()
    base = AMI_TYPES.get(label, ())
    if first is None:
        return base
    opener = first.source_label.lower()
    reply = (utterance.reply_type or "").upper()
    if opener in _AMI_ELICIT and label in _AMI_CONTENT:
        return ("respuesta",)
    if (reply == "POS" and label in ("ass", "und")
            and opener in ("sug", "off", "ass")):
        return ("aceptacion",)
    if (reply == "NEG" and label in ("ass", "und", "inf", "oth")
            and opener in ("sug", "off", "ass", "inf")):
        return ("rechazo",)
    return base


# --- ICSI MRDA (general tag + specific tags, e.g. ``s^aa``) ------------

_MRDA_SPECIFIC: Tuple[Tuple[str, Categories], ...] = (
    ("aa", ("aceptacion",)), ("aap", ("aceptacion",)),
    ("ar", ("rechazo",)), ("arp", ("rechazo",)),
    ("cc", ("compromiso",)), ("co", ("peticion",)),
    ("cs", ("propuesta",)),
    ("ny", ("respuesta",)), ("nn", ("respuesta",)),
    ("na", ("respuesta",)), ("ng", ("respuesta",)),
    ("no", ("respuesta",)), ("nd", ("respuesta",)),
    ("am", ("respuesta",)),
    ("ft", ("relacion",)), ("fa", ("relacion",)),
    ("fw", ("relacion",)), ("by", ("relacion",)),
    ("j", ("relacion",)),
    ("tc", ("gestion",)),
    ("bu", ("pregunta",)), ("br", ("pregunta",)),
    ("d", ("pregunta",)), ("g", ("pregunta",)),
    ("bk", ("escucha",)), ("ba", ("escucha",)),
    ("m", ("escucha",)), ("r", ("escucha",)),
)
"""MRDA specific tags in priority order (first found wins)."""

_MRDA_GENERAL: Mapping[str, Categories] = {
    "s": ("informacion", "opinion"),
    "qy": ("pregunta",), "qw": ("pregunta",), "qr": ("pregunta",),
    "qrr": ("pregunta",), "qo": ("pregunta",),
    "qh": (),
    "b": ("escucha",),
    "fg": ("gestion",),
    "fh": ("otro",), "h": ("otro",),
    "%": ("otro",), "%-": ("otro",), "%--": ("otro",),
}
"""MRDA general tags → our categories when no specific tag decides."""


_MRDA_DISRUPTIONS = ("%", "%-", "%--")


def mrda_tags(label: str) -> Tuple[str, List[str]]:
    """Split an MRDA label into its general tag and its specific tags.

    Only the first of several ``|``-separated labels is read; ``:``
    attachments are dropped, and a ``.%-``-style disruption mark is
    returned as one more specific tag (``s.%--`` → ``s``, ``[%--]``).
    """
    first = label.strip().split("|")[0].split(":")[0]
    head, *marks = first.split(".")
    disruptions = [m for m in marks if m in _MRDA_DISRUPTIONS]
    parts = [p for p in head.split("^") if p] + disruptions
    if not parts:
        return "", []
    general = parts[0]
    specific = [p.split(".")[0].rstrip("*") for p in parts[1:]]
    return general, specific


def mrda_categories(
    utterance: SourceUtterance, first: Optional[SourceUtterance]
) -> Categories:
    """Map an MRDA label: decisive specific tag first, else the general.

    An act marked as disrupted or abandoned (``s.%--``) may also be
    ``otro``, so ``otro`` joins whatever the rest of the label maps to.
    """
    general, specific = mrda_tags(utterance.source_label)
    found: Categories = _MRDA_GENERAL.get(general, ())
    for tag, categories in _MRDA_SPECIFIC:
        if tag in specific:
            found = categories
            break
    if found and any(m in specific for m in _MRDA_DISRUPTIONS):
        return tuple(c for c in CATEGORIES if c in found or c == "otro")
    return found


# --- Switchboard-DA (SWBD-DAMSL, clustered to 43 tags) -----------------

SWDA_TAGS: Mapping[str, Categories] = {
    "sd": ("informacion",), "sv": ("opinion",),
    "b": ("escucha",), "bk": ("escucha",), "ba": ("escucha",),
    "bh": ("escucha",), "b^m": ("escucha",),
    "aa": ("aceptacion",), "aap_am": ("aceptacion",),
    "ar": ("rechazo",), "arp_nd": ("rechazo", "respuesta"),
    "qy": ("pregunta",), "qw": ("pregunta",), "qo": ("pregunta",),
    "qrr": ("pregunta",), "qy^d": ("pregunta",), "qw^d": ("pregunta",),
    "^g": ("pregunta",), "br": ("pregunta",),
    "qh": (),
    "ny": ("respuesta",), "nn": ("respuesta",), "na": ("respuesta",),
    "ng": ("respuesta",), "no": ("respuesta",),
    "ad": ("peticion",),
    "oo_co_cc": ("compromiso", "propuesta"),
    "bf": ("gestion",),
    "fp": ("relacion",), "fc": ("relacion",), "fa": ("relacion",),
    "ft": ("relacion",), "bd": ("relacion",),
    "%": ("otro",), "h": ("otro",), "^h": ("otro",), "t1": ("otro",),
    "t3": ("otro",), "^2": ("otro",),
    "^q": ("informacion", "opinion"),
    "fo_o_fw_\"_by_bc": (),
    "x": (), "+": (),
}
"""Clustered SwDA tag → our categories."""


def swda_cluster(act_tag: str) -> str:
    """Cluster a raw SWBD-DAMSL tag into the usual 43 classes.

    Follows the clustering of Jurafsky et al. as implemented in Potts'
    ``swda`` reader: the first of several tags, ``^`` diacritics
    dropped except for ``qy^d``, ``qw^d`` and ``b^m``, and the merged
    classes (``qr``→``qy``, ``fe``→``ba``, offers/options/commits, …).
    """
    tags = [t for t in re.split(r"\s*[,;]\s*", act_tag.strip()) if t]
    if not tags:
        return ""
    tag = tags[0]
    if tag in ("qy^d", "qw^d", "b^m"):
        return tag
    if tag == "nn^e":
        return "ng"
    if tag == "ny^e":
        return "na"
    tag = re.sub(r"(.)\^.*", r"\1", tag)
    tag = re.sub(r"[()@*]", "", tag)
    merged = {
        "qr": "qy", "fe": "ba", "oo": "oo_co_cc", "co": "oo_co_cc",
        "cc": "oo_co_cc", "fx": "sv", "aap": "aap_am", "am": "aap_am",
        "arp": "arp_nd", "nd": "arp_nd", "fo": "fo_o_fw_\"_by_bc",
        "o": "fo_o_fw_\"_by_bc", "fw": "fo_o_fw_\"_by_bc",
        "\"": "fo_o_fw_\"_by_bc", "by": "fo_o_fw_\"_by_bc",
        "bc": "fo_o_fw_\"_by_bc",
    }
    return merged.get(tag, tag)


def swda_categories(
    utterance: SourceUtterance, first: Optional[SourceUtterance]
) -> Categories:
    """Map a raw Switchboard-DA tag through its clustered class."""
    return SWDA_TAGS.get(swda_cluster(utterance.source_label), ())


# --- DIHANA (Spanish, level 1 of its three-level scheme) ---------------

DIHANA_LEVEL1: Mapping[str, Categories] = {
    "pregunta": ("pregunta",),
    "respuesta": ("respuesta",),
    "confirmacion": ("pregunta",),
    "nueva_consulta": ("pregunta",),
    "espera": ("gestion",),
    "cierre": ("relacion",),
    "afirmacion": ("respuesta", "aceptacion"),
    "apertura": ("relacion",),
    "no_entendido": ("pregunta",),
    "negacion": ("respuesta", "rechazo"),
    "indefinida": (),
}
"""DIHANA level-1 label (lower case, no accents) → our categories."""

_ACCENTS = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")


def dihana_categories(
    utterance: SourceUtterance, first: Optional[SourceUtterance]
) -> Categories:
    """Map the level-1 part of a DIHANA label (``L1:L2:L3``)."""
    level1 = utterance.source_label.split(":")[0].strip()
    key = level1.translate(_ACCENTS).lower().replace(" ", "_")
    return DIHANA_LEVEL1.get(key, ())


MAPPERS: Mapping[str, Mapper] = {
    "own": own_categories,
    "ami": ami_categories,
    "mrda": mrda_categories,
    "swda": swda_categories,
    "dihana": dihana_categories,
}
"""Mapper of each corpus the importer knows."""
