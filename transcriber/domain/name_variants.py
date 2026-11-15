"""Names as a recogniser writes them: honorifics, variants, clusters.

Real recordings write the same person's name in several ways: with an
honorific ("señor Ruiz", "Mr Ruiz"), split in two words or joined
("Ber Mejo" / "Bermejo"), with one letter changed ("Bermeja"), or cut
short ("Berme"). This module (TR-21, issue #116) holds the pure string
work that makes those one name:

- :func:`fold` compares words without case and accents;
- :func:`strip_honorifics` removes the leading honorifics of a name;
- :func:`similar_names` tells whether two spellings may be the same
  name (joined-vs-split words, edit distance that grows with the
  length, a long common prefix), but never a change of the final
  vowel (``Mario`` / ``María``, ``Juan`` / ``Juana`` are two people);
- :func:`cluster_names` groups the spellings of a session into names,
  each shown with its most frequent well-formed spelling.

Nothing here does I/O or imports another layer.
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence, Tuple

HONORIFICS = frozenset({
    "senor", "senora", "senorita", "sr", "sra", "srta", "don", "dona",
    "doctor", "doctora", "profesor", "profesora", "prof",
    "mr", "mrs", "ms", "miss", "dr", "dra", "sir",
})
"""Words that may precede a name and are not part of it (folded)."""

VOWEL_ENDINGS = frozenset({"a", "o", "e"})
"""Final letters whose change makes another name (gender pairs)."""

MIN_PREFIX = 5
"""Letters two spellings must share at the start to be one name cut
short ("Berme" / "Bermejo")."""


def fold(text: str) -> str:
    """Return ``text`` in lower case without accents, for comparisons."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _letters(word: str) -> str:
    """Return the folded letters of ``word`` (no dots or apostrophes)."""
    return "".join(c for c in fold(word) if c.isalpha())


def strip_honorifics(name: str) -> str:
    """Return ``name`` without its leading honorifics ("señor", "Mr.")."""
    words = name.split()
    while words and _letters(words[0]) in HONORIFICS:
        words = words[1:]
    return " ".join(words)


def edit_distance(a: str, b: str) -> int:
    """Return the Levenshtein distance between ``a`` and ``b``."""
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            previous, row[j] = row[j], min(
                row[j] + 1, row[j - 1] + 1, previous + (ca != cb)
            )
    return row[-1]


def _allowed(longest: int) -> int:
    """Return the edits allowed between words of ``longest`` letters."""
    if longest <= 4:
        return 0
    if longest <= 6:
        return 1
    if longest <= 9:
        return 2
    return 3


def _gender_pair(a: str, b: str) -> bool:
    """Tell whether ``a`` and ``b`` differ only in a final vowel.

    ``Mario`` / ``María``, ``Juan`` / ``Juana``: different people.
    """
    short, long_ = sorted((a, b), key=len)
    if len(long_) == len(short) + 1 and long_.startswith(short):
        return long_[-1] in VOWEL_ENDINGS
    return (
        len(a) == len(b) and a[:-1] == b[:-1]
        and a[-1] in VOWEL_ENDINGS and b[-1] in VOWEL_ENDINGS
    )


def _common_prefix(a: str, b: str) -> int:
    """Return how many letters ``a`` and ``b`` share at the start."""
    count = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        count += 1
    return count


SOUNDS: Tuple[Tuple[str, str], ...] = (
    ("tz", "ch"), ("tx", "ch"), ("ll", "y"), ("qu", "k"), ("ce", "se"),
    ("ci", "si"), ("ge", "je"), ("gi", "ji"), ("z", "s"), ("v", "b"),
    ("w", "b"), ("ph", "f"), ("ck", "k"), ("c", "k"), ("x", "ks"),
)
"""Spellings that sound alike in Spanish (and a few in English),
replaced in order by one of them, so a misheard ``Itziar`` / ``Ichiar``
or ``Valdivieso`` / ``Baldiviezo`` meet."""


def sound_key(word: str) -> str:
    """Return a rough key of how a folded word sounds.

    The pairs of :data:`SOUNDS` are made one, a silent ``h`` (not in
    ``ch``) and doubled letters are dropped.
    """
    key = word.replace("ch", "\0")
    for written, sound in SOUNDS:
        key = key.replace(written, sound).replace("ch", "\0")
    key = key.replace("h", "").replace("\0", "ch")
    return re.sub(r"(.)\1+", r"\1", key)


def similar_words(a: str, b: str) -> bool:
    """Tell whether two folded spellings may be the same word.

    Equal, or sounding the same (:func:`sound_key`); or not a gender
    pair and either within the edits allowed for their length or
    sharing a long prefix (one cut short).
    """
    if a == b:
        return True
    if not a or not b or _gender_pair(a, b):
        return False
    a, b = sound_key(a), sound_key(b)
    if a == b:
        return True
    if min(len(a), len(b)) >= MIN_PREFIX and (
        _common_prefix(a, b) >= MIN_PREFIX
    ):
        return True
    return edit_distance(a, b) <= _allowed(max(len(a), len(b)))


def _words(name: str) -> List[str]:
    """Return the folded words of ``name`` without honorifics."""
    return [w for w in (_letters(x) for x in strip_honorifics(name).split())
            if w]


def similar_names(a: str, b: str) -> bool:
    """Tell whether two names may be the same person's.

    Honorifics, case and accents do not count. Two one-word names are
    compared as words; names split differently are compared joined
    ("Ber Mejo" / "Bermejo"); a one-word name matches a longer one
    when it is like one of its words ("Ruiz" / "Ana Ruiz"); two longer
    names match when their first and last words are alike, or when
    one is the other with a first name before it ("Ana Bermejo" /
    "Ber Mejo").
    """
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    if similar_words("".join(wa), "".join(wb)):
        return True
    if len(wa) == 1 or len(wb) == 1:
        one, many = (wa, wb) if len(wa) == 1 else (wb, wa)
        return any(similar_words(one[0], w) for w in many)
    return (
        similar_words(wa[0], wb[0]) and similar_words(wa[-1], wb[-1])
    ) or similar_words("".join(wa[1:]), "".join(wb)) or similar_words(
        "".join(wa), "".join(wb[1:])
    )


def _well_formed(spelling: str) -> Tuple[int, int]:
    """Rank a spelling: capitalised words first, then the fewest words."""
    words = spelling.split()
    capitalised = all(w[:1].isupper() for w in words)
    return int(capitalised), -len(words)


def cluster_names(spellings: Sequence[str]) -> Dict[str, str]:
    """Group spellings into names; return each spelling's shown name.

    ``spellings`` are the names found in a session, one per sighting
    (repeated spellings count as votes for how the name is shown).
    Spellings of several words are grouped first; a one-word spelling
    joins them only when it is like exactly one group, so "Juan" does
    not chain "Juan Ruiz" and "Juan Gil" into one person. A group is
    shown with its most frequent spelling (ties: well formed, fewer
    words, first seen).
    """
    tidy = [strip_honorifics(s).strip() for s in spellings]
    counts = Counter(t for t in tidy if t)
    order = list(counts)
    longer = [s for s in order if len(_words(s)) > 1]
    single = [s for s in order if len(_words(s)) == 1]
    groups: List[List[str]] = []
    for spelling in longer + single:
        matching = [
            g for g in groups if any(similar_names(spelling, s) for s in g)
        ]
        if len(matching) == 1:
            matching[0].append(spelling)
        else:
            groups.append([spelling])
    shown: Dict[str, str] = {}
    for group in groups:
        best = max(group, key=lambda s: (
            counts[s], _well_formed(s), -order.index(s)
        ))
        for spelling in group:
            shown[spelling] = best
    return {
        original: shown[strip_honorifics(original).strip()]
        for original in spellings if strip_honorifics(original).strip()
    }
