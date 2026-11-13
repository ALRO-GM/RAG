"""Tokenizer shared by indexing and querying for lexical retrieval."""

import re
from typing import List

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be
    because been before being below between both but by can could did do
    does doing down during each few for from further had has have having he
    her here hers him his how i if in into is it its itself just me more
    most my no nor not now of off on once only or other our out over own
    same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what
    when where which while who whom why will with would you your vllm use
    used using""".split()
)


def _stem(token: str) -> str:
    """Apply a light English suffix stemmer (plural and -ing/-ed forms)."""
    if len(token) <= 4 or token.isdigit():
        return token
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, bigrams: bool = False) -> List[str]:
    """Split text into lowercase lexical terms.

    Identifiers are kept whole *and* split into their snake_case and
    camelCase parts, so ``get_kv_cache`` matches both the exact
    identifier and a paraphrase mentioning ``kv cache``.

    With ``bigrams`` every pair of consecutive word stems is also a
    term (``data~parallel``), which rewards chunks where the words of
    the question appear together and in the same order.
    """
    tokens: List[str] = []
    stems: List[str] = []
    for word in _WORD.findall(text):
        lower = word.lower()
        parts = [
            p.lower()
            for chunk in word.split("_")
            for p in _CAMEL.findall(chunk)
        ]
        if len(parts) > 1 and lower not in STOPWORDS:
            tokens.append(lower)
        for part in parts:
            if part not in STOPWORDS and len(part) > 1:
                stems.append(_stem(part))
                tokens.append(stems[-1])
    if bigrams:
        tokens.extend(f"{a}~{b}" for a, b in zip(stems, stems[1:]))
    return tokens


_IDENTIFIER = re.compile(
    r"[A-Za-z0-9]+_[A-Za-z0-9_]*"        # snake_case, FP8_MIN
    r"|[a-z]+[A-Z][A-Za-z0-9]*"           # camelCase, vLLM
    r"|[A-Z]+[a-z0-9]+[A-Z][A-Za-z0-9]*"  # PascalCase, KVCacheManager
    r"|[A-Za-z_]\w*\.[A-Za-z_]\w*"        # module.attribute, file.py
    r"|\w+\(\)?"                          # call()
    r"|`[^`]+`"                           # `code`
    r"|--[a-z][\w-]*"                     # --cli-flag
    r"|[\w.-]*/[\w./-]+"                  # path/to/file
)


def _bare(word: str) -> str:
    """Strip sentence punctuation and a possessive from a word."""
    core = word.strip(".,;:!?\"')]}>[{<")
    for suffix in ("'s", "\u2019s"):
        if core.endswith(suffix):
            return core[:-2]
    return core


def identifier_share(text: str) -> float:
    """Share of the words of ``text`` that look like code identifiers.

    Identifiers (``snake_case``, ``camelCase``, ``module.attr``,
    ``call()``, flags, paths, back-quoted code) are tokens a reader
    expects to match *exactly*: for them a lexical match is strong
    evidence, while an embedding only knows them approximately. Plain
    prose scores 0.
    """
    words = text.split()
    if not words:
        return 0.0
    hits = sum(1 for word in words if _IDENTIFIER.fullmatch(_bare(word)))
    return hits / len(words)


def adaptive_weight(
    weight: float, identifier_weight: float, share: float,
    saturation: float,
) -> float:
    """Semantic weight for a query, given its identifier share.

    Interpolates from ``weight`` (prose, share 0) to
    ``identifier_weight`` (share at or above ``saturation``): the more
    exact tokens a query carries, the more the lexical ranking counts.
    """
    if saturation <= 0:
        return identifier_weight if share > 0 else weight
    mix = min(max(share / saturation, 0.0), 1.0)
    return weight + (identifier_weight - weight) * mix
