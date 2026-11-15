"""Import dialogue-act corpora into one common JSONL format.

Each corpus is read by an adapter behind :class:`DialogueActCorpus`;
its labels are mapped to our 13 categories by the corpus's mapper in
:mod:`src.domain.skills.dialogue_acts`, and the records go to an
:class:`ActRecordSink` (a JSONL file under the git-ignored ``data/``).
The summary returned holds only aggregates (counts per category and per
unmapped source label), which is all the documentation may quote.
"""

from collections import Counter
from typing import Dict, Iterable, Iterator, Optional, Protocol

from pydantic import BaseModel

from src.domain.errors import InvalidInputError
from src.domain.skills.dialogue_acts import (
    MAPPERS,
    ActRecord,
    CorpusInfo,
    SourceDialogue,
    category_key,
    to_records,
)


class DialogueActCorpus(Protocol):
    """A corpus of labelled dialogues read from disk."""

    @property
    def info(self) -> CorpusInfo:
        """Return the corpus's name, licence and language."""
        ...

    def dialogues(self) -> Iterator[SourceDialogue]:
        """Yield the dialogues in a stable order."""
        ...


class ActRecordSink(Protocol):
    """Where imported records are written."""

    def write(self, records: Iterable[ActRecord]) -> int:
        """Write ``records`` and return how many were written."""
        ...


class ImportSummary(BaseModel):
    """Aggregates of one import (safe to publish)."""

    corpus: str
    licence: str
    non_commercial: bool
    dialogues: int
    records: int
    mapped: int
    distribution: Dict[str, int]
    unmapped_labels: Dict[str, int]


def import_corpus(
    corpus: DialogueActCorpus,
    sink: ActRecordSink,
    context: int = 3,
    limit: Optional[int] = None,
) -> ImportSummary:
    """Map every dialogue of ``corpus`` and write it to ``sink``.

    ``limit`` stops after that many dialogues (``None``: all).
    """
    info = corpus.info
    mapper = MAPPERS.get(info.name)
    if mapper is None:
        raise InvalidInputError(
            f"No label mapping for corpus {info.name!r}; known: "
            f"{', '.join(sorted(MAPPERS))}.")
    if context < 0:
        raise InvalidInputError("--context must be >= 0.")
    if limit is not None and limit < 1:
        raise InvalidInputError("--limit must be >= 1.")
    distribution: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    counted = {"dialogues": 0}

    def records() -> Iterator[ActRecord]:
        """Yield the records of every dialogue, counting as they go."""
        for dialogue in corpus.dialogues():
            if limit is not None and counted["dialogues"] >= limit:
                return
            counted["dialogues"] += 1
            for record in to_records(info, dialogue, mapper, context):
                distribution[category_key(record.categories)] += 1
                if not record.categories:
                    unmapped[record.source_label] += 1
                yield record

    written = sink.write(records())
    return ImportSummary(
        corpus=info.name, licence=info.licence,
        non_commercial=info.non_commercial,
        dialogues=counted["dialogues"], records=written,
        mapped=written - distribution.get("-", 0),
        distribution=dict(sorted(distribution.items(),
                                 key=lambda kv: (-kv[1], kv[0]))),
        unmapped_labels=dict(unmapped.most_common()),
    )
