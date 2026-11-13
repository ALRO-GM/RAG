"""Voiceprint repository: one JSON file per enrolled person.

``<root>/voiceprints/<speaker id>.json`` holds the voiceprint (vector,
voice model, consent version, date) and the consent record, written
atomically. Nothing of the enrolment sample is stored.

Vectors of different voice models cannot be compared, so the
repository is bound to the voice model in use: it refuses to store a
voiceprint of another model, :meth:`FileSystemVoiceprintRepository.get`
raises :class:`VoiceModelMismatchError` for one found on disk (the
person must enrol again) and :meth:`FileSystemVoiceprintRepository.all`
leaves such voiceprints out, as the transcription would anyway.
"""

import re
import threading
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from transcriber.adapters.outbound.fs_storage import (
    child_path,
    read_json,
    write_json,
)
from transcriber.domain.errors import (
    InvalidInputError,
    StorageError,
    VoiceModelMismatchError,
)
from transcriber.domain.models import (
    SPEAKER_ID_PATTERN,
    ConsentRecord,
    Voiceprint,
)

SUFFIX = ".json"


class _Enrolment(BaseModel):
    """What is stored for one person: the voiceprint and the consent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    voiceprint: Voiceprint
    consent: ConsentRecord


class FileSystemVoiceprintRepository:
    """Enrolled voiceprints of one voice model under ``root``."""

    def __init__(self, root: Path, model: str) -> None:
        """Keep files under ``root / "voiceprints"``; accept ``model``."""
        self.root = Path(root) / "voiceprints"
        self.model = model
        self._lock = threading.Lock()

    def _path(self, speaker_id: str) -> Path:
        """Return the file of ``speaker_id`` after checking the id."""
        return child_path(
            self.root, speaker_id, SPEAKER_ID_PATTERN, "speaker", SUFFIX
        )

    def _read(self, speaker_id: str) -> Optional[_Enrolment]:
        """Return what is stored for ``speaker_id``, if anything."""
        path = self._path(speaker_id)
        try:
            data = read_json(path)
        except FileNotFoundError:
            return None
        try:
            stored = _Enrolment.model_validate(data)
        except ValidationError as exc:
            raise StorageError(
                f"Damaged voiceprint file {path.name}: {exc.errors()[0]}"
            ) from exc
        if stored.voiceprint.speaker_id != speaker_id:
            raise StorageError(
                f"{path.name} holds the voiceprint of"
                f" {stored.voiceprint.speaker_id}."
            )
        return stored

    def _check_model(self, voiceprint: Voiceprint) -> None:
        """Raise :class:`VoiceModelMismatchError` for another model."""
        if voiceprint.model != self.model:
            raise VoiceModelMismatchError(
                f"The voiceprint of {voiceprint.speaker_id} was made with"
                f" the voice model {voiceprint.model}, but this"
                f" transcriber uses {self.model}; vectors of different"
                f" models cannot be compared. Enrol"
                f" {voiceprint.speaker_id} again."
            )

    def save(self, voiceprint: Voiceprint, consent: ConsentRecord) -> None:
        """Store a voiceprint and its consent, replacing earlier ones."""
        if consent.speaker_id != voiceprint.speaker_id:
            raise InvalidInputError(
                "The consent record belongs to another speaker."
            )
        self._check_model(voiceprint)
        path = self._path(voiceprint.speaker_id)
        with self._lock:
            write_json(path, _Enrolment(
                voiceprint=voiceprint, consent=consent,
            ).model_dump(mode="json"))

    def get(self, speaker_id: str) -> Optional[Voiceprint]:
        """Return the voiceprint of ``speaker_id``, if enrolled.

        Raises :class:`VoiceModelMismatchError` when it was made with
        another voice model.
        """
        stored = self._read(speaker_id)
        if stored is None:
            return None
        self._check_model(stored.voiceprint)
        return stored.voiceprint

    def consent(self, speaker_id: str) -> Optional[ConsentRecord]:
        """Return the consent record of ``speaker_id``, if enrolled."""
        stored = self._read(speaker_id)
        return stored.consent if stored is not None else None

    def all(self) -> List[Voiceprint]:
        """Return every voiceprint of this voice model, by speaker id.

        Files whose name is not a speaker id are not voiceprints and
        are ignored.
        """
        found: List[Voiceprint] = []
        if not self.root.is_dir():
            return found
        for path in sorted(self.root.glob(f"*{SUFFIX}")):
            speaker_id = path.name[:-len(SUFFIX)]
            if not re.fullmatch(SPEAKER_ID_PATTERN, speaker_id):
                continue
            stored = self._read(speaker_id)
            if stored is not None and stored.voiceprint.model == self.model:
                found.append(stored.voiceprint)
        return found

    def delete(self, speaker_id: str) -> bool:
        """Remove voiceprint and consent; tell whether they existed."""
        path = self._path(speaker_id)
        with self._lock:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
        return True
