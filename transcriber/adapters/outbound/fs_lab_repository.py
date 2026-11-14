"""The lab's storage: one directory per session under ``<data_dir>/lab``.

Layout (``docs/transcriber/LAB.md`` §6)::

    lab/<session_id>/
      original.json             key of the original audio
      audio/<key>.wav           16 kHz mono 16-bit WAVs: the original
                                (key: SHA-256 of its samples) and each
                                denoiser's output (key: its stage key)
      stages/<key>/outcome.json one stage result, with the files the
                                tool wrote (``runner.log``, ...)
      mappings/<key>.json       speaker mappings
      runs/<run_id>.json        a run; ``.cancel`` beside it asks it to
                                stop; ``.results.json`` its comparison
      labels.json               the owner's reference labels
      work/<random>/            workspaces of stages being computed

Every id and key is checked against its pattern before a path is made,
every JSON file is written atomically (:mod:`fs_storage`) and a stage's
directory appears in one rename, so an interrupted stage leaves only a
workspace (removed by :meth:`discard` or ignored).
"""

import hashlib
import os
import shutil
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from transcriber.adapters.outbound.fs_storage import (
    atomic_file,
    child_path,
    read_json,
    write_json,
)
from transcriber.adapters.outbound.wav_playback import (
    PCM_RATE,
    FilePlayback,
    write_wav,
)
from transcriber.application.ports import Audio, PlaybackAudio
from transcriber.domain.errors import (
    LabStepError,
    NotFoundError,
    StorageError,
)
from transcriber.domain.lab import (
    LAB_RUN_ID_PATTERN,
    STAGE_KEY_PATTERN,
    LabRun,
    ReferenceLabel,
    SpeakerMapping,
    StageOutcome,
)
from transcriber.domain.models import SESSION_ID_PATTERN

LAB_DIR = "lab"
AUDIO_KEY_PATTERN = r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$"
"""A stage key (40 hex) or an original's SHA-256 (64 hex)."""
WORK_PATTERN = r"^[0-9a-f]{32}$"
BLOCK_SAMPLES = 1 << 16


class FileSystemLabRepository:
    """Lab results stored under ``<data_dir>/lab`` (see the module)."""

    def __init__(self, data_dir: Path) -> None:
        """Use ``data_dir``; nothing is created until the first write."""
        self._root = Path(data_dir) / LAB_DIR

    def _session(self, session_id: str) -> Path:
        """Return the session's lab directory (checked id)."""
        self._root.mkdir(parents=True, exist_ok=True)
        return child_path(self._root, session_id, SESSION_ID_PATTERN,
                          "session")

    def _dir(self, session_id: str, name: str) -> Path:
        """Return a subdirectory of the session's lab directory."""
        path = self._session(session_id) / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ----------------------------------------------------------- audio

    def store_original(self, session_id: str, audio: Audio) -> str:
        """Store the session's audio as WAV; return its SHA-256 key."""
        pcm = _pcm(audio)
        key = hashlib.sha256(pcm.tobytes()).hexdigest()
        target = child_path(self._dir(session_id, "audio"), key,
                            AUDIO_KEY_PATTERN, "audio", ".wav")
        if not target.is_file():
            with atomic_file(target) as stream:
                write_wav(stream, _blocks(pcm), PCM_RATE)
        write_json(self._session(session_id) / "original.json",
                   {"key": key})
        return key

    def original_key(self, session_id: str) -> Optional[str]:
        """Return the key of the stored original audio, if any."""
        path = self._session(session_id) / "original.json"
        try:
            key = read_json(path).get("key")
        except FileNotFoundError:
            return None
        if not isinstance(key, str):
            raise StorageError("original.json has no key.")
        if not self._audio(session_id, key).is_file():
            return None
        return key

    def _audio(self, session_id: str, key: str) -> Path:
        """Return the path of the audio ``key`` (checked key)."""
        return child_path(self._dir(session_id, "audio"), key,
                          AUDIO_KEY_PATTERN, "audio", ".wav")

    def audio_path(self, session_id: str, audio_key: str) -> Path:
        """Return the WAV of ``audio_key`` or raise NotFoundError."""
        path = self._audio(session_id, audio_key)
        if not path.is_file():
            raise NotFoundError("No such lab audio.")
        return path

    def load_audio(self, session_id: str, audio_key: str) -> Audio:
        """Return the samples of ``audio_key``."""
        return read_wav(self.audio_path(session_id, audio_key))

    def playback(self, session_id: str, audio_key: str) -> PlaybackAudio:
        """Return the WAV of ``audio_key`` for the player."""
        return FilePlayback(self.audio_path(session_id, audio_key))

    # ---------------------------------------------------------- stages

    def new_workspace(self, session_id: str) -> Path:
        """Return a new, empty directory for one stage."""
        path = self._dir(session_id, "work") / uuid.uuid4().hex
        path.mkdir()
        return path

    def discard(self, session_id: str, workspace: Path) -> None:
        """Remove a workspace made by :meth:`new_workspace`."""
        work = self._dir(session_id, "work")
        path = child_path(work, workspace.name, WORK_PATTERN, "workspace")
        shutil.rmtree(path, ignore_errors=True)

    def _stage(self, session_id: str, key: str) -> Path:
        """Return the directory of the stage ``key`` (checked key)."""
        return child_path(self._dir(session_id, "stages"), key,
                          STAGE_KEY_PATTERN, "stage")

    def keep(
        self, session_id: str, key: str, outcome: StageOutcome,
        workspace: Path,
    ) -> StageOutcome:
        """Store ``outcome`` and the workspace's files under ``key``.

        A denoiser's ``audio.wav`` must be a 16 kHz mono 16-bit WAV
        (:class:`LabStepError` otherwise); it becomes audio ``key``.
        """
        work = child_path(self._dir(session_id, "work"), workspace.name,
                          WORK_PATTERN, "workspace")
        if outcome.stage == "denoise":
            produced = work / "audio.wav"
            check_wav(produced)
            os.replace(produced, self._audio(session_id, key))
            outcome = outcome.model_copy(update={"audio_key": key})
        write_json(work / "outcome.json", outcome.model_dump(mode="json"))
        target = self._stage(session_id, key)
        if target.exists():
            shutil.rmtree(work, ignore_errors=True)
        else:
            os.replace(work, target)
        return outcome

    def outcome(self, session_id: str, key: str) -> Optional[StageOutcome]:
        """Return the stored outcome of ``key``, if any."""
        path = self._stage(session_id, key) / "outcome.json"
        try:
            outcome = StageOutcome.model_validate(read_json(path))
        except FileNotFoundError:
            return None
        except ValueError as exc:
            raise StorageError(f"Cannot read a lab stage: {exc}") from exc
        if outcome.audio_key and not self._audio(
            session_id, outcome.audio_key
        ).is_file():
            return None
        return outcome

    def mapping(
        self, session_id: str, key: str
    ) -> Optional[List[SpeakerMapping]]:
        """Return the stored speaker mapping ``key``, if any."""
        path = child_path(self._dir(session_id, "mappings"), key,
                          STAGE_KEY_PATTERN, "mapping", ".json")
        try:
            data = read_json(path)
        except FileNotFoundError:
            return None
        return [SpeakerMapping.model_validate(m) for m in data]

    def save_mapping(
        self, session_id: str, key: str, mapping: List[SpeakerMapping]
    ) -> None:
        """Store a speaker mapping."""
        path = child_path(self._dir(session_id, "mappings"), key,
                          STAGE_KEY_PATTERN, "mapping", ".json")
        write_json(path, [m.model_dump(mode="json") for m in mapping])

    # ------------------------------------------------------------ runs

    def _run_file(self, session_id: str, run_id: str, suffix: str) -> Path:
        """Return a file of run ``run_id`` (checked id)."""
        return child_path(self._dir(session_id, "runs"), run_id,
                          LAB_RUN_ID_PATTERN, "lab run", suffix)

    def save_run(self, run: LabRun) -> None:
        """Store (create or replace) a run."""
        write_json(self._run_file(run.session_id, run.id, ".json"),
                   run.model_dump(mode="json"))

    def run(self, session_id: str, run_id: str) -> Optional[LabRun]:
        """Return the run, if it exists."""
        try:
            data = read_json(self._run_file(session_id, run_id, ".json"))
        except FileNotFoundError:
            return None
        try:
            return LabRun.model_validate(data)
        except ValueError as exc:
            raise StorageError(f"Cannot read a lab run: {exc}") from exc

    def runs(self, session_id: str) -> List[LabRun]:
        """Return the session's readable runs, newest first."""
        found = []
        for path in self._run_paths(session_id):
            run = self.run(session_id, path.name[:-len(".json")])
            if run is not None:
                found.append(run)
        return sorted(found, key=lambda r: r.created_at, reverse=True)

    def _run_paths(self, session_id: str) -> Iterator[Path]:
        """Yield the run files of a session."""
        for path in sorted(self._dir(session_id, "runs").glob("*.json")):
            name = path.name[:-len(".json")]
            if len(name) == 32 and all(c in "0123456789abcdef"
                                       for c in name):
                yield path

    def request_cancel(self, session_id: str, run_id: str) -> None:
        """Leave the marker that asks the run to stop."""
        path = self._run_file(session_id, run_id, ".cancel")
        path.touch()

    def cancel_requested(self, session_id: str, run_id: str) -> bool:
        """Tell whether the run was asked to stop."""
        return self._run_file(session_id, run_id, ".cancel").exists()

    def clear_cancel(self, session_id: str, run_id: str) -> None:
        """Remove the stop marker."""
        self._run_file(session_id, run_id, ".cancel").unlink(
            missing_ok=True
        )

    def save_results(
        self, session_id: str, run_id: str, results: Dict[str, Any]
    ) -> None:
        """Store a run's comparison."""
        write_json(self._run_file(session_id, run_id, ".results.json"),
                   results)

    def results(
        self, session_id: str, run_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return a run's comparison, if stored."""
        try:
            data = read_json(
                self._run_file(session_id, run_id, ".results.json")
            )
        except FileNotFoundError:
            return None
        if not isinstance(data, dict):
            raise StorageError("Lab results are not an object.")
        return data

    # ---------------------------------------------------------- labels

    def labels(self, session_id: str) -> List[ReferenceLabel]:
        """Return the reference labels (empty if none)."""
        try:
            data = read_json(self._session(session_id) / "labels.json")
        except FileNotFoundError:
            return []
        try:
            return [ReferenceLabel.model_validate(x) for x in data]
        except (TypeError, ValueError) as exc:
            raise StorageError(f"Cannot read the labels: {exc}") from exc

    def save_labels(
        self, session_id: str, labels: List[ReferenceLabel]
    ) -> None:
        """Replace the reference labels."""
        write_json(self._session(session_id) / "labels.json",
                   [label.model_dump(mode="json") for label in labels])

    # --------------------------------------------------------- session

    def delete_session(self, session_id: str) -> None:
        """Remove the session's lab directory (absent: nothing to do).

        The id is checked before the path is made. A file written
        meanwhile (a stage finishing) may survive the first pass: the
        tree is removed once more, and then any error is raised.
        """
        directory = child_path(self._root, session_id, SESSION_ID_PATTERN,
                               "session")
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            if directory.exists():
                shutil.rmtree(directory)


def _pcm(audio: Audio) -> "np.ndarray[Any, np.dtype[np.int16]]":
    """Return ``audio`` as 16-bit PCM at 16 kHz (resampled if needed)."""
    samples = np.asarray(audio.samples, dtype=np.float64)
    if audio.sample_rate != PCM_RATE and samples.size:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(audio.sample_rate, PCM_RATE))
        samples = np.asarray(resample_poly(
            samples, PCM_RATE // gcd, audio.sample_rate // gcd
        ), dtype=np.float64)
    clipped = np.clip(samples, -1.0, 1.0)
    return np.round(clipped * 32767).astype("<i2")


def _blocks(pcm: "np.ndarray[Any, np.dtype[np.int16]]") -> Iterator[bytes]:
    """Yield ``pcm`` in blocks of bytes."""
    for first in range(0, int(pcm.shape[0]), BLOCK_SAMPLES):
        yield pcm[first:first + BLOCK_SAMPLES].tobytes()


def check_wav(path: Path) -> None:
    """Raise :class:`LabStepError` unless ``path`` is 16 kHz mono PCM16."""
    try:
        with wave.open(str(path), "rb") as reader:
            shape = (reader.getframerate(), reader.getnchannels(),
                     reader.getsampwidth())
    except (OSError, EOFError, wave.Error) as exc:
        raise LabStepError(
            "failed", f"The denoiser wrote no readable WAV ({exc})."
        ) from exc
    if shape != (PCM_RATE, 1, 2):
        raise LabStepError(
            "failed",
            f"The denoiser wrote {shape[0]} Hz, {shape[1]} channel(s),"
            f" {8 * shape[2]}-bit audio; 16 kHz mono 16-bit is required.",
        )


def read_wav(path: Path) -> Audio:
    """Return a 16-bit PCM WAV file as samples in [-1, 1]."""
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise StorageError("Lab audio must be 16-bit PCM.")
        rate = reader.getframerate()
        channels = reader.getnchannels()
        data = reader.readframes(reader.getnframes())
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return Audio(pcm, rate)
