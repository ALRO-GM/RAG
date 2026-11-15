"""The lab's diarizers as the transcription's :class:`Diarizer` (TR-18).

A transcription can separate the voices with a diarizer of the lab's
catalogue instead of the built-in grouping. The model runs exactly as
in the lab: its runner script (``lab/runners/<family>.py``) in the
family's own environment, as a subprocess
(:class:`~transcriber.adapters.outbound.subprocess_lab_tool.
SubprocessLabTool`, the runner contract of ``docs/transcriber/LAB.md``
§3). The model is therefore loaded for one recording and its GPU
memory is freed when the process ends.

The session's audio is written as a 16 kHz mono 16-bit WAV in a
temporary directory under ``work_dir`` (inside the service's data
directory, so the recording never leaves it) and deleted with the
runner's output once the segments are read. ``before_run`` is called
before the runner starts: the composition root passes a function that
unloads the service's speech recogniser, so the diarizer has the GPU
memory the recogniser held (the GPU is shared with the RAG and another
project's worker).
"""

import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from transcriber.adapters.outbound.subprocess_lab_tool import (
    FileSystemLabEnvironment,
    JsonCatalogue,
    SubprocessLabTool,
)
from transcriber.application.lab_ports import ToolRequest
from transcriber.application.ports import Audio
from transcriber.domain.diarization import Diarization
from transcriber.domain.errors import TranscriberError
from transcriber.domain.lab import CatalogueEntry, merged_params


def write_wav(path: Path, audio: Audio) -> None:
    """Write ``audio`` as a mono 16-bit PCM WAV at its own rate."""
    pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(int(audio.sample_rate))
        writer.writeframes(pcm.tobytes())


class LabRunnerDiarizer:
    """Run a catalogue diarizer on a session's audio (see the module)."""

    def __init__(
        self,
        tool: SubprocessLabTool,
        environment: FileSystemLabEnvironment,
        catalogue: JsonCatalogue,
        work_dir: Path,
        before_run: Callable[[], None] = lambda: None,
    ) -> None:
        """Use ``tool``'s runners; write audio under ``work_dir``.

        ``work_dir`` is made absolute here: the runner starts in its own
        workspace, so a relative path (the default data directory is
        ``data/transcriber``) would not lead it to the audio.
        """
        self._tool = tool
        self._environment = environment
        self._catalogue = catalogue
        self._work_dir = Path(work_dir).absolute()
        self._before_run = before_run

    def _entry(self, model: str) -> CatalogueEntry:
        """Return the catalogue entry of diarizer ``model``."""
        return self._catalogue.load().require(model, "diarize")

    def problem(self, model: str) -> Optional[str]:
        """Return why ``model`` cannot run here, or ``None`` if it can.

        The model must be a diarizer of the catalogue, run by a runner
        (not the service's own family) whose environment is installed
        and verified, with a token when it is gated.
        """
        try:
            entry = self._entry(model)
        except TranscriberError as exc:
            return str(exc)
        if entry.family == "project":
            return f"{model} runs in the service, not as a runner"
        problem = self._environment.family_problem(entry.family)
        if problem is not None:
            return f"{entry.family}: {problem}"
        if entry.gated and not self._environment.token_present():
            return f"{model} is gated and there is no Hugging Face token"
        return None

    def diarize(
        self, model: str, audio: Audio, num_speakers: Optional[int]
    ) -> Diarization:
        """Run ``model`` on ``audio``; return its segments and terms.

        ``num_speakers`` is passed when the model declares it. A failure
        of the runner raises
        :class:`~transcriber.domain.errors.LabStepError`.
        """
        entry = self._entry(model)
        params = merged_params(entry, {}, {"num_speakers": num_speakers})
        self._before_run()
        self._work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="diarize-", dir=self._work_dir
        ) as scratch:
            wav = Path(scratch) / "audio.wav"
            write_wav(wav, audio)
            workspace = Path(scratch) / "out"
            workspace.mkdir()
            outcome = self._tool.run(ToolRequest(
                entry=entry, stage="diarize", params=params, audio=wav,
                workspace=workspace, load=lambda: audio,
            ), lambda: False)
        return Diarization(
            model=entry.id, licence=entry.licence,
            commercial=entry.licence_commercial,
            version=outcome.model_version, segments=outcome.segments,
            overlaps=outcome.overlap, peak_gpu_mb=outcome.peak_gpu_mb,
        )
