"""Voice activity adapter: Silero VAD (ONNX) on the CPU, one frame at a time.

The model is the Silero VAD ONNX file bundled with ``faster-whisper``
(``faster_whisper/assets/silero_vad_v6.onnx``, MIT), run with
``onnxruntime`` on one CPU thread: no GPU memory and no torch. Each
32 ms frame (512 samples at 16 kHz) is given to the model with the
last 64 samples of the previous frame in front of it, and the model's
recurrent state (``h``, ``c``) is carried from frame to frame, as
Silero's own streaming wrapper does. Install with the ``live``
dependency group.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from transcriber.domain.live import FRAME_SAMPLES

FloatArray = npt.NDArray[np.float32]

CONTEXT_SAMPLES = 64
"""Samples of the previous frame the model reads before each frame."""

STATE_SHAPE = (1, 1, 128)

SessionFactory = Callable[[Path], Any]


def bundled_model() -> Path:  # pragma: no cover - needs faster-whisper
    """Return the Silero VAD ONNX file shipped with faster-whisper."""
    from faster_whisper.utils import get_assets_path

    return Path(get_assets_path()) / "silero_vad_v6.onnx"


def onnx_session(path: Path) -> Any:  # pragma: no cover - needs onnxruntime
    """Open ``path`` with onnxruntime on one CPU thread."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"], sess_options=options
    )


class SileroVad:
    """Speech probability of each 32 ms frame of one stream.

    ``session_factory`` opens the ONNX model (tests give a fake one);
    the model is opened on the first frame.
    """

    def __init__(
        self,
        model: Optional[Path] = None,
        session_factory: SessionFactory = onnx_session,
    ) -> None:
        """Keep where the model is; open it lazily."""
        self._model = model
        self._factory = session_factory
        self._session: Any = None
        self.reset()

    def reset(self) -> None:
        """Forget the previous frames (a new stream starts)."""
        self._h = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context: FloatArray = np.zeros(
            CONTEXT_SAMPLES, dtype=np.float32
        )

    def probability(self, frame: FloatArray) -> float:
        """Return the probability that ``frame`` (512 samples) is speech."""
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(
                f"a VAD frame has {FRAME_SAMPLES} samples, not {frame.shape}"
            )
        if self._session is None:
            self._session = self._factory(self._model or bundled_model())
        audio: FloatArray = np.concatenate(
            [self._context, frame.astype(np.float32)]
        )
        output, self._h, self._c = self._session.run(
            None, {"input": audio[np.newaxis, :], "h": self._h, "c": self._c}
        )
        self._context = audio[-CONTEXT_SAMPLES:]
        return float(np.asarray(output).reshape(-1)[0])
