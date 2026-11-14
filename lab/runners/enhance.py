"""Lab runner of the ``enhance`` family (audio in, cleaner audio out).

Models:

``deepfilternet3`` (denoise)
    DeepFilterNet3 (``deepfilternet`` 0.5.6, MIT/Apache-2.0; weights
    from the project's GitHub release, fetched on first use into
    ``~/.cache/DeepFilterNet``), full band at 48 kHz. Run on pieces of
    ``piece_seconds`` (30 s) with 1 s cross-faded overlaps: on a whole
    AMI meeting (25-40 min) at once it tried to allocate 1.8-5.3 GiB
    more and ran out of GPU memory. Pieces are not rescaled (as before,
    the model's output level is kept).
``mossformer2-se-48k`` / ``mossformergan-se-16k`` (denoise)
    ClearerVoice-Studio (``clearvoice`` 0.1.2, Apache-2.0) speech
    enhancement models ``alibabasglab/MossFormer2_SE_48K`` and
    ``alibabasglab/MossFormerGAN_SE_16K``; checkpoints are fetched on
    first use into ``~/.local/share/ragputin/lab-models/clearvoice``.
``rnnoise`` (denoise)
    Xiph RNNoise (BSD-3-Clause) through ``pyrnnoise`` 0.4.5 (its bundled
    library and weights), 48 kHz, 10 ms frames, CPU only.

ClearVoice cuts long audio into windows itself (10 s for the 16 kHz
models) and only a window's features reach the model, so its memory
does not grow with the recording and it is not cut here (written as
PCM16 because its reader cannot take float WAV without ffprobe).
RNNoise runs 10 ms frames on the CPU. Every output is resampled back
to 16 kHz mono with the input's length. See ``_common`` for the contract.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict

# The script's directory must not shadow a package of the same name
# (nemo, pyannote...): drop it from the path, load the helpers by file.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
_SPEC = importlib.util.spec_from_file_location("_common", HERE / "_common.py")
assert _SPEC is not None and _SPEC.loader is not None
_common = importlib.util.module_from_spec(_SPEC)
sys.modules["_common"] = _common
_SPEC.loader.exec_module(_common)
Context = _common.Context
RunnerError = _common.RunnerError

MODELS = Path.home() / ".local" / "share" / "ragputin" / "lab-models"
CLEARVOICE = {
    "mossformer2-se-48k": "MossFormer2_SE_48K",
    "mossformergan-se-16k": "MossFormerGAN_SE_16K",
}


def _torchaudio_backend_shim() -> None:
    """Provide ``torchaudio.backend.common.AudioMetaData``.

    DeepFilterNet 0.5.6 imports it (for a type) from a module torchaudio
    removed in 2.1; the lab never uses DeepFilterNet's file loading, so
    an empty class is enough.
    """
    if "torchaudio.backend.common" in sys.modules:
        return
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:
        """Stand-in for torchaudio's removed metadata class."""

    common.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]
    backend.common = common  # type: ignore[attr-defined]
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def deepfilternet(context: Context) -> Dict[str, Any]:
    """Enhance with DeepFilterNet3 at 48 kHz, ``piece_seconds`` at a time."""
    _torchaudio_backend_shim()
    import numpy as np
    import torch
    from df.enhance import enhance, init_df

    context.ignore_unknown(("atten_lim_db", "post_filter", "piece_seconds"))
    model, state, _ = init_df(
        post_filter=bool(context.param("post_filter", False)),
        log_level="WARNING", log_file=None,
    )
    model = model.to(context.device).eval()
    context.model_version = (
        f"DeepFilterNet3 deepfilternet=={_common.version_of('deepfilternet')}"
    )
    rate = state.sr()
    samples = _common.read_audio(context.source)
    wide = _common.resample(samples, _common.RATE, rate)
    context.loaded()
    limit = context.param("atten_lim_db")
    piece = int(float(context.param("piece_seconds", 30.0)) * rate)

    def run(chunk: "np.ndarray") -> "np.ndarray":
        """Enhance one piece."""
        with torch.inference_mode():
            out = enhance(
                model, state, torch.from_numpy(chunk).unsqueeze(0),
                atten_lim_db=None if limit is None else float(limit),
            )
        return out.squeeze(0).cpu().numpy()[:len(chunk)]

    enhanced = _common.pieces(wide, piece, rate, run)
    _common.write_audio(
        context.out / "audio.wav", enhanced, rate, len(samples),
    )
    return {"audio": "audio.wav", "sample_rate": _common.RATE}


def clearvoice(context: Context) -> Dict[str, Any]:
    """Enhance with a ClearerVoice-Studio model."""
    import tempfile

    import numpy as np
    import soundfile

    context.ignore_unknown(())
    name = CLEARVOICE[context.model]
    home = MODELS / "clearvoice"
    home.mkdir(parents=True, exist_ok=True)
    os.chdir(home)  # ClearVoice keeps checkpoints under ./checkpoints
    if context.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from clearvoice import ClearVoice

    voice = ClearVoice(task="speech_enhancement", model_names=[name])
    rate = int(voice.models[0].args.sampling_rate)
    context.model_version = (
        f"alibabasglab/{name} clearvoice=={_common.version_of('clearvoice')}"
    )
    samples = _common.read_audio(context.source)
    context.loaded()
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "input.wav"
        soundfile.write(
            str(source),
            np.clip(_common.resample(samples, _common.RATE, rate), -1, 1),
            rate, subtype="PCM_16",
        )
        out = voice(input_path=str(source), online_write=False)
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    _common.write_audio(context.out / "audio.wav", out, rate, len(samples))
    return {"audio": "audio.wav", "sample_rate": _common.RATE}


def rnnoise(context: Context) -> Dict[str, Any]:
    """Denoise with RNNoise, 10 ms frames at 48 kHz (CPU)."""
    import numpy as np
    from pyrnnoise.rnnoise import (
        FRAME_SIZE,
        SAMPLE_RATE,
        create,
        destroy,
        process_mono_frame,
    )

    context.ignore_unknown(())
    context.model_version = f"pyrnnoise=={_common.version_of('pyrnnoise')}"
    samples = _common.read_audio(context.source)
    wide = _common.resample(samples, _common.RATE, SAMPLE_RATE)
    pcm = (np.clip(wide, -1.0, 1.0) * 32767).astype(np.int16)
    context.loaded()
    state = create()
    try:
        out = np.concatenate([
            process_mono_frame(state, pcm[i:i + FRAME_SIZE])[0]
            for i in range(0, len(pcm), FRAME_SIZE)
        ]).astype(np.float32) / 32767.0
    finally:
        destroy(state)
    context.device = "cpu"
    _common.write_audio(
        context.out / "audio.wav", out, SAMPLE_RATE, len(samples)
    )
    return {"audio": "audio.wav", "sample_rate": _common.RATE}


HANDLERS = {
    ("denoise", "deepfilternet3"): deepfilternet,
    ("denoise", "rnnoise"): rnnoise,
    **{("denoise", model): clearvoice for model in CLEARVOICE},
}


def imports() -> Dict[str, str]:
    """Import DeepFilterNet, ClearVoice and pyrnnoise (no weights)."""
    _torchaudio_backend_shim()
    from clearvoice import ClearVoice  # noqa: F401
    from df.enhance import enhance, init_df  # noqa: F401
    from pyrnnoise.rnnoise import create  # noqa: F401

    return {p: str(_common.version_of(p)) for p in (
        "deepfilternet", "clearvoice", "pyrnnoise", "torch", "torchaudio",
    )}


if __name__ == "__main__":
    sys.exit(_common.main("enhance", HANDLERS, imports))
