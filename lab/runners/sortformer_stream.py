"""Streaming Sortformer v2 as a long-lived worker (live mono sessions, LV-6).

Run with the ``nemo`` lab environment's interpreter (NeMo does not
install in the service's environment)::

    <envs>/nemo/bin/python lab/runners/sortformer_stream.py \
        [--device cuda|cpu] [--chunk 6] [--right 7] [--fifo 188] \
        [--update 144] [--cache 188]

The defaults are the model card's low-latency setting (1.04 s from a
sample to the frame that covers it). The worker speaks JSON lines: one
request per line on stdin, one answer per request on stdout (stdout
carries nothing else; library output goes to stderr).

Requests and answers:

``{"type": "start"}``
    Forget the stream (a new session). Answer ``{"type": "started"}``.
``{"type": "audio", "pcm": "<base64 s16le 16 kHz mono>"}``
    Append audio; run every chunk it completes. Answer
    ``{"type": "frames", "first": n, "probs": [[p0, p1, p2, p3], ...]}``:
    the speaker probabilities of frames ``n``, ``n + 1``, ... (80 ms
    each, frame ``k`` covers ``[0.08 k, 0.08 (k + 1))`` s); no frames
    when no chunk was completed.
``{"type": "flush"}``
    The stream ended: run what is left with the right context there
    is. Answer ``frames`` as above.
``{"type": "stop"}``
    Exit (end of input does the same).

The first line the worker writes, once the model is loaded, is
``{"type": "ready", "model", "revision", "frame_seconds", "speakers",
"latency_seconds", "device"}``. A failure answers ``{"type": "error",
"message"}`` (the worker goes on for a bad request, exits for a broken
model). Slots are numbered in the order speakers are first heard.
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np

# The script's directory holds ``nemo.py`` (the lab runner), which
# would shadow the NeMo package: drop it from the path.
HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]

MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"
RATE = 16000
HOP = 160
"""Samples per feature frame (10 ms)."""
EDGE = 10
"""Feature frames of audio taken before a window and dropped (STFT edge)."""


class Stream:
    """One stream's audio, its features and the model's streaming state."""

    def __init__(self, model: Any, device: str) -> None:
        """Start an empty stream."""
        self.model = model
        self.device = device
        self.modules = model.sortformer_modules
        self.factor = int(self.modules.subsampling_factor)
        self.samples = np.zeros(0, dtype=np.float32)
        self.offset = 0
        self.chunk = 0
        self.frames = 0
        self.state = self.modules.init_streaming_state(
            batch_size=1, async_streaming=model.async_streaming,
            device=model.device,
        )

    def _empty(self) -> Any:
        """Return an empty prediction tensor."""
        import torch

        return torch.zeros(
            (1, 0, self.modules.n_spk), device=self.model.device
        )

    def feature_frames(self) -> int:
        """Return the feature frames the audio received so far gives."""
        return (self.offset + len(self.samples)) // HOP + 1

    def add(self, samples: np.ndarray) -> None:
        """Append ``samples`` (float32, mono, 16 kHz)."""
        self.samples = np.concatenate([self.samples, samples])

    def run(self, final: bool) -> List[List[float]]:
        """Run every chunk the audio completes; return their frames."""
        out: List[List[float]] = []
        step = self.modules.chunk_len * self.factor
        right = self.modules.chunk_right_context * self.factor
        while True:
            total = self.feature_frames()
            start = self.chunk * step
            if start >= total or (not final and start + step + right
                                  + EDGE > total):
                break
            out.extend(self._chunk(start, step, right, total, final))
            self.chunk += 1
        return out

    def _chunk(
        self, start: int, step: int, right: int, total: int, final: bool
    ) -> List[List[float]]:
        """Run the chunk of feature frames ``start`` .. ``start + step``."""
        import torch

        left = min(self.modules.chunk_left_context * self.factor, start)
        end = min(start + step, total)
        rc = min(right, total - end) if final else right
        first = start - left
        last = end + rc
        lead = min(EDGE, first)
        begin = (first - lead) * HOP - self.offset
        stop = (last - 1) * HOP + 512 // 2 + 1 - self.offset
        audio = self.samples[max(begin, 0):max(stop, 0)]
        signal = torch.from_numpy(np.ascontiguousarray(audio)).to(
            self.model.device
        ).unsqueeze(0)
        length = torch.tensor([signal.shape[1]], device=self.model.device)
        feats, _ = self.model.preprocessor(
            input_signal=signal, length=length
        )
        feats = feats[:, :, lead:lead + (last - first)]
        count = feats.shape[2]
        lengths = torch.tensor([count], device=self.model.device)
        chunk = torch.transpose(feats, 1, 2)
        # The model appends the chunk's frames to ``total_preds``; an
        # empty one each time keeps only the new frames.
        self.state, preds = self.model.forward_streaming_step(
            processed_signal=chunk, processed_signal_length=lengths,
            streaming_state=self.state, total_preds=self._empty(),
            left_offset=left, right_offset=rc,
        )
        new = preds[0].float().cpu().numpy()
        self.frames += new.shape[0]
        self._drop(start + step - self.modules.chunk_left_context
                   * self.factor - EDGE)
        return [[round(float(p), 3) for p in row] for row in new]

    def _drop(self, keep_frame: int) -> None:
        """Forget audio before feature frame ``keep_frame``."""
        drop = max(keep_frame, 0) * HOP - self.offset
        if drop > 0:
            self.samples = self.samples[drop:]
            self.offset += drop


def load(device: str, args: argparse.Namespace) -> Any:
    """Load the model with the streaming setting of ``args``."""
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel(logging.WARNING)
    model = SortformerEncLabelModel.from_pretrained(
        MODEL, map_location=device
    ).eval()
    modules = model.sortformer_modules
    modules.chunk_len = args.chunk
    modules.chunk_right_context = args.right
    modules.fifo_len = args.fifo
    modules.spkcache_update_period = args.update
    modules.spkcache_len = args.cache
    modules._check_streaming_parameters()
    return model


def revision() -> Optional[str]:
    """Return the cached snapshot of the model, if known."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == MODEL:
                return sorted(r.commit_hash for r in repo.revisions)[0]
    except Exception:  # noqa: BLE001 - only informative
        return None
    return None


def answer(out: TextIO, message: Dict[str, Any]) -> None:
    """Write one answer line."""
    out.write(json.dumps(message, separators=(",", ":")) + "\n")
    out.flush()


def serve(model: Any, device: str, source: TextIO, out: TextIO) -> None:
    """Answer requests until ``stop`` or the end of input."""
    import torch

    stream = Stream(model, device)
    for line in source:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            kind = request.get("type")
            if kind == "stop":
                return
            if kind == "start":
                stream = Stream(model, device)
                answer(out, {"type": "started"})
                continue
            if kind not in ("audio", "flush"):
                raise ValueError(f"unknown request {kind!r}")
            first = stream.frames
            if kind == "audio":
                pcm = base64.b64decode(request.get("pcm", ""))
                samples = np.frombuffer(pcm, dtype="<i2").astype(
                    np.float32) / 32768.0
                stream.add(samples)
            with torch.inference_mode():
                probs = stream.run(final=kind == "flush")
            answer(out, {"type": "frames", "first": first, "probs": probs})
        except (ValueError, TypeError, KeyError) as exc:
            answer(out, {"type": "error", "message": str(exc)[:300]})


def main() -> int:
    """Load the model, say ready, serve."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=6)
    parser.add_argument("--right", type=int, default=7)
    parser.add_argument("--fifo", type=int, default=188)
    parser.add_argument("--update", type=int, default=144)
    parser.add_argument("--cache", type=int, default=188)
    args = parser.parse_args()
    # The protocol owns the real stdout; anything a library prints goes
    # to stderr.
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        model = load(args.device, args)
    except Exception as exc:  # noqa: BLE001 - reported, then exit
        answer(out, {"type": "error", "message": f"{type(exc).__name__}:"
                     f" {str(exc)[:300]}"})
        return 1
    frame = 0.01 * int(model.sortformer_modules.subsampling_factor)
    answer(out, {
        "type": "ready", "model": MODEL, "revision": revision(),
        "frame_seconds": frame,
        "speakers": int(model.sortformer_modules.n_spk),
        "latency_seconds": round(frame * (args.chunk + args.right), 3),
        "device": args.device,
    })
    serve(model, args.device, sys.stdin, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
