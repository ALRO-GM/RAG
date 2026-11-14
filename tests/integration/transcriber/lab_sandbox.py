"""A lab with a fake runner, for the adapter, API and CLI tests.

:func:`make_lab` writes into a temporary directory a catalogue, a
runner script (``fake.py``, standard library only, run by this
interpreter through an "environment" whose ``bin/python`` links to it)
and the marker ``make lab-env`` leaves after a self-test. The fake
runner follows the runner contract of ``docs/transcriber/LAB.md`` §3:

* ``fake-denoise`` copies the input WAV at half the volume;
* ``fake-diarize`` gives the first half to ``A`` and the second to
  ``B``, overlapping by one second; ``fake-split`` cuts at a third;
* ``fake-asr`` hears one word per second;
* ``fake-gated`` needs ``HF_TOKEN``; ``fake-crash`` exits 1 with no
  result; ``fake-slow`` sleeps (for time-outs and cancelling);
  ``fake-env`` echoes its environment into its notes.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from transcriber.adapters.outbound.subprocess_lab_tool import READY_MARKER
from transcriber.config import LabSettings

FAKE_RUNNER = r'''
import json, os, sys, time, wave, struct

def write(out, data, code=0):
    with open(os.path.join(out, "result.json.tmp"), "w") as f:
        json.dump(data, f)
    os.replace(os.path.join(out, "result.json.tmp"),
               os.path.join(out, "result.json"))
    sys.exit(code)

if sys.argv[1] == "--selftest":
    print(json.dumps({"ok": True, "family": "fake"}))
    sys.exit(0)
stage = sys.argv[1]
args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
model, src, out = args["--model"], args["--in"], args["--out"]
params = json.loads(args.get("--params", "{}"))
with wave.open(src, "rb") as r:
    rate, frames = r.getframerate(), r.getnframes()
    data = r.readframes(frames)
seconds = frames / rate
base = {"schema": "lab.result.v1", "ok": True, "stage": stage,
        "model": model, "model_version": "fake==1", "device": "cpu",
        "seconds": 0.01, "load_seconds": 0.0, "peak_gpu_mb": None,
        "audio_seconds": seconds, "notes": []}
fail = dict(base, ok=False)
if model == "fake-crash":
    print("something broke in " + src)
    sys.exit(1)
if model == "fake-slow":
    time.sleep(60)
if model == "fake-gated" and not os.environ.get("HF_TOKEN"):
    write(out, dict(fail, error={"code": "needs_token",
                                 "message": "needs token"}), 3)
if model == "fake-env":
    base["notes"] = sorted(os.environ)
    base["segments"] = []
    write(out, base)
if stage == "denoise":
    samples = struct.unpack("<%dh" % frames, data)
    with wave.open(os.path.join(out, "audio.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % frames,
                                  *[s // 2 for s in samples]))
    write(out, dict(base, audio="audio.wav", sample_rate=16000))
if stage == "diarize":
    cut = seconds / (3 if model == "fake-split" else 2)
    write(out, dict(base, segments=[
        {"start": 0.0, "end": cut + 0.5, "speaker": "A"},
        {"start": cut - 0.5, "end": seconds, "speaker": "B"}]))
if stage == "recognise":
    words = [{"start": i + 0.1, "end": i + 0.6, "text": "w%d" % i,
              "probability": 0.9} for i in range(int(seconds))]
    write(out, dict(base, words=words,
                    language=params.get("language")))
write(out, dict(fail, error={"code": "unknown_model", "message": model}), 2)
'''


def entry(model_id: str, stage: str, family: str = "fake",
          **extra: Any) -> Dict[str, Any]:
    """Return one catalogue entry."""
    return {
        "id": model_id, "stage": stage, "family": family,
        "display_name": model_id, "source_url": "https://example.org",
        "licence": "MIT", "licence_commercial": True, "gated": False,
        "languages": ["any"], "parameters": {}, "weights": [],
        "notes": "", **extra,
    }


def catalogue() -> Dict[str, Any]:
    """Return the sandbox's catalogue."""
    entries: List[Dict[str, Any]] = [
        entry("none", "denoise", "project"),
        entry("builtin", "diarize", "project", parameters={
            "num_speakers": None, "expected_speakers": None,
        }),
        entry("project-asr", "recognise", "project",
              weights=["fake/recogniser"],
              parameters={"language": None}),
        entry("fake-denoise", "denoise"),
        entry("fake-diarize", "diarize", parameters={"num_speakers": None}),
        entry("fake-split", "diarize", lineage="fake-diarize",
              licence_commercial=False),
        entry("fake-asr", "recognise", parameters={"language": None}),
        entry("fake-heavy", "recognise", weights=["org/not-downloaded"]),
        entry("fake-gated", "diarize", gated=True,
              token_url="https://example.org/accept"),
        entry("fake-crash", "diarize"),
        entry("fake-slow", "diarize"),
        entry("fake-env", "diarize", gated=True),
        entry("missing-model", "diarize", family="absent"),
    ]
    return {"schema": "lab.catalogue.v1", "entries": entries}


def make_lab(root: Path, ready: bool = True) -> LabSettings:
    """Write the sandbox under ``root``; return its lab settings.

    A sandbox already there is reused as it is.
    """
    runners = root / "runners"
    envs = root / "envs"
    path = root / "catalogue.json"
    if root.exists():
        return LabSettings(catalogue=path, runners_dir=runners,
                           envs_dir=envs, step_timeout=20.0)
    runners.mkdir(parents=True)
    (runners / "fake.py").write_text(FAKE_RUNNER, encoding="utf-8")
    (runners / "absent.py").write_text("", encoding="utf-8")
    (envs / "fake" / "bin").mkdir(parents=True)
    os.symlink(sys.executable, envs / "fake" / "bin" / "python")
    if ready:
        (envs / "fake" / READY_MARKER).write_text("{}", encoding="utf-8")
    path.write_text(json.dumps(catalogue()), encoding="utf-8")
    return LabSettings(catalogue=path, runners_dir=runners, envs_dir=envs,
                       step_timeout=20.0)
