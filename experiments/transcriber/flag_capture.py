"""Capture what the certainty flags are computed from (TR-19, issue #95).

The doubtful and overlap flags of a sentence depend on the short voice
windows' cosines, which no stored lab output keeps. This script runs the
service's own path (:class:`~transcriber.application.transcribe.
TranscribeSession` with the diarizer the settings' rule picks, as
``service_diarizer_eval.py corpus`` does) on the corpus sessions the lab
imported, and writes for each one a ``<session>.json.gz`` with the
inputs of the flags: the words before the "kept whole" rule, the voice
evidence (windows, cosines, voices), the names the voices got, the
stretches the diarizer and the detector heard overlapped, the diarizer's
segments and the sentences the service produced. ``flag_retune.py``
recomputes the flags from these files with any thresholds, without
running a model again.

Usage (GPU, outside the shell; see the ticket's notes)::

    PYTHONPATH=. python experiments/transcriber/flag_capture.py \\
        --data-dir data/transcriber --out <dir>

The script may import :mod:`transcriber`; it never imports ``src``.
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from transcriber.adapters.outbound.fs_lab_repository import (
    FileSystemLabRepository,
)
from transcriber.application import transcribe as transcribe_module
from transcriber.application.lab_builtin import (
    SCRATCH_ID,
    _GivenAudio,
    _NoLocks,
    _NoVoiceprints,
    _ScratchSessions,
)
from transcriber.application.transcribe import (
    TranscribeSession,
    TranscriptionOptions,
)
from transcriber.bootstrap import build_diarizer, build_models
from transcriber.config import load_settings
from transcriber.domain.certainty import VoiceEvidence
from transcriber.domain.models import Session, SessionState, Utterance, Word

Capture = Dict[str, Any]


def _evidence(evidence: VoiceEvidence) -> Capture:
    """Return ``evidence`` as JSON-able lists."""
    return {
        "windows": [[round(a, 3), round(b, 3)] for a, b in evidence.windows],
        "scores": [[round(float(x), 5) for x in row]
                   for row in evidence.scores],
        "voices": list(evidence.voices),
    }


def main() -> None:
    """Capture the flag inputs of every corpus session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", default="")
    parser.add_argument("--calls-speakers", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = load_settings(os.environ).model_copy(
        update={"data_dir": args.data_dir})
    recogniser, encoder = build_models(settings)
    built, rule = build_diarizer(settings, os.environ, recogniser.release)
    if built is None:
        raise SystemExit("The settings give no lab diarizer.")
    repo = FileSystemLabRepository(args.data_dir)
    names = sorted(
        p.name for p in (args.data_dir / "lab").iterdir()
        if p.name.startswith(("ami-", "callhome-"))
    )
    if args.sessions:
        names = [n for n in names if n in set(args.sessions.split(","))]
    captured: Capture = {}
    original_keep = getattr(transcribe_module, "keep_sentences_whole")
    original_assess = getattr(transcribe_module, "assess_utterances")
    original_add = getattr(transcribe_module, "add_detected_overlaps")

    def keep(words: Sequence[Word], evidence: Any, *a: Any, **k: Any) -> Any:
        """Record the words and evidence the rule starts from."""
        captured["words"] = [w.model_dump(mode="json") for w in words]
        if evidence is not None:
            captured["evidence_labels"] = _evidence(evidence)
        return original_keep(words, evidence, *a, **k)

    def assess(
        utterances: Sequence[Utterance], evidence: Any, *a: Any, **k: Any
    ) -> Any:
        """Record the names the voices got."""
        if evidence is not None:
            captured["evidence_names"] = list(evidence.voices)
        return original_assess(utterances, evidence, *a, **k)

    def add(
        utterances: Sequence[Utterance],
        detected: Sequence[Tuple[float, float]], *a: Any, **k: Any
    ) -> Any:
        """Record the overlaps the diarizer and the detector heard."""
        captured["detected"] = [[x, y] for x, y in detected]
        return original_add(utterances, detected, *a, **k)

    setattr(transcribe_module, "keep_sentences_whole", keep)
    setattr(transcribe_module, "assess_utterances", assess)
    setattr(transcribe_module, "add_detected_overlaps", add)
    p = settings.transcription
    for name in names:
        target = args.out / f"{name}.json.gz"
        if target.exists():
            continue
        reference = repo.reference(name)
        if reference is None or reference.excluded() is not None:
            print(f"{name}: skipped")
            continue
        key = repo.original_key(name)
        assert key is not None
        audio = repo.load_audio(name, key)
        count = args.calls_speakers if name.startswith("callhome-") \
            else None
        scratch = _ScratchSessions(Session(
            id=SCRATCH_ID, language=reference.language,
            state=SessionState.QUEUED,
        ))
        session = TranscribeSession(
            scratch, _NoVoiceprints(), _GivenAudio(audio), recogniser,
            encoder, _NoLocks(), p, None, built, rule,
        )
        captured.clear()
        session(SCRATCH_ID, TranscriptionOptions(num_speakers=count))
        recogniser.release()
        encoder.release()
        transcript = scratch.transcript(SCRATCH_ID)
        assert transcript is not None
        row: Capture = {
            "session": name, "corpus": reference.corpus,
            "duration": audio.duration, "num_speakers": count,
            "diarizer": transcript.pipeline.diarizer,
            "parameters": p.model_dump(mode="json"),
            "reference": [s.model_dump(mode="json")
                          for s in reference.segments],
            "utterances": [u.model_dump(mode="json")
                           for u in transcript.utterances],
        }
        row.update(captured)
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(row, handle)
        print(name, transcript.pipeline.diarizer,
              len(transcript.utterances), flush=True)


if __name__ == "__main__":
    main()
