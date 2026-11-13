"""Measure transcription and diarization on synthetic meetings.

A "meeting" is built by concatenating read-speech utterances of several
speakers from Multilingual LibriSpeech (public, with transcripts and
speaker ids), with short silences in between. That gives exact ground
truth (who spoke when, and what) without any private recording.

It is an optimistic test: read speech, studio-like audio, no overlap,
no interruptions. It checks that the pipeline works and calibrates the
voice threshold; it does not predict accuracy on real meetings.

Usage::

    uv run python experiments/audio/meeting_eval.py DEV.parquet \
        --meetings 6 --speakers 4 --turns 16 --language es
"""

import argparse
import io
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile

from src.adapters.outbound.soundfile_audio_reader import (
    DEFAULT_SAMPLE_RATE,
    resample,
    to_mono,
)
from src.adapters.outbound.whisper_recogniser import WhisperRecogniser
from src.adapters.outbound.xvector_diarizer import XVectorDiarizer
from src.application.skills.ports import Audio
from src.bootstrap import resolve_runtime
from src.config import SkillsSettings
from src.domain.skills.speech import (
    SpeakerSpan,
    assign_speakers,
    smooth_speakers,
    speech_regions,
)
from src.domain.skills.speech_metrics import (
    speaker_accuracy,
    word_error_rate,
)

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def build_meeting(
    table: Any, speakers: List[str], turns: int, rng: random.Random
) -> Tuple[Audio, List[SpeakerSpan], str]:
    """Concatenate ``turns`` utterances, never the same speaker twice."""
    by_speaker: Dict[str, List[int]] = {}
    for row, speaker in enumerate(table.column("speaker_id").to_pylist()):
        if speaker in speakers:
            by_speaker.setdefault(speaker, []).append(row)
    pieces: List[np.ndarray[Any, np.dtype[np.float32]]] = []
    spans: List[SpeakerSpan] = []
    texts: List[str] = []
    cursor, previous = 0.0, ""
    for _ in range(turns):
        speaker = rng.choice([s for s in speakers if s != previous])
        previous = speaker
        row = by_speaker[speaker].pop(
            rng.randrange(len(by_speaker[speaker]))
        )
        record = table.slice(row, 1).to_pylist()[0]
        data, rate = soundfile.read(
            io.BytesIO(record["audio"]["bytes"]), dtype="float32",
            always_2d=True,
        )
        samples = resample(to_mono(data), int(rate), DEFAULT_SAMPLE_RATE)
        gap = np.zeros(
            int(rng.uniform(0.2, 1.0) * DEFAULT_SAMPLE_RATE), np.float32
        )
        start = cursor + gap.size / DEFAULT_SAMPLE_RATE
        end = start + samples.size / DEFAULT_SAMPLE_RATE
        spans.append(SpeakerSpan(speaker=speaker, start=start, end=end))
        texts.append(record["transcript"])
        pieces += [gap, samples]
        cursor = end
    audio = Audio(np.concatenate(pieces), DEFAULT_SAMPLE_RATE)
    return audio, spans, " ".join(texts)


def main() -> None:
    """Run the evaluation and append one row per meeting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--meetings", type=int, default=6)
    parser.add_argument("--speakers", type=int, default=4)
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--language", default="es")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/audio/meetings.jsonl"),
    )
    args = parser.parse_args()
    settings = SkillsSettings()
    plan = resolve_runtime(
        settings.device, settings.torch_threads, settings.cpu_cores,
        settings.runtime_plan_path,
    )[1]
    recogniser = WhisperRecogniser(settings.speech_model, plan=plan)
    diarizer = XVectorDiarizer(settings.voice_model, plan=plan)
    table = pq.read_table(args.parquet)
    everyone = sorted(set(table.column("speaker_id").to_pylist()))
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for meeting in range(args.meetings):
        people = rng.sample(everyone, args.speakers)
        audio, truth, reference = build_meeting(
            table, people, args.turns, rng
        )
        began = time.perf_counter()
        speech = recogniser.recognise(audio, args.language)
        asr_seconds = time.perf_counter() - began
        words = [w for w in speech.words if w.text.strip()]
        regions = speech_regions(words)

        def score(count: Any, threshold: float) -> Tuple[float, int]:
            diarizer.threshold = threshold
            spans = diarizer.diarize(audio, regions, count)
            labelled = smooth_speakers(assign_speakers(words, spans))
            return (
                speaker_accuracy(labelled, truth)[0],
                len({s.speaker for s in spans}),
            )

        began = time.perf_counter()
        known, _ = score(args.speakers, 0.5)
        voice_seconds = time.perf_counter() - began
        sweep = {}
        for threshold in THRESHOLDS:
            accuracy, found = score(None, threshold)
            sweep[str(threshold)] = {
                "accuracy": round(accuracy, 4), "speakers": found,
            }
        row = {
            "meeting": meeting, "seed": args.seed,
            "language": args.language, "speakers": args.speakers,
            "turns": args.turns, "duration": round(audio.duration, 1),
            "speech_model": recogniser.model_name,
            "voice_model": diarizer.model_name,
            "wer": round(word_error_rate(
                reference, " ".join(w.text for w in words)
            ), 4),
            "speaker_accuracy_known_count": round(known, 4),
            "threshold_sweep": sweep,
            "asr_seconds": round(asr_seconds, 1),
            "voice_seconds": round(voice_seconds, 1),
            "device": plan.device,
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        best = max(sweep.items(), key=lambda kv: kv[1]["accuracy"])
        print(f"meeting {meeting}: {row['duration']:.0f}s  WER "
              f"{row['wer']:.3f}  speakers right (count known) "
              f"{known:.3f}  best threshold {best[0]} -> "
              f"{best[1]['accuracy']:.3f} with {best[1]['speakers']} "
              f"voices  [asr {asr_seconds:.1f}s, voices "
              f"{voice_seconds:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
