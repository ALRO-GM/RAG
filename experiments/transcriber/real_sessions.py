"""Rerun the transcriber on copies of real sessions; print aggregates.

The recordings are private. This script copies the first ``--first``
finished sessions (in the order of their ids) and the voiceprints from
``--data-dir`` into ``--work`` (a scratch directory outside the
repository, deleted by the caller afterwards), queues each copy again
with the options it was started with and runs the current pipeline on
it with the real models. It prints aggregate counts only: sessions are
``R1``, ``R2``... in the order of their ids and enrolled people ``A``,
``B``... in the order of their ids; no audio, text, id or name is
printed.

Per recording it prints:

* who was named (``A``/``B``/``anonymous``), and who only through the
  expected-speakers floor (``floor``, when the pipeline has that rule);
* utterances, the doubtful and overlap shares, sentences holding a
  stretch ``keep_sentences_whole`` moved and how many stretches moved;
* utterances out of time order (a start before the previous one);
* words the recogniser returned, and how many the seam step dropped;
* the short voice windows: how many, the share whose two closest
  voices are within the doubt margin, and the 10% quantile of that gap;
* recognition: the share of words with a probability, and the seconds
  spent recognising and scoring.

Usage (the source directory is only read)::

    PYTHONPATH=. python experiments/transcriber/real_sessions.py \\
        --data-dir data/transcriber --work /tmp/real-sessions --first 3
    rm -r /tmp/real-sessions

``--set name=value`` overrides a transcription parameter (repeatable),
``--language`` the sessions' language. The script may import
:mod:`transcriber`; it never imports ``src``.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import transcriber.application.transcribe as transcribe_module
from transcriber.adapters.outbound.av_audio_decoder import AvAudioDecoder
from transcriber.adapters.outbound.fs_session_repository import (
    FileSystemSessionRepository,
)
from transcriber.adapters.outbound.fs_voiceprint_repository import (
    FileSystemVoiceprintRepository,
)
from transcriber.adapters.outbound.thread_session_locks import (
    ThreadSessionLocks,
)
from transcriber.adapters.outbound.wavlm_voice_encoder import (
    WavLMVoiceEncoder,
)
from transcriber.adapters.outbound.whisper_recogniser import (
    WhisperRecogniser,
)
from transcriber.application.ports import (
    Audio,
    RecognisedSpeech,
    WordScores,
)
from transcriber.application.transcribe import (
    TranscribeSession,
    TranscriptionOptions,
    with_session_options,
)
from transcriber.domain.certainty import VoiceEvidence
from transcriber.domain.models import Session, SessionState, Transcript, Word
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.seams import resolve_seams
from transcriber.domain.sessions import move

Row = Dict[str, Any]


class Watched:
    """What one run passed between its stages, kept for the counts."""

    def __init__(self) -> None:
        """Start empty."""
        self.raw: List[Word] = []
        self.evidence: Optional[VoiceEvidence] = None
        self.moved: Sequence[Tuple[float, float]] = ()
        self.recognise_seconds = 0.0
        self.score_seconds = 0.0


class WatchedRecogniser(WhisperRecogniser):
    """Whisper that keeps the raw words and times its two passes."""

    def __init__(self, watched: Watched, device: str) -> None:
        """Remember where to keep what is seen."""
        super().__init__(device=device)  # type: ignore[arg-type]
        self.watched = watched

    def recognise(
        self, audio: Audio, language: Optional[str] = None
    ) -> RecognisedSpeech:
        """Recognise, keeping the words as returned."""
        began = time.perf_counter()
        speech = super().recognise(audio, language)
        self.watched.recognise_seconds = time.perf_counter() - began
        self.watched.raw = list(speech.words)
        return speech

    def score_words(
        self, audio: Audio, words: Sequence[Word],
        language: Optional[str] = None,
    ) -> WordScores:
        """Score, timing it."""
        began = time.perf_counter()
        scores = super().score_words(audio, words, language)
        self.watched.score_seconds = time.perf_counter() - began
        return scores


def watch_certainty(watched: Watched) -> None:
    """Keep the evidence and moved stretches the use case assesses with."""
    original = getattr(transcribe_module, "assess_utterances")

    def assess(*args: Any, **kwargs: Any) -> Any:
        """Keep the arguments, then assess as usual."""
        watched.evidence = args[1] if len(args) > 1 else kwargs["evidence"]
        watched.moved = args[3] if len(args) > 3 else kwargs.get("moved", ())
        return original(*args, **kwargs)

    setattr(transcribe_module, "assess_utterances", assess)


def copy_sessions(source: Path, work: Path, first: int) -> List[str]:
    """Copy the first ``first`` finished sessions and the voiceprints."""
    if work.exists():
        raise SystemExit(f"{work} exists; give a new scratch directory.")
    chosen: List[str] = []
    for folder in sorted((source / "sessions").iterdir()):
        stored = folder / "session.json"
        if not stored.exists():
            continue
        if json.loads(stored.read_text())["state"] != "done":
            continue
        chosen.append(folder.name)
        if len(chosen) == first:
            break
    for name in chosen:
        shutil.copytree(source / "sessions" / name,
                        work / "sessions" / name)
    shutil.copytree(source / "voiceprints", work / "voiceprints")
    return chosen


def window_gaps(evidence: Optional[VoiceEvidence]) -> List[float]:
    """Return, per short window, the gap between its two closest voices."""
    if evidence is None:
        return []
    named = [n for n, v in enumerate(evidence.voices) if v is not None]
    if len(named) < 2:
        return []
    block = np.sort(evidence.scores[:, named].astype(np.float64), axis=1)
    return [float(x) for x in block[:, -1] - block[:, -2]]


def summary(transcript: Transcript, watched: Watched,
            letters: Dict[str, str], margin: float,
            words: Sequence[Word]) -> Row:
    """Return the aggregate counts of one rerun."""
    utterances = transcript.utterances
    count = len(utterances)
    named: Dict[str, str] = {}
    for speaker in transcript.speakers:
        if speaker.enrolled:
            how = "floor" if getattr(speaker, "named_by", None) == \
                "expected_floor" else "threshold"
            named[letters.get(speaker.id, "?")] = how
    raw = [w for w in watched.raw if w.text.strip()]
    gaps = window_gaps(watched.evidence)
    moved_in = sum(
        1 for u in utterances if any(
            min(r, u.end) > max(left, u.start) for left, r in watched.moved)
    )

    def share(flag: str) -> float:
        """Share of utterances whose ``flag`` is true."""
        return round(sum(
            1 for u in utterances if getattr(u, flag, None)) / count, 4)

    row: Row = {
        "named": dict(sorted(named.items())),
        "anonymous_speakers": sum(
            1 for s in transcript.speakers if not s.enrolled),
        "utterances": count,
        "doubtful_share": share("doubtful"),
        "doubtful": sum(1 for u in utterances if u.doubtful),
        "overlap_share": share("overlap"),
        "overlap": sum(1 for u in utterances if u.overlap),
        "moved_stretches": len(watched.moved),
        "utterances_with_moved_stretch": moved_in,
        "out_of_order": sum(
            1 for a, b in zip(utterances, utterances[1:])
            if b.start < a.start),
        "words_returned": len(raw),
        "words_dropped_by_seams": len(raw) - len(resolve_seams(raw)),
        "short_windows": len(gaps),
        "windows_gap_below_margin_share": round(
            sum(1 for g in gaps if g < margin) / len(gaps), 4)
        if gaps else None,
        "windows_gap_p10": round(float(np.quantile(gaps, 0.1)), 3)
        if gaps else None,
        "words_with_probability_share": round(sum(
            1 for w in words if w.probability is not None)
            / max(len(words), 1), 4),
        "recognise_seconds": round(watched.recognise_seconds, 1),
        "score_seconds": round(watched.score_seconds, 1),
    }
    return row


def parameters_of(settings: Sequence[str]) -> TranscriptionParameters:
    """Return the default parameters with ``name=value`` overrides."""
    fields: Dict[str, Any] = {}
    for item in settings:
        name, _, value = item.partition("=")
        fields[name] = json.loads(value)
    return TranscriptionParameters(**fields)


def main() -> None:
    """Copy, rerun and print the aggregates of each session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--first", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    parameters = parameters_of(args.set)
    chosen = copy_sessions(args.data_dir, args.work, args.first)
    watched = Watched()
    watch_certainty(watched)
    encoder = WavLMVoiceEncoder(device=args.device)
    sessions = FileSystemSessionRepository(args.work)
    voiceprints = FileSystemVoiceprintRepository(args.work,
                                                 encoder.model_name)
    letters = {
        v.speaker_id: chr(ord("A") + n)
        for n, v in enumerate(sorted(voiceprints.all(),
                                     key=lambda v: v.speaker_id))
    }
    recogniser = WatchedRecogniser(watched, args.device)
    use_case = TranscribeSession(
        sessions, voiceprints, AvAudioDecoder(), recogniser, encoder,
        ThreadSessionLocks(), parameters,
    )
    for number, session_id in enumerate(chosen, start=1):
        session = sessions.get(session_id)
        assert session is not None
        if args.language is not None:
            session = Session.model_validate(
                {**session.model_dump(), "language": args.language})
        sessions.save(move(session, SessionState.QUEUED))
        options = with_session_options(TranscriptionOptions(), session)
        transcript = use_case(session_id, options)
        assert transcript is not None
        row = {"recording": f"R{number}", "language": args.language
               or "as recorded",
               **summary(transcript, watched, letters,
                         parameters.doubt_margin,
                         sessions.words(session_id))}
        print(json.dumps(row), flush=True)
    recogniser.release()
    encoder.release()


if __name__ == "__main__":
    main()
