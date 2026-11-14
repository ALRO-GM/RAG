"""Scoring pipelines against the owner's reference labels.

Only the labelled time counts (``unsure`` labels are left out), and
every score carries the labelled duration it rests on
(``docs/transcriber/LAB.md`` §5). Per pipeline:

* the diarization error components on the labelled frames, with the
  names as given (an enrolled id must be the right one: no mapping);
* on the frames labelled ``both``: how much the pipeline marks two or
  more speakers, and how much it gives exactly the labelled people;
* words on the right speaker: words whose middle lies in a ``person``
  or ``both`` label and whose speaker is one of the labelled people;
  words in ``nobody`` labels are counted apart;
* word error rate against the corrected text of the labels that have
  one.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from transcriber.domain.comparison import (
    ComparisonParameters,
    frame_index,
    align,
    error_components,
    frame_count,
    normalise_text,
    to_frames,
)
from transcriber.domain.lab import ReferenceLabel, Segment
from transcriber.domain.models import Word


class PipelineScore(BaseModel):
    """One pipeline measured against the labels (see the module).

    Seconds are of labelled time; ``error_rate`` is (missed + false
    alarm + confusion) over ``reference_speech`` (speaker time, so a
    frame of two people counts twice). ``None`` where nothing was
    labelled to measure it on. ``rank`` is 1 for the lowest error rate.
    """

    model_config = ConfigDict(frozen=True)

    pipeline: str
    rank: Optional[int] = None
    labelled_seconds: float = 0.0
    reference_speech: float = 0.0
    missed: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    error_rate: Optional[float] = None
    overlap_labelled_seconds: float = 0.0
    overlap_detected: Optional[float] = None
    overlap_exact: Optional[float] = None
    words_counted: int = 0
    words_right: int = 0
    words_right_share: Optional[float] = None
    words_in_silence: int = 0
    text_labels: int = 0
    reference_tokens: int = 0
    wer: Optional[float] = None


def score_pipeline(
    pipeline: str,
    segments: Sequence[Segment],
    words: Sequence[Word],
    labels: Sequence[ReferenceLabel],
    duration: float,
    p: ComparisonParameters = ComparisonParameters(),
) -> PipelineScore:
    """Score one pipeline: its named ``segments`` and attributed ``words``.

    ``words`` carry the speaker the pipeline gave them (``None`` when it
    has no diarizer). Labels must not overlap
    (:func:`~transcriber.domain.lab.check_labels`).
    """
    frames = frame_count(duration, p.frame)
    scored_labels = [
        label for label in labels if label.reference() is not None
    ]
    names = sorted(
        {s.speaker for s in segments}
        | {n for label in scored_labels for n in label.speakers}
    )
    hyp = to_frames(segments, frames, p.frame, names).active
    ref = np.zeros_like(hyp)
    labelled = np.zeros(frames, dtype=bool)
    both = np.zeros(frames, dtype=bool)
    for label in scored_labels:
        first = min(frame_index(label.start, p.frame), frames)
        last = min(frame_index(label.end, p.frame), frames)
        labelled[first:last] = True
        for name in label.speakers:
            ref[first:last, names.index(name)] = True
        if label.kind == "both":
            both[first:last] = True
    total, missed, false_alarm, confusion = error_components(
        ref, hyp, labelled
    )
    overlap_frames = float(np.sum(both))
    detected = exact = None
    if overlap_frames:
        marked = hyp[both].sum(axis=1) >= 2
        detected = float(np.sum(marked)) / overlap_frames
        same = np.all(hyp[both] == ref[both], axis=1)
        exact = float(np.sum(same)) / overlap_frames
    counted = right = silent = 0
    errors = tokens = with_text = 0
    for label in scored_labels:
        inside = [
            w for w in words
            if label.start <= (w.start + w.end) / 2 < label.end
        ]
        if label.kind == "nobody":
            silent += len(inside)
        else:
            counted += len(inside)
            right += sum(1 for w in inside if w.speaker in label.speakers)
        if label.text is not None:
            reference = normalise_text(label.text)
            hypothesis = [
                t for w in inside for t in normalise_text(w.text)
            ]
            result = align(reference, hypothesis)
            errors += result.errors
            tokens += result.reference
            with_text += 1
    return PipelineScore(
        pipeline=pipeline,
        labelled_seconds=float(np.sum(labelled)) * p.frame,
        reference_speech=total * p.frame,
        missed=missed * p.frame, false_alarm=false_alarm * p.frame,
        confusion=confusion * p.frame,
        error_rate=(missed + false_alarm + confusion) / total
        if total else None,
        overlap_labelled_seconds=overlap_frames * p.frame,
        overlap_detected=detected, overlap_exact=exact,
        words_counted=counted, words_right=right,
        words_right_share=right / counted if counted else None,
        words_in_silence=silent, text_labels=with_text,
        reference_tokens=tokens,
        wer=errors / tokens if tokens else None,
    )


def rank_scores(scores: Sequence[PipelineScore]) -> List[PipelineScore]:
    """Return ``scores`` best first, each with its ``rank``.

    Lowest error rate first; then most words on the right speaker;
    then lowest word error rate. Missing values sort last; equal keys
    share a rank.
    """
    def key(score: PipelineScore) -> Tuple[float, float, float]:
        """Sort key: missing values last."""
        return (
            score.error_rate if score.error_rate is not None else 9e9,
            -(score.words_right_share or -1.0),
            score.wer if score.wer is not None else 9e9,
        )

    ordered = sorted(scores, key=key)
    ranked: List[PipelineScore] = []
    for n, score in enumerate(ordered):
        rank = n + 1
        if ranked and key(ranked[-1]) == key(score):
            rank = ranked[-1].rank or rank
        ranked.append(score.model_copy(update={"rank": rank}))
    return ranked
