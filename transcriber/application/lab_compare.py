"""Assemble the comparison of a lab run from its stored stage results.

:func:`compare_run` takes what the steps produced (named diarizations,
transcripts) and returns :class:`LabResults`, the document the API
serves and the CLI summarises (``docs/transcriber/LAB.md`` §4): one row
per diarization track, transcript and pipeline, the pairwise matrices,
the consensus, the effect of each denoiser and the fragments. All the
arithmetic is in :mod:`transcriber.domain.comparison`; this module only
wires it.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from transcriber.domain.comparison import (
    ComparisonParameters,
    Consensus,
    Fragment,
    StateSpan,
    Track,
    TrackComparison,
    TranscriptComparison,
    agreement_with,
    attribute,
    compare_tracks,
    compare_transcripts,
    consensus,
    consensus_coverage,
    consensus_spans,
    distinct_texts,
    fragments,
    summarise_track,
    text_of,
    tokens_of,
    word_agreement,
    words_on_consensus,
)
from transcriber.domain.lab import (
    NO_DENOISER,
    CatalogueEntry,
    PipelineRef,
    Segment,
    SpeakerMapping,
    StageOutcome,
    track_id,
)
from transcriber.domain.lab_reference import (
    REFERENCE_TRACK,
    ReferencePipelineRow,
    ReferenceResults,
    ReferenceTrackRow,
    ReferenceTranscriptRow,
    SessionReference,
    score_track,
)
from transcriber.domain.models import Word


class ModelInfo(BaseModel):
    """What the results show of a model: name and licence."""

    model_config = ConfigDict(frozen=True)

    id: str
    stage: str
    display_name: str
    family: str
    licence: str
    licence_commercial: bool
    gated: bool


class TrackRow(BaseModel):
    """One diarization: a diarizer on one denoised audio."""

    model_config = ConfigDict(frozen=True)

    id: str
    denoiser: str
    diarizer: str
    vote_group: str
    speakers: List[str]
    speaker_seconds: Dict[str, float]
    speech_seconds: float
    overlap_seconds: float
    labels: int
    mapping: List[SpeakerMapping]
    agreement_with_consensus: Optional[float] = None
    consensus_seconds: float = 0.0
    seconds: Optional[float] = None
    peak_gpu_mb: Optional[float] = None
    device: Optional[str] = None
    model_version: Optional[str] = None


class TranscriptRow(BaseModel):
    """One recognition: a recogniser on one denoised audio."""

    model_config = ConfigDict(frozen=True)

    id: str
    denoiser: str
    recogniser: str
    words: int
    language: Optional[str] = None
    seconds: Optional[float] = None
    peak_gpu_mb: Optional[float] = None
    device: Optional[str] = None
    model_version: Optional[str] = None


class PipelineRow(BaseModel):
    """One pipeline: denoiser + diarizer + recogniser.

    ``seconds`` adds the processing time of its steps; ``peak_gpu_mb``
    is the largest of theirs. ``agreement_with_consensus`` is its
    track's; ``words_on_consensus`` the share of its words given a
    speaker of the consensus. ``licence_commercial`` is false if any of
    its models is non-commercial.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    denoiser: str
    diarizer: Optional[str] = None
    recogniser: Optional[str] = None
    track: Optional[str] = None
    transcript: Optional[str] = None
    speakers: List[str] = []
    speech_seconds: Optional[float] = None
    overlap_seconds: Optional[float] = None
    words: int = 0
    seconds: Optional[float] = None
    peak_gpu_mb: Optional[float] = None
    agreement_with_consensus: Optional[float] = None
    words_on_consensus: Optional[float] = None
    licence_commercial: bool = True


class WordAgreement(BaseModel):
    """Two pipelines: share of words given the same speaker.

    Each pipeline's words are attributed by both pipelines'
    diarizations; the two directions are averaged.
    """

    model_config = ConfigDict(frozen=True)

    left: str
    right: str
    agreement: Optional[float] = None


class DenoiserEffect(BaseModel):
    """What one denoiser changed for one model, against ``none``.

    For a recogniser: ``wer`` of its words on the denoised audio with
    its words on the original as reference, ``time_agreement``. For a
    diarizer: ``agreement`` / ``name_agreement`` of the two tracks,
    ``error_rate`` with the original's track as reference, and the
    change in agreement with the consensus.
    """

    model_config = ConfigDict(frozen=True)

    denoiser: str
    stage: str
    model: str
    wer: Optional[float] = None
    time_agreement: Optional[float] = None
    agreement: Optional[float] = None
    name_agreement: Optional[float] = None
    error_rate: Optional[float] = None
    consensus_change: Optional[float] = None


class StoredFragment(Fragment):
    """A fragment with how many different texts the recognisers give."""

    distinct_texts: int = 0


class ConsensusInfo(BaseModel):
    """The consensus track: coverage, vote groups and its stretches."""

    model_config = ConfigDict(frozen=True)

    coverage: Optional[float] = None
    vote_groups: Dict[str, str]
    weights: Dict[str, float]
    spans: List[StateSpan]


class LabResults(BaseModel):
    """The comparison of one run (see the module)."""

    model_config = ConfigDict(frozen=True)

    schema_name: str = "lab.results.v1"
    run_id: str
    session_id: str
    duration: float
    parameters: ComparisonParameters
    models: List[ModelInfo]
    tracks: List[TrackRow]
    transcripts: List[TranscriptRow]
    pipelines: List[PipelineRow]
    diarization: List[TrackComparison]
    words: List[WordAgreement]
    recognition: List[TranscriptComparison]
    consensus: ConsensusInfo
    denoiser_effect: List[DenoiserEffect]
    fragments: List[StoredFragment]
    reference: Optional[ReferenceResults] = None


class Diarization(BaseModel):
    """A finished diarize step with its names."""

    model_config = ConfigDict(frozen=True)

    denoiser: str
    entry: CatalogueEntry
    outcome: StageOutcome
    mapping: List[SpeakerMapping]
    segments: List[Segment]

    @property
    def id(self) -> str:
        """Return the track id."""
        return track_id(self.denoiser, self.entry.id)


class Recognition(BaseModel):
    """A finished recognise step."""

    model_config = ConfigDict(frozen=True)

    denoiser: str
    entry: CatalogueEntry
    outcome: StageOutcome

    @property
    def id(self) -> str:
        """Return the transcript id."""
        return track_id(self.denoiser, self.entry.id)


def _model_info(entry: CatalogueEntry) -> ModelInfo:
    """Return what the results show of ``entry``."""
    return ModelInfo(
        id=entry.id, stage=entry.stage, display_name=entry.display_name,
        family=entry.family, licence=entry.licence,
        licence_commercial=entry.licence_commercial, gated=entry.gated,
    )


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round a number that may be missing."""
    return None if value is None else round(value, digits)


def compare_run(
    run_id: str,
    session_id: str,
    duration: float,
    pipelines: Sequence[PipelineRef],
    diarizations: Sequence[Diarization],
    recognitions: Sequence[Recognition],
    entries: Mapping[str, CatalogueEntry],
    denoise_costs: Mapping[str, StageOutcome],
    p: ComparisonParameters = ComparisonParameters(),
    reference: Optional[SessionReference] = None,
) -> LabResults:
    """Compare everything a run produced (see the module).

    Only the ``pipelines`` whose diarization and recognition (when they
    have one) are among the finished ones are listed. ``denoise_costs``
    holds each denoiser's outcome (its time and memory count in every
    pipeline that reads its audio). With the session's ``reference``
    (a corpus recording) every track, transcript and pipeline is also
    measured against it (``reference`` of the results); the reference
    takes no part in the consensus or the fragments.
    """
    tracks = [
        Track(d.id, d.segments, d.entry.vote_group) for d in diarizations
    ]
    by_track = {d.id: d for d in diarizations}
    by_text = {r.id: r for r in recognitions}
    agreed = consensus(tracks, duration, p)
    track_rows = [_track_row(d, agreed, duration, p) for d in diarizations]
    rows = {row.id: row for row in track_rows}
    transcript_rows = [_transcript_row(r) for r in recognitions]
    attributions = _attributions(diarizations, recognitions)
    finished = [
        ref for ref in pipelines
        if (ref.track is None or ref.track in by_track)
        and (ref.transcript is None or ref.transcript in by_text)
        and (ref.track is not None or ref.transcript is not None)
    ]
    pipeline_rows = [
        _pipeline_row(
            ref, rows, by_text, attributions, agreed, entries,
            denoise_costs, p,
        )
        for ref in finished
    ]
    used = {
        e for ref in finished
        for e in (ref.denoiser, ref.diarizer, ref.recogniser) if e
    }
    return LabResults(
        run_id=run_id, session_id=session_id, duration=round(duration, 3),
        parameters=p,
        models=[_model_info(entries[e]) for e in sorted(used)
                if e in entries],
        tracks=track_rows,
        transcripts=transcript_rows,
        pipelines=pipeline_rows,
        diarization=[
            compare_tracks(a, b, duration, p)
            for a in tracks for b in tracks if a.id != b.id
        ],
        words=_word_matrix(finished, attributions),
        recognition=[
            compare_transcripts(
                a.id, a.outcome.words, b.id, b.outcome.words, p
            )
            for a in recognitions for b in recognitions if a.id != b.id
        ],
        consensus=ConsensusInfo(
            coverage=_round(consensus_coverage(agreed)),
            vote_groups={t.id: t.vote_group for t in tracks},
            weights={k: round(v, 4) for k, v in agreed.weights.items()},
            spans=consensus_spans(agreed, p.frame),
        ),
        denoiser_effect=_denoiser_effect(
            diarizations, recognitions, rows, duration, p
        ),
        fragments=_fragments(tracks, recognitions, duration, p),
        reference=None if reference is None else _reference_results(
            reference, diarizations, recognitions, pipeline_rows, p,
        ),
    )


def _reference_results(
    reference: SessionReference,
    diarizations: Sequence[Diarization],
    recognitions: Sequence[Recognition],
    pipelines: Sequence[PipelineRow],
    p: ComparisonParameters,
) -> ReferenceResults:
    """Measure every track, transcript and pipeline against the reference.

    The reference is one more track for :func:`score_track` (DER, JER,
    overlap, speaker count) and one more transcript for
    :func:`compare_transcripts` (WER with the lab's normalisation).
    """
    summary = summarise_track(reference.track(), reference.duration, p)
    tracks = {
        d.id: ReferenceTrackRow(
            denoiser=d.denoiser, diarizer=d.entry.id,
            **score_track(
                reference, Track(d.id, d.segments, d.entry.vote_group), p,
            ).model_dump(),
        )
        for d in diarizations
    }
    texts: Dict[str, ReferenceTranscriptRow] = {}
    tokens = None
    if reference.words is not None:
        tokens = len(tokens_of(reference.words)[0])
        for r in recognitions:
            compared = compare_transcripts(
                REFERENCE_TRACK, reference.words, r.id, r.outcome.words, p,
            )
            texts[r.id] = ReferenceTranscriptRow(
                denoiser=r.denoiser, recogniser=r.entry.id,
                **compared.model_dump(),
            )
    rows = []
    for row in pipelines:
        track = tracks.get(row.track or "")
        text = texts.get(row.transcript or "")
        rows.append(ReferencePipelineRow(
            id=row.id, track=row.track, transcript=row.transcript,
            der=track.der if track else None,
            jer=track.jer if track else None,
            overlap_precision=track.overlap_precision if track else None,
            overlap_recall=track.overlap_recall if track else None,
            speaker_count_error=(
                track.speaker_count_error if track else None
            ),
            wer=_round(text.wer) if text else None,
            licence_commercial=(
                row.licence_commercial and reference.licence_commercial
            ),
        ))
    return ReferenceResults(
        corpus=reference.corpus, recording=reference.recording,
        split=reference.split, licence=reference.licence,
        licence_commercial=reference.licence_commercial,
        source_url=reference.source_url, collar=p.collar,
        speakers=reference.speakers(),
        speech_seconds=round(summary.speech_seconds, 3),
        overlap_seconds=round(summary.overlap_seconds, 3),
        reference_tokens=tokens,
        tracks=list(tracks.values()), transcripts=list(texts.values()),
        pipelines=rows,
    )


def _track_row(
    d: Diarization, agreed: Consensus, duration: float,
    p: ComparisonParameters,
) -> TrackRow:
    """Return the row of one diarization."""
    summary = summarise_track(
        Track(d.id, d.segments, d.entry.vote_group), duration, p
    )
    agreement = agreement_with(agreed, d.id, p.frame)
    return TrackRow(
        id=d.id, denoiser=d.denoiser, diarizer=d.entry.id,
        vote_group=d.entry.vote_group, speakers=summary.speakers,
        speaker_seconds=summary.speaker_seconds,
        speech_seconds=round(summary.speech_seconds, 3),
        overlap_seconds=round(summary.overlap_seconds, 3),
        labels=len({s.speaker for s in d.outcome.segments}),
        mapping=d.mapping,
        agreement_with_consensus=_round(agreement.agreement),
        consensus_seconds=round(agreement.scored_seconds, 3),
        seconds=_round(d.outcome.seconds, 3),
        peak_gpu_mb=_round(d.outcome.peak_gpu_mb, 1),
        device=d.outcome.device, model_version=d.outcome.model_version,
    )


def _transcript_row(r: Recognition) -> TranscriptRow:
    """Return the row of one recognition."""
    return TranscriptRow(
        id=r.id, denoiser=r.denoiser, recogniser=r.entry.id,
        words=len(r.outcome.words), language=r.outcome.language,
        seconds=_round(r.outcome.seconds, 3),
        peak_gpu_mb=_round(r.outcome.peak_gpu_mb, 1),
        device=r.outcome.device, model_version=r.outcome.model_version,
    )


Attributions = Dict[Tuple[str, str], List[Optional[str]]]
"""(transcript, track) → the speaker each of the transcript's words gets."""


def _attributions(
    diarizations: Sequence[Diarization], recognitions: Sequence[Recognition]
) -> Attributions:
    """Attribute every transcript's words with every diarization."""
    return {
        (r.id, d.id): attribute(r.outcome.words, d.segments)
        for r in recognitions for d in diarizations
    }


def attributed_words(
    words: Sequence[Word], segments: Sequence[Segment]
) -> List[Word]:
    """Return ``words`` with the speaker ``segments`` give each."""
    return [
        w.model_copy(update={"speaker": s})
        for w, s in zip(words, attribute(words, segments))
    ]


def _pipeline_row(
    ref: PipelineRef,
    tracks: Mapping[str, TrackRow],
    texts: Mapping[str, Recognition],
    attributions: Attributions,
    agreed: Consensus,
    entries: Mapping[str, CatalogueEntry],
    denoise_costs: Mapping[str, StageOutcome],
    p: ComparisonParameters,
) -> PipelineRow:
    """Return the row of one finished pipeline."""
    track = tracks.get(ref.track or "")
    text = texts.get(ref.transcript or "")
    costs = [denoise_costs.get(ref.denoiser)]
    if track is not None:
        costs.append(StageOutcome(
            stage="diarize", model=track.diarizer, seconds=track.seconds,
            peak_gpu_mb=track.peak_gpu_mb,
        ))
    if text is not None:
        costs.append(text.outcome)
    seconds = [c.seconds for c in costs if c is not None and c.seconds]
    memory = [
        c.peak_gpu_mb for c in costs if c is not None and c.peak_gpu_mb
    ]
    on_consensus = None
    if text is not None and track is not None:
        on_consensus = words_on_consensus(
            text.outcome.words,
            attributions[(text.id, track.id)], agreed, p.frame,
        )
    models = [m for m in (ref.denoiser, ref.diarizer, ref.recogniser) if m]
    return PipelineRow(
        id=ref.id, denoiser=ref.denoiser, diarizer=ref.diarizer,
        recogniser=ref.recogniser, track=ref.track,
        transcript=ref.transcript,
        speakers=track.speakers if track is not None else [],
        speech_seconds=track.speech_seconds if track else None,
        overlap_seconds=track.overlap_seconds if track else None,
        words=len(text.outcome.words) if text is not None else 0,
        seconds=round(sum(seconds), 3) if seconds else None,
        peak_gpu_mb=max(memory) if memory else None,
        agreement_with_consensus=(
            track.agreement_with_consensus if track else None
        ),
        words_on_consensus=_round(on_consensus),
        licence_commercial=all(
            entries[m].licence_commercial for m in models if m in entries
        ),
    )


def _word_matrix(
    pipelines: Sequence[PipelineRef], attributions: Attributions
) -> List[WordAgreement]:
    """Return the word-level speaker agreement of every pair."""
    with_both = [
        ref for ref in pipelines
        if ref.track is not None and ref.transcript is not None
    ]
    out: List[WordAgreement] = []
    for n, left in enumerate(with_both):
        for right in with_both[n + 1:]:
            lt, lk = left.transcript or "", left.track or ""
            rt, rk = right.transcript or "", right.track or ""
            one = word_agreement(attributions[(lt, lk)],
                                 attributions[(lt, rk)])
            two = word_agreement(attributions[(rt, rk)],
                                 attributions[(rt, lk)])
            values = [v for v in (one, two) if v is not None]
            out.append(WordAgreement(
                left=left.id, right=right.id,
                agreement=_round(sum(values) / len(values))
                if values else None,
            ))
    return out


def _denoiser_effect(
    diarizations: Sequence[Diarization],
    recognitions: Sequence[Recognition],
    rows: Mapping[str, TrackRow],
    duration: float,
    p: ComparisonParameters,
) -> List[DenoiserEffect]:
    """Return what each denoiser changed against ``none``."""
    out: List[DenoiserEffect] = []
    plain_texts = {
        r.entry.id: r for r in recognitions if r.denoiser == NO_DENOISER
    }
    for r in recognitions:
        base = plain_texts.get(r.entry.id)
        if r.denoiser == NO_DENOISER or base is None:
            continue
        compared = compare_transcripts(
            base.id, base.outcome.words, r.id, r.outcome.words, p
        )
        out.append(DenoiserEffect(
            denoiser=r.denoiser, stage="recognise", model=r.entry.id,
            wer=_round(compared.wer),
            time_agreement=_round(compared.time_agreement),
        ))
    plain_tracks = {
        d.entry.id: d for d in diarizations if d.denoiser == NO_DENOISER
    }
    for d in diarizations:
        base_track = plain_tracks.get(d.entry.id)
        if d.denoiser == NO_DENOISER or base_track is None:
            continue
        pair = compare_tracks(
            Track(base_track.id, base_track.segments, ""),
            Track(d.id, d.segments, ""), duration, p,
        )
        before = rows[base_track.id].agreement_with_consensus
        after = rows[d.id].agreement_with_consensus
        out.append(DenoiserEffect(
            denoiser=d.denoiser, stage="diarize", model=d.entry.id,
            agreement=_round(pair.agreement),
            name_agreement=_round(pair.name_agreement),
            error_rate=_round(pair.error_rate),
            consensus_change=_round(after - before)
            if after is not None and before is not None else None,
        ))
    return out


def _fragments(
    tracks: Sequence[Track],
    recognitions: Sequence[Recognition],
    duration: float,
    p: ComparisonParameters,
) -> List[StoredFragment]:
    """Return the fragments with how many texts differ in each."""
    out: List[StoredFragment] = []
    order = {}
    for r in recognitions:
        middles = np.array(
            [(w.start + w.end) / 2 for w in r.outcome.words], dtype=float
        )
        ranked = np.argsort(middles, kind="stable")
        order[r.id] = (middles[ranked], ranked)
    for fragment in fragments(tracks, duration, p):
        texts = {}
        for r in recognitions:
            middles, ranked = order[r.id]
            first, last = np.searchsorted(
                middles, [fragment.start, fragment.end], side="left"
            )
            inside = sorted(int(i) for i in ranked[first:last])
            texts[r.id] = text_of(r.outcome.words, inside)
        out.append(StoredFragment(
            **fragment.model_dump(),
            distinct_texts=distinct_texts(texts) if texts else 0,
        ))
    return out
