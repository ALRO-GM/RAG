"""Parameters of a transcription: every threshold in one place.

The use cases take them from the settings instead of hard-coding them.
The window lengths, the clustering threshold, smoothing and turn gaps
are the values the analyzer's audio commands use under ``src/``. The
match threshold, the single-voice rule and the speech-activity level
were measured by ticket TR-05 on synthetic meetings of read speech
(``docs/transcriber/IDENTIFICATION.md``); a phone on a table has not
been measured yet.
"""

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcriber.domain.certainty import DEFAULT_DOUBT_MARGIN
from transcriber.domain.voices import (
    DEFAULT_MATCH_THRESHOLD,
    Normalisation,
    centre_on_session,
    unit_vectors,
)

NormalisationName = Literal["unit", "centre_on_session"]
IdentificationName = Literal["seeded", "cluster_then_match"]


class TranscriptionParameters(BaseModel):
    """How speech becomes voice groups, speakers, turns and sentences.

    ``region_gap``: words closer than this (seconds) are one speech
    region. ``window``/``step``: the long analysis windows that find
    the voices (and that enrolment samples are cut into, so voiceprints
    and session groups are averaged over windows of the same length).
    ``fine_window``/``fine_step``: the short windows given to the
    nearest voice for time resolution. ``window_minimum``: regions
    shorter than this are not embedded. ``centre_windows``: subtract the
    recording's mean voice before grouping (as the analyzer does).
    ``single_voice_threshold``/``single_voice_quantile``: when the
    number of people is unknown, a recording whose long windows have,
    at that quantile, at least this cosine with the windows' mean
    direction (raw unit vectors, before centring) is one voice and is
    not clustered (centring one voice would leave only its own
    variation to split).
    ``group_threshold``/``max_speakers``: clustering when the number of
    people is unknown. ``identification``: how voices get names when the
    session has at least one voiceprint to compare with. ``seeded``
    (:func:`~transcriber.domain.voices.seed_voices`): the voiceprints
    claim the long windows at ``match_threshold`` first and only the
    unclaimed windows are clustered, into anonymous voices;
    ``cluster_then_match``: the voices are found first
    (:func:`~transcriber.domain.voices.group_voices`) and each group is
    then matched to a voiceprint. Without voiceprints the voices are
    always found by clustering. ``match_threshold``/``normalisation``: matching
    groups to voiceprints. ``smoothing_run``: longest run of words
    flipped back to its neighbours' speaker. ``turn_gap``: silence
    (seconds) that starts a new turn. ``activity_frame``/
    ``activity_threshold``: the audio is cut into frames of this length
    (seconds) and a frame whose RMS level reaches the threshold (full
    scale is 1) is speech; a recognised word with no such frame in its
    span is dropped (recognisers invent words over silence).
    ``doubt_margin``: a sentence whose speaker margin (the voice windows'
    cosine with its speaker minus that with the closest other voice;
    :mod:`transcriber.domain.certainty`) is below this is ``doubtful``;
    a window inside a sentence that prefers another voice by at least
    this much makes it an ``overlap``; a speaker change inside a
    sentence supported by less than this (or by a single window) is
    undone.

    Measured in TR-05 (``docs/transcriber/IDENTIFICATION.md``; figures
    on the readers held out from the choice): ``match_threshold`` 0.95
    on raw unit vectors, the best of the sweep for cluster-then-match
    on the tuning readers (held out, number of speakers given, mean of
    the four enrolment conditions: 86.1% of words on the right id
    against 84.5% at the old 0.5). The single-voice rule 0.95 at
    quantile 0.1 merged 3 of 20 two-reader recordings into one voice
    and split none of 40 one-reader recordings; the old 0.9 on the
    minimum merged 10 of 20, split 2 of 40 and merged one four-reader
    meeting. ``activity_frame`` 0.025 s and ``activity_threshold``
    0.003 (about -50 dBFS) were kept: they dropped 5 of 22,379 words
    said by a reader and 7 of the 56 words Whisper timed entirely
    inside a pause (clean read speech with digital silence between
    turns, unlike a phone on a table). ``identification`` ``seeded``:
    on the held-out readers, mean of the four enrolment conditions, it
    put 91.4% of words on the right id against 86.1% for
    cluster-then-match with the number of speakers given (both at 0.95,
    the default ``match_threshold``, unit vectors); with it estimated
    88.2% (at 0.92) against 82.6% (at 0.94), each at its own best
    threshold on the tuning readers.
    """

    model_config = ConfigDict(frozen=True)

    region_gap: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    window: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    step: float = Field(default=2.5, gt=0, allow_inf_nan=False)
    fine_window: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    fine_step: float = Field(default=0.75, gt=0, allow_inf_nan=False)
    window_minimum: float = Field(default=0.4, ge=0, allow_inf_nan=False)
    centre_windows: bool = True
    single_voice_threshold: float = Field(default=0.95, ge=-1, le=1)
    single_voice_quantile: float = Field(default=0.1, ge=0, lt=1)
    group_threshold: float = Field(default=0.5, ge=-1, le=1)
    max_speakers: int = Field(default=10, ge=1)
    match_threshold: float = Field(
        default=DEFAULT_MATCH_THRESHOLD, ge=-1, le=1
    )
    normalisation: NormalisationName = "unit"
    identification: IdentificationName = "seeded"
    smoothing_run: int = Field(default=1, ge=0)
    turn_gap: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    activity_frame: float = Field(default=0.025, gt=0, allow_inf_nan=False)
    activity_threshold: float = Field(default=0.003, ge=0, le=1)
    doubt_margin: float = Field(
        default=DEFAULT_DOUBT_MARGIN, ge=0, le=2, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def _windows(self) -> "TranscriptionParameters":
        """Reject a window shorter than the shortest region embedded."""
        for length in (self.window, self.fine_window):
            if length < self.window_minimum:
                raise ValueError(
                    "windows cannot be shorter than window_minimum"
                )
        return self

    def coarse(self) -> Tuple[float, float, float]:
        """Return ``(length, step, minimum)`` of the long windows."""
        return self.window, self.step, self.window_minimum

    def fine(self) -> Tuple[float, float, float]:
        """Return ``(length, step, minimum)`` of the short windows."""
        return self.fine_window, self.fine_step, self.window_minimum

    def normaliser(self) -> Normalisation:
        """Return the normalisation function named by ``normalisation``."""
        if self.normalisation == "centre_on_session":
            return centre_on_session
        return unit_vectors
