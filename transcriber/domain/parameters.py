"""Parameters of a transcription: every threshold in one place.

The use cases take them from the settings instead of hard-coding them.
The defaults are the values the analyzer's audio commands use under
``src/`` (window lengths, the clustering threshold, smoothing and turn
gaps) and the untuned starting values of this package (the match
threshold). None of them has been measured on the transcriber yet:
ticket TR-05 measures and sets them.
"""

from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcriber.domain.voices import (
    DEFAULT_MATCH_THRESHOLD,
    Normalisation,
    centre_on_session,
    unit_vectors,
)

NormalisationName = Literal["unit", "centre_on_session"]


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
    ``group_threshold``/``max_speakers``: clustering when the number of
    people is unknown. ``match_threshold``/``normalisation``: matching
    groups to voiceprints. ``smoothing_run``: longest run of words
    flipped back to its neighbours' speaker. ``turn_gap``: silence
    (seconds) that starts a new turn.
    """

    model_config = ConfigDict(frozen=True)

    region_gap: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    window: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    step: float = Field(default=2.5, gt=0, allow_inf_nan=False)
    fine_window: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    fine_step: float = Field(default=0.75, gt=0, allow_inf_nan=False)
    window_minimum: float = Field(default=0.4, ge=0, allow_inf_nan=False)
    centre_windows: bool = True
    group_threshold: float = Field(default=0.5, ge=-1, le=1)
    max_speakers: int = Field(default=10, ge=1)
    match_threshold: float = Field(
        default=DEFAULT_MATCH_THRESHOLD, ge=-1, le=1
    )
    normalisation: NormalisationName = "unit"
    smoothing_run: int = Field(default=1, ge=0)
    turn_gap: float = Field(default=1.0, ge=0, allow_inf_nan=False)

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
