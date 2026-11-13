"""Use case: turn a recording into a diarized transcript.

``audio -> words with times -> who spoke when -> turns``. The speech
recogniser and the diarizer are ports, so the models behind them can
change (or run elsewhere) without touching this module. The result is
the same :class:`Transcript` the rest of the analyzer reads, with the
timed words kept on each turn so that sentences get exact times.
"""

from pathlib import PurePath
from typing import Mapping, Optional

from src.application.skills.ports import (
    AudioReader,
    Diarizer,
    SpeechRecogniser,
)
from src.domain.errors import InvalidInputError
from src.domain.skills.speech import (
    assign_speakers,
    rename_speakers,
    segments_from_words,
    smooth_speakers,
    speech_regions,
)
from src.domain.skills.transcript import Transcript, build_transcript


class TranscriptionService:
    """Transcribe a recording and attribute every word to a speaker.

    Args:
        reader: Decodes the audio file.
        recogniser: Speech-to-text with word timestamps.
        diarizer: "Who spoke when"; ``None`` attributes everything to
            one unknown speaker (a monologue, or diarization done
            elsewhere).
        turn_gap: Silence in seconds that splits a speaker's words
            into two turns.
        smooth: Longest run of words (see
            :func:`~src.domain.skills.speech.smooth_speakers`) that is
            given back to the surrounding speaker; 0 disables it.
    """

    def __init__(
        self,
        reader: AudioReader,
        recogniser: SpeechRecogniser,
        diarizer: Optional[Diarizer] = None,
        turn_gap: float = 1.0,
        smooth: int = 1,
    ):
        """Wire the ports."""
        if turn_gap < 0 or smooth < 0:
            raise InvalidInputError(
                "turn_gap and smooth must not be negative."
            )
        self.reader = reader
        self.recogniser = recogniser
        self.diarizer = diarizer
        self.turn_gap = turn_gap
        self.smooth = smooth

    def transcribe(
        self,
        path: str,
        session_id: Optional[str] = None,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        names: Optional[Mapping[str, str]] = None,
    ) -> Transcript:
        """Return the diarized transcript of the recording at ``path``.

        Args:
            path: Audio file.
            session_id: Identifier of the session (default: file name).
            language: Spoken language code; ``None`` lets the
                recogniser detect it.
            num_speakers: Number of people in the recording, when
                known: it makes diarization far more reliable.
            names: Participant names by diarizer label
                (``{"SPEAKER_01": "Ana"}``).

        Raises:
            InvalidInputError: no speech was recognised.
        """
        if num_speakers is not None and num_speakers < 1:
            raise InvalidInputError("num_speakers must be at least 1.")
        audio = self.reader.read(path)
        speech = self.recogniser.recognise(audio, language)
        words = [w for w in speech.words if w.text.strip()]
        if not words:
            raise InvalidInputError(f"No speech recognised in {path}.")
        spans = []
        if self.diarizer is not None and num_speakers != 1:
            spans = self.diarizer.diarize(
                audio, speech_regions(words), num_speakers
            )
        words = assign_speakers(words, spans)
        if self.smooth:
            words = smooth_speakers(words, self.smooth)
        transcript = build_transcript(
            session_id or PurePath(path).stem,
            segments_from_words(words, self.turn_gap),
            language=speech.language or language,
            merge_gap=self.turn_gap,
        )
        return rename_speakers(transcript, names) if names else transcript
