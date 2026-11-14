"""Enrolment use cases: keep a person's voiceprint, or forget it.

A person reads the enrolment text (``docs/transcriber/ARCHITECTURE.md``
§6); the sample is decoded, checked by the domain rules and reduced to
one voiceprint. Only the voiceprint and the consent record are stored:
the sample audio is never handed to a repository.
"""

import re
from typing import Optional

from transcriber.application.common import check_speaker_id
from transcriber.application.ports import (
    AudioDecoder,
    Clock,
    RecordedAudio,
    SpeechRecogniser,
    VoiceEncoder,
    VoiceprintRepository,
)
from transcriber.domain.enrolment import (
    SampleRules,
    build_voiceprint,
    enrolment_text,
)
from transcriber.domain.errors import (
    ConsentRequiredError,
    InvalidInputError,
    NotFoundError,
)
from transcriber.domain.models import (
    ANONYMOUS_LABEL_PATTERN,
    ConsentRecord,
    Voiceprint,
)
from transcriber.domain.parameters import TranscriptionParameters
from transcriber.domain.speakers import analysis_windows, speech_regions


class EnrolSpeaker:
    """Turn a person's reading of the enrolment text into a voiceprint.

    The sample is cut into the same long windows a transcription uses
    to find voices (``parameters.window``), so a voiceprint and a
    session's voice groups are averages over windows of one length.
    The recogniser's word times are the speech regions, as in a
    transcription.
    """

    def __init__(
        self,
        decoder: AudioDecoder,
        recogniser: SpeechRecogniser,
        encoder: VoiceEncoder,
        voiceprints: VoiceprintRepository,
        clock: Clock,
        parameters: TranscriptionParameters = TranscriptionParameters(),
        rules: SampleRules = SampleRules(),
    ) -> None:
        """Wire the ports and take the thresholds from the settings."""
        self._decoder = decoder
        self._recogniser = recogniser
        self._encoder = encoder
        self._voiceprints = voiceprints
        self._clock = clock
        self._parameters = parameters
        self._rules = rules

    def __call__(
        self,
        speaker_id: str,
        name: str,
        language: str,
        recording: RecordedAudio,
        consent: bool,
        consent_version: str,
        text_version: Optional[str] = None,
    ) -> Voiceprint:
        """Enrol ``speaker_id`` from ``recording``; return the voiceprint.

        ``consent`` must be ``True`` (the person agreed to the consent
        statement of version ``consent_version``); anything else raises
        :class:`ConsentRequiredError` before the audio is touched.
        ``text_version`` is the enrolment text read (the latest of
        ``language`` if omitted). Enrolling an enrolled person again
        replaces the voiceprint and the consent record. Ids shaped like
        anonymous labels (``S1``, ``S2``...) are reserved and refused.
        Raises :class:`~transcriber.domain.errors.SampleRejectedError`
        when the sample is unfit.
        """
        if consent is not True:
            raise ConsentRequiredError(
                "A voice is only enrolled with the person's explicit"
                " consent."
            )
        if not consent_version.strip():
            raise InvalidInputError("The consent version is required.")
        check_speaker_id(speaker_id)
        if re.fullmatch(ANONYMOUS_LABEL_PATTERN, speaker_id):
            raise InvalidInputError(
                f"Speaker id {speaker_id} is reserved for anonymous"
                " voices; choose another id."
            )
        text = enrolment_text(language, text_version)
        text.text(name)  # the text must be readable with this name
        now = self._clock.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise InvalidInputError("The clock must give an aware time.")
        audio = self._decoder.decode(recording)
        speech = self._recogniser.recognise(audio, language)
        p = self._parameters
        regions = speech_regions(speech.words, p.region_gap)
        windows = analysis_windows(regions, *p.coarse())
        vectors = self._encoder.embed(audio, windows)
        voiceprint = build_voiceprint(
            speaker_id, self._encoder.model_name, consent_version, now,
            vectors, windows, regions, audio.duration, self._rules,
        )
        self._voiceprints.save(voiceprint, ConsentRecord(
            speaker_id=speaker_id, name=" ".join(name.split()),
            consent_version=consent_version, text_version=text.version,
            given_at=now,
        ))
        return voiceprint


class ForgetSpeaker:
    """Delete a person's voiceprint and consent record.

    Transcripts already made keep the person's id: they are text, and
    deleting them is a separate decision (``DeleteSession``).
    """

    def __init__(self, voiceprints: VoiceprintRepository) -> None:
        """Wire the port."""
        self._voiceprints = voiceprints

    def __call__(self, speaker_id: str) -> None:
        """Forget ``speaker_id``; :class:`NotFoundError` if not enrolled.

        An invalid id is an :class:`InvalidInputError`.
        """
        if not self._voiceprints.delete(check_speaker_id(speaker_id)):
            raise NotFoundError(f"No enrolled speaker {speaker_id}.")
