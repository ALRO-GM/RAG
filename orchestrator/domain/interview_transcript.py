"""The interview's ``transcript.v1`` and the job text for the analysis.

The transcriber hears only the candidate's microphone (the bot speaks
into a headset), so the orchestrator writes the interview's transcript
itself: the candidate's finals (stream times of the transcriber) and
the bot's spoken turns (times of the orchestrator's clock from the
session start, which starts within a fraction of a second of the
transcriber's stream). :func:`analysis_job` is the "job sought" text
handed to the post-call analysis: the offer's title and summary and the
filled slot sheet with quotes. Plain data: no I/O.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.domain.interview import InterviewPlan, Sheet

BOT_ID, CANDIDATE_ID = "BOT", "S1"
MAX_JOB = 4000
"""The analysis's longest job text (``MAX_JOB`` of the RAG)."""
LEVEL_NAMES = {"minimo": "mínimo", "deseable": "deseable",
               "opcional": "opcional"}
STATUS_NAMES = {"filled": "cubierto", "partial": "parcial",
                "empty": "sin respuesta", "skipped": "saltado",
                "unknown": "no lo conoce"}


@dataclass(frozen=True)
class Said:
    """One utterance of the interview, by the bot or the candidate."""

    speaker: str
    text: str
    start: float
    end: float
    ref: str = ""


def interview_transcript(
    session_id: str,
    said: List[Said],
    language: str,
    title: str,
    candidate: Optional[str],
    started_at: Optional[datetime],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Return the interview as ``transcript.v1`` and its utterance ids.

    The second value maps each :attr:`Said.ref` (the live utterance id)
    to the transcript's utterance id.
    """
    refs: Dict[str, str] = {}
    ordered = sorted(
        (s for s in said if s.text.strip()), key=lambda s: (s.start, s.end)
    )
    utterances: List[Dict[str, Any]] = []
    turn = -1
    index = 0
    previous: Optional[Dict[str, Any]] = None
    for item in ordered:
        if previous is None or previous["speaker"] != item.speaker:
            turn += 1
            index = 0
        else:
            index += 1
        start = max(0.0, round(item.start, 3))
        utterance: Dict[str, Any] = {
            "id": f"t{turn}.s{index}", "turn": turn, "index": index,
            "speaker": item.speaker, "start": start,
            "end": max(start, round(item.end, 3)),
            "text": " ".join(item.text.split()),
            "previous_speaker": previous["speaker"] if previous else None,
            "previous_utterance": previous["id"] if previous else None,
        }
        utterances.append(utterance)
        if item.ref:
            refs[item.ref] = utterance["id"]
        previous = utterance
    duration = max((u["end"] for u in utterances), default=0.0)
    return {
        "schema": "transcript.v1",
        "session": {
            "id": session_id, "title": f"Entrevista IA: {title}"[:120],
            "language": language, "duration": duration,
            "started_at": started_at.isoformat() if started_at else None,
            "type": "entrevista",
            "goal": f"Entrevista para el puesto «{title}»"[:500],
        },
        "speakers": [
            {"id": BOT_ID, "name": "Entrevistador IA", "enrolled": False},
            {"id": CANDIDATE_ID, "name": candidate or None,
             "enrolled": False},
        ],
        "utterances": utterances,
        "pipeline": {
            "recogniser": "faster-whisper (live)", "voice_model": "none",
            "transcriber": "orchestrator interview (voicebot.interview)",
            "diarizer": "channels",
        },
    }, refs


def analysis_job(plan: InterviewPlan, sheet: Sheet) -> str:
    """Return the job text handed to the analysis (at most 4000 chars)."""
    lines = [f"Puesto: {plan.title}"]
    if plan.summary:
        lines.append(plan.summary)
    lines.append("Cobertura de la entrevista con IA (borrador):")
    for slot in sheet.slots:
        head = (f"- [{LEVEL_NAMES[slot.level]}, peso {slot.weight}] "
                f"{slot.label}: {STATUS_NAMES[slot.status]}")
        if slot.score is not None:
            head += f", {slot.score}/100"
        lines.append(head)
        for answer in slot.answers[:2]:
            lines.append(f"  «{answer.quote}» ({answer.utterance})")
    text = "\n".join(lines)
    return text if len(text) <= MAX_JOB else text[:MAX_JOB - 1] + "…"
