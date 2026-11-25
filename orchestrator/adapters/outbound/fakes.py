"""In-process fakes of the knowledge and reasoner ports.

They let a replay run end to end with no service: deterministic, no
network, no model. The fake reasoner answers with a follow-up question
about the focus utterance, citing it in ``about`` (and the first
retrieved chunk in ``cites``), and can sleep to imitate a service's
latency. It also plays the AI interviewer's templates (#208): the
planner drafts a plan from the offer's bullet lines, the slot filler
fills the slot asked when the answer has twelve words or more, the
report scores the slots by status (``@2`` also judges every answer
``correct_basic``), and ``interviewer@1`` words each
step with fixed phrases. Their ``@2`` versions (#258) add the concepts
of each slot: the planner gives generic ones, the filler marks those
the rules of :func:`~orchestrator.domain.interview_points.point_hits`
find, and the interviewer names the pending ones in its follow-up.
"""

import json
import math
import time
import wave
from array import array
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional

from orchestrator.application.ports import (
    CancelToken,
    Document,
)
from orchestrator.domain.interview import Point
from orchestrator.domain.interview_points import join_words, point_hits
from orchestrator.domain.speech import SpeechChunk, split_sentences


class FakeKnowledge:
    """A knowledge service with a fixed list of chunks.

    Args:
        chunks: ``(id, text)`` pairs; a chunk is returned when it shares
            a word of four letters or more with the query (all of them
            when ``everything`` is set).
        everything: Return every chunk whatever the query.
        delay: Seconds to wait per query.
        sleep: Waits (replaced in tests).
    """

    def __init__(
        self,
        chunks: Optional[List[Dict[str, str]]] = None,
        everything: bool = False,
        delay: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Keep the chunks."""
        self.chunks = chunks or []
        self.everything = everything
        self.delay = delay
        self._sleep = sleep
        self.queries: List[Document] = []

    def query(self, query: Document) -> Document:
        """Return the chunks that match ``query['text']``."""
        self.queries.append(query)
        if self.delay:
            self._sleep(self.delay)
        words = {w.lower() for w in query["text"].split() if len(w) >= 4}
        found = [
            c for c in self.chunks
            if self.everything
            or words & {w.lower() for w in c["text"].split()}
        ]
        top_k = int(query.get("top_k", 5))
        return {
            "schema": "knowledge.result.v1",
            "request_id": query["request_id"], "domain": query["domain"],
            "index_version": "fake",
            "chunks": [
                {
                    "id": c["id"], "text": c["text"], "score": 1.0,
                    "source": {"file_path": "fake.md",
                               "first_character_index": 0,
                               "last_character_index": len(c["text"])},
                }
                for c in found[:top_k]
            ],
        }


class FakeReasoner:
    """A reasoner that echoes a follow-up question about the focus.

    As the voice bot's ``conversational`` reasoner it reads the first
    retrieved chunk aloud (or says it does not know), streamed in
    pieces by :meth:`reason_stream`.

    Args:
        delay: Seconds to wait per call.
        status: The ``status`` to answer with (``ok`` by default).
        sleep: Waits (replaced in tests).
        piece: Characters per piece of a streamed reply.
    """

    def __init__(
        self,
        delay: float = 0.0,
        status: str = "ok",
        sleep: Callable[[float], None] = time.sleep,
        piece: int = 7,
    ):
        """Keep the options."""
        self.delay = delay
        self.status = status
        self.piece = max(1, piece)
        self._sleep = sleep
        self.requests: List[Document] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Answer ``request`` without a model."""
        self.requests.append(request)
        if self.delay:
            self._sleep(self.delay)
        if request["role"] == "structured":
            return self._structured(request)
        if request["role"] == "conversational":
            if str(request.get("template") or "").startswith("interviewer@"):
                return self._answer(request, _interviewer_text(request))
            return self._reply(request)
        window = request["context"]["window"]
        focus = request["context"].get("focus")
        said = next((w["text"] for w in window if w["id"] == focus), "")
        suggestions: List[Dict[str, Any]] = []
        if self.status == "ok":
            cites = [r["id"] for r in request.get("retrieved", [])[:1]]
            suggestions = [{
                "kind": "follow_up",
                "text": "¿Puedes contar más sobre: " + " ".join(
                    said.split()[:8]) + "?",
                "cites": cites, "about": [focus] if focus else [],
            }]
        return {
            "schema": "reasoner.response.v1",
            "request_id": request["request_id"], "role": request["role"],
            "status": self.status, "suggestions": suggestions,
            "model": "fake", "backend": "local",
            "latency": {"total_ms": self.delay * 1000.0},
        }

    def reason_stream(
        self, request: Document, cancel: CancelToken,
        on_text: Callable[[str], None],
    ) -> Document:
        """Answer like :meth:`reason`, writing the reply in pieces."""
        response = self.reason(request, cancel)
        reply = response.get("reply")
        if reply:
            raw = json.dumps({"text": reply["text"],
                              "cites": reply["cites"]}, ensure_ascii=False)
            for at in range(0, len(raw), self.piece):
                if cancel.cancelled:
                    break
                on_text(raw[at:at + self.piece])
        return response

    def _answer(
        self, request: Document, text: Optional[str],
        data: Optional[Document] = None,
    ) -> Document:
        """A response with ``text`` as the reply, or ``data``."""
        ok = self.status == "ok"
        return {
            "schema": "reasoner.response.v1",
            "request_id": request["request_id"], "role": request["role"],
            "status": self.status,
            "reply": {"text": text, "cites": []} if ok and text else None,
            "data": data if ok else None,
            "model": "fake", "backend": "local",
            "latency": {"total_ms": self.delay * 1000.0},
        }

    def _structured(self, request: Document) -> Document:
        """The interview's planner, slot filler and report, without AI."""
        payload = request.get("payload") or {}
        template = request.get("template")
        if template in ("interview.planner@1", "interview.planner@2"):
            data = _fake_plan(str(payload.get("offer") or ""),
                              points=template.endswith("@2"))
        elif template in ("interview.slots@1", "interview.slots@2"):
            data = _fake_fill(payload, points=template.endswith("@2"))
        elif template == "interview.report@2":
            data = _fake_judged(payload)
        elif template == "interview.attribution@1":
            data = _fake_attribution(payload)
        elif template == "interview.ticks@1":
            data = {"points": []}
        else:
            score = {"filled": 80, "partial": 45, "unknown": 0,
                     "skipped": 10}
            data = {
                "slots": [
                    {"id": s.get("id"), "score": score.get(
                        str(s.get("status")), 0),
                     "above": False, "confidence": 0.5,
                     "notes": "Puntuación de prueba (sin modelo)."}
                    for s in payload.get("slots") or []
                    if isinstance(s, dict) and s.get("status") != "empty"
                ],
                "other_skills": [],
                "summary": "Resumen de prueba generado sin modelo.",
            }
        return self._answer(request, None, data)

    def _reply(self, request: Document) -> Document:
        """A voice bot's reply: the first retrieved chunk, read aloud."""
        retrieved = request.get("retrieved") or []
        if self.status != "ok":
            reply = None
        elif retrieved:
            first = " ".join(str(retrieved[0]["text"]).split()[:30])
            reply = {"text": f"Te cuento lo que sé. {first}",
                     "cites": [retrieved[0]["id"]]}
        else:
            reply = {"text": "No tengo ese dato ahora mismo. "
                             "¿Quieres que lo consulte?", "cites": []}
        return {
            "schema": "reasoner.response.v1",
            "request_id": request["request_id"], "role": request["role"],
            "status": self.status, "reply": reply,
            "model": "fake", "backend": "local",
            "latency": {"total_ms": self.delay * 1000.0},
        }


def tone_wav(seconds: float, rate: int = 16000, pitch: float = 220.0) -> bytes:
    """Return a quiet tone as a 16-bit mono WAV file (tiny test audio)."""
    count = max(1, int(seconds * rate))
    samples = array("h", (
        int(3000 * math.sin(2 * math.pi * pitch * i / rate))
        for i in range(count)
    ))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


class FakeTts:
    """A synthesiser that answers each sentence with a short tone.

    Args:
        seconds_per_word: Length of the tone per word of the sentence.
        delay: Seconds to wait per sentence (imitates Piper's time).
        rate: Sample rate of the tones.
        sleep: Waits (replaced in tests).
    """

    def __init__(
        self,
        seconds_per_word: float = 0.05,
        delay: float = 0.0,
        rate: int = 16000,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Keep the options."""
        self.seconds_per_word = seconds_per_word
        self.delay = delay
        self.rate = rate
        self._sleep = sleep
        self.requests: List[Document] = []

    def synthesize(
        self, request: Document, cancel: CancelToken
    ) -> Iterator[SpeechChunk]:
        """Yield one tone per sentence until ``cancel`` is set."""
        self.requests.append(request)
        for seq, sentence in enumerate(split_sentences(request["text"])):
            if cancel.cancelled:
                return
            if self.delay:
                self._sleep(self.delay)
            wav = tone_wav(self.seconds_per_word * len(sentence.split()),
                           self.rate)
            audio = wav if request.get("format") == "wav" else wav[44:]
            yield SpeechChunk(
                request_id=request["request_id"], seq=seq, text=sentence,
                audio=audio, format=request.get("format") or "pcm_s16le",
                sample_rate=self.rate, synth_ms=self.delay * 1000.0,
            )


def _fake_fill(payload: Document, points: bool) -> Document:
    """The fake slot filler: the slot asked, by the answer's length.

    With ``points`` (``@2``) it also says which concepts of the slot
    asked the rules find in the answer.
    """
    answer = str(payload.get("answer") or "")
    target = str(payload.get("target") or "")
    words = len(answer.split())
    if not target or words < 4:
        return {"slots": []}
    item: Document = {
        "id": target, "status": "complete" if words >= 12 else "partial",
        "quote": " ".join(answer.split()[:20]),
    }
    if points:
        slot = next((s for s in payload.get("slots") or []
                     if isinstance(s, dict) and s.get("id") == target), {})
        concepts = [Point.model_validate(p) for p in slot.get("points") or []]
        item["points"] = [
            {"id": h.point, "status": h.status, "quote": h.quote}
            for h in point_hits(answer, concepts)
        ]
    return {"slots": [item]}


def _words(text: str) -> "set[str]":
    """Lower-case words of five letters or more (the fake's topic words)."""
    return {w.strip("¿?¡!.,;:") for w in text.lower().split()
            if len(w.strip("¿?¡!.,;:")) >= 5}


def _fake_attribution(payload: Document) -> Document:
    """The fake attribution (#268): each answer to the point last asked.

    An interviewer line asks the point sharing most words (five letters
    or more) with its label and questions; each candidate line after it
    answers that point, and the rules say which concepts it treats.
    """
    slots = [s for s in payload.get("slots") or [] if isinstance(s, dict)]
    topic = {str(s.get("id")): _words(" ".join(
        [str(s.get("label")), *map(str, s.get("questions") or [])]))
        for s in slots}
    found: Dict[str, Document] = {}
    asked: Optional[Document] = None
    for line in payload.get("round") or []:
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "")
        if line.get("role") == "interviewer":
            heard = _words(text)
            best = max(slots, default=None,
                       key=lambda s: len(heard & topic[str(s.get("id"))]))
            if best is not None and heard & topic[str(best.get("id"))]:
                asked = best
        elif line.get("role") == "candidate" and asked is not None:
            item = found.setdefault(str(asked.get("id")), {
                "id": asked.get("id"), "answers": [], "status": "partial",
                "points": []})
            item["answers"].append(line.get("id"))
            concepts = [Point.model_validate(p)
                        for p in asked.get("points") or []]
            item["points"] += [
                {"id": h.point, "status": h.status,
                 "utterance": line.get("id"), "quote": h.quote}
                for h in point_hits(text, concepts)]
    return {"slots": list(found.values())}


def _fake_judged(payload: Document) -> Document:
    """The fake report ``@2``: scores by status, answers ``correct_basic``.

    Each answered slot and treated concept is judged on its first
    answer; nothing is listed to check.
    """
    score = {"filled": 80, "partial": 45, "unknown": 0, "skipped": 10}
    slots: List[Document] = []
    for slot in payload.get("slots") or []:
        if not isinstance(slot, dict) or slot.get("status") == "empty":
            continue
        answers = slot.get("answers") or []
        first = answers[0] if answers else {}
        said: Document = {
            "label": "correct_basic", "kind": "fact",
            "utterance": first.get("utterance"),
            "quote": " ".join(str(first.get("text") or "").split()[:12]),
            "reason": "Juicio de prueba (sin modelo).", "cites": [],
        }
        slots.append({
            "id": slot.get("id"),
            "score": score.get(str(slot.get("status")), 0),
            "above": False, "confidence": 0.5,
            "notes": "Puntuación de prueba (sin modelo).",
            "judgement": said if answers else None,
            "points": [{"id": p.get("id"), **said}
                       for p in slot.get("points") or []
                       if isinstance(p, dict) and p.get("status") == "treated"
                       and answers],
        })
    return {"slots": slots, "to_check": [], "other_skills": [],
            "summary": "Resumen de prueba generado sin modelo."}


_FAKE_POINTS = ["Dónde y en qué puesto", "Cuánto tiempo", "Tareas que hacía"]
_FAKE_STAR = ["Situación concreta", "Lo que hizo él o ella", "Resultado"]


def _fake_plan(offer: str, points: bool = False) -> Document:
    """A plan draft from the offer's bullet lines (fake planner).

    With ``points`` (``@2``) every slot gets generic concepts.
    """
    bullets = [
        line.strip(" -*•\t") for line in offer.splitlines()
        if line.strip().startswith(("-", "*", "•"))
        and len(line.split()) >= 3
    ][:4]
    slots: List[Document] = [
        {"label": text[:80], "kind": "requirement",
         "level": "minimo" if i < 2 else "deseable", "phase": "experience",
         "questions": [f"Háblame de tu experiencia con esto: {text[:80]}"],
         "evidence": "Dónde, cuánto tiempo y qué hacía.", "weight": 4}
        for i, text in enumerate(bullets)
    ]
    slots += [
        {"label": "Comunicación clara", "kind": "competency",
         "competency": "COMM", "level": "minimo", "phase": "behavioural",
         "questions": ["Cuéntame una vez en la que tuviste que explicar "
                       "algo difícil. ¿Qué hiciste tú?"],
         "evidence": "Situación, acción propia y resultado.", "weight": 3},
        {"label": "Resolución de problemas", "kind": "competency",
         "competency": "SOLVE", "level": "deseable",
         "phase": "behavioural",
         "questions": ["Cuéntame un problema que resolviste en el "
                       "trabajo. ¿Cómo acabó?"],
         "evidence": "Problema, pasos y resultado.", "weight": 2},
    ]
    if points:
        for slot in slots:
            slot["points"] = list(_FAKE_STAR if slot["kind"] == "competency"
                                  else _FAKE_POINTS)
    first = next((line.strip() for line in offer.splitlines()
                  if line.strip()), "Puesto")
    return {"title": first[:80], "role_type": "otro",
            "summary": " ".join(offer.split())[:200], "slots": slots}


def _interviewer_text(request: Document) -> str:
    """The interviewer's fixed phrase for a step (fake interviewer)."""
    payload = request.get("payload") or {}
    action = str(payload.get("action") or "ask")
    questions = payload.get("questions") or []
    question = str(questions[0]) if questions else "¿Algo más?"
    if action == "open":
        return (f"Hola, soy el entrevistador de inteligencia artificial. "
                f"La entrevista dura unos {payload.get('minutes', 20)} "
                f"minutos. {question}")
    asked = [str(p) for p in payload.get("pending_points") or []]
    said = [str(p) for p in payload.get("treated_points") or []][-2:]
    if action == "probe" and asked:
        head = f"Me has hablado de {join_words(said)}. " if said else ""
        return f"{head}¿Y de {join_words(asked)}?"
    if action == "clarify" and asked:
        return (f"No me ha quedado claro lo de {join_words(asked)}; "
                "¿quieres contarme algo o pasamos a la siguiente pregunta?")
    if action == "probe":
        missing = payload.get("missing", "un ejemplo")
        return f"¿Me puedes concretar un poco más? Me falta {missing}."
    if action == "clarify":
        return ("No me ha quedado del todo claro esto; ¿quieres matizarlo "
                "o pasamos a la siguiente pregunta?")
    if action == "answer":
        return f"Eso te lo concretará el equipo. {question}"
    if action == "reask":
        return f"Sin prisa. Te lo pregunto de otra forma: {question}"
    if action == "hint":
        return ("Puedes contarlo con un ejemplo concreto. ¿Te referías a "
                "tu último trabajo, o prefieres hablar de otro?")
    if action == "close":
        return ("Muchas gracias por tu tiempo. El equipo se pondrá en "
                "contacto contigo con los siguientes pasos. Hasta pronto.")
    return f"Gracias. {question}"
