"""In-process fakes of the knowledge and reasoner ports.

They let a replay run end to end with no service: deterministic, no
network, no model. The fake reasoner answers with a follow-up question
about the focus utterance, citing it in ``about`` (and the first
retrieved chunk in ``cites``), and can sleep to imitate a service's
latency.
"""

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

    Args:
        delay: Seconds to wait per call.
        status: The ``status`` to answer with (``ok`` by default).
        sleep: Waits (replaced in tests).
    """

    def __init__(
        self,
        delay: float = 0.0,
        status: str = "ok",
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Keep the options."""
        self.delay = delay
        self.status = status
        self._sleep = sleep
        self.requests: List[Document] = []

    def reason(self, request: Document, cancel: CancelToken) -> Document:
        """Answer ``request`` without a model."""
        self.requests.append(request)
        if self.delay:
            self._sleep(self.delay)
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
