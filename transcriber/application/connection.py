"""Use case: which services a live session depends on are available.

The connection check of the live page (#247, ``docs/live/CONNECTION.md``)
asks the transcriber, behind the access code, whether the services it
cannot reach from the browser answer: the live transcription (this
service), the live assistant (the orchestrator) and the knowledge
search (the RAG). Each one is a :class:`ServiceProbe`; they are asked at
the same time, so the answer takes as long as the slowest probe (each
adapter keeps its own short timeout). Only one word per service leaves
this module: no address, version or error text.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Mapping, Optional

from transcriber.application.ports import ServiceProbe

SERVICES = ("transcription", "assistant", "knowledge")
"""The services the live page shows, in its order."""
AVAILABLE = "available"
UNAVAILABLE = "unavailable"
NOT_SET_UP = "not_set_up"
"""The service is not part of this installation (no probe given)."""


class FixedProbe:
    """A probe whose answer is known when the service starts.

    The live transcription runs in this process: when the page can ask
    this question the transcriber answers, so its row says whether live
    sessions were set up.
    """

    def __init__(self, answer: bool) -> None:
        """Keep the answer."""
        self.answer = answer

    def available(self) -> bool:
        """Return the answer given at start."""
        return self.answer


class CheckServices:
    """Ask every service probe; return one word per service.

    Args:
        probes: A probe per name of :data:`SERVICES`; a missing or
            ``None`` one gives :data:`NOT_SET_UP`.
    """

    def __init__(self, probes: Mapping[str, Optional[ServiceProbe]]) -> None:
        """Keep the probes; nothing is asked yet."""
        self.probes = dict(probes)

    def __call__(self) -> Dict[str, str]:
        """Return ``{service: available|unavailable|not_set_up}``."""
        asked = {
            name: probe for name in SERVICES
            if (probe := self.probes.get(name)) is not None
        }
        answers: Dict[str, bool] = {}
        if asked:
            with ThreadPoolExecutor(max_workers=len(asked)) as pool:
                futures = {
                    name: pool.submit(safe, probe)
                    for name, probe in asked.items()
                }
                answers = {name: f.result() for name, f in futures.items()}
        return {
            name: (
                NOT_SET_UP if name not in answers
                else AVAILABLE if answers[name] else UNAVAILABLE
            )
            for name in SERVICES
        }


def safe(probe: ServiceProbe) -> bool:
    """Return the probe's answer; an exception counts as unavailable."""
    try:
        return bool(probe.available())
    except Exception:  # noqa: BLE001 - a broken probe is a down service
        return False
