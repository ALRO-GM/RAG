"""The local model for names (#102) on a few sentences, real weights.

Loads the model the way the service does (Qwen3-8B at 4 bits when it
fits the free GPU memory, else Qwen3-4B, waiting a minute at most;
on a machine without CUDA the 4B in float32) from the local cache
(``HF_HUB_OFFLINE=1``) and runs the whole search on a short Spanish
conversation. It checks the wiring and the JSON the model gives back,
not the accuracy (``experiments/transcriber/speaker_names_eval.py``
measures that).
"""

from pathlib import Path

import pytest

from transcriber.adapters.outbound.local_name_model import LocalNameModel
from transcriber.domain.errors import ModelLoadError
from transcriber.domain.models import (
    Pipeline,
    Session,
    SessionState,
    Speaker,
    Transcript,
    Utterance,
)
from transcriber.domain.sentences import regroup_utterances
from transcriber.domain.speaker_names import (
    SYSTEM_PROMPT,
    decide,
    find_candidates,
    model_prompt,
    parse_reply,
)

pytestmark = pytest.mark.slow

CACHE = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B"


@pytest.mark.skipif(not CACHE.exists(), reason="Qwen3-4B is not cached")
def test_a_self_introduction_and_a_look_alike() -> None:
    lines = [
        ("S1", "Buenos días, soy Marta y coordino el proyecto."),
        ("S2", "Yo soy de Madrid, por cierto."),
        ("S2", "Gracias, Marta."),
    ]
    rough = [
        Utterance(id="t0.s0", turn=0, index=0, speaker=s, start=3.0 * n,
                  end=3.0 * n + 2.0, text=t, previous_speaker=None,
                  previous_utterance=None)
        for n, (s, t) in enumerate(lines)
    ]
    transcript = Transcript(
        session=Session(id="s1", language="es", duration=10.0,
                        state=SessionState.DONE),
        speakers=[Speaker(id="S1", enrolled=False),
                  Speaker(id="S2", enrolled=False)],
        utterances=regroup_utterances(rough),
        pipeline=Pipeline(recogniser="r", voice_model="v", transcriber="t"),
    )
    model = LocalNameModel(wait_seconds=60.0)
    try:
        evidence = []
        for candidate in find_candidates(transcript):
            reply = model.complete(SYSTEM_PROMPT, model_prompt(candidate), 96)
            item = parse_reply(transcript, candidate, reply)
            if item is not None:
                evidence.append(item)
    except ModelLoadError as exc:
        pytest.skip(f"the model cannot run now: {exc}")
    finally:
        model.release()
    decision = decide(transcript.speakers, evidence)
    assert [(a.speaker, a.name) for a in decision.auto] == [("S1", "Marta")]
