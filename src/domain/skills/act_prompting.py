"""Prompt and reply format of the dialogue-act classifier.

One sentence, the sentences said just before it (with their speakers)
and the 13 categories of ``docs/skills/DIALOGUE_ACTS_GUIDE.md`` §2,
with the order of questions of §3; the answer is one category id. It
is how the judges are compared on the Spanish set of
``examples/skills/dialogue_acts`` (``DESIGN.md`` §4), and the first
layer of the analyzer once it is trusted.
"""

import json
import re
from typing import Optional, Sequence, Tuple

ACT_DEFINITIONS: Tuple[Tuple[str, str], ...] = (
    ("pregunta", "pide información, una opinión, una confirmación o una "
     "elección"),
    ("respuesta", "contesta a una pregunta anterior"),
    ("informacion", "cuenta un hecho, un dato o una explicación que se "
     "podría comprobar"),
    ("opinion", "valora, juzga o toma postura"),
    ("propuesta", "plantea una idea o forma de actuar para que el grupo "
     "decida"),
    ("peticion", "pide a alguien (o al grupo) que haga algo"),
    ("aceptacion", "acepta o apoya lo que dijo otro"),
    ("rechazo", "rechaza, discrepa o corrige lo que dijo otro"),
    ("compromiso", "quien habla se obliga a hacer algo"),
    ("escucha", "«ajá», «ya, ya»: sigue la conversación sin aportar"),
    ("gestion", "habla de la propia reunión: orden, turnos, tiempo, "
     "resúmenes"),
    ("relacion", "saludos, gracias, disculpas, elogios o críticas a la "
     "persona, bromas"),
    ("otro", "frase cortada, ininteligible o sin función clara"),
)
"""The 13 categories with their one-line definition (guide §2)."""

ACT_IDS: Tuple[str, ...] = tuple(act for act, _ in ACT_DEFINITIONS)

ORDER = (
    "otro → escucha → gestion → relacion → respuesta → aceptacion o "
    "rechazo → compromiso → peticion → propuesta → pregunta → opinion → "
    "informacion"
)

ACT_SYSTEM_PROMPT = (
    "Clasificas frases de conversaciones de trabajo en español según lo "
    "que hacen en la conversación. Recibes UNA frase objetivo y las "
    "frases dichas justo antes, con quién las dijo.\n"
    "Categorías:\n{categories}\n"
    "Reglas:\n"
    "1. Clasifica solo la frase objetivo; el contexto sirve para saber a "
    "qué responde.\n"
    "2. Si dudas entre varias, hazte las preguntas en este orden y para "
    "en la primera que encaje: {order}.\n"
    "3. Clasifica lo que se dice, no a la persona.\n"
    "4. Responde con un único objeto JSON y nada más:\n"
    '{{"act": "<id de una categoría>"}}'
)


def act_system_prompt() -> str:
    """Return the classifier's system prompt."""
    lines = "\n".join(f"- {act}: {text}" for act, text in ACT_DEFINITIONS)
    return ACT_SYSTEM_PROMPT.format(categories=lines, order=ORDER)


def act_user_prompt(
    context: Sequence[Tuple[str, str]], speaker: str, text: str
) -> str:
    """Return the user turn: context lines, then the target sentence.

    ``context`` holds ``(speaker, text)`` pairs, oldest first.
    """
    before = "\n".join(f"{who}: {said}" for who, said in context)
    return (
        f"Contexto (no se clasifica):\n{before or '(inicio)'}\n\n"
        f"Frase objetivo, dicha por {speaker}:\n{text}\n\nJSON:"
    )


_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_act(reply: str) -> Optional[str]:
    """Return the category id in a reply, or ``None`` if there is none.

    The first JSON object in the reply is read; a bare category id is
    accepted too. Anything that is not one of the 13 ids is ``None``.
    """
    match = _OBJECT.search(reply)
    value: object = None
    if match:
        try:
            data = json.loads(match.group(0))
        except ValueError:
            data = None
        if isinstance(data, dict):
            value = data.get("act")
    else:
        value = reply.strip().strip('".').lower()
    if isinstance(value, str) and value.strip().lower() in ACT_IDS:
        return value.strip().lower()
    return None
