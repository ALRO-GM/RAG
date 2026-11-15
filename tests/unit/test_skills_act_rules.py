"""Approach A of the dialogue-act tagger: the rule keys (#82)."""

from typing import Optional

import pytest

from src.domain.skills.act_rules import normalise, rule_key, rule_keys
from src.domain.skills.act_tagging import ActInput
from src.domain.skills.dialogue_acts import ContextLine


def item(text: str, before: Optional[str] = None,
         same: bool = False) -> ActInput:
    """Return a sentence by Ana, after ``before`` (by Luis or Ana)."""
    context = [] if before is None else [
        ContextLine(speaker="Ana" if same else "Luis", text=before)]
    return ActInput(id="1", speaker="Ana", text=text, context=context)


@pytest.mark.parametrize("text, before, key", [
    ("", None, "otro:empty"),
    ("Pues el tema es que...", None, "otro:cut"),
    ("Eh, em...", None, "otro:cut"),
    ("Eh, em", None, "otro:filler"),
    ("Sí.", "¿Lo tienes?", "respuesta:yes_no_after_question"),
    ("Ajá.", "Te cuento lo de ayer.", "escucha:backchannel_after_other"),
    ("Mm hmm", None, "escucha:backchannel"),
    ("Muchas gracias a todos.", None, "relacion:formula"),
    ("¿Puedes mandarme el informe?", None, "peticion:can_you_question"),
    ("¿Y si lo dejamos para el lunes?", None, "propuesta:question"),
    ("¿Cuánto cuesta?", None, "pregunta:question_mark"),
    ("¿Y el plazo?", "¿Cuánto cuesta?", "pregunta:question_mark_after_question"),
    ("what time is it", None, "pregunta:interrogative_word"),
    ("Pasamos al siguiente punto del orden del día.", None, "gestion:cue"),
    ("Me encargo yo de llamarles.", None, "compromiso:cue"),
    ("De acuerdo, lo vemos así.", "Hagámoslo el lunes.", "aceptacion:opener"),
    ("No estoy de acuerdo con eso.", "Subimos el precio.",
     "rechazo:opener"),
    ("Podríamos probar con otro proveedor.", None, "propuesta:cue"),
    ("Revisa los números antes del viernes.", None, "peticion:cue"),
    ("Creo que el diseño es demasiado caro.", None, "opinion:cue"),
    ("El pedido llegó el martes.", "¿Cuándo llegó?", "default:after_question"),
    ("El pedido llegó el martes por la tarde.", None, "default:statement"),
    ("El martes.", None, "default:short"),
])
def test_rule_keys(text: str, before: Optional[str], key: str) -> None:
    assert rule_key(item(text, before)) == key


def test_openers_need_somebody_else_before() -> None:
    assert rule_key(item("De acuerdo, lo vemos así.", "Hola", same=True)
                    ) != "aceptacion:opener"
    assert rule_key(item("Sí.", "¿Lo tienes?", same=True)) == (
        "escucha:backchannel")


def test_normalise_and_many_keys() -> None:
    assert normalise("  ¿Qué   TAL? ") == "¿que tal?"
    assert rule_keys([item("Hola"), item("¿Sí?")]) == (
        "relacion:formula", "pregunta:question_mark")
