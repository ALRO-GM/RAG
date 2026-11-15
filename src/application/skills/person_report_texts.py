"""Fixed wording of the person report, in Spanish and English.

Every sentence a reader sees comes from here, from the rubric
catalogue or from the transcript; nothing is written by a model when
the report is rendered.
"""

from typing import Dict

TEXTS: Dict[str, Dict[str, str]] = {
    "es": {
        "title": "Informe de {participant}",
        "subtitle": "Conversación de grupo «{session}»",
        "synthetic": (
            "Datos de ejemplo inventados. Esta página no describe a una "
            "persona real: sirve para ver cómo es el informe."
        ),
        "summary": "Resumen",
        "meaning": "Qué significa y cómo leerlo",
        "charts": "El perfil de un vistazo",
        "strengths": "Puntos fuertes",
        "weaknesses": "Puntos a mejorar",
        "signals": "Conductas aisladas a vigilar",
        "notes": "Notas",
        "s_session": (
            "{participant} participó en una conversación de grupo"
            "{duration}."
        ),
        "s_duration": " de unos {minutes} minutos",
        "s_duration_short": " de menos de un minuto",
        "s_rated": (
            "Hay nivel en {rated} de {total} competencias{n1}; en las "
            "otras {missing} no hubo pruebas suficientes, y eso no "
            "significa un nivel bajo."
        ),
        "s_rated_all": "Hay nivel en las {total} competencias{n1}.",
        "s_rated_none": (
            "No hubo pruebas suficientes para valorar ninguna "
            "competencia{n1}. Eso no significa un nivel bajo: solo que "
            "esta conversación no las mostró."
        ),
        "s_strengths": "Sus puntos fuertes en esta conversación: {names}.",
        "s_no_strengths": (
            "No se vio ningún punto fuerte con pruebas suficientes."
        ),
        "s_weaknesses": "Puntos a mejorar: {names}.",
        "s_no_weaknesses": (
            "No se vio ningún punto a mejorar con pruebas suficientes."
        ),
        "s_signals": (
            "No se vio ningún punto a mejorar con pruebas suficientes, "
            "pero sí {count} conductas aisladas que conviene vigilar."
        ),
        "s_signal": (
            "No se vio ningún punto a mejorar con pruebas suficientes, "
            "pero sí una conducta aislada que conviene vigilar."
        ),
        "s_draft": (
            "Es un borrador para un evaluador humano{n2}: una sola "
            "conversación no basta para un perfil fiable."
        ),
        "and": "y",
        "observed": "Qué se observó",
        "observed_text": (
            "Una conversación{n3} («{session}»){duration_text}. "
            "{participant} intervino {turns} veces y dijo unas {words} "
            "palabras."
        ),
        "observed_duration": ", de unos {minutes} minutos",
        "observed_duration_short": ", de menos de un minuto",
        "level": "Qué es un nivel",
        "level_text": (
            "Cada competencia se valora de {minimum} a {maximum}{n4}. Solo "
            "se da un nivel si la conducta se vio al menos {needed} veces "
            "en momentos distintos de la conversación."
        ),
        "limits_title": "Límites de este informe",
        "limits_text": (
            "Borrador para un evaluador humano; una sola conversación no "
            "basta para un perfil fiable. Aquí se ha visto {seen} "
            "situación{plural}; para un perfil fiable hacen falta unas "
            "{recommended}."
        ),
        "limits_more": (
            "No se deduce ninguna emoción ni rasgo de personalidad a "
            "partir de la voz. El tiempo que alguien habla no se usa como "
            "medida de liderazgo."
        ),
        "legend": "Cómo leer las gráficas",
        "band_strong": "Punto fuerte",
        "band_middle": "Intermedio",
        "band_weak": "A mejorar",
        "band_none": "Sin datos suficientes",
        "legend_strong": "Punto fuerte: nivel {low} o más.",
        "legend_middle": "Intermedio: nivel {mid}.",
        "legend_weak": "A mejorar: nivel {high} o menos.",
        "legend_none": (
            "Sin datos suficientes: no se vio lo bastante; no es un "
            "nivel bajo."
        ),
        "legend_evidence": (
            "Los puntos pequeños cuentan las pruebas{n5}: cada punto "
            "relleno es una vez que se vio la conducta."
        ),
        "evidence": "{count} de {needed} pruebas necesarias",
        "evidence_enough": "{count} pruebas (mínimo {needed})",
        "radar_title": "Perfil de las competencias valoradas",
        "radar_note": (
            "Solo aparecen las competencias con nivel. Cuanto más lejos "
            "del centro, más alto el nivel."
        ),
        "no_radar": (
            "No hay gráfico de araña: hacen falta al menos {needed} "
            "competencias con nivel y aquí hay {rated}."
        ),
        "level_label": "Nivel {rating} de {maximum}",
        "seen": "Lo que se vio",
        "helps": "Cómo ayuda en la práctica",
        "hurts": "Cómo puede perjudicar en la práctica",
        "also_for": "También se vio a favor",
        "said": "{speaker}, minuto {clock}",
        "said_turn": "{speaker}, intervención {turn}",
        "none_strengths": (
            "No hay ninguna competencia con nivel alto y pruebas "
            "suficientes."
        ),
        "none_weaknesses": (
            "No hay ninguna competencia con nivel bajo y pruebas "
            "suficientes."
        ),
        "signals_text": (
            "Se vieron pocas veces: no bastan para dar un nivel, pero un "
            "evaluador debería revisarlas."
        ),
        "how_made": (
            "Cómo se hizo: rúbrica {catalogue} {version}; las citas se "
            "comprobaron palabra por palabra en la transcripción; "
            "modelo que propuso las valoraciones: {judge}."
        ),
        "note_competency": (
            "Competencia: una forma de comportarse en el trabajo que se "
            "puede observar, por ejemplo escuchar o repartir tareas."
        ),
        "note_evaluator": (
            "Evaluador humano: una persona formada que revisa este "
            "borrador y decide. El informe no decide nada."
        ),
        "note_situation": (
            "Situación: una conversación o ejercicio distinto. Una "
            "persona se comporta distinto según la situación; por eso "
            "hacen falta varias."
        ),
        "note_level": "Nivel: {labels}.",
        "note_level_plain": (
            "Nivel: {minimum} es el más bajo y {maximum} el más alto."
        ),
        "tile_situations": "Situaciones observadas",
        "tile_of": "de {count} recomendadas",
        "tile_rated": "Competencias con nivel",
        "tile_rated_of": "de {count}",
        "tile_duration": "Duración",
        "tile_minutes": "minutos",
        "tile_unknown": "sin hora",
        "note_evidence": (
            "Prueba: una frase dicha por la persona, copiada tal cual de "
            "la transcripción, que muestra la conducta."
        ),
    },
    "en": {
        "title": "Report on {participant}",
        "subtitle": "Group conversation “{session}”",
        "synthetic": (
            "Made-up sample data. This page does not describe a real "
            "person: it shows what the report looks like."
        ),
        "summary": "Summary",
        "meaning": "What it means and how to read it",
        "charts": "The profile at a glance",
        "strengths": "Strengths",
        "weaknesses": "Points to improve",
        "signals": "Isolated behaviours to watch",
        "notes": "Notes",
        "s_session": (
            "{participant} took part in a group conversation{duration}."
        ),
        "s_duration": " of about {minutes} minutes",
        "s_duration_short": " of less than a minute",
        "s_rated": (
            "{rated} of {total} competencies{n1} could be rated; for the "
            "other {missing} there was not enough evidence, which does "
            "not mean a low level."
        ),
        "s_rated_all": "All {total} competencies{n1} could be rated.",
        "s_rated_none": (
            "There was not enough evidence to rate any competency{n1}. "
            "That does not mean a low level: only that this conversation "
            "did not show them."
        ),
        "s_strengths": "Strengths in this conversation: {names}.",
        "s_no_strengths": "No strength was seen with enough evidence.",
        "s_weaknesses": "Points to improve: {names}.",
        "s_no_weaknesses": (
            "No point to improve was seen with enough evidence."
        ),
        "s_signals": (
            "No point to improve was seen with enough evidence, but "
            "{count} isolated behaviours are worth watching."
        ),
        "s_signal": (
            "No point to improve was seen with enough evidence, but one "
            "isolated behaviour is worth watching."
        ),
        "s_draft": (
            "This is a draft for a human assessor{n2}: one conversation "
            "is not enough for a reliable profile."
        ),
        "and": "and",
        "observed": "What was observed",
        "observed_text": (
            "One conversation{n3} (“{session}”){duration_text}. "
            "{participant} spoke {turns} times and said about {words} "
            "words."
        ),
        "observed_duration": ", about {minutes} minutes long",
        "observed_duration_short": ", less than a minute long",
        "level": "What a level is",
        "level_text": (
            "Each competency is rated from {minimum} to {maximum}{n4}. A "
            "level is only given when the behaviour was seen at least "
            "{needed} times at different moments of the conversation."
        ),
        "limits_title": "Limits of this report",
        "limits_text": (
            "Draft for a human assessor; one conversation is not enough "
            "for a reliable profile. {seen} situation{plural} seen here; "
            "a reliable profile needs about {recommended}."
        ),
        "limits_more": (
            "No emotion or personality trait is inferred from the voice. "
            "How long someone speaks is not used as a measure of "
            "leadership."
        ),
        "legend": "How to read the charts",
        "band_strong": "Strength",
        "band_middle": "Middle",
        "band_weak": "To improve",
        "band_none": "Not enough evidence",
        "legend_strong": "Strength: level {low} or above.",
        "legend_middle": "Middle: level {mid}.",
        "legend_weak": "To improve: level {high} or below.",
        "legend_none": (
            "Not enough evidence: not seen enough; it is not a low level."
        ),
        "legend_evidence": (
            "The small dots count the evidence{n5}: each filled dot is "
            "one time the behaviour was seen."
        ),
        "evidence": "{count} of {needed} pieces of evidence needed",
        "evidence_enough": "{count} pieces of evidence (minimum {needed})",
        "radar_title": "Profile of the rated competencies",
        "radar_note": (
            "Only rated competencies appear. The further from the "
            "centre, the higher the level."
        ),
        "no_radar": (
            "No radar chart: it needs at least {needed} rated "
            "competencies and there are {rated}."
        ),
        "level_label": "Level {rating} of {maximum}",
        "seen": "What was seen",
        "helps": "How it helps in practice",
        "hurts": "How it can hurt in practice",
        "also_for": "Also seen in favour",
        "said": "{speaker}, minute {clock}",
        "said_turn": "{speaker}, turn {turn}",
        "none_strengths": (
            "No competency has a high level with enough evidence."
        ),
        "none_weaknesses": (
            "No competency has a low level with enough evidence."
        ),
        "signals_text": (
            "Seen only a few times: not enough for a level, but an "
            "assessor should check them."
        ),
        "how_made": (
            "How it was made: rubric {catalogue} {version}; quotes were "
            "checked word by word against the transcript; model that "
            "proposed the ratings: {judge}."
        ),
        "note_competency": (
            "Competency: an observable way of behaving at work, for "
            "example listening or sharing out tasks."
        ),
        "note_evaluator": (
            "Human assessor: a trained person who reviews this draft and "
            "decides. The report decides nothing."
        ),
        "note_situation": (
            "Situation: a different conversation or exercise. People "
            "behave differently from one situation to another; that is "
            "why several are needed."
        ),
        "note_level": "Level: {labels}.",
        "note_level_plain": (
            "Level: {minimum} is the lowest and {maximum} the highest."
        ),
        "tile_situations": "Situations observed",
        "tile_of": "of {count} recommended",
        "tile_rated": "Competencies with a level",
        "tile_rated_of": "of {count}",
        "tile_duration": "Length",
        "tile_minutes": "minutes",
        "tile_unknown": "not timed",
        "note_evidence": (
            "Evidence: a sentence said by the person, copied exactly from "
            "the transcript, that shows the behaviour."
        ),
    },
}
"""Report wording per language; ``{nN}`` marks the footnote calls."""
