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
        "context_type": "Tipo de conversación: {type}.",
        "context_goal": "Objetivo: {goal}.",
        "context_note": (
            "Lo indicó quien grabó la conversación; es contexto y no "
            "cambia ningún nivel."
        ),
        "type.reunion_decision": "reunión de decisión",
        "type.lluvia_ideas": "lluvia de ideas",
        "type.resolucion_problemas": "resolución de problemas",
        "type.negociacion": "negociación",
        "type.entrevista": "entrevista",
        "type.feedback": "feedback / evaluación",
        "type.conflicto": "conflicto / mediación",
        "type.coaching": "coaching",
        "type.venta": "venta",
        "type.conversacion_libre": "conversación libre",
        "type.otro": "otro",
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
        "band_provisional": "Estimación provisional — revisar",
        "legend_provisional": (
            "Estimación provisional: círculo hueco. Hay alguna prueba, "
            "pero no basta para un nivel firme; es una hipótesis para "
            "revisar."
        ),
        "estimate_score": "~{rating}",
        "estimate_label": (
            "Estimación provisional — revisar: nivel {rating} de "
            "{maximum}, confianza {confidence}"
        ),
        "confidence_low": "baja",
        "confidence_medium": "media",
        "s_provisional": (
            "En {count} competencias hay una estimación provisional: una "
            "hipótesis para que el entrevistador la revise, no una "
            "valoración."
        ),
        "s_provisional_one": (
            "En una competencia hay una estimación provisional: una "
            "hipótesis para que el entrevistador la revise, no una "
            "valoración."
        ),
        "s_tiers": (
            "Valoraciones firmes: {firm} · estimaciones provisionales: "
            "{provisional} · observaciones: {observations}."
        ),
        "review": "Estimaciones provisionales y observaciones",
        "review_text": (
            "Aquí hubo algo de evidencia, pero no la suficiente para un "
            "nivel firme. Cada estimación es una hipótesis; cada "
            "observación es una frase real de la persona, con su minuto, "
            "para comprobarla en la transcripción."
        ),
        "review_none": (
            "No hay estimaciones provisionales ni observaciones: lo que "
            "se vio tiene nivel firme o no se vio nada."
        ),
        "observations_only": "Sin estimación: solo observaciones",
        "why_not_firm": "Por qué no es firme",
        "observed_items": "Observaciones",
        "reason.few_quotes": "{quotes} de {needed} citas necesarias",
        "reason.only_confirmations": (
            "solo frases confirmadas por el juez una a una, sin cita en "
            "la valoración"
        ),
        "reason.single_episode": "un solo momento de la conversación",
        "reason.weak_match": (
            "alguna cita se parece poco a la conducta descrita"
        ),
        "reason.no_judge_rating": (
            "el modelo no propuso nivel; se estima por el signo (+/−) de "
            "las frases"
        ),
        "reason.short_participation": "participación corta: {words} palabras",
        "source.below_minimum": "cita comprobada; no llega al mínimo",
        "source.weak_match": "cita comprobada; poco parecida a la conducta",
        "source.confirmation": "confirmada por el juez frase a frase",
        "why_title": "Por qué hay pocas valoraciones firmes",
        "why.short_recording": "La grabación es corta.",
        "why.little_participation": (
            "{participant} habló poco: unas {words} palabras."
        ),
        "why.below_floor": (
            "{participant} habló menos del mínimo para valorar; no se "
            "pidió ninguna valoración al modelo."
        ),
        "why.few_quotes_proposed": (
            "El modelo propuso pocas citas: menos de las {needed} por "
            "competencia que pide un nivel firme."
        ),
        "why.quotes_filtered": (
            "La mitad o más de las citas propuestas no pasaron las "
            "comprobaciones (no estaban tal cual en la transcripción o se "
            "parecían poco a la conducta)."
        ),
        "why.not_shown": (
            "La conversación no mostró lo bastante de estas conductas."
        ),
        "limits_provisional": (
            "Las estimaciones provisionales son hipótesis para el "
            "entrevistador, no valoraciones: se apoyan en menos pruebas "
            "de las necesarias y hay que comprobarlas en la transcripción."
        ),
        "tile_provisional": "Estimaciones provisionales",
        "tile_observations": "{count} observaciones",
        "syn_title": "Síntesis",
        "syn_note": (
            "Borrador generado por Claude a partir de las evidencias; es "
            "una ayuda para la entrevista, no una decisión."
        ),
        "syn_job": "Puesto que se busca: {job}",
        "syn_strengths": "En qué destaca",
        "syn_unobserved": "Qué no se ha podido observar",
        "syn_unobserved_none": (
            "Todas las competencias tienen al menos alguna observación."
        ),
        "syn_fit": "Encaje con el puesto",
        "syn_tasks": "Para qué tipo de tareas parece útil",
        "syn_questions": "Qué preguntar o comprobar en la entrevista",
        "syn_confidence": "Confianza: {level}",
        "syn_conf_low": "baja",
        "syn_conf_medium": "media",
        "syn_conf_high": "alta",
        "syn_conf_reason": (
            "{firm} competencias con nivel firme ({firm_quotes} citas), "
            "{provisional} con estimación provisional, {quotes} citas en "
            "total y {words} palabras dichas."
        ),
        "syn_basis_firm": "nivel firme",
        "syn_basis_provisional": "estimación provisional",
        "syn_basis_observation": "observación",
        "syn_none": "El modelo no dejó ningún punto que cite evidencias.",
        "syn_skipped": "No se ha escrito la síntesis: {reason}",
        "syn_no_evidence": "no hay ninguna cita de esta persona.",
        "syn_dropped": (
            "Se han descartado {count} textos del modelo (puntos, preguntas, "
            "conclusiones por competencia o la línea global; las "
            "conclusiones y la línea se sustituyen por un texto de reglas) "
            "por no citar evidencias del análisis o por saltarse las reglas "
            "(veredicto o recomendación, comparación, etiqueta de "
            "personalidad o dato personal)."
        ),
        "syn_model": "Escrito por {model}.",
    },
    "en": {
        "title": "Report on {participant}",
        "subtitle": "Group conversation “{session}”",
        "context_type": "Kind of conversation: {type}.",
        "context_goal": "Goal: {goal}.",
        "context_note": (
            "Given by whoever recorded the conversation; it is context "
            "and changes no level."
        ),
        "type.reunion_decision": "decision meeting",
        "type.lluvia_ideas": "brainstorming",
        "type.resolucion_problemas": "problem solving",
        "type.negociacion": "negotiation",
        "type.entrevista": "interview",
        "type.feedback": "feedback / appraisal",
        "type.conflicto": "conflict / mediation",
        "type.coaching": "coaching",
        "type.venta": "sales",
        "type.conversacion_libre": "casual conversation",
        "type.otro": "other",
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
        "band_provisional": "Provisional estimate — review",
        "legend_provisional": (
            "Provisional estimate: hollow circle. There is some evidence, "
            "but not enough for a firm level; it is a hypothesis to "
            "review."
        ),
        "estimate_score": "~{rating}",
        "estimate_label": (
            "Provisional estimate — review: level {rating} of {maximum}, "
            "{confidence} confidence"
        ),
        "confidence_low": "low",
        "confidence_medium": "medium",
        "s_provisional": (
            "{count} competencies have a provisional estimate: a "
            "hypothesis for the interviewer to check, not an assessment."
        ),
        "s_provisional_one": (
            "One competency has a provisional estimate: a hypothesis for "
            "the interviewer to check, not an assessment."
        ),
        "s_tiers": (
            "Firm levels: {firm} · provisional estimates: {provisional} · "
            "observations: {observations}."
        ),
        "review": "Provisional estimates and observations",
        "review_text": (
            "There was some evidence here, but not enough for a firm "
            "level. Each estimate is a hypothesis; each observation is a "
            "real sentence of the person, with its minute, to check in "
            "the transcript."
        ),
        "review_none": (
            "No provisional estimates or observations: what was seen has "
            "a firm level, or nothing was seen."
        ),
        "observations_only": "No estimate: observations only",
        "why_not_firm": "Why it is not firm",
        "observed_items": "Observations",
        "reason.few_quotes": "{quotes} of {needed} quotes needed",
        "reason.only_confirmations": (
            "only sentences the judge confirmed one by one, no quote in "
            "the assessment"
        ),
        "reason.single_episode": "a single moment of the conversation",
        "reason.weak_match": (
            "a quote looks little like the behaviour described"
        ),
        "reason.no_judge_rating": (
            "the model proposed no level; estimated from the sign (+/−) "
            "of the sentences"
        ),
        "reason.short_participation": "short participation: {words} words",
        "source.below_minimum": "checked quote; below the minimum",
        "source.weak_match": "checked quote; little like the behaviour",
        "source.confirmation": "confirmed by the judge sentence by sentence",
        "why_title": "Why there are few firm levels",
        "why.short_recording": "The recording is short.",
        "why.little_participation": (
            "{participant} spoke little: about {words} words."
        ),
        "why.below_floor": (
            "{participant} spoke less than the minimum to assess; the "
            "model was not asked for any level."
        ),
        "why.few_quotes_proposed": (
            "The model proposed few quotes: fewer than the {needed} per "
            "competency a firm level needs."
        ),
        "why.quotes_filtered": (
            "Half or more of the proposed quotes failed the checks (not "
            "verbatim in the transcript, or little like the behaviour)."
        ),
        "why.not_shown": (
            "The conversation did not show enough of these behaviours."
        ),
        "limits_provisional": (
            "Provisional estimates are hypotheses for the interviewer, not "
            "assessments: they rest on less evidence than needed and must "
            "be checked in the transcript."
        ),
        "tile_provisional": "Provisional estimates",
        "tile_observations": "{count} observations",
        "syn_title": "Synthesis",
        "syn_note": (
            "Draft written by Claude from the evidence; an aid for the "
            "interview, not a decision."
        ),
        "syn_job": "Job sought: {job}",
        "syn_strengths": "What they stand out in",
        "syn_unobserved": "What could not be observed",
        "syn_unobserved_none": (
            "Every competency has at least some observation."
        ),
        "syn_fit": "Fit with the job sought",
        "syn_tasks": "What kinds of tasks they look useful for",
        "syn_questions": "What to ask or check in the interview",
        "syn_confidence": "Confidence: {level}",
        "syn_conf_low": "low",
        "syn_conf_medium": "medium",
        "syn_conf_high": "high",
        "syn_conf_reason": (
            "{firm} competencies with a firm level ({firm_quotes} quotes), "
            "{provisional} with a provisional estimate, {quotes} quotes in "
            "all and {words} words said."
        ),
        "syn_basis_firm": "firm level",
        "syn_basis_provisional": "provisional estimate",
        "syn_basis_observation": "observation",
        "syn_none": "The model left no point that cites evidence.",
        "syn_skipped": "No synthesis was written: {reason}",
        "syn_no_evidence": "there is no quote of this person.",
        "syn_dropped": (
            "{count} texts of the model (points, questions, per-skill "
            "conclusions or the overall line; conclusions and the line are "
            "replaced by a rules-written text) were dropped for citing no "
            "evidence of the analysis or breaking the rules (a verdict or "
            "recommendation, a comparison, a personality label or a "
            "personal attribute)."
        ),
        "syn_model": "Written by {model}.",
    },
}
"""Report wording per language; ``{nN}`` marks the footnote calls."""


def conversation_type_label(language: str, kind: str) -> str:
    """Return the name of conversation type ``kind`` in ``language``.

    A type this wording does not know is shown as it came.
    """
    return TEXTS[language].get(f"type.{kind}", kind)
