"""Fixed wording of the profile mode of the person report.

The profile page joins several sessions of one person. It reuses the
band names, the limits and the footnotes of ``person_report_texts``;
what is specific to several sessions is here, in Spanish and English.
Nothing is written by a model when the page is rendered.
"""

from typing import Dict

PROFILE_TEXTS: Dict[str, Dict[str, str]] = {
    "es": {
        "title": "Perfil de {participant}",
        "subtitle": "{count} conversaciones de grupo{dates}",
        "subtitle_one": "Una conversación de grupo{dates}",
        "subtitle_dates": ", del {first} al {last}",
        "subtitle_date": ", el {first}",
        "months": "ene feb mar abr may jun jul ago sep oct nov dic",
        "date": "{day} {month} {year}",
        "sessions": "Las conversaciones",
        "s_seen": (
            "{participant} aparece en {count} conversaciones{dates}, de "
            "{kinds} tipo{plural} distinto{plural}."
        ),
        "s_seen_one": "{participant} aparece en una sola conversación.",
        "s_seen_untyped": (
            "{participant} aparece en {count} conversaciones{dates}."
        ),
        "s_dates": " grabadas entre el {first} y el {last}",
        "s_aggregated": (
            "Hay un nivel de conjunto{n1} en {count} de {total} "
            "competencias: las que se vieron con pruebas en al menos "
            "{needed} conversaciones."
        ),
        "s_aggregated_none": (
            "Todavía ninguna competencia se vio con pruebas en {needed} "
            "conversaciones, así que no hay ningún nivel de conjunto{n1}."
        ),
        "s_building": (
            "Otras {count} están en construcción: se vieron en menos de "
            "{needed} conversaciones. Que falten pruebas no significa un "
            "nivel bajo."
        ),
        "s_building_one": (
            "Otra está en construcción: se vio en menos de {needed} "
            "conversaciones. Que falten pruebas no significa un nivel "
            "bajo."
        ),
        "s_repeated_strengths": (
            "Puntos fuertes que se repiten en varias conversaciones: "
            "{names}."
        ),
        "s_no_repeated_strengths": (
            "Ningún punto fuerte se repite en dos conversaciones o más."
        ),
        "s_repeated_weaknesses": "Puntos a mejorar que se repiten: {names}.",
        "s_no_repeated_weaknesses": (
            "Ningún punto a mejorar se repite en dos conversaciones o más."
        ),
        "s_draft_short": (
            "Es un borrador para un evaluador humano{n2}: para un perfil "
            "fiable hacen falta unas {recommended} situaciones distintas"
            "{n3} y aquí hay {count}."
        ),
        "s_draft": "Es un borrador para un evaluador humano{n2}.",
        "tile_situations": "Conversaciones",
        "tile_of": "de {count} recomendadas",
        "tile_kinds": "Tipos de conversación",
        "tile_kinds_of": "distintos",
        "tile_aggregated": "Con nivel de conjunto",
        "tile_aggregated_of": "de {count} competencias",
        "aggregate_title": "Qué es el nivel de conjunto",
        "aggregate_text": (
            "En cada conversación, una competencia tiene nivel solo si se "
            "vio al menos {needed} veces. El nivel de conjunto{n1} es el "
            "valor del medio de esos niveles, y solo se da cuando la "
            "competencia se vio en {situations} conversaciones o más. Las "
            "conversaciones sin pruebas no cuentan: no son un cero."
        ),
        "limits_text": (
            "Borrador para un evaluador humano. Se han visto {seen} "
            "situaciones; para un perfil fiable hacen falta unas "
            "{recommended}, y para una sola competencia unas {target}."
        ),
        "limits_change": (
            "Una persona se comporta distinto según la situación: que el "
            "nivel cambie de una conversación a otra es normal y no es un "
            "error del informe."
        ),
        "strip_title": "Cada competencia, conversación a conversación",
        "strip_text": (
            "Cada casilla es una conversación, en orden de fecha (el "
            "número de arriba). Dentro, el nivel que se vio en ella; una "
            "casilla con guiones es una conversación sin pruebas "
            "suficientes, no un nivel bajo."
        ),
        "status_aggregated": "Nivel de conjunto",
        "status_building": "En construcción",
        "status_none": "Sin datos suficientes",
        "aggregate_of": "valor del medio de {rated} conversaciones",
        "building_text": (
            "Pruebas en {rated} de {count} conversaciones; faltan "
            "{missing} con pruebas para dar un nivel de conjunto."
        ),
        "building_text_one": (
            "Pruebas en {rated} de {count} conversaciones; falta una más "
            "con pruebas para dar un nivel de conjunto."
        ),
        "none_list_title": "Sin datos suficientes en ninguna conversación",
        "none_list_text": (
            "Ninguna conversación mostró estas competencias lo bastante "
            "para darles un nivel. No es un nivel bajo: no se vieron."
        ),
        "legend_empty": (
            "Casilla con guiones: en esa conversación no hubo pruebas "
            "suficientes; no es un nivel bajo."
        ),
        "none_text": (
            "Ninguna conversación la mostró lo bastante; no es un nivel "
            "bajo."
        ),
        "consistent": (
            "Niveles parecidos en las conversaciones con nivel: de {low} "
            "a {high}."
        ),
        "same": "El mismo nivel en las {rated} conversaciones con nivel: "
                "{low}.",
        "varies": "Cambia según la conversación: de {low} a {high}.",
        "trend_up": (
            "Con el tiempo, sube (con tan pocas conversaciones puede ser "
            "casualidad)."
        ),
        "trend_down": (
            "Con el tiempo, baja (con tan pocas conversaciones puede ser "
            "casualidad)."
        ),
        "trend_flat": "Con el tiempo, se mantiene.",
        "reliability": (
            "Confianza en este nivel{n6}: {word} ({value} de 1); con unas "
            "{target} conversaciones llegaría a 0,70."
        ),
        "reliability_ok": (
            "Confianza en este nivel{n6}: {word} ({value} de 1)."
        ),
        "measured": (
            " Con los datos de las personas guardadas sale {value}."
        ),
        "word_low": "baja",
        "word_middle": "media",
        "word_good": "aceptable",
        "cell_level": "Conversación {number}: nivel {rating}, {band}",
        "cell_none": "Conversación {number}: sin datos suficientes",
        "kinds_seen": "Tipos de conversación con pruebas: {kinds}.",
        "repeated_strengths": "Puntos fuertes que se repiten",
        "repeated_weaknesses": "Puntos a mejorar que se repiten",
        "repeated_count": (
            "Punto fuerte en {count} de {rated} conversaciones con pruebas."
        ),
        "repeated_count_weak": (
            "Punto a mejorar en {count} de {rated} conversaciones con "
            "pruebas."
        ),
        "none_repeated_strengths": (
            "Ninguna competencia ha sido punto fuerte en dos "
            "conversaciones o más."
        ),
        "none_repeated_weaknesses": (
            "Ninguna competencia ha sido punto a mejorar en dos "
            "conversaciones o más."
        ),
        "in_session": "Conversación {number}",
        "session_turns": "{turns} intervenciones",
        "no_date": "sin fecha",
        "excluded_title": "Conversaciones que no se incluyeron",
        "excluded.no_person_ids": (
            "«{session}»: la grabación no identifica a las personas, así "
            "que no se puede saber cuál de las voces es {participant}."
        ),
        "excluded.absent": "«{session}»: {participant} no aparece.",
        "legend_building": (
            "En construcción: se vio en menos de {needed} conversaciones; "
            "no es un nivel bajo."
        ),
        "how_made": (
            "Cómo se hizo: rúbrica {catalogue} {version}; las citas se "
            "comprobaron palabra por palabra en cada transcripción; "
            "modelo que propuso las valoraciones: {judges}. Las "
            "conversaciones se unieron por el identificador de la "
            "persona «{person}»."
        ),
        "note_aggregate": (
            "Nivel de conjunto: el valor del medio (la mediana) de los "
            "niveles de las conversaciones en que hubo pruebas "
            "suficientes. Si hay un número par, la media de los dos del "
            "medio."
        ),
        "note_reliability": (
            "Confianza: de 0 a 1, cuánto se parecería este nivel al que "
            "saldría con otras conversaciones distintas. Es una "
            "estimación a partir de estudios publicados (una misma "
            "competencia coincide un 0,27 entre dos situaciones; Putka y "
            "Hoffman, 2013), no una medida hecha con estos datos."
        ),
    },
    "en": {
        "title": "Profile of {participant}",
        "subtitle": "{count} group conversations{dates}",
        "subtitle_one": "One group conversation{dates}",
        "subtitle_dates": ", {first} to {last}",
        "subtitle_date": ", {first}",
        "months": "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec",
        "date": "{day} {month} {year}",
        "sessions": "The conversations",
        "s_seen": (
            "{participant} appears in {count} conversations{dates}, of "
            "{kinds} different kind{plural}."
        ),
        "s_seen_one": "{participant} appears in one conversation only.",
        "s_seen_untyped": (
            "{participant} appears in {count} conversations{dates}."
        ),
        "s_dates": " recorded between {first} and {last}",
        "s_aggregated": (
            "{count} of {total} competencies have an overall level{n1}: "
            "those seen with evidence in at least {needed} conversations."
        ),
        "s_aggregated_none": (
            "No competency has been seen with evidence in {needed} "
            "conversations yet, so there is no overall level{n1}."
        ),
        "s_building": (
            "Another {count} are under construction: seen in fewer than "
            "{needed} conversations. Missing evidence does not mean a low "
            "level."
        ),
        "s_building_one": (
            "Another one is under construction: seen in fewer than "
            "{needed} conversations. Missing evidence does not mean a low "
            "level."
        ),
        "s_repeated_strengths": (
            "Strengths that repeat across conversations: {names}."
        ),
        "s_no_repeated_strengths": (
            "No strength repeats in two conversations or more."
        ),
        "s_repeated_weaknesses": "Points to improve that repeat: {names}.",
        "s_no_repeated_weaknesses": (
            "No point to improve repeats in two conversations or more."
        ),
        "s_draft_short": (
            "This is a draft for a human assessor{n2}: a reliable profile "
            "needs about {recommended} different situations{n3} and there "
            "are {count} here."
        ),
        "s_draft": "This is a draft for a human assessor{n2}.",
        "tile_situations": "Conversations",
        "tile_of": "of {count} recommended",
        "tile_kinds": "Kinds of conversation",
        "tile_kinds_of": "different",
        "tile_aggregated": "With an overall level",
        "tile_aggregated_of": "of {count} competencies",
        "aggregate_title": "What the overall level is",
        "aggregate_text": (
            "In each conversation, a competency has a level only if it "
            "was seen at least {needed} times. The overall level{n1} is "
            "the middle value of those levels, and it is only given when "
            "the competency was seen in {situations} conversations or "
            "more. Conversations without evidence do not count: they are "
            "not a zero."
        ),
        "limits_text": (
            "Draft for a human assessor. {seen} situations seen; a "
            "reliable profile needs about {recommended}, and a single "
            "competency about {target}."
        ),
        "limits_change": (
            "People behave differently from one situation to another: a "
            "level that changes between conversations is normal and not "
            "a fault of the report."
        ),
        "strip_title": "Each competency, conversation by conversation",
        "strip_text": (
            "Each box is one conversation, in date order (the number on "
            "top). Inside, the level seen in it; a dashed box is a "
            "conversation without enough evidence, not a low level."
        ),
        "status_aggregated": "Overall level",
        "status_building": "Under construction",
        "status_none": "Not enough evidence",
        "aggregate_of": "middle value of {rated} conversations",
        "building_text": (
            "Evidence in {rated} of {count} conversations; {missing} more "
            "with evidence needed for an overall level."
        ),
        "building_text_one": (
            "Evidence in {rated} of {count} conversations; one more with "
            "evidence needed for an overall level."
        ),
        "none_list_title": "Not enough evidence in any conversation",
        "none_list_text": (
            "No conversation showed these competencies enough to give "
            "them a level. It is not a low level: they were not seen."
        ),
        "legend_empty": (
            "Dashed box: not enough evidence in that conversation; it is "
            "not a low level."
        ),
        "none_text": (
            "No conversation showed it enough; it is not a low level."
        ),
        "consistent": (
            "Similar levels in the conversations with a level: {low} to "
            "{high}."
        ),
        "same": "The same level in the {rated} conversations with a "
                "level: {low}.",
        "varies": "Changes with the conversation: from {low} to {high}.",
        "trend_up": (
            "Over time, it goes up (with so few conversations this may be "
            "chance)."
        ),
        "trend_down": (
            "Over time, it goes down (with so few conversations this may "
            "be chance)."
        ),
        "trend_flat": "Over time, it stays the same.",
        "reliability": (
            "Confidence in this level{n6}: {word} ({value} out of 1); "
            "about {target} conversations would bring it to 0.70."
        ),
        "reliability_ok": (
            "Confidence in this level{n6}: {word} ({value} out of 1)."
        ),
        "measured": " With the data of the stored people it is {value}.",
        "word_low": "low",
        "word_middle": "medium",
        "word_good": "acceptable",
        "cell_level": "Conversation {number}: level {rating}, {band}",
        "cell_none": "Conversation {number}: not enough evidence",
        "kinds_seen": "Kinds of conversation with evidence: {kinds}.",
        "repeated_strengths": "Strengths that repeat",
        "repeated_weaknesses": "Points to improve that repeat",
        "repeated_count": (
            "Strength in {count} of {rated} conversations with evidence."
        ),
        "repeated_count_weak": (
            "Point to improve in {count} of {rated} conversations with "
            "evidence."
        ),
        "none_repeated_strengths": (
            "No competency was a strength in two conversations or more."
        ),
        "none_repeated_weaknesses": (
            "No competency was a point to improve in two conversations or "
            "more."
        ),
        "in_session": "Conversation {number}",
        "session_turns": "{turns} turns",
        "no_date": "no date",
        "excluded_title": "Conversations left out",
        "excluded.no_person_ids": (
            "“{session}”: the recording does not identify people, so "
            "there is no knowing which voice is {participant}."
        ),
        "excluded.absent": "“{session}”: {participant} does not appear.",
        "legend_building": (
            "Under construction: seen in fewer than {needed} "
            "conversations; it is not a low level."
        ),
        "how_made": (
            "How it was made: rubric {catalogue} {version}; quotes were "
            "checked word by word against each transcript; model that "
            "proposed the ratings: {judges}. The conversations were "
            "joined by the person id “{person}”."
        ),
        "note_aggregate": (
            "Overall level: the middle value (median) of the levels of "
            "the conversations with enough evidence. With an even number, "
            "the mean of the two middle ones."
        ),
        "note_reliability": (
            "Confidence: from 0 to 1, how close this level would be to "
            "the one other conversations would give. It is an estimate "
            "from published studies (the same competency agrees .27 "
            "between two situations; Putka and Hoffman, 2013), not a "
            "measurement on these data."
        ),
    },
}
"""Profile wording per language; ``{nN}`` marks the footnote calls."""
