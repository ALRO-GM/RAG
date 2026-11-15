# Guía de anotación: qué hace cada frase en una conversación

Para quien anota. No hace falta ser lingüista. Las 13 categorías y la
razón de cada una están en [`CONVERSATION.md`](CONVERSATION.md) §5 (decisión
1 del dueño, 2026-11-15: categorías separadas, 13 en total). Esta guía es
distinta de [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md): aquella enseña a
marcar **competencias** (escucha, conflicto, liderazgo...) y a valorarlas;
esta enseña a marcar **qué hace cada frase** (pregunta, propuesta,
acuerdo...). Las dos usan las mismas normas de fondo: se anota lo que se
dijo, no lo que la persona pensaba ni cómo se sentía, y se anota antes de
ver lo que diga el sistema.

Estado: guía de trabajo, escrita antes de anotar nada. Los números de
acuerdo que aparecen son **prestados** de otros estudios; se cambian por
los nuestros cuando los midamos (§11). Decisión 2 del dueño (2026-11-15):
una frase puede llevar una **segunda y una tercera etiqueta, cada una con
su peso en %** (§7).

---

## 1. Qué se anota

Cada **frase** de la transcripción recibe **una etiqueta principal** (la
categoría que mejor dice qué hace esa frase en la conversación), y como
mucho **dos más con su peso** si hace claramente varias cosas (§7), y, si
la frase responde a otra anterior, el **identificador de esa frase
anterior**.

- La unidad es la frase tal como viene en `transcript.v1`: tiene un
  identificador como `t3.s0` (turno 3, frase 0), un hablante y un texto.
  Se ve con `uv run python -m src skills sentences --transcript ...`.
- **No se cortan ni se unen frases.** Si una frase hace dos cosas, ver §7.
- Se anota **qué hace** la frase, no cómo se siente quien habla. "Gracias"
  es una muestra de cortesía; no se anota si parecía contento o enfadado.
  Tampoco se anota el acento, la gramática ni la fluidez.
- Se mira el **contexto**: las frases de antes y quién las dijo. La misma
  frase ("Vale") puede ser cosas distintas según lo que haya pasado antes.
  No se mira lo que viene después, salvo para saber si algo se contestó
  (§5, categorías 1 y 2).
- Todavía no hay formulario para esto. Mientras tanto, una tabla con
  columnas `frase` (id), `etiquetas` y `responde_a` (id o vacío). En
  `etiquetas` va una categoría sola (`compromiso`, que vale 100 %) o
  hasta tres con su peso, la principal primero:
  `compromiso:70 aceptacion:30` (§7). Es la misma forma que usa el
  conjunto de ejemplos (`examples/skills/dialogue_acts/`), así que una
  tabla se puede comprobar con el mismo programa.

## 2. Las 13 categorías de un vistazo

| # | Categoría | En una frase | Responde a otra frase |
|---|---|---|---|
| 1 | Pregunta | Pide información, una opinión, una confirmación o una elección | no (pide respuesta) |
| 2 | Respuesta | Contesta a una pregunta anterior | sí, a la pregunta |
| 3 | Información | Cuenta un hecho, un dato o una explicación que se podría comprobar | no |
| 4 | Opinión | Valora, juzga o toma postura | no |
| 5 | Propuesta | Plantea una idea o forma de actuar para que el grupo decida | no (pide respuesta) |
| 6 | Petición | Pide a alguien (o al grupo) que haga algo | no (pide cumplimiento) |
| 7 | Aceptación | Acepta o apoya lo que dijo otro | sí |
| 8 | Rechazo | Rechaza, discrepa o corrige lo que dijo otro | sí |
| 9 | Compromiso | Quien habla se obliga a hacer algo | a veces (si responde a una petición) |
| 10 | Señal de escucha | "Ajá", "ya, ya": sigue la conversación sin aportar | no |
| 11 | Gestión de la reunión | Habla de la propia reunión: orden, turnos, tiempo, resúmenes | no |
| 12 | Relación | Saludos, gracias, disculpas, elogios o críticas a la persona, bromas | no |
| 13 | Otro | Frase cortada, ininteligible o sin función clara | no |

## 3. Cómo decidir: el orden de las preguntas

Ante una frase, hazte las preguntas en este orden y **para en la primera
que sea clara**:

1. ¿Está cortada, no se entiende o no dice nada? → **Otro** (§8).
2. ¿Es solo un "ajá / ya / mm" mientras otro habla? → **Señal de escucha**.
3. ¿Habla de la reunión misma (orden del día, turnos, tiempo, resumen,
   cambio de tema)? → **Gestión**.
4. ¿Es cortesía o trato personal (saludo, gracias, perdón, elogio o
   crítica a una persona, broma)? → **Relación**.
5. ¿Contesta a una pregunta que acaba de hacerse? → **Respuesta**.
6. ¿Reacciona a una propuesta, petición u opinión de otro, diciendo que sí
   o que no? → **Aceptación** o **Rechazo**.
7. ¿Quien habla se obliga a hacer algo? → **Compromiso**.
8. ¿Pide algo a una persona concreta o al grupo? → **Petición**.
9. ¿Plantea una idea o forma de actuar para que el grupo la considere?
   → **Propuesta**.
10. ¿Pide información, opinión o confirmación? → **Pregunta**.
11. ¿Valora, juzga o toma postura? → **Opinión**.
12. ¿Cuenta algo que se podría comprobar? → **Información**.

Es una ayuda, no una ley: cuando el orden choque con una regla de una
categoría concreta (§4), manda la regla de la categoría.

## 4. Las categorías, una a una

En cada una: qué es, cómo reconocerla, ejemplos, con qué se confunde y la
regla para decidir. Los ejemplos están escritos para esta guía; no son
datos reales.

### 4.1 Pregunta

**Qué es.** Pide información, una opinión, una confirmación o una
elección. Incluye "¿cómo?" o "¿a qué te refieres?" cuando se pide que lo
repitan o lo aclaren.

**Cómo reconocerla.** Suele llevar interrogación o empezar con qué, quién,
cuánto, cuándo, por qué. Quien pregunta espera una respuesta.

**Ejemplos.**
- "¿Cuántas horas nos quedan para la entrega?"
- "¿Y tú qué opinas de cambiar el proveedor, Elena?"
- "O sea, ¿entonces la reunión es el jueves?" (pide confirmar)
- "¿A qué te refieres con lo del segundo contrato?"

**Se confunde con.**
- *Petición*: "¿Me pasas el acta?" no pregunta nada, pide una acción.
- *Propuesta*: "¿Y si lo dejamos para el lunes?" propone.
- *Rechazo*: "¿Y para qué sirve eso?" dicho para quitar valor.

**Regla.** Pregunta solo si lo que se quiere es una **respuesta**. Si lo
que se quiere es una acción o se ofrece una idea, es Petición o Propuesta
aunque lleve interrogación. Sin interrogación (el reconocimiento de voz a
veces la quita): "Entonces lo entregamos el viernes" es Pregunta solo si
la frase siguiente la contesta; si no, es Información.

### 4.2 Respuesta

**Qué es.** Da la información o la opinión que se pidió en una pregunta
anterior. Incluye "no lo sé". Se anota `responde_a` con la pregunta.

**Ejemplos.**
- "Unas veinte horas, más o menos." (a "¿Cuántas horas nos quedan?")
- "Yo prefiero seguir con el proveedor de siempre." (a "¿Qué opinas?")
- "No lo sé, tendría que mirarlo." 
- "Sí, el jueves a las diez." (a una pregunta de confirmación)

**Se confunde con.**
- *Aceptación*: "Sí, hagámoslo" contesta a una **propuesta**, no a una
  pregunta.
- *Información* / *Opinión*: una frase que no responde a ninguna pregunta.

**Regla.** Si la frase contesta una pregunta, es Respuesta, **aunque lo
que diga sea una opinión o un dato**. Una contestación que se va por las
ramas sin dar lo pedido ("Bueno, ya sabes cómo es esto...") no es
Respuesta: es Información u Opinión sin `responde_a`.

### 4.3 Información

**Qué es.** Cuenta un hecho, un suceso, una cifra, una explicación o una
razón que **se podría comprobar**: alguien podría mirarlo y decir si es
verdad o mentira. No contesta a una pregunta y no propone nada.

**Ejemplos.**
- "El proveedor dijo que entrega el día quince."
- "Ayer cambiaron la contraseña del servidor."
- "El presupuesto ya está aprobado por dirección."
- "Esa tabla se calcula con los datos de marzo."

**Se confunde con.**
- *Opinión*: ver 4.4.
- *Respuesta*: si contesta a una pregunta, es Respuesta.
- *Compromiso*: "Ya lo mandé" es Información (cuenta algo pasado);
  "Lo mando mañana" es Compromiso.

**Regla.** Pregunta de control: **¿se podría comprobar?** Si sí,
Información. Si lo que se dice depende de cómo lo ve quien habla, es
Opinión. Lo que alguien dijo que opina ("Pablo cree que no llegamos") es
Información sobre lo que dijo Pablo; la opinión es de Pablo, no de quien
lo cuenta.

### 4.4 Opinión

**Qué es.** Expresa una valoración, un juicio, una creencia, una
preferencia o una postura sobre el tema, una opción o el trabajo de
alguien. Dice **cómo lo ve** quien habla, no qué ocurrió.

**Cómo reconocerla.** Suele llevar "creo que", "me parece", "para mí",
"no me convence", o palabras que valoran (mejor, peor, arriesgado,
injusto, fácil, importante).

**Ejemplos.**
- "Me parece demasiado arriesgado cambiar ahora."
- "Para mí lo urgente es el informe, no la presentación."
- "No me convence esa herramienta."
- "El problema no es el plazo, es que nadie se hace cargo."

**Se confunde con.**
- *Información*: la confusión más probable. "Creo que" no decide por sí
  solo: "Creo que la reunión era a las diez" es un dato dicho con duda
  (Información); "Creo que la reunión es una pérdida de tiempo" es un
  juicio (Opinión). Una predicción que depende de cómo lee la situación
  quien habla ("el proveedor no va a llegar") es Opinión; un hecho
  previsto en un calendario ("llega el jueves") es Información.
- *Aceptación / Rechazo*: si la valoración es una **reacción** a lo que
  acaba de decir otro ("Eso me parece una buena idea", "No lo veo
  claro"), es Aceptación o Rechazo, no Opinión.
- *Relación*: valorar a la persona ("eres muy desordenado") es Relación;
  valorar su trabajo o una idea no lo es.
- *Propuesta*: "Lo mejor sería esperar" propone una acción; es Propuesta.

**Regla.** Opinión es una valoración que **se abre sola**, sin responder a
una pregunta ni a una propuesta. Si responde, manda la otra categoría.
Esta categoría y Información se midieron por separado a petición del
dueño; si al medir no se distinguen bien se volverán a unir (§11).

### 4.5 Propuesta

**Qué es.** Plantea una idea, solución, plan u oferta para que el grupo la
considere y la acepte o la rechace. Quien la dice **no obliga a nadie en
concreto** a hacerla.

**Ejemplos.**
- "Podríamos repartir el informe en dos partes."
- "¿Y si probamos primero con un solo cliente?" (una propuesta en forma
  de pregunta)
- "Propongo que votemos ahora."
- "Habría que revisar también las cifras de ventas."

**Se confunde con.**
- *Petición*: ver 4.6. Es la confusión principal.
- *Pregunta*: "¿Y si...?" es Propuesta, no Pregunta.
- *Compromiso*: "Si queréis, lo hago yo" es Propuesta (depende de que
  acepten); "Lo hago yo" es Compromiso.
- *Gestión*: proponer cómo llevar la **reunión** ("vamos punto por
  punto") es Gestión; proponer cómo hacer el **trabajo** es Propuesta.

**Regla.** Propuesta cuando la acción es una **idea para decidir entre
todos** y no hay una persona señalada para hacerla.

### 4.6 Petición

**Qué es.** Pide o manda a una persona concreta (o al grupo) que haga algo.

**Ejemplos.**
- "Lucía, ¿me mandas los datos esta tarde?"
- "Revisad el documento antes del lunes."
- "Necesito que alguien llame al proveedor."
- "Pásame el enlace, por favor."

**Se confunde con.**
- *Pregunta*: "¿Me mandas...?" es Petición, aunque lleve interrogación.
- *Propuesta*: "Habría que llamar al proveedor" no señala a nadie:
  Propuesta.

**Regla.** Pregunta de control: **¿hay alguien a quien se le pide que
haga algo?** Si hay destinatario (nombrado, o claro por el contexto:
"tú", "vosotros", "alguien"), Petición. Si es una idea para el grupo sin
destinatario, Propuesta. Esta pareja (5 y 6) se medirá por separado; si no
se distingue bien, se unirán (§11).

### 4.7 Aceptación

**Qué es.** Acepta, apoya o está de acuerdo con una propuesta, petición,
oferta u opinión anterior de otra persona. Incluye "sí, y además...". Se
anota `responde_a`.

**Ejemplos.**
- "Vale, me parece bien."
- "De acuerdo, lo hacemos así."
- "Totalmente, y además nos ahorra una semana."
- "Sí, eso es justo lo que yo pensaba."

**Se confunde con.**
- *Señal de escucha*: "vale" o "sí" **mientras otro cuenta algo** es
  Señal de escucha; "vale" que cierra una propuesta o una petición es
  Aceptación.
- *Rechazo*: "Sí, pero..." es Rechazo (ver 4.8).
- *Compromiso*: "Vale, lo hago yo" acepta y se compromete (§7: primaria
  Compromiso).

**Regla.** Aceptación necesita **algo anterior** que aceptar. Si no se
puede decir a qué responde, no es Aceptación.

### 4.8 Rechazo

**Qué es.** Rechaza una propuesta o petición, discrepa de una opinión,
corrige lo dicho o pone una objeción. Incluye el rechazo parcial ("sí,
pero...") y la pregunta retórica que objeta. Se anota `responde_a`.

**Ejemplos.**
- "No, eso no va a funcionar: el cliente no lo acepta."
- "Sí, pero con ese plazo no nos da."
- "No es así, la reunión era el martes, no el jueves." (corrige)
- "Eso no me cuadra con lo que dijo el cliente."

**Se confunde con.**
- *Opinión*: una valoración negativa que no responde a nada ("Esa
  herramienta es mala") es Opinión; si responde a una propuesta, Rechazo.
- *Relación*: un ataque a la persona sin contenido ("no tienes ni idea")
  es Relación; si objeta la idea con contenido, Rechazo.
- *Aceptación*: "Sí, pero..." empieza con sí pero **es Rechazo**: lo que
  pesa es la objeción.

**Regla.** Rechazo necesita algo anterior a lo que se oponga. Con razón o
sin ella, es Rechazo; si lleva razón, se puede añadir Información como
segunda etiqueta (§7).

### 4.9 Compromiso

**Qué es.** Quien habla se obliga a hacer algo en el futuro ("lo hago yo",
"te lo mando mañana"). Es poco frecuente y fácil de confundir: léelo con
cuidado.

**Ejemplos.**
- "Yo me encargo de llamar al proveedor."
- "Te lo mando mañana antes de las diez."
- "Esta semana lo reviso y os digo algo."
- "Vale, lo preparo yo para el jueves."

**Se confunde con.**
- *Propuesta*: "Habría que mirarlo" (sin dueño) es Propuesta; "Lo miro
  yo" es Compromiso.
- *Petición / Información*: "Juan lo hará" es una promesa sobre otro: se
  anota Petición o Información según el contexto; Compromiso solo cuando
  habla la propia persona.
- *Información*: "Ya lo mandé" cuenta algo pasado: Información.
- *Aceptación*: ver §7.

**Regla.** Compromiso necesita **tres cosas**: quien habla es quien lo
hará, es una acción futura, y suena a obligación (no a "quizá" ni a "me
gustaría"). Si falta alguna, no es Compromiso.

### 4.10 Señal de escucha

**Qué es.** Una palabra o dos que muestran atención o que se ha entendido,
sin quitarle la palabra a nadie y sin añadir contenido: "ajá", "sí, sí",
"ya, ya", "mm", "claro" mientras otro habla.

**Ejemplos.**
- "Ajá."
- "Ya, ya."
- "Sí, sí."
- "Ah, vale." (solo entiende lo que se acaba de contar)

**Se confunde con.**
- *Aceptación*: si el "vale" cierra una propuesta o petición, es
  Aceptación.
- *Otro*: "eh...", "bueno..." para ganar tiempo, que no son escucha, son
  Otro (o Gestión si anuncian que va a hablar).

**Regla.** Señal de escucha es corta, no añade nada y **no responde a
una propuesta ni a una pregunta**. Si después del "vale" la frase sigue con
contenido ("Vale, y entonces ¿cuándo?"), se anota el contenido (Pregunta).
Las transcripciones automáticas pierden muchas de estas señales; que haya
pocas no dice nada de la persona (§8).

### 4.11 Gestión de la reunión

**Qué es.** Habla de la conversación misma: el orden del día, quién habla,
el tiempo, cambiar de tema, abrir o cerrar, y los **resúmenes** de lo dicho
o decidido.

**Ejemplos.**
- "Pasamos al siguiente punto."
- "Nos quedan diez minutos."
- "Entonces, resumiendo: Marta entrega el viernes y Luis lo revisa."
- "Antes de seguir, ¿te parece que dejemos esto y lo retomamos luego?"

**Se confunde con.**
- *Propuesta*: proponer cómo hacer el **trabajo** es Propuesta; proponer
  cómo llevar la **reunión** es Gestión.
- *Pregunta*: "Irene, no has dicho nada, ¿cómo lo ves?" da la palabra:
  Gestión 60, Pregunta 40 (§7).
- *Información*: un resumen que repite lo hablado es Gestión, no
  Información.

**Regla.** Gestión si la frase trata de **cómo va la reunión**, no del
tema que se discute.

### 4.12 Relación

**Qué es.** Cuida la relación, no la tarea: saludos y despedidas, gracias,
disculpas, elogios o críticas **a una persona**, ánimos, bromas y charla
personal. Se nombra por **lo que hace** la frase (agradecer, disculparse),
nunca por lo que parece sentir quien habla.

**Ejemplos.**
- "Buenos días a todos."
- "Gracias por traerlo tan preparado, Paula."
- "Perdona, antes te he cortado."
- "Qué bien que hayas podido venir hoy." 

**Se confunde con.**
- *Opinión / Información*: elogiar un **trabajo concreto** ("la tabla de
  la página cuatro está muy clara") es Opinión; el elogio general a la
  persona ("eres un fenómeno") es Relación.
- *Rechazo*: criticar una idea es Rechazo; criticar a la persona, Relación.

**Regla.** Relación si el destino de la frase es **la persona o el trato
entre personas**; si es la idea o el trabajo, la categoría es otra.

### 4.13 Otro

**Qué es.** Lo que no cabe en las demás: frases cortadas, arrancadas y
abandonadas, muletillas para ganar tiempo, texto ininteligible, ruido
transcrito, charla sin relación con la reunión.

**Ejemplos.**
- "Es que el... bueno, no."
- "Eh..."
- "[ininteligible]"
- "Pues eso, que..."

**Regla.** Otro es la **última salida**, no la primera. Antes de usarlo,
mira si, leyendo con las frases de alrededor, la frase cumple alguna
función (§8).

## 5. Respuestas, aceptaciones y rechazos: a qué responden

Cuatro categorías son la "segunda mitad" de un par: Respuesta, Aceptación,
Rechazo y, a veces, Compromiso. Para ellas se apunta en `responde_a` el
identificador de la frase a la que contestan.

| Frase | Responde a | Ejemplo |
|---|---|---|
| Respuesta | la Pregunta | `t7.s0` "Unas veinte horas." → `t6.s0` |
| Aceptación | la Propuesta, Petición u Opinión | `t9.s0` "Vale, me parece bien." → `t8.s0` |
| Rechazo | la Propuesta, Petición, Opinión o Información que se corrige | `t10.s0` "Sí, pero no nos da el plazo." → `t8.s0` |
| Compromiso | la Petición, si la hubo; si no, queda vacío | `t11.s0` "Te lo mando mañana." → `t10.s1` |

Reglas:

1. **Se apunta la frase concreta**, no el turno. Si la propuesta y su
   motivo están en dos frases, se apunta a la que contiene la propuesta.
2. **Si lo que contesta está a varias frases de distancia**, se apunta a
   la correcta igualmente ("Sí, hagámoslo", tres frases después de la
   propuesta, apunta a la propuesta).
3. **Si no se puede decir a qué responde**, probablemente no es una segunda
   mitad: vuelve a §3 (¿Aceptación sin nada aceptado? Mira si es una
   Señal de escucha o una Opinión).
4. **Una propuesta puede tener varias respuestas** (aceptación de uno,
   rechazo de otro): cada una apunta a la propuesta. Lo que más tarde se
   deduce (que hay una decisión, una tarea asignada, una pregunta sin
   contestar) lo calcula el sistema a partir de estas marcas; **quien anota
   no marca decisiones ni acuerdos**, solo frases y enlaces.
5. Una Pregunta o una Propuesta **sin respuesta** no se marcan de ninguna
   forma especial. Es útil dejar la propuesta o la pregunta sin segunda
   mitad: es justo la información que se busca.

## 6. Cuando la frase es dudosa

- **Léela otra vez con las dos frases de antes.** Muchas dudas se resuelven
  con el contexto.
- Elige **la categoría que mejor explica por qué se dijo la frase en ese
  momento** (no la más llamativa).
- Si dudas entre dos, **aplica la regla de las dos categorías** (§4), que
  siempre la dice. Si las dos reglas no deciden, elige y marca la frase
  como "dudosa" en una columna aparte; se revisan al final (§10).
- **Nunca se deja una frase sin etiqueta.** Otro es una etiqueta, no un
  hueco.

## 7. Segunda y tercera etiqueta, con peso

**Decisión 2 del dueño (2026-11-15):** una frase puede llevar una
segunda y una tercera etiqueta, cada una con su % de certeza. El peso
dice **qué parte de lo que hace la frase** corresponde a cada acto.

**Cuándo se añade.** Solo cuando la frase hace **claramente dos (o tres)
cosas** de las categorías de esta guía. Hay que poder decir cada una con
la regla de §4; no vale "a lo mejor también". Casi todas las frases
llevan una sola etiqueta; si te ves poniendo dos en muchas, relee §4.
La tercera es rara: una frase corta casi nunca hace tres cosas.

**Cómo se reparte el peso.**

1. Los pesos son **números enteros que suman 100**, en **pasos de 10**
   (10, 20, 30...). No hay más precisión que esa: nadie distingue un 65
   de un 70.
2. La **principal va primero** y es la que más pesa (si dos pesan lo
   mismo, va primero la que se dice antes). Con estas reglas la
   principal siempre tiene **40 o más**.
3. Repartos de partida: **70 / 30** si la segunda cosa es claramente
   menor (lo normal); **60 / 40** si las dos pesan casi igual; **50 /
   50** solo si de verdad no se puede decir cuál pesa más (y entonces
   la principal es la primera que se dice); **40 / 30 / 30** o
   **60 / 20 / 20** con tres. 90 / 10 no merece la pena: si la segunda
   cosa pesa tan poco, no se pone.
4. Una etiqueta sola se escribe sin peso y vale 100.
5. Si la frase responde a algo, la regla de §5 vale para **todas** sus
   etiquetas: un `aceptacion:30` también necesita `responde_a`.

**Cuál es la principal.** La que cambia lo que pasa a continuación en la
conversación (la que pide o hace algo); la segunda es la que lo sostiene.
Si hay duda, la **primera cosa que dice la frase**. La principal es la
que cuenta para κ (§10), así que es la decisión más importante.

**Ejemplos resueltos.**

| Frase | Etiquetas | Por qué |
|---|---|---|
| "Vale, te lo mando a las cuatro." (a una petición) | `compromiso:70 aceptacion:30` | lo que cambia la conversación es la promesa; el "vale" acepta, pero es menor |
| "Irene, no has dicho nada del reparto, ¿cómo lo ves?" | `gestion:60 pregunta:40` | da la palabra (Gestión) y a la vez pide su opinión; las dos pesan casi igual |
| "No, el cliente quiere un solo documento y lo dijo por escrito." | `rechazo:70 informacion:30` | rechaza; la razón es un dato comprobable que lo sostiene |
| "Podríamos dividirlo en dos, porque así se revisa antes." | `propuesta:70 opinion:30` | propone; "se revisa antes" valora, no es un dato |
| "Gracias, Luis; Irene, ¿tú cómo lo ves?" | `gestion:40 pregunta:30 relacion:30` | da la palabra (lo que cambia la conversación), pide opinión y agradece; tres cosas claras, ninguna pesa mucho más |
| "Sí, sí." (alguien está contando algo) | `escucha` | una sola cosa: no se añade nada |

**En ningún caso se usa la segunda etiqueta para evitar decidir.** Una
frase dudosa entre Información y Opinión no es "información:50
opinion:50": se decide con la regla de §4.4 y se marca como dudosa (§6).
El peso dice cuánto de la frase hace cada cosa, **no lo seguro que
estás**. (El etiquetador automático sí usa el peso como probabilidad:
§11.)

## 8. Errores de transcripción, frases cortadas y señales de escucha

Las transcripciones salen de un reconocedor de voz y se equivocan. En el
fichero `transcript.v1` (JSON) cada frase puede traer `confidence`,
`doubtful` (el sistema duda de la atribución o del texto) y `overlap`; el
listado de `skills sentences` solo enseña tiempos, id, hablante y texto.

1. **Error evidente que no cambia la función** ("Podemos devidir el
   informe"): se anota con la categoría que corresponde y no se corrige el
   texto.
2. **Error que impide entender la frase** o frase que se corta en medio:
   **Otro**, aunque tenga rastros de otra categoría. No se adivina.
3. **Palabras mal reconocidas que cambian el sentido** ("no" donde se dijo
   "sí"): si no se puede saber lo que se dijo, **Otro**. Esas frases
   se cuentan aparte para medir cuánto estropea el reconocedor.
4. **Frase cortada que se completa en la siguiente**
   ("Es que si lo hacemos así..." / "...no llegamos"): cada frase se anota
   por separado. La primera, si no se entiende su función sola, Otro; la
   segunda, por lo que dice.
5. **Frases cortas sin contenido** ("sí", "mm", "ajá"): ver 4.10. Si
   responden a una pregunta de sí/no ("¿Lo tienes?" "Sí.") son
   Respuesta; si responden a una propuesta, Aceptación; si no responden a
   nada, Señal de escucha.
6. **Solapamientos y cambios de hablante mal puestos**: si una frase parece
   de otra persona por una atribución dudosa, se anota igual por lo que
   dice; se marca como dudosa.
7. **Falta de interrogaciones o de puntuación**: no cambia la categoría.
   Se decide por el sentido, no por el signo.
8. **Pocas señales de escucha**: es normal. El reconocedor no las recoge
   bien. No se infiere que alguien "no escucha".

## 9. Práctica: 20 frases con las etiquetas esperadas

Reunión de cuatro personas (Marta, Luis, Irene, Pablo). Anota primero tú,
sin mirar la columna de la derecha; después compara y entiende cada
diferencia.

| id | Quién | Frase | Etiqueta esperada | Responde a | Por qué |
|---|---|---|---|---|---|
| p1 | Marta | Buenos días a todos. | Relación | | saludo |
| p2 | Marta | Tenemos media hora y dos puntos: el plazo y el reparto. | Gestión | | orden y tiempo |
| p3 | Luis | El cliente llamó ayer y dice que necesita la entrega el día quince. | Información | | se podría comprobar |
| p4 | Irene | Me parece demasiado pronto. | Opinión | | valora un dato, sin proponer ni rechazar una propuesta |
| p5 | Marta | ¿Cuánto nos falta para terminar el informe? | Pregunta | | pide información |
| p6 | Pablo | Más o menos una semana. | Respuesta | p5 | contesta la pregunta |
| p7 | Irene | Podríamos dividir el informe en dos entregas. | Propuesta | | idea para el grupo, sin destinatario |
| p8 | Luis | Sí, y así el cliente ve algo antes. | Aceptación | p7 | acepta y añade |
| p9 | Pablo | No, el cliente quiere un solo documento y lo dijo por escrito. | Rechazo 70, Información 30 | p7 | rechaza con razón |
| p10 | Marta | Ajá. | Señal de escucha | | solo muestra atención |
| p11 | Irene | Pablo, mándame el borrador antes de las cinco. | Petición | | pide a Pablo |
| p12 | Pablo | Vale, te lo mando a las cuatro. | Compromiso 70, Aceptación 30 | p11 | se obliga, con hora; el "vale" acepta |
| p13 | Luis | Es que el... bueno, no. | Otro | | frase abandonada |
| p14 | Marta | Irene, no has dicho nada del reparto, ¿cómo lo ves? | Gestión 60, Pregunta 40 | | da la palabra y pide opinión |
| p15 | Irene | Para mí el reparto actual está bien. | Respuesta | p14 | contesta a una pregunta, aunque sea una opinión |
| p16 | Luis | Perdona, Pablo, antes te he interrumpido. | Relación | | disculpa |
| p17 | Marta | Entonces, resumiendo: entrega única el día quince y Pablo manda el borrador a las cuatro. | Gestión | | resume lo decidido |
| p18 | Irene | Habría que revisar también las cifras de ventas. | Propuesta | | idea para el grupo, sin dueño |
| p19 | Pablo | Sí, pero ya no nos da tiempo. | Rechazo | p18 | "sí, pero" con objeción |
| p20 | Luis | Yo me encargo de las cifras. | Compromiso | | quien habla se obliga |

Las 13 categorías aparecen al menos una vez. Si fallas más de cuatro,
repasa los puntos de confusión de las categorías que fallaste. Al
comparar cuenta primero la **principal**; en p9, p12 y p14 un reparto
distinto (60/40 en vez de 70/30) no es un fallo, pero poner una segunda
etiqueta en las demás frases sí lo es.

## 10. Acuerdo entre anotadores y cómo se resuelven las diferencias

- **Objetivo: κ ≥ .60 por categoría** entre dos anotadores entrenados,
  antes de usar una categoría como prueba de nada. Qué es κ y cuánta
  certeza se espera de cada categoría: §11.
- **Cómo se mide.** Dos anotadores marcan las mismas frases por separado,
  sin hablar entre ellos y sin ver nada del sistema. Se calcula κ de la
  **etiqueta principal** por categoría y global, con su intervalo de
  confianza del 95 %. Los pesos se miden aparte: cuánto coinciden los
  repartos de los dos anotadores (solapamiento medio) y lo mismo
  descontando el azar (α de Krippendorff); ver
  [`CONVERSATION.md`](CONVERSATION.md) §6.3.1. Para esto
  hacen falta **al menos 50 ejemplos de cada categoría** (por eso la
  muestra incluye reuniones con tareas y desacuerdos); con menos, la cifra
  baila demasiado.
- **Entrenamiento.** Estudiar esta guía; anotar el ejercicio de §9; hablar
  de cada diferencia; después una prueba de unas 100 frases distintas, en
  la que cada anotador debe llegar a κ ≥ .60 con el consenso de quien
  forma. (Propuesta; el procedimiento no se ha ensayado.)
- **Cómo se resuelven las diferencias.** Cuando termina la anotación
  independiente:
  1. Se listan las frases en las que no coinciden.
  2. Los dos anotadores las discuten **con la regla de la categoría en la
     mano** (§4), no por quién tiene razón.
  3. Si se ponen de acuerdo, esa es la etiqueta de consenso.
  4. Si no, decide **una tercera persona** (quien tenga más experiencia en
     análisis de conversación o evaluación, p. ej. la psicóloga) y deja
     por escrito el motivo.
  5. Cada decisión que resuelve una confusión nueva se añade a esta guía
     como caso resuelto (§4, "se confunde con").
- **Se guardan siempre las dos anotaciones originales** además del
  consenso: κ se calcula con las originales, no con el consenso.
- **Las dudas de la guía se corrigen en la guía**, no en cada cabeza. Si dos
  categorías no se distinguen aunque se afine la guía, se miden y se unen
  (§11).

## 11. κ en una frase y cuánta certeza tiene cada categoría

**Qué es κ.** Si dos personas etiquetan lo mismo, parte del acuerdo es
casualidad: aunque lo hicieran al azar coincidirían a veces. κ (kappa)
**descuenta esa casualidad**: **0 significa que se ponen de acuerdo solo
por azar y 1 que coinciden siempre**. κ × 100 se lee como "el % del
acuerdo posible por encima del azar": κ = .70 es el 70 %. Se usan estas
bandas: **≥ .60 mantener, entre .40 y .60 provisional, por debajo de .40
en riesgo de eliminar o unir**.

**Lo que se espera hoy (prestado).** Estas cifras vienen de otros
estudios, con otros esquemas y otros idiomas (detalle y fuentes en
[`CONVERSATION.md`](CONVERSATION.md) §5.3.1). **No son lo que nosotros
vamos a conseguir**; se sustituyen por las nuestras cuando las midamos.
La banda se decide con el valor **más bajo** del rango.

| # | Categoría | κ prestado | % del acuerdo posible sobre el azar | Veredicto (prestado) |
|---|---|---|---|---|
| 1 | Pregunta | .75–.80 | 75–80 % | mantener |
| 2 | Respuesta | .76 | 76 % | mantener |
| 3 | Información | .47–.71 | 47–71 % | provisional |
| 4 | Opinión | sin cifra | — | provisional por defecto |
| 5 | Propuesta | .70 (junto con Petición) | 70 % | mantener (la separación no está probada) |
| 6 | Petición | .70 (junto con Propuesta) | 70 % | mantener (la separación no está probada) |
| 7 | Aceptación | .42–.71 | 42–71 % | provisional |
| 8 | Rechazo | .42–.71 | 42–71 % | provisional |
| 9 | Compromiso | .15–.78 | 15–78 % | en riesgo |
| 10 | Señal de escucha | .21–.57 | 21–57 % | en riesgo |
| 11 | Gestión de la reunión | .58–.82 | 58–82 % | provisional |
| 12 | Relación | sin cifra fiable | — | provisional por defecto |
| 13 | Otro | .14–.64 | 14–64 % | en riesgo |

Los veredictos de Opinión y Relación son una decisión de este documento
(no hay cifra), no un resultado de la literatura.

**Cuando midamos lo nuestro.** Cada categoría recibe su κ medido con su
intervalo de confianza del 95 %, y la banda se decide con ese κ. **Se
elimina o se une con otra una categoría cuyo intervalo, aun con más
ejemplos, queda entero por debajo de .40** (no se puede demostrar que
supere el azar con margen). Las uniones previstas son Propuesta + Petición,
Información + Opinión y Aceptación + Señal de escucha; el veredicto de la
unión se calcula sobre la categoría unida.

**Cuando haya etiquetador automático.** Dará sus **tres etiquetas más
probables** con la misma forma que una anotación, más un **resto** (lo
que queda hasta 100 para las demás categorías). Cada etiqueta que ponga
irá con una **confianza calibrada en %** (de las etiquetas con 70 %, unas 70 de
cada 100 deben ser correctas). Si la confianza **no supera con claridad la
frecuencia habitual de esa categoría** (su porcentaje entre todas las
frases), la etiqueta se muestra como **"dudosa"** o no se muestra. El
margen exacto se fija con los datos anotados; todavía no está decidido.

## 12. Lo que no se anota

- Emociones, intención, carácter ni nada que no esté en el texto de la
  frase. Las categorías dicen **qué hace** una frase.
- Si alguien "habla mucho" o "poco": no se anota.
- Decisiones, acuerdos, tareas asignadas: las deduce el sistema (§5).
- Competencias: otra guía ([`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md)).
  Una etiqueta de acto no es una valoración de nadie y no se usa para
  evaluar a una persona por sí sola; es una prueba que una persona
  evaluadora puede mirar.
- El tono de voz (el texto no lo tiene) y la ironía: si una frase solo se
  entiende en broma o con tono, anótala por lo que dice y márcala como
  dudosa.

Límites de esta guía: escrita sin haber anotado datos reales; las
confusiones previstas son las de la literatura y las que se nos ocurren a
nosotros; se corregirá con las diferencias reales de la primera ronda.
