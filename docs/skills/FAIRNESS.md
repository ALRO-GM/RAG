# Equidad: lista de comprobación para auditar sesgos

Fecha: 2026-11-15. Ticket #59 (SK-3). Primeras acciones hechas el
2026-11-16 en #86 (SK-13): sección 6.
Esto no es asesoramiento jurídico. Marcas: **[abierto]** = leí la fuente;
**[no verificado]** = no pude abrirla (o sólo vi un resumen) y va de
memoria o de fuente secundaria; **[verificado, pág. N]** = comprobado el
2026-11-15 en una copia oficial aportada por el dueño (AI Act del DOUE,
regla final de la DCWP, NIST AI 100-1 y 600-1, SIOP 2023).

## 1. Resumen para el dueño

Una auditoría de sesgos profesional hace una pregunta simple: **¿los
errores o las puntuaciones del sistema caen más sobre unos grupos de
personas que sobre otros?** Se responde en cuatro pasos que podemos copiar:

1. **Partir los resultados por grupo** (sexo, edad, lengua materna o
   acento...) y mirar cada medida por separado, no sólo la media.
2. **Comparar con una regla fija.** La más usada es la de los cuatro
   quintos (EEOC): si la tasa de un grupo es menos del 80 % de la del
   grupo mejor tratado, es un indicio de impacto adverso. Es una alarma,
   no una sentencia. Nueva York (LL144) pide algo parecido ("cociente de
   impacto"), un auditor independiente y un resumen público.
3. **Guardar la evidencia**: qué datos, qué versión, qué cifras, quién
   decidió qué.
4. **Repetir** cada vez que cambie un modelo, un umbral o los datos.

Tres hechos de nuestro proyecto:

- **Hoy no podemos medir sesgo en personas reales**: no hay sesiones
  reales ni datos de grupo (`docs/ESTADO.md`, fases 9 y 10). Sí podemos
  medir **la parte de audio** con corpus públicos y **dejar preparado**
  todo lo demás para que el día del piloto sea rellenar tablas.
- **Buena parte del riesgo está en cómo se usa**, no en el modelo: el
  informe es un borrador para un evaluador formado (`DESIGN.md` §2).
- Con empleados, el AI Act lo trata como **alto riesgo** (Anexo III 4,
  `RESEARCH.md` §7) [verificado, pág. 127]. Los artículos de datos,
  supervisión humana y precisión (10, 14, 15) recaerían sobre el
  proveedor, es decir, nosotros [verificado, págs. 57–62]. La excepción
  del art. 6(3) no vale si el sistema elabora perfiles de personas
  [verificado, pág. 54].

Primeras acciones sin datos reales: sección 4.

## 2. Qué marcos aplican y por qué

| Marco | Qué aporta | Aplica a nosotros | Fuente |
|---|---|---|---|
| EEOC, 29 CFR 1607.4(D) | Regla 4/5 sobre tasas de selección por grupo; indicio, no prueba; mirar significación y números pequeños | Sí, como umbral de alarma si hay una decisión (selección, ascenso) | [Cornell LII](https://www.law.cornell.edu/cfr/text/29/1607.4) [abierto] |
| NYC Local Law 144 (DCWP) | Auditoría anual independiente; cociente de impacto por sexo, raza/etnia e intersecciones; resumen público; aviso 10 días hábiles antes | Sólo con clientes en Nueva York; es el **modelo de informe** a copiar | [DCWP](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page), [regla](https://rules.cityofnewyork.us/rule/automated-employment-decision-tools-updated/) [verificado en la regla final, §§ 5-300 a 5-304: cociente = tasa del grupo / tasa del grupo más seleccionado, o, si hay nota, % por encima de la mediana de la muestra / el del grupo que más puntúa (pág. 4); el auditor *puede* excluir un grupo con < 2 % de los datos, justificándolo y publicando igualmente su n y su tasa (pág. 7); auditoría de menos de un año antes del uso (pág. 5); resumen publicado antes del uso y mantenido al menos 6 meses tras el último uso (pág. 9); aviso 10 días hábiles antes (págs. 9–10)] |
| EU AI Act (Reg. 2024/1689) | Alto riesgo en RR. HH.: art. 10 (examinar y mitigar sesgos, representatividad; 10(5) permite tratar datos de categorías especiales sólo para detectar y corregir sesgos), 13 (transparencia e instrucciones al usuario), 14 (supervisión humana; 14(4)(b) nombra el sesgo de automatización), 15 (precisión declarada), 26 (deberes del usuario; 26(7): el empleador informa a los representantes y a los trabajadores antes del uso), 27 (FRIA) | Sí, si se usa con trabajadores o candidatos. Art. 5(1)(f): no inferir emociones en el trabajo; el límite a datos biométricos (voz) viene del art. 3(39) y los considerandos 18 y 44, no del texto del 5(1)(f). *Corrección:* la FRIA del art. 27 sólo es obligatoria para organismos públicos, entidades privadas que prestan servicios públicos y los usos del Anexo III 5(b)-(c); una empresa privada que use la herramienta con su plantilla no está obligada por el art. 27 (sí puede necesitar una DPIA) | [verificado en el texto del DOUE: arts. 3(39) pág. 48, 5(1)(f) pág. 51, 6(3) pág. 54, 10 págs. 57–58, 13 pág. 59, 14 págs. 60–61, 15 pág. 61, 26 págs. 67–69, 27 pág. 69, Anexo III pág. 127; considerandos 18 pág. 5 y 44 pág. 12] |
| NIST AI RMF 1.0 y SP 1270 | Govern/Map/Measure/Manage; sesgo sistémico, estadístico-computacional y humano-cognitivo; enfoque sociotécnico. El perfil de IA generativa (NIST AI 600-1, §2.6) avisa de que un LLM puede rendir peor en lenguas distintas del inglés o en ciertos dialectos | Sí, como estructura de proceso (voluntario); el aviso de 600-1 toca a nuestro juez en español (J4) | [AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) [verificado en NIST AI 100-1: tres categorías de sesgo pág. 18, cuatro funciones pág. 20, voluntario pág. 2; NIST AI 600-1 §2.6 pág. 8]; [SP 1270](https://nvlpubs.nist.gov/nistpubs/specialpublications/NIST.SP.1270.pdf) [abierto, sólo categorías de sesgo] |
| SIOP 2023 (evaluaciones con IA) | Mismos estándares que un test tradicional; un algoritmo validado en otro contexto necesita pruebas en el nuevo; micrófonos, cámaras o conexión distintos por grupo son sesgo de medida; quitar diferencias de grupo dentro del algoritmo no arregla ese sesgo y puede ser ilegal en EE. UU. | Sí, como norma profesional para S1-S3 y T1 | [verificado: págs. 6, 13, 19] |
| ISO/IEC TR 24027:2021 | Fuentes de sesgo, sesgo de automatización y de confirmación, técnicas de medida, todo el ciclo | Sí, como vocabulario y guía de medida | [ISO](https://www.iso.org/standard/77607.html) [sólo resumen vía buscador; norma de pago] |
| ISO/IEC 42001:2023 | Sistema de gestión de IA certificable | No ahora; útil si un cliente lo exige | [ISO](https://www.iso.org/standard/81230.html) [no abierto: 403] |
| ICO (Reino Unido) | Equidad = exactitud estadística + no discriminar; documentar compromisos; DPIA viva | Guía práctica aunque no estemos en UK | [ICO](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/how-do-we-ensure-fairness-in-ai/) [abierto] |
| Fairlearn / Aequitas | Medir por grupo: `MetricFrame`, paridad demográfica, odds igualadas; Aequitas: métricas por grupo, disparidad frente a un grupo de referencia, umbral configurable | Sí, para el cálculo cuando haya datos de grupo | [Fairlearn](https://fairlearn.org/main/user_guide/assessment/index.html), [Aequitas](https://dssg.github.io/aequitas/) [abiertos] |
| RGPD arts. 22 y 9 | Sin decisión sólo automática con efecto significativo; voz para identificar = biométrico | Sí; ya en `RESEARCH.md` §7 | no reabierto aquí |
| IEEE 7003, model cards, datasheets | Proceso de consideraciones de sesgo; fichas de modelo y de datos | Model card y datasheet = "evidencia a guardar" | **no verificado** |

Matiz: el AI Act sólo prohíbe inferir emociones desde datos **biométricos**
(voz), no desde texto (`RESEARCH.md` §7). Nuestra regla "puntuar sólo
texto" es también una medida de equidad: acento y timbre no entran en la
nota.

## 3. Lista de comprobación por etapa

Estado hoy: **hecho** / **no aplica aún** (necesita personas reales o un
piloto) / **pendiente**. Cuándo: **P** = antes del piloto, **R** = en cada
versión (modelo, umbral o rúbrica), **D** = en cada conjunto de datos.
Los grupos se toman de lo que la persona declare con consentimiento; el
sistema nunca los infiere. Los umbrales marcados "nuestro" son provisionales.

### 3.1 Transcripción

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| T1 | El error de palabras (WER) no depende del grupo | WER por grupo en corpus con grupo y texto de referencia | Cociente WER grupo / mejor grupo ≤ 1,25 (nuestro), con intervalo de confianza | Tabla por grupo, modelo, corpus | R, D | pendiente: sólo hay WER de habla leída sintética |
| T2 | Acentos y lenguas del uso real están en los datos de prueba | Inventario de lenguas, acentos y micrófonos frente al uso previsto (art. 10(3)-(4)) | Cobertura declarada, huecos listados | Datasheet del conjunto de prueba | P, D | pendiente |
| T3 | Los errores que cambian el sentido (negaciones, nombres) no caen más en un grupo | Revisión manual de muestra estratificada | Tasa de errores relevantes por grupo | Muestra, revisor, fecha | P | no aplica aún |
| T4 | Habla ininteligible: se marca, no se descarta en silencio | Regla escrita y prueba | % marcado por grupo | Regla en `DESIGN.md` | P | pendiente: Whisper no da confianza |

### 3.2 Diarización y asignación de persona

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| D1 | Error de diarización (DER) por grupo y nº de hablantes | AMI y CALLHOME (`LAB.md` §13), por sexo si el corpus lo da | DER por grupo; cociente ≤ 1,25 (nuestro) | Tabla del laboratorio | R, D | hecho en AMI por sexo (§6.1), sin conclusión: 12 personas y el signo cambia con la reunión TS3003a; CALLHOME no da sexo |
| D2 | Una voz no se atribuye siempre a otra persona (se puntúa a quien no es) | Confusión de hablante por grupo | % de frases mal atribuidas por grupo | Matriz de confusión | R | pendiente |
| D3 | Quien habla poco o solapa no queda mal representado | Error de tiempo de habla por persona | Error por persona | Tabla | R | pendiente: el solape no se detecta; ya hay error por hablante de referencia (`score_speakers`, §6.1) |
| D4 | Huella de voz con consentimiento y retirada | Flujo de inscripción y `speakers forget` | Borrado funcional | Consentimiento versionado | P | hecho (`ESTADO.md` §4.11) |
| D5 | Voz no inscrita no se identifica | Prueba con voz desconocida | Sale "desconocido" | Prueba automática | R | pendiente: confirmar |

### 3.3 Etiquetado y juez

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| J1 | La etiqueta no cambia por rasgos irrelevantes | Prueba de intercambio: misma frase con otro nombre, género o registro | Cambios de etiqueta < 2 % (nuestro) | Pares y resultados por modelo | R | hecho, primera medida (§6.2): 60 frases sintéticas; con 15 pares por variante no se puede afirmar < 2 % |
| J2 | Sesgo de orden, longitud y verbosidad del juez | Orden aleatorio (T-50); correlación nota-longitud | Variación por orden; correlación | Informe del juez | R | pendiente: T-50 |
| J3 | Citas literales existen y son de la persona | Anclaje (`DESIGN.md` §2) | Tasa de rechazo; por grupo cuando haya | Contadores `proposed/rejected/unsupported` | R | hecho (global) |
| J4 | Mismo comportamiento, misma etiqueta entre español e inglés | Frases equivalentes | Acuerdo entre variantes | Pares y resultados | R | pendiente |
| J5 | Juez por API: qué texto sale y a quién | Revisar el envío y el contrato | Sin identificadores; decisión del dueño | Decisión escrita | P | pendiente (`AUDIO.md` §5) |
| J6 | Tasa de rechazo del juez por grupo | Contadores por grupo | Diferencia entre grupos | Tabla | P | no aplica aún |

### 3.4 Puntuación

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| S1 | Cociente de impacto: tasa de personas sobre un corte, por grupo | Fórmula tipo LL144 y 4/5, con Fairlearn o Aequitas; grupos pequeños sólo descritos | Cociente ≥ 0,80 con intervalo | Tabla, tamaños, fecha, versión | P, R | no aplica aún; plantilla y cálculo listos (§6.3) |
| S2 | Brecha de nota entre grupos frente a la brecha de evaluadores humanos | Comparar con el consenso humano | Brecha sistema − humana, con intervalo | Tabla | P | no aplica aún (T-36) |
| S3 | Acuerdo con humanos igual para todos | `skills evaluate` por grupo | Diferencia de acuerdo con intervalo | Informe | P, R | no aplica aún |
| S4 | Abstención justa ("evidencia insuficiente") | Tabla evaluado/no evaluado por grupo | Diferencia de tasa | Tabla | P | no aplica aún |
| S5 | Quien habla poco no sale como deficiente | Regla `min_participant_words` | Revisar umbral con datos | Regla y prueba | R | hecho (umbral provisional) |
| S6 | Constantes y cortes documentados | Registro con justificación | Lista completa | `DESIGN.md` §9 | R | hecho |
| S7 | Fiabilidad por persona suficiente | ~4 situaciones por persona para .70 (`ESTADO.md` fase 9) | Nº de situaciones, fiabilidad | Informe | P | no aplica aún |

### 3.5 Informe

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| I1 | Citas, tiempos, límites y rótulo de borrador | Revisar la plantilla | Todo ítem con cita; aviso visible | Plantilla versionada | R | hecho (`report.py`) |
| I2 | Sin lenguaje de rasgos ni emociones | Lista de palabras y prueba sobre plantillas y salida del juez | 0 apariciones | Prueba automática | R | pendiente |
| I3 | Incertidumbre y datos que faltan visibles | "Evidencia insuficiente"; confianza de transcripción | Presente en todo informe | Muestra de informes | R | pendiente: lo primero hecho, lo segundo no |
| I4 | Model card del juez y datasheets de rúbrica y datos de prueba | Fichas cortas: uso, límites, datos, métricas por grupo | Ficha completa | `docs/skills/` | P, R | hecho (§6.4); sin métricas por grupo, que no existen |
| I5 | Explicación comprensible de la nota | Cita + indicador + regla | Revisión por un evaluador | Prueba de usuario | P | no aplica aún |

### 3.6 Uso humano

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| H1 | Evaluador formado que no delega la decisión (RGPD 22, AI Act 14) | Formación documentada | Registro | Lista de evaluadores | P | no aplica aún |
| H2 | Sesgo de automatización (AI Act art. 14(4)(b), pág. 60) | El humano anota su juicio antes de ver el del sistema | % de cambio hacia el sistema | Registro | P | no aplica aún |
| H3 | Derecho a ver, corregir y recurrir | Proceso y canal de contacto | Caso de prueba resuelto | Texto y registro | P | pendiente |
| H4 | Aviso a participantes antes del uso (NYC: 10 días hábiles, § 5-304; UE: el empleador informa a representantes y trabajadores, art. 26(7)) | Texto y fecha de entrega | Aviso entregado | Registro versionado | P | pendiente: hay consentimiento de grabación, no aviso de evaluación |
| H5 | Usos prohibidos: decisión única sobre empleo; emociones desde la voz | Política de uso aceptable | Cláusula | Política | P | pendiente |

### 3.7 Gobierno

| # | Qué comprobar | Cómo | Métrica y umbral | Evidencia | Cuándo | Estado |
|---|---|---|---|---|---|---|
| G1 | Clasificación de riesgo decidida con razones | Nota jurídica | Decisión escrita | Documento fechado | P | pendiente (T-48) |
| G2 | FRIA (art. 27, como plantilla voluntaria: sólo obliga a organismos públicos y servicios públicos) y DPIA para el piloto | Plantilla; alinear ambas | Seis elementos del art. 27(1)(a)-(f) cubiertos [verificado, pág. 69] | Documento | P | pendiente |
| G3 | Registro de riesgos: sesgos conocidos y medidas | Tabla viva (Map/Measure/Manage) | Cada riesgo con dueño y fecha | `docs/` | R | pendiente (acción 5) |
| G4 | Auditoría independiente y resumen público | Un tercero ejecuta S1 con datos reales | Informe tipo LL144 | Informe | antes de uso comercial | no aplica aún |
| G5 | Gatillos de reauditoría | Cambio de modelo, umbral, rúbrica o población; anual como mínimo | Calendario | Registro de versiones | R | pendiente |
| G6 | Registro de qué versión puntuó a quién | Modelo, rúbrica y constantes en cada scorecard | Metadatos en cada informe | Scorecards | R | pendiente: confirmar qué guarda hoy |

Recuento de las 38 filas (2026-11-16, tras #86): 8 hechas, 11 no aplican
aún, 19 pendientes. Antes de #86: 5, 11 y 22.
Casi todo lo "hecho" son reglas del pipeline; ninguna medida de equidad
sobre personas está hecha.

## 4. Primeras acciones sin datos de participantes

1. **WER y DER por grupo con corpus públicos (T1, D1, D2).** Elegir un
   corpus con etiquetas de sexo/lengua y texto de referencia (p. ej.
   Common Voice o FLEURS para WER, [no verificado]; AMI y CALLHOME, ya
   importados, para DER) y sacar la tabla por grupo del laboratorio. Es
   lo único sobre voces reales que se puede medir hoy. Respetar licencias
   (CALLHOME: sólo uso interno, `LAB.md` §13).
2. **Prueba de intercambio para el juez (J1, J4).** 50-100 frases con
   variantes de nombre, género y registro; contar cambios de etiqueta. No
   necesita personas.
3. **Plantilla de informe de equidad tipo LL144 + 4/5 (S1-S4).** Tablas
   (grupo, n, tasa, cociente, intervalo) y regla para grupos pequeños,
   probada con datos simulados, para rellenar el día del piloto.
4. **Model card del juez y datasheets de rúbrica y de datos de prueba
   (I4, T2).** Una página cada una.
5. **Registro de riesgos y hoja de clasificación de riesgo (G1, G3).** El
   dueño la firma antes de grabar a nadie.

## 6. Primeras acciones hechas (#86, 2026-11-16)

Sin datos de participantes. La acción 5 (registro de riesgos y hoja de
clasificación) sigue pendiente.

### 6.1 Diarización por sexo del hablante en AMI (D1)

- **Sexo en el corpus.** AMI lo da en el id de participante:
  `[MF][IET][EDO]nnn`, primera letra M o F (página oficial del corpus,
  <https://groups.inf.ed.ac.uk/ami/corpus/participantids.shtml>
  [abierto 2026-11-16]). Es la etiqueta del corpus, binaria; el sistema
  no la infiere. CALLHOME (`A`/`B`) no da sexo: no se mide en él.
- **Datos.** Las 4 reuniones AMI de #44 (`ami-EN2002b`, `ES2004b`,
  `ES2004c`, `TS3003a`, test, sdm), salidas ya guardadas del
  laboratorio (no se volvió a ejecutar ningún diarizador). 12 personas
  distintas: 4 mujeres, 8 hombres; ES2004b y c son las mismas 4
  personas (se promedian y cuentan una vez); TS3003a son 4 hombres, dos
  de los cuales hablan 26 y 40 s.
- **Medida.** `score_speakers` (`transcriber/domain/lab_reference.py`)
  reparte el error por hablante de referencia con el mismo collar
  (0,25 s) y el mismo emparejamiento uno a uno del DER: *no atribuido* =
  parte del habla de la persona que la pista no da a la etiqueta
  emparejada con ella (perdida o dada a otra); *perdido* = parte en que
  la pista no marca a nadie. La falsa alarma no es de nadie y no se
  reparte. Comprobado: en las 95 pistas, la suma de lo no atribuido
  coincide con perdido + confusión del DER guardado (diferencia < 0,05 s).
- **Cálculo.** `experiments/transcriber/speaker_groups.py` →
  `experiments/skills/fairness_audit.py`; resultados en
  `experiments/transcriber/results/ami_speaker_sex*.{json,md}` (también
  la parte perdida). Media por grupo sobre personas, IC bootstrap de
  percentiles (2.000 remuestreos), cociente frente al grupo con menos
  error, umbral 1,25, grupos de al menos 3 personas.

Parte no atribuida, audio original (media F / media M; cociente del
grupo con más error frente al otro, IC 95 %):

| Diarizador | 4 reuniones, F 4 / M 8 | Cociente | Sólo reuniones mixtas, F 4 / M 4 | Cociente |
|---|---|---|---|---|
| builtin | 0,523 / 0,562 | M 1,07 (0,58–1,88) | 0,523 / 0,272 | F 1,92 (1,00–4,45) |
| pyannote-3.1 | 0,128 / 0,349 | M 2,73 (0,79–7,46) | 0,128 / 0,097 | F 1,32 (0,54–3,04) |
| pyannote-community-1 | 0,128 / 0,346 | M 2,71 (0,79–7,02) | 0,128 / 0,097 | F 1,31 (0,57–3,20) |
| diarizen-wavlm-large-s80-md-v2 | 0,081 / 0,214 | M 2,63 (0,68–8,48) | 0,081 / 0,054 | F 1,51 (0,66–3,18) |
| nemotron-3-diarization | 0,252 / 0,471 | M 1,87 (1,15–2,93) | 0,252 / 0,331 | M 1,31 (0,77–2,16) |
| sortformer-streaming-v2 | 0,408 / 0,374 | F 1,09 (0,53–2,19) | 0,408 / 0,174 | F 2,34 (1,25–3,79) |
| ecapa-spectral | 0,549 / 0,608 | M 1,11 (0,75–1,50) | 0,549 / 0,352 | F 1,56 (1,24–2,08) |
| nemo-clustering | 0,571 / 0,526 | F 1,08 (0,63–2,03) | 0,571 / 0,238 | F 2,40 (1,53–3,85) |

**Lectura.** No hay conclusión sobre el sexo. Con las 4 reuniones 6 de
8 diarizadores fallan más con los hombres; quitando TS3003a (sólo
hombres, dos de ellos casi callados) el signo se invierte en 5 de 8 y
7 de 8 fallan más con las mujeres. La diferencia puede deberse a la
reunión o al tiempo de habla de cada persona
tanto como al sexo: con 4 reuniones y 4 personas por grupo no se
pueden separar. Con 4 personas el bootstrap tiene muy pocas muestras distintas:
los intervalos son orientativos. Para medir D1 de verdad hace falta un
corpus con muchas más personas por grupo (p. ej. todo el test de AMI, 16
grabaciones, o CALLHOME con metadatos de sexo), no disponible sin
descargar más (#44 importó 4).

### 6.2 Prueba de intercambio (J1)

`examples/skills/fairness/swap_sentences.json`: 15 frases base de
reunión, en español, inventadas, cada una en 4 variantes: `f` (nombre y
concordancia femeninos, «Lucía… segura»), `m` (masculinos, «Pablo…
seguro»), `origin` (nombre de otro origen con concordancia femenina:
Fátima, Aisha, Mei, Ngozi, Svetlana) y `colloquial` (la de `f` en
registro coloquial). 60 frases, una frase de contexto cada una. Cuenta
(`experiments/skills/swap_test.py`, `swap_changes`): bases cuya etiqueta
de acto cambia frente a la variante `f` (15 pares por variante, IC de
Wilson). Resultados en `experiments/skills/results/swap_test/`.

| Modelo | `m` | `origin` | `colloquial` | Acierto con la etiqueta pretendida (f/m/origin/coloquial, de 15) |
|---|---|---|---|---|
| Reglas (`rules-1`) | 0 (0–0,20) | 0 (0–0,20) | 8 (0,30–0,75) | 10 / 10 / 10 / 5 |
| Codificador XLM-R (`xlmr-base-ami-mrda-es`, por defecto en el paso d) | 1 (0,01–0,30) | 0 (0–0,20) | 1 (0,01–0,30) | 10 / 11 / 10 / 10 |
| Qwen3-8B 4 bits (juez local) | 0 (0–0,20) | 0 (0–0,20) | 2 (0,04–0,38) | 13 / 13 / 13 / 14 |
| Claude Code `opus` (60 llamadas, suscripción) | 0 (0–0,20) | 0 (0–0,20) | 0 (0–0,20) | 11 / 11 / 11 / 11 |

- El único cambio por género: el codificador etiqueta «Tranquila,
  Lucía, yo me encargo y lo llamo mañana…» como `relacion` y la misma
  frase con «Tranquilo, Pablo» como `compromiso` (b07). El
  codificador cambia la distribución de probabilidad poco con el
  género y el origen (variación total media 0,032 y 0,026, máxima 0,165
  y 0,084) y más con el registro (media 0,134, máxima 0,537).
- El registro coloquial es lo que más cambia las etiquetas, sobre todo
  en las reglas (8 de 15), que dependen de fórmulas y signos.
- Límites: 15 pares por variante sólo permiten decir que la tasa de
  cambio está por debajo de ~20 % cuando es 0; el umbral de 2 % no se
  puede comprobar con este tamaño. Frases escritas por nosotros, sin
  revisión humana de las etiquetas pretendidas. Sólo actos de diálogo:
  la confirmación de etiquetas de la rúbrica (paso f) y la evaluación
  (paso g) no se probaron. Una sola pasada por modelo (sin medir la
  variación de una pasada a otra de los jueces). J4 (español frente a
  inglés) no se hizo.

### 6.3 Plantilla de informe de equidad (S1-S4)

`src/domain/skills/fairness.py` (sin E/S) calcula las tablas:

- **Tasa** (`rate_audit`): por grupo n, seleccionados, tasa con IC de
  Wilson, cociente frente al grupo con mejor tasa (cociente de impacto
  de LL144; 4/5 de la EEOC) con IC de Katz (logaritmo; 0,5 por celda si
  una está vacía). `above_median` da la tasa de puntuación de LL144
  (proporción del grupo sobre la mediana de toda la muestra).
- **Media** (`mean_audit`): para errores u otras medidas por persona;
  cada persona cuenta una vez; IC bootstrap sobre personas, semilla fija.
- **Reglas**: dirección (más es mejor o peor), umbral (0,80 para tasas,
  1,25 para errores), grupos con menos de `min_units` (5) personas o
  menos del 2 % sólo se describen (n, valor, IC) y no reciben cociente
  ni son referencia; se marca si el IC del cociente cruza el umbral.
  El umbral por omisión sigue la dirección y se rechaza uno del lado
  equivocado de 1. Si la referencia vale 0 no hay cociente: un grupo
  peor que ella queda como «sin cociente: la referencia es 0», con la
  diferencia absoluta, y nunca cuenta como dentro del umbral. En las
  tablas de medias, un grupo con menos de dos valores distintos no
  tiene intervalo (—); con pocas personas los intervalos son
  orientativos.

`src/application/skills/fairness_report.py` llena la plantilla a
partir de un documento `fairness.results.v1` (contexto: sistema, datos,
origen de los grupos, fecha, notas; y por medida las observaciones por
persona) y la escribe en Markdown, en español o inglés:
`experiments/skills/fairness_audit.py results.json informe.md`. Probado
con datos simulados (`tests/unit/test_skills_fairness*.py`); el primer
uso real es §6.1.

### 6.4 Fichas (I4)

`MODEL_CARD_JUDGES.md` (jueces: Claude Code y local, cuándo se usa cada
uno, cifras sintéticas medidas), `DATASHEET_RUBRIC.md` (rúbrica `core`
0.1.0, en y es) y `DATASHEET_DIALOGUE_ACTS.md` (las 13 categorías, el
conjunto español sintético y los corpus públicos). Ninguna tiene
métricas por grupo, porque no existen; dicen qué no se sabe (licencias
de los modelos Qwen no anotadas, demografía de los hablantes no
registrada). T2 sigue pendiente: falta el inventario de acentos y
micrófonos de los datos de prueba del audio.

## 5. Límites

- Los umbrales de este documento (1,25 en WER/DER, 2 % en el
  intercambio) son **nuestros y provisionales**. Sólo el 4/5 viene de una
  fuente oficial, y ésta lo define como indicio, no como prueba.
- Con pocas personas por grupo no hay potencia estadística: hace falta
  intervalo de confianza y a veces sólo se puede describir.
- Medir sesgo exige datos de grupo, que son sensibles (RGPD art. 9) y
  necesitan base legal y voluntariedad. El AI Act art. 10(5) permite al
  proveedor tratarlos excepcionalmente sólo para detectar y corregir
  sesgos, con garantías [verificado, pág. 58].
- Equidad no es validez: una nota igual de mala para todos pasa la prueba
  de equidad y no predice nada. La validez sigue en `RESEARCH.md` y en la
  fase 10 de `ESTADO.md`.
- Varias medidas de equidad no se pueden cumplir a la vez (paridad frente
  a odds igualadas); hay que elegir y documentar por qué (ICO).
- No se pudieron abrir: ISO/IEC 42001, ISO/IEC TR 24027 (sólo resumen) e
  IEEE 7003 (normas de pago). El texto oficial del AI Act, NIST AI 100-1 y
  la regla LL144 completa se comprobaron el 2026-11-15.

## Referencias

- 29 CFR 1607.4(D): <https://www.law.cornell.edu/cfr/text/29/1607.4>
- NYC DCWP: <https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page>,
  regla <https://rules.cityofnewyork.us/rule/automated-employment-decision-tools-updated/>
- AI Act arts. 10 y 27 (copia no oficial): <https://artificialintelligenceact.eu/article/10/>,
  <https://artificialintelligenceact.eu/article/27/>; oficial (DOUE L,
  12.7.2024; páginas citadas = página n/144 del PDF)
  <https://eur-lex.europa.eu/eli/reg/2024/1689/oj>
- NYC DCWP, Notice of Adoption of Final Rule (§§ 5-300 a 5-304).
- NIST AI 100-1 (AI RMF 1.0, 2023) y NIST AI 600-1 (perfil de IA generativa, 2024).
- SIOP (2023). Considerations and Recommendations for the Validation and
  Use of AI-Based Assessments for Employee Selection.
- NIST AI RMF: <https://www.nist.gov/itl/ai-risk-management-framework>;
  SP 1270: <https://nvlpubs.nist.gov/nistpubs/specialpublications/NIST.SP.1270.pdf>
- ISO/IEC TR 24027: <https://www.iso.org/standard/77607.html>;
  ISO/IEC 42001: <https://www.iso.org/standard/81230.html> (no abierto)
- ICO: <https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/how-do-we-ensure-fairness-in-ai/>
- Fairlearn: <https://fairlearn.org/main/user_guide/assessment/index.html>;
  Aequitas: <https://dssg.github.io/aequitas/>
