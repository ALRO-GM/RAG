# Tutorial: grabar una reunión y leer lo que se dijo

Este tutorial explica cómo usar el transcriptor
(`docs/transcriber/ARCHITECTURE.md`). Tiene dos lectores:

- **Quien arranca el ordenador** (la persona que organiza): sección 1.
- **Quien se sienta a la mesa** (participantes): secciones 2 y 3.

Después vienen lo que puede fallar (3), la privacidad (4) y una lista
de comprobación (5).

En resumen: un teléfono graba la mesa. El ordenador de quien organiza
convierte el audio en una lista de frases. Cada frase tiene la hora, el
nombre de quien habla y el texto, y se puede escuchar con un clic.

Los nombres de botones y pantallas van entre comillas, tal como los
ve la persona en la página.

---

## 1. Quien arranca el ordenador

### Preparar el ordenador (una vez)

El ordenador llega al servidor por un túnel sobre HTTPS (un programa
llamado chisel), con una contraseña propia. Se prepara una sola vez:

```sh
make tunnel-client   # descarga chisel y comprueba su huella SHA-256
make tunnel-secret   # crea la contraseña del túnel (modo 600)
```

`make tunnel-secret` escribe la contraseña en
`~/.config/ragputin/tunnel-credential` y, al lado, el fichero que el
servidor necesita (`tunnel-users.json`). Ese fichero lo sube al
servidor quien lo administra (`make infra-up`, ver
`docs/DEPLOYMENT.md`). No compartas la contraseña: con ella alguien
podría hacerse pasar por tu ordenador ante los teléfonos.

### Arrancar

En la carpeta del proyecto:

```sh
make transcriber-serve TORCH=cuda SERVE_ARGS='--keep-code'
```

`TORCH=cuda` usa la tarjeta gráfica. `SERVE_ARGS` pasa opciones a
`python -m transcriber serve`.

Si la página dice «El ordenador está cargando los modelos. Inténtalo de
nuevo en un minuto.», espera y prueba otra vez.

### Qué escribe la terminal

Cuando el servicio está listo, la terminal muestra, en este orden:

1. `Transcriber listening on http://127.0.0.1:8770` (el puerto es el de
   la configuración; `--port` lo cambia).
2. `Open this link on the phone (it holds the access code):` y debajo
   el **enlace**, con esta forma: `https://rag.peer4u.app/#k=<código>`.
3. El **código QR** de ese mismo enlace, dibujado con bloques.
4. `Ctrl-C to stop.`

Después la terminal escribe una línea cada vez que algo cambia:

| Línea | Qué significa |
|---|---|
| `HH:MM:SS  tunnel  connected` | El túnel con el servidor está abierto y se ha comprobado que `https://rag.peer4u.app/api/health` llega a tu ordenador. Las personas pueden entrar. |
| `HH:MM:SS  tunnel  connecting` | Se está abriendo. Si sigue un motivo entre paréntesis, chisel ya está conectado pero la comprobación aún falla. |
| `HH:MM:SS  tunnel  reconnecting (motivo)` | Se ha cortado y se reintenta solo, con esperas cada vez más largas. Mientras tanto la web dice que el ordenador no está disponible. También sale si chisel sigue conectado pero la comprobación falla 3 veces seguidas (se repite cada 5 s): entonces se reinicia el cliente. |
| `HH:MM:SS  <sesión>  queued` | Una sesión espera su turno (el ordenador transcribe de una en una). |
| `HH:MM:SS  <sesión>  running  recognise 5%` | Está transcribiendo. Se escribe una línea por cada etapa y cada 10 %. Las etapas son `decode`, `recognise`, `voices`, `identify`, `sentences` y `save` (en la web aparecen traducidas: «leyendo el audio», «reconociendo el habla», «separando las voces», «poniendo nombres», «formando frases», «guardando»). |
| `HH:MM:SS  <sesión>  done` | Terminada. |
| `HH:MM:SS  <sesión>  failed  <motivo>` | Ha fallado. En la web aparece «Ha fallado» y se puede pulsar «Transcribir» otra vez. |

Si falta chisel o la contraseña, `serve` se para al momento y dice qué
`make` ejecutar.

Si usas `--no-tunnel` (solo para probar en el mismo ordenador), no hay
líneas de túnel y el enlace es `http://127.0.0.1:<puerto>/#k=<código>`.

### El código de acceso y `--keep-code`

El enlace lleva un **código de acceso**. Sin él, la página se abre pero
no deja hacer nada.

- **Sin `--keep-code`**: el código es nuevo cada vez que arrancas.
  Los QR y enlaces de la vez anterior dejan de valer.
- **Con `--keep-code`**: el código se guarda en
  `data/transcriber/access_code` (solo lo puede leer tu usuario) y se
  reutiliza. Un QR impreso o un enlace guardado siguen valiendo.

Cuándo cambiar el código:

- Si el enlace se ha compartido con quien no debía, o se ha perdido un
  teléfono con la página abierta.
- Para cambiarlo, arranca con `--new-code` (descarta el guardado y
  guarda uno nuevo), o arranca sin `--keep-code`.
  Todos los enlaces anteriores dejan de valer. Para ver el nuevo, mira
  la terminal.

Una persona con un enlace que ya no vale ve «El enlace de acceso ya
no vale». Si eso pasa **mientras graba**, la grabación no se para: el
teléfono guarda el audio (dice cuántos segundos y MB tiene pendientes)
y muestra un recuadro para pegar el enlace nuevo o el código; al
pegarlo sigue enviando desde donde se quedó. El teléfono guarda como
mucho 30 minutos de audio sin enviar; si se llena, deja de grabar y
envía lo que tiene en cuanto puede. Con un código equivocado repetido, la web pide esperar unos
segundos («Demasiados intentos»).

### Parar

Pulsa **Ctrl-C** en la terminal. Escribe `Stopped.` y termina.

- Una transcripción que estaba en marcha se abandona: al arrancar de
  nuevo, esa sesión queda como «Ha fallado» (interrumpida) y se puede
  volver a transcribir. El audio no se pierde.
- Mientras el servicio esté parado, la web dice que el ordenador no
  está disponible.

### Dónde están los datos

Todo está en `data/transcriber/` (git no lo sube):

| Carpeta o archivo | Qué hay |
|---|---|
| `sessions/` | Una carpeta por sesión: el audio y la transcripción. |
| `voiceprints/` | Una huella de voz por persona registrada, con su consentimiento. |
| `published/` | Transcripciones entregadas con `publish`. |
| `access_code` | El código guardado por `--keep-code`. |

La ruta puede cambiarse con la variable `TRANSCRIBER_DATA_DIR`.

### Ver, exportar y borrar desde la terminal

Estos comandos se ejecutan en la carpeta del proyecto con el mismo
Python del proyecto (`python -m transcriber ...`). Pueden usarse con el
servicio en marcha. Todos aceptan `--help`.

| Qué quieres | Comando |
|---|---|
| Listar las sesiones | `python -m transcriber sessions` (una línea por sesión: id, estado, duración, idioma, título) |
| Ver las frases de una sesión | `python -m transcriber show ID` (añade `--lead 3` para ver cada hora 3 s antes; la cabecera lo avisa) |
| Exportar la transcripción | `python -m transcriber export ID --output ARCHIVO.json` (sin `--output` la escribe en la terminal) |
| Entregarla al RAG | `python -m transcriber publish ID` (la escribe en `data/transcriber/published/`) |
| Borrar una sesión | `python -m transcriber sessions delete ID` |
| Listar las voces registradas | `python -m transcriber speakers` |
| Olvidar una voz | `python -m transcriber speakers forget ID` |

Borrar una sesión elimina su audio y su transcripción. Olvidar una voz
elimina su huella y su consentimiento. Ninguna de las dos acciones se
puede deshacer.

Hay otros dos comandos que cargan los modelos **en ese mismo
proceso**: `enrol` (registrar una voz desde un archivo) e `import --wait`
(transcribir un archivo ya grabado). No los uses mientras el servicio
está transcribiendo: solo hay una tarjeta gráfica.

### Si la web dice «El ordenador no está disponible»

La web dice «El ordenador no está disponible» cuando no puede llegar a
tu ordenador. Comprueba, por orden:

1. ¿Sigue abierta la terminal con `make transcriber-serve`? Si no,
   arranca de nuevo.
2. ¿Qué dice la última línea `tunnel`? Si es `reconnecting (...)`,
   el ordenador no llega al servidor: mira la conexión a internet. Se
   reintenta solo. Si el motivo dice `unable to authenticate`, la
   contraseña del túnel no coincide con la del servidor: `make
   tunnel-secret` y que quien administra el servidor haga `make
   infra-up`.
3. Si dice `connected` y la web sigue sin responder, para con Ctrl-C y
   arranca otra vez.
4. En la web, el botón «Reintentar» vuelve a comprobarlo.

Lo que ya estaba guardado no se pierde. Para los fallos del servidor
(no del ordenador), mira `docs/DEPLOYMENT.md`, sección «Troubleshooting».

---

## 2. Quien se sienta a la mesa

### Entrar

1. Abre en el teléfono el **enlace** que te pasa quien organiza, o
   escanea el **código QR** que sale en su pantalla con la cámara.
2. No hay que instalar nada. Sirve un navegador reciente.
3. Si la página dice «Falta el enlace de acceso», abre el enlace
   completo otra vez. La página recuerda el código solo mientras la
   pestaña está abierta: si cierras la pestaña, vuelve a abrir el enlace.

La primera vez, el navegador pedirá permiso para usar el micrófono.
Acepta. Si lo negaste sin querer, actívalo en los ajustes del navegador
para ese sitio.

Arriba hay tres enlaces: «Sesiones», «Grabar» y «Registrar voz».

### Registrar tu voz

**Para qué sirve.** El programa que escribe las palabras no necesita
saber quién eres. El registro sirve para otra cosa: con tu voz
registrada, el sistema puede poner tu **nombre** en cada frase tuya, en
vez de «S1» o «S2». Sin registro todo funciona, pero las personas salen
con esas etiquetas y hay que poner los nombres a mano después.

**Cómo se hace.** Se hace una vez por persona, mejor justo antes de la
reunión.

1. Pulsa «Registrar mi voz» (en «Sesiones») o entra en «Registrar voz».
2. Escribe «Tu nombre». El «Identificador» se rellena solo (minúsculas
   y sin acentos); puedes cambiarlo. No valen identificadores como
   `S1` o `S2`.
3. Lee el aviso de consentimiento y marca la casilla.
   Pulsa «Continuar».
4. Aparece «Lee este texto en voz alta». Pulsa «Grabar».
5. **Di tu nombre primero**: el texto ya empieza con «Me llamo
   (tu nombre)». Léelo entero, en voz alta, con tu tono normal y sin
   parar. Es un texto corto, de unos **30 segundos**.
6. Una cuenta atrás cuenta hasta 30. Cuando pone «Ya hay suficiente»
   se activa «Terminar». Pulsa «Terminar».
7. La página dice «Enviando y analizando tu voz…». Si el ordenador
   está ocupado transcribiendo, puede tardar. Al final: «Listo, (nombre):
   tu voz está registrada.»

**Para que salga bien:**

- Usa **el mismo teléfono** y **la misma sala** que en la reunión, si
  puedes. Así la voz suena igual y el reconocimiento es mejor.
- Busca un sitio sin ruido y donde nadie más hable.
- Si no sale, la página explica por qué: grabación demasiado corta,
  casi sin voz (acerca el teléfono) o más de una voz. Pulsa «Grabar otra
  vez».

**Qué se guarda.** Solo una **huella numérica** de tu voz (una lista de
números, no un sonido) y tu **consentimiento** (cuándo y qué texto del
aviso aceptaste). **No se guarda el audio** del registro: se borra en
cuanto se calcula la huella. Todo queda en el ordenador de quien
organiza.

**Cómo borrarla.** En «Registrar voz», abajo, en «Voces registradas»,
pulsa «Olvidar» junto a tu nombre y confirma. También puedes pedírselo a
quien organiza (`speakers forget`). Se borran la huella y el
consentimiento. Para volver a ser reconocido, tendrías que registrarte
otra vez.

### Grabar la reunión

1. Una persona abre «Nueva grabación» (o «Grabar»).
2. Rellena, si quiere: «Título (opcional)», «Idioma de la reunión»,
   «Quién está presente» (marca a las personas con voz registrada) y
   «Número de personas (opcional)». Si sabes cuántas hablan, escríbelo:
   ayuda a separar las voces. Con voces registradas, ese número es un
   mínimo, no un tope: cada persona registrada cuya voz se reconoce con
   seguridad sale con su nombre aunque así haya más personas que el
   número escrito; el número solo decide en cuántas voces se separa el
   resto.
3. Lee el aviso «Antes de empezar», con todas las personas delante. La
   casilla dice que **las personas presentes lo saben y están de acuerdo
   en que se grabe**. Márcala solo si es verdad.
4. Pulsa «Empezar a grabar».

**Dónde poner el teléfono.**

- En el **centro de la mesa**, a una distancia parecida de todas las
  personas.
- Con la **pantalla encendida** y la página abierta. Si el navegador no
  puede mantener la pantalla encendida, la página lo avisa: entonces no
  la apagues ni cambies de aplicación.
- **Sin tapar el micrófono** (cuidado con fundas, la mano o un papel
  encima).
- Una **conversación a la vez**: cuanta menos gente hable a la vez, mejor.
- No uses el teléfono para otra cosa mientras graba. Una llamada corta
  el micrófono: la página lo avisa y hay que terminar y empezar otra
  grabación.

**Mientras graba** la página muestra el tiempo, un indicador del nivel
del micrófono (si se mueve, te oye) y los botones «Pausa» y «Terminar».

**El indicador de segundos pendientes.** La página manda el audio al
ordenador poco a poco, en trozos de unos segundos.

- «Todo enviado.»: no queda nada por enviar.
- «Pendiente de enviar: 6 s de audio.»: hay audio en el teléfono que
  aún no ha llegado al ordenador. Unos pocos segundos es normal.
- «Sin conexión con el ordenador; se reintenta solo.»: se ha cortado la
  conexión. **No cierres la página.** El audio sigue en la
  página y se envía cuando vuelve la conexión, sin que tengas que
  hacer nada.
- «Hay N s de audio sin enviar. Comprueba la conexión y no cierres
  esta página.»: la cola ya es larga. Mira el wifi o los datos.

Si sales de la página con audio sin enviar, el navegador pregunta si
quieres salir. Di que no.

**Terminar.** Pulsa «Terminar». La página dice «Enviando lo que falta…»
y, cuando todo ha llegado, te lleva a «Sesiones». La transcripción
empieza sola. Si algo falla al final, aparece «Reintentar».

Si cerraste la página antes de terminar, en «Sesiones» la sesión
aparece como «Grabando» con los botones «Continuar grabando» y «Terminar
grabación».

### Después: la transcripción

**Progreso.** En «Sesiones» cada reunión es una tarjeta con su estado:
«Grabando», «Subida, sin transcribir», «En cola», «Transcribiendo»
(con una barra y un porcentaje), «Lista» o «Ha fallado». La lista se
actualiza sola. Una reunión larga tarda un rato, y si hay otra
sesión delante, espera en cola.

En una reunión «Lista», pulsa «Ver transcripción». Si quedó «Subida,
sin transcribir» o «Ha fallado», pulsa «Transcribir».

**La lista de frases.** Cada frase muestra:

- la **hora** en que empieza (minutos y segundos desde el inicio de la
  grabación);
- **quién habla**: el nombre, o «S1», «S2»… si no se reconoció;
- el **texto**;
- «**tras X**»: quién habló justo antes. Sale cuando cambia el que habla;
  ayuda a ver quién responde a quién.
- una marca amarilla cuando la atribución es **dudosa**: «hablante
  dudoso: ¿quizá X?» si la voz de esa frase se parece casi tanto a la
  de otra persona, o «dos voces» si dentro de la frase se oye a otra
  persona (hablar a la vez, o un cambio rápido). Al pasar el ratón se
  ve el «margen de la voz»: cuánto se parece más a quien se le atribuye
  que a la siguiente persona. **No es una probabilidad**; solo sirve
  para saber qué frases revisar. Si es una persona sola, no hay marcas.

**Escuchar una frase.** Pulsa una frase y suena. La frase que suena se
marca, y la lista la sigue (se puede desmarcar «Seguir la frase que
suena»).

- «Segundos de margen antes de cada frase»: al pulsar, el audio
  empieza esos segundos **antes** de la frase, para oír el contexto.
  Por defecto son 3. El número se queda guardado en tu navegador.
- «Reproducir» / «Pausa», «−5 s» y «+5 s» mueven el audio.
- «Velocidad»: de 0.75× a 2×.
- «Mostrar a»: elige una persona para ver solo sus frases. «Todos» las
  muestra todas.
- «Solo frases dudosas»: deja solo las frases marcadas, para revisarlas
  de una en una (escucharlas y, si hace falta, «Cambiar hablante»).

**Corregir.**

- **Una persona desconocida** («S1», «S2»…): abre «Personas de esta
  sesión», escribe el nombre en la casilla «Nombre de S1» y pulsa
  «Poner nombre». Se cambia en todas sus frases. Esto solo pone un
  nombre en esta sesión: **no crea ninguna huella de voz**.
- **Dos etiquetas que son la misma persona**: en «Personas de esta
  sesión», usa «Unir S2 con» y elige a una persona registrada. Pide
  confirmación.
- **Una sola frase mal atribuida**: pulsa «Cambiar hablante» junto a
  la frase y elige quién habla.
- Después de un cambio sale «Cambio guardado.».

Si pulsas «Transcribir» en una sesión «Lista», la transcripción nueva
**sustituye a la actual y a las correcciones hechas a mano**. La
página pide confirmación.

**Descargar.** Pulsa «Descargar transcripción (JSON)». Es un archivo
con las frases, las horas y los hablantes, ya con tus correcciones.
Quien organiza también puede exportarlo desde la terminal (sección 1).

**Borrar.** En «Sesiones», «Borrar» elimina el audio y la transcripción
de esa sesión, para todos. No se puede deshacer. Mientras una sesión se
está transcribiendo no aparece «Borrar»: hay que esperar a que acabe.

**Una grabación hecha con otra app.** En «Sesiones», abajo, «Subir una
grabación hecha con otra app» deja subir un archivo de audio o vídeo.
Tiene su propio aviso de consentimiento: las personas que se oyen deben
saber que se transcribe.

---

## 3. Qué puede fallar

La transcripción **no es perfecta**. Léela como un borrador y
comprueba lo importante con el audio.

| Problema | Qué pasa | Qué hacer |
|---|---|---|
| **Atribuciones cruzadas** | Una frase aparece con el nombre de otra persona, o dos personas con voces parecidas se mezclan. | Pulsa la frase y escucha. Corrígela con «Cambiar hablante», o une etiquetas en «Personas de esta sesión». |
| **Dos personas a la vez** | El sistema no puede representar a dos personas hablando a la vez: **da ese tramo a una de ellas**. La otra pierde esas palabras o aparecen mal escritas. | Evita hablar encima. Si pasó, escucha el tramo. |
| **Ruido y distancia** | Con ruido, eco o una persona lejos del teléfono, las palabras salen peor y la voz se reconoce peor. | Pon el teléfono en el centro y cerca; elige una sala tranquila. |
| **Una voz no registrada** | Esa persona sale como «S1», «S2»… | Ponle nombre después con «Poner nombre», o regístrala antes de la próxima reunión. |
| **Voz cambiada** | Un catarro, otro teléfono u otra sala pueden hacer que la voz suene distinta del registro. | Registra otra vez en las mismas condiciones que la reunión. |
| **Varias etiquetas para una persona** | A veces una misma voz sale como «S1» y «S3». | «Unir … con» en «Personas de esta sesión». |

**Cómo comprobar con el audio.** Pulsa la frase en duda (suena desde
unos segundos antes), escucha, y mira si el nombre y el texto
coinciden. Aumenta «Segundos de margen antes de cada frase» si necesitas
más contexto. Corrige lo que no coincida.

**Qué se ha medido hasta ahora (y qué no).** Muy poco, y en condiciones
fáciles. Las cifras salen de `docs/skills/AUDIO.md`, sección 2:

- Fueron **reuniones sintéticas** montadas pegando frases leídas de
  audiolibros (seis reuniones, cuatro lectores, unos cuatro minutos
  cada una). No eran personas conversando.
- Error en las palabras: **5,4 %** de media (entre 2,2 % y 8,6 %).
- Palabras atribuidas a la persona correcta: **91,5 %** de media (entre
  76,3 % y 99,7 %) cuando se le dice al sistema cuántas personas hay;
  **90,2 %** de media (79,0 % la peor) si lo tiene que estimar.
- Esas cifras son **optimistas**: voz leída, limpia, una persona cada
  vez, con pausas, sin interrupciones. Un teléfono en una mesa, con
  distancia, ruido y gente hablando a la vez, será peor.
- **Con voces registradas** hay un estudio (`docs/transcriber/IDENTIFICATION.md`),
  también con voz leída: 20 reuniones sintéticas de cuatro lectores
  reservadas para medir (no se usaron para elegir nada). Con el método
  que usa ahora el programa (las voces registradas sirven de punto de
  partida para separar las voces), el **91,4 %** de las palabras quedó
  con la persona correcta cuando se dice cuántas personas hay (86,1 %
  con el método anterior), y el **88,2 %** si lo tiene que estimar
  (medido con el umbral 0,92; el programa usa 0,95).
- Cuando una persona **sin registrar** está en la reunión, el 17,5 % de
  sus palabras acabó con el nombre de alguien registrado (40,8 % con el
  método anterior), contando las veces en que se dijo cuántas personas
  había.
- **Aún no se ha medido en una reunión real.** Las mismas reservas de
  arriba valen aquí, y una más: en el estudio cada lector tenía su
  propio micrófono, y eso ayuda a distinguir las voces; alrededor de un
  solo teléfono será más difícil. Trata los nombres como una propuesta
  y compruébalos con el audio.

---

## 4. Privacidad y consentimiento

En palabras sencillas:

- **Qué se graba.** El sonido del micrófono del teléfono: las voces de
  todas las personas presentes. Es información personal. Una voz
  identifica a una persona.
- **Dónde se transcribe.** En el ordenador de quien organiza. No se usa
  un servicio externo para transcribir.
- **Por dónde pasa el audio.** Pasa por el servidor de la página
  (`rag.peer4u.app`) **solo de camino** hacia ese ordenador. El servidor
  no lo guarda.
- **Dónde se guarda.** Solo en el ordenador de quien organiza
  (`data/transcriber/`): el audio, el texto y las huellas de voz.
- **Quién puede abrir el enlace.** **Cualquiera que tenga el enlace o el
  código.** Es como una llave: no lo publiques, no lo pongas en un chat
  grande. Quien lo tenga puede ver la lista de sesiones, escuchar el
  audio y leer las transcripciones, y también grabar y borrar. Si dudas
  de quién lo tiene, quien organiza cambia el código (sección 1).
- **Cómo se borra.**
  - Una **sesión** (audio y texto): botón «Borrar» en «Sesiones», o
    `python -m transcriber sessions delete ID`.
  - Una **voz** (huella y consentimiento): «Olvidar» en «Registrar
    voz», o `python -m transcriber speakers forget ID`.
  - Cualquier persona puede pedírselo a quien organiza en cualquier
    momento.
- **Antes de grabar**, todas las personas presentes deben saberlo y
  estar de acuerdo. La casilla del aviso lo dice, y se marca por todas
  ellas.
- **Valorar a las personas.** Si lo que se escribe se va a usar para
  **evaluar a las personas** (por ejemplo, sus habilidades blandas), eso
  necesita el **acuerdo previo de todas** y decidir antes cómo se
  guardan los datos y cuánto tiempo (ver `docs/skills/AUDIO.md`,
  sección 5). No lo hagas con una grabación sin ese acuerdo.

---

## 5. Lista de comprobación antes de la primera reunión

Quien organiza:

- [ ] Haber hecho una vez `make tunnel-client` y `make tunnel-secret`,
      y que el servidor tenga el fichero del túnel (`make infra-up`).
- [ ] Arrancar con `make transcriber-serve TORCH=cuda SERVE_ARGS='--keep-code'`
      y ver el enlace, el QR y `tunnel  connected`.
- [ ] Abrir el enlace en un teléfono de prueba: sale «Sesiones».
- [ ] Hacer una grabación de prueba de un minuto y comprobar que la
      transcripción llega («Lista») y que al pulsar una frase suena.
- [ ] Borrar esa sesión de prueba.
- [ ] Avisar a todas las personas de qué se graba, dónde se guarda y
      cómo se borra (sección 4), y tener su acuerdo.
- [ ] Que cada persona registre su voz con el mismo teléfono y en la
      misma sala, justo antes.
- [ ] Teléfono cargado (o con cargador), en el centro de la mesa,
      con la pantalla encendida y el micrófono sin tapar.
- [ ] Ordenador sin suspensión automática y con internet.

Participantes:

- [ ] Tengo el enlace o el QR.
- [ ] He aceptado el micrófono en el navegador.
- [ ] He registrado mi voz (o sé que saldré como «S1», «S2»…).
- [ ] Sé que puedo pedir que se borren mi voz y la sesión.
