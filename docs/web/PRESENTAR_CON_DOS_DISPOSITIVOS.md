# Presentar con dos dispositivos

Un ordenador conectado al proyector por HDMI y una tableta en la mano.
Los dos muestran la misma aplicación. No es vídeo: la aplicación funciona
en **un** dispositivo, el que comparte, y el otro recibe la página tal
cual. Los dos van por internet a través de rag.peer4u.app. Pueden estar en
redes distintas: el wifi del lugar, datos móviles o una mezcla. Cómo
funciona por dentro: [`SCREEN_SHARE.md`](SCREEN_SHARE.md).

**Regla de oro:** el sonido y el micrófono son siempre los del
dispositivo que comparte.

## Elige una de las dos formas

| | A. El ordenador comparte, la tableta maneja | B. La tableta comparte, el ordenador mira |
|---|---|---|
| Comparte | el ordenador del proyector | la tableta |
| Se une | la tableta, con «Envía acciones» activado | el ordenador, solo mira |
| Sonido y micrófono | los del ordenador (los altavoces de la sala) | los de la tableta |
| Cuándo | **recomendada** si el público tiene que oír las voces de los ejemplos | si el sonido no importa o la tableta va al altavoz |

### A. El ordenador comparte y la tableta maneja (recomendada)

1. **Ordenador**: pulsa el botón de las dos pantallas, arriba a la
   derecha («Compartir pantalla»), y luego «Compartir». Pon un nombre
   (por ejemplo «Presentación») y apunta la contraseña de 4 cifras. Pulsa
   «Empezar a compartir».
2. **Tableta**: «Compartir pantalla», «Unirse», el mismo nombre y las 4
   cifras, «Unirse».
3. **Ordenador**: toca el indicador de abajo a la izquierda («Compartiendo
   …») y activa «Envía acciones» en la fila de la tableta. Si quieres que
   nadie toque el ordenador sin querer, desactiva también «Envía acciones»
   en la fila del ordenador. Cierra el panel.
4. **Tableta**: en la barra de arriba elige cómo la ves. «Ver a mi tamaño»
   muestra el diseño de tableta. «Ver como el que comparte» muestra el
   escritorio del proyector, en pequeño. Desde ahora todo se maneja desde
   la tableta: tocar, desplazarse, escribir con el teclado de la pantalla,
   elegir en los desplegables, ordenar y filtrar la tabla de una oferta y
   empezar los ejemplos. Lo que tocas se marca en el proyector con un
   círculo discreto.

### B. La tableta comparte y el ordenador mira

1. **Tableta**: «Compartir pantalla», «Compartir», nombre y contraseña,
   «Empezar a compartir».
2. **Ordenador**: «Compartir pantalla», «Unirse», el mismo nombre y las 4
   cifras. Arriba elige «Ver a mi tamaño» para el diseño de escritorio a
   todo el ancho del proyector, o «Ver como el que comparte» para ver la
   pantalla de la tableta tal cual.
3. El ordenador solo mira: desde él no se puede cambiar nada.

## «Estudio» e «Investigación»

Son páginas aparte de la aplicación, pero la sesión compartida sigue al
ir a ellas y al volver, en los dos sentidos. No hay que volver a poner el
nombre ni la contraseña. El otro dispositivo sigue mostrando la página
anterior un momento (en la simulación del enlace público, entre 0,8 y
1,4 s) y luego muestra la nueva entera. Desde un dispositivo
con «Envía acciones» se puede ir a ellas desde el menú lateral, usar su
índice de contenidos, desplazarse, abrir las tablas de datos, buscar y
filtrar las referencias, y volver a la aplicación con «Inicio». En las dos
páginas está el mismo botón «Compartir pantalla»: se puede empezar,
gestionar o dejar de compartir, y unirse, desde ellas.

- Un enlace a otra web (las referencias de «Investigación», las fuentes
  de «Estudio») tocado en el dispositivo unido se abre **en ese
  dispositivo**, en una pestaña nueva; el proyector no se mueve. Tocado en
  el dispositivo que comparte se abre allí en una pestaña nueva: vuelve a
  la pestaña de la aplicación para seguir.
- En el dispositivo que comparte, **no abras páginas de la aplicación en
  una pestaña nueva** mientras compartes: la pestaña nueva se quedaría con
  la sesión y la antigua dejaría de compartir.

## Antes de empezar (lista de comprobación)

- [ ] Los **dos** dispositivos han entrado en la aplicación antes
      (enlace con `#k=` o contraseña).
- [ ] **Bloqueo automático de pantalla desactivado** en la tableta y en el
      ordenador (Ajustes → Pantalla). La aplicación pide al navegador que
      no apague la pantalla, pero no todos los navegadores lo cumplen.
- [ ] En el dispositivo que comparte, abre «En directo», empieza la
      entrevista de ejemplo y comprueba que se oyen las voces. Así el
      navegador ya tiene permiso para el sonido. Si vas a usar el
      micrófono, acéptalo aquí una vez.
- [ ] Haz la prueba completa: comparte, únete, toca algo desde la tableta
      y míralo en el proyector.
- [ ] Prueba un corte: apaga el wifi de la tableta 5 segundos y vuelve a
      encenderlo. Las dos pantallas deben recuperarse solas.
- [ ] Para el momento clave, evita el minuto **:20** de cada hora. El
      enlace público del servidor se corta unos segundos a esa hora. Se
      recupera solo, pero mejor que no coincida.
- [ ] En el proyector, pon el navegador a pantalla completa (F11).

## Si se va la conexión

- No hagas nada. Cada dispositivo indica «sin conexión, se reconecta
  solo» y vuelve sin pedir la contraseña. El que comparte sigue
  funcionando normalmente. Al volver la conexión, el otro muestra el
  estado actual. En una simulación del enlace público, un corte de 3–5 s
  se recuperó en 4–6 s. No está medido en rag.peer4u.app real.
- Si el dispositivo que comparte se bloquea o cambia de aplicación, el
  otro indica «el dispositivo que comparte se está reconectando». Tienes
  **10 minutos** para volver sin perder la sesión.
- Si se recarga la página del dispositivo unido, vuelve sola a la sesión,
  sin contraseña. Lo mismo el dispositivo que comparte, al recargar o al
  pasar a «Estudio» o «Investigación» y volver.
- Si pasan más de 10 minutos, la sesión se cierra. Vuelve a compartir y
  únete otra vez.
- Sin wifi en el lugar: comparte los datos de un móvil con los dos
  dispositivos. Consumo medido en la simulación: unos 55 KB por minuto en
  una página normal y unos 170 KB por minuto durante la entrevista de
  ejemplo.

## Lo que no se ve en el otro dispositivo

La lista abierta de un desplegable solo se ve en el dispositivo donde se
toca; el otro ve el resultado al elegir. Las ventanitas propias del
navegador (avisos de «¿Seguro que…?», el permiso del micrófono) solo
aparecen en el dispositivo que comparte. Si la tableta maneja (forma A), esos
avisos salen en el **ordenador** y hay que contestarlos allí. Los abren
estas acciones; mejor evitarlas durante la presentación:

- «Cerrar sesión» mientras se graba, y salir de una página que está
  grabando.
- En «Sesiones»: volver a procesar o borrar una sesión.
- En una transcripción: unir a dos personas.
- En «Registrar voz»: retirar el permiso de clonar la voz u olvidar una
  voz.
- En un análisis: cancelar o volver a empezar.
- Empezar una sesión con micrófono por primera vez (el permiso del
  navegador).

Tampoco se comparten nunca el código de acceso, las contraseñas, lo que
guarda cada navegador ni el panel «Compartir pantalla». Nada de lo que
pasa por el servidor se guarda ni se anota.
