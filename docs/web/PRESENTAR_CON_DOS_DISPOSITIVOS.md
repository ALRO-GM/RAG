# Presentar con dos dispositivos

Una tableta en la mano y un ordenador conectado al proyector por HDMI.
La tableta maneja la aplicación y el proyector muestra lo mismo. No es
vídeo: la aplicación funciona en la tableta y el proyector recibe la
página tal cual. Los dos van por internet, a través de rag.peer4u.app. No
hace falta que estén en la misma red: el wifi del lugar, datos móviles o
una mezcla sirven igual. Cómo funciona por dentro: [`SCREEN_SHARE.md`](SCREEN_SHARE.md).

## Antes de la presentación (5 minutos)

1. En los **dos** dispositivos, abre la aplicación y entra (enlace con
   `#k=` o contraseña).
2. En la **tableta**, abre «En directo», empieza la entrevista de ejemplo
   y escucha una voz. Así el navegador de la tableta ya tiene permiso para
   el sonido. Si vas a usar el micrófono, acéptalo aquí una vez.
3. Conecta el altavoz o el sonido de la sala a la **tableta**. Las voces
   suenan en el dispositivo que comparte.
4. En el proyector, pon el navegador a pantalla completa (F11).

## Empezar

1. **Tableta**: pulsa el botón de las dos pantallas, arriba a la
   derecha («Compartir pantalla»), y luego «Compartir». Deja el nombre
   («Presentación») o pon otro. Apunta la contraseña de 4 cifras que
   aparece. Pulsa «Empezar a compartir». Abajo a la izquierda verás
   «Compartiendo «Presentación» · dispositivos unidos: 0».
2. **Ordenador del proyector**: pulsa «Compartir pantalla», luego
   «Unirse». Escribe el mismo nombre y las 4 cifras y pulsa «Unirse». La
   pantalla pasa a mostrar la tableta. Arriba pone «Viendo «Presentación»
   · solo ves».
3. En la barra de arriba del proyector eliges cómo se ve:
   - «Ver a mi tamaño» (vista de este dispositivo): la aplicación con el
     diseño de escritorio, a todo el ancho del proyector. Recomendado.
   - «Ver como el que comparte»: la pantalla de la tableta tal cual,
     escalada para que quepa.

Por defecto solo la tableta envía acciones. El proyector solo mira y no
se puede cambiar nada desde él: así nadie toca algo sin querer. Las
pulsaciones del presentador se ven en el proyector como un círculo
discreto.

## Durante la presentación

- Todo lo que hagas en la tableta aparece en el proyector: páginas,
  menús, tablas, formularios, la entrevista y la llamada de ejemplo.
- Si quieres manejar la aplicación desde el ordenador, toca el indicador
  de la tableta y activa «Envía acciones» en la fila del ordenador. Si los
  dos envían acciones, se aplican en el orden en que llegan.
- Para terminar: en la tableta, toca el indicador y luego «Dejar de
  compartir». El proyector vuelve a su propia aplicación.

## Si se va la conexión

- No hagas nada. Cada dispositivo indica «sin conexión, se reconecta
  solo» y se vuelve a conectar sin pedir la contraseña. La tableta sigue
  funcionando normalmente. Al volver la conexión, el proyector muestra el
  estado actual. Un corte de 3–5 s se recupera en 4–6 s (medido).
- El túnel del servidor se corta unos segundos una vez por hora. Es lo
  mismo: se recupera solo.
- Si el corte dura más de 2 minutos, la sesión compartida se cierra.
  Vuelve a compartir con una contraseña nueva y únete otra vez.
- Si se recarga la página del proyector, vuelve sola a la sesión, sin
  contraseña, mientras siga abierta.
- Sin wifi en el lugar: comparte los datos del móvil con los dos
  dispositivos. Consumo medido: unos 55 KB por minuto en una página
  normal y unos 170 KB por minuto durante la entrevista de ejemplo.

## Lo que no se comparte

El micrófono y el sonido se quedan en la tableta. El código de acceso,
las contraseñas y lo que guarda cada navegador no se envían. El panel
«Compartir pantalla» tampoco aparece en el proyector. Nada de lo que
pasa por el servidor se guarda ni se anota.
