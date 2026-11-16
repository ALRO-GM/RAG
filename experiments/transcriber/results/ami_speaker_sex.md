# Diarización por sexo del hablante (AMI)

- Fecha: 2026-11-16
- Sistema auditado: diarizadores del laboratorio sobre el audio original, salidas de las ejecuciones de #44 (sin volver a ejecutar); collar 0,25 s, emparejamiento uno a uno de etiquetas del DER
- Datos: AMI sdm, test, CC BY 4.0: ami-EN2002b, ami-ES2004b, ami-ES2004c, ami-TS3003a
- Origen de los grupos: sexo en el id de participante de AMI (primera letra M/F), metadato del corpus: https://groups.inf.ed.ac.uk/ami/corpus/participantids.shtml

## Resumen

- `unattributed:none+builtin`: Ningún grupo pasa del umbral.
- `unattributed:none+diarizen-wavlm-large-s80-md-v2`: Pasan del umbral: M.
- `unattributed:none+ecapa-spectral`: Ningún grupo pasa del umbral.
- `unattributed:none+nemo-clustering`: Ningún grupo pasa del umbral.
- `unattributed:none+nemotron-3-diarization`: Pasan del umbral: M.
- `unattributed:none+pyannote-3.1`: Pasan del umbral: M.
- `unattributed:none+pyannote-community-1`: Pasan del umbral: M.
- `unattributed:none+sortformer-streaming-v2`: Ningún grupo pasa del umbral.
- `missed:none+builtin`: Pasan del umbral: M.
- `missed:none+diarizen-wavlm-large-s80-md-v2`: Pasan del umbral: M.
- `missed:none+ecapa-spectral`: Ningún grupo pasa del umbral.
- `missed:none+nemo-clustering`: Ningún grupo pasa del umbral.
- `missed:none+nemotron-3-diarization`: Pasan del umbral: M.
- `missed:none+pyannote-3.1`: Pasan del umbral: M.
- `missed:none+pyannote-community-1`: Pasan del umbral: M.
- `missed:none+sortformer-streaming-v2`: Pasan del umbral: M.

## `unattributed:none+builtin`

Parte del habla puntuada de cada persona que `none+builtin` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,523 | 0,334–0,769 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,562 | 0,344–0,768 | 1,074 | 0,582–1,883 | dentro (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+diarizen-wavlm-large-s80-md-v2`

Parte del habla puntuada de cada persona que `none+diarizen-wavlm-large-s80-md-v2` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,081 | 0,041–0,155 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,214 | 0,067–0,449 | 2,633 | 0,680–8,479 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+ecapa-spectral`

Parte del habla puntuada de cada persona que `none+ecapa-spectral` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,549 | 0,497–0,623 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,608 | 0,419–0,806 | 1,107 | 0,745–1,496 | dentro (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+nemo-clustering`

Parte del habla puntuada de cada persona que `none+nemo-clustering` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,571 | 0,400–0,859 | 1,084 | 0,626–2,027 | dentro (el IC cruza el umbral) |
| M | 8 | 66,667 | 0,526 | 0,314–0,758 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+nemotron-3-diarization`

Parte del habla puntuada de cada persona que `none+nemotron-3-diarization` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,252 | 0,182–0,340 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,471 | 0,318–0,657 | 1,872 | 1,149–2,933 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+pyannote-3.1`

Parte del habla puntuada de cada persona que `none+pyannote-3.1` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,128 | 0,061–0,229 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,349 | 0,113–0,650 | 2,725 | 0,785–7,464 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+pyannote-community-1`

Parte del habla puntuada de cada persona que `none+pyannote-community-1` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,128 | 0,065–0,224 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,346 | 0,115–0,652 | 2,711 | 0,788–7,021 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `unattributed:none+sortformer-streaming-v2`

Parte del habla puntuada de cada persona que `none+sortformer-streaming-v2` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,408 | 0,223–0,602 | 1,091 | 0,528–2,187 | dentro (el IC cruza el umbral) |
| M | 8 | 66,667 | 0,374 | 0,215–0,575 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+builtin`

Parte del habla puntuada de cada persona que `none+builtin` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,161 | 0,087–0,254 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,257 | 0,132–0,399 | 1,593 | 0,716–3,333 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+diarizen-wavlm-large-s80-md-v2`

Parte del habla puntuada de cada persona que `none+diarizen-wavlm-large-s80-md-v2` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,022 | 0,013–0,036 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,094 | 0,034–0,172 | 4,320 | 1,385–9,988 | pasa el umbral |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+ecapa-spectral`

Parte del habla puntuada de cada persona que `none+ecapa-spectral` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,384 | 0,320–0,436 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,425 | 0,276–0,582 | 1,108 | 0,704–1,581 | dentro (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+nemo-clustering`

Parte del habla puntuada de cada persona que `none+nemo-clustering` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,395 | 0,324–0,490 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,428 | 0,222–0,639 | 1,083 | 0,562–1,734 | dentro (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+nemotron-3-diarization`

Parte del habla puntuada de cada persona que `none+nemotron-3-diarization` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,181 | 0,143–0,216 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,287 | 0,216–0,362 | 1,587 | 1,141–2,174 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+pyannote-3.1`

Parte del habla puntuada de cada persona que `none+pyannote-3.1` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,025 | 0,011–0,048 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,103 | 0,035–0,188 | 4,033 | 1,217–12,282 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+pyannote-community-1`

Parte del habla puntuada de cada persona que `none+pyannote-community-1` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,025 | 0,011–0,048 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,103 | 0,035–0,188 | 4,033 | 1,217–12,282 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## `missed:none+sortformer-streaming-v2`

Parte del habla puntuada de cada persona que `none+sortformer-streaming-v2` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 33,333 | 0,150 | 0,123–0,177 | 1,000 | — | referencia |
| M | 8 | 66,667 | 0,243 | 0,157–0,335 | 1,620 | 1,003–2,380 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 12 personas.

## Notas

- ES2004b y ES2004c tienen las mismas cuatro personas: cada persona se promedia sobre sus reuniones y cuenta una vez.
- 4 reuniones, 12 personas: los intervalos son anchos y dicen más que los cocientes. No es un contraste de hipótesis.
