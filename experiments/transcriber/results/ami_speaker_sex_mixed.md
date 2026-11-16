# Diarización por sexo del hablante (AMI, sólo reuniones mixtas)

- Fecha: 2026-11-16
- Sistema auditado: diarizadores del laboratorio sobre el audio original, salidas de las ejecuciones de #44 (sin volver a ejecutar); collar 0,25 s, emparejamiento uno a uno de etiquetas del DER
- Datos: AMI sdm, test, CC BY 4.0: ami-EN2002b, ami-ES2004b, ami-ES2004c
- Origen de los grupos: sexo en el id de participante de AMI (primera letra M/F), metadato del corpus: https://groups.inf.ed.ac.uk/ami/corpus/participantids.shtml

## Resumen

- `unattributed:none+builtin`: Pasan del umbral: F.
- `unattributed:none+diarizen-wavlm-large-s80-md-v2`: Pasan del umbral: F.
- `unattributed:none+ecapa-spectral`: Pasan del umbral: F.
- `unattributed:none+nemo-clustering`: Pasan del umbral: F.
- `unattributed:none+nemotron-3-diarization`: Pasan del umbral: M.
- `unattributed:none+pyannote-3.1`: Pasan del umbral: F.
- `unattributed:none+pyannote-community-1`: Pasan del umbral: F.
- `unattributed:none+sortformer-streaming-v2`: Pasan del umbral: F.
- `missed:none+builtin`: Pasan del umbral: F.
- `missed:none+diarizen-wavlm-large-s80-md-v2`: Ningún grupo pasa del umbral.
- `missed:none+ecapa-spectral`: Pasan del umbral: F.
- `missed:none+nemo-clustering`: Pasan del umbral: F.
- `missed:none+nemotron-3-diarization`: Ningún grupo pasa del umbral.
- `missed:none+pyannote-3.1`: Pasan del umbral: F.
- `missed:none+pyannote-community-1`: Pasan del umbral: F.
- `missed:none+sortformer-streaming-v2`: Ningún grupo pasa del umbral.

## `unattributed:none+builtin`

Parte del habla puntuada de cada persona que `none+builtin` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,523 | 0,334–0,769 | 1,921 | 1,001–4,453 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,272 | 0,122–0,422 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+diarizen-wavlm-large-s80-md-v2`

Parte del habla puntuada de cada persona que `none+diarizen-wavlm-large-s80-md-v2` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,081 | 0,041–0,155 | 1,508 | 0,657–3,176 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,054 | 0,037–0,071 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+ecapa-spectral`

Parte del habla puntuada de cada persona que `none+ecapa-spectral` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,549 | 0,497–0,623 | 1,560 | 1,237–2,081 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,352 | 0,279–0,426 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+nemo-clustering`

Parte del habla puntuada de cada persona que `none+nemo-clustering` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,571 | 0,400–0,859 | 2,402 | 1,525–3,854 | pasa el umbral |
| M | 4 | 50,000 | 0,238 | 0,181–0,294 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+nemotron-3-diarization`

Parte del habla puntuada de cada persona que `none+nemotron-3-diarization` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,252 | 0,182–0,340 | 1,000 | — | referencia |
| M | 4 | 50,000 | 0,331 | 0,222–0,490 | 1,313 | 0,772–2,159 | pasa el umbral (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+pyannote-3.1`

Parte del habla puntuada de cada persona que `none+pyannote-3.1` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,128 | 0,061–0,229 | 1,316 | 0,544–3,043 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,097 | 0,047–0,147 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+pyannote-community-1`

Parte del habla puntuada de cada persona que `none+pyannote-community-1` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,128 | 0,065–0,224 | 1,311 | 0,565–3,202 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,097 | 0,044–0,151 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `unattributed:none+sortformer-streaming-v2`

Parte del habla puntuada de cada persona que `none+sortformer-streaming-v2` no dio a la etiqueta emparejada con esa persona (perdida o dada a otra).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,408 | 0,223–0,602 | 2,344 | 1,246–3,793 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,174 | 0,139–0,214 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+builtin`

Parte del habla puntuada de cada persona que `none+builtin` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,161 | 0,087–0,254 | 1,719 | 0,782–5,281 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,094 | 0,034–0,154 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+diarizen-wavlm-large-s80-md-v2`

Parte del habla puntuada de cada persona que `none+diarizen-wavlm-large-s80-md-v2` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,022 | 0,013–0,036 | 1,202 | 0,535–3,582 | dentro (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,018 | 0,006–0,031 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+ecapa-spectral`

Parte del habla puntuada de cada persona que `none+ecapa-spectral` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,384 | 0,320–0,436 | 1,816 | 1,415–2,380 | pasa el umbral |
| M | 4 | 50,000 | 0,211 | 0,167–0,247 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+nemo-clustering`

Parte del habla puntuada de cada persona que `none+nemo-clustering` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,395 | 0,324–0,490 | 2,753 | 2,062–3,657 | pasa el umbral |
| M | 4 | 50,000 | 0,143 | 0,116–0,171 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+nemotron-3-diarization`

Parte del habla puntuada de cada persona que `none+nemotron-3-diarization` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,181 | 0,143–0,216 | 1,000 | — | referencia |
| M | 4 | 50,000 | 0,189 | 0,160–0,218 | 1,045 | 0,823–1,362 | dentro (el IC cruza el umbral) |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+pyannote-3.1`

Parte del habla puntuada de cada persona que `none+pyannote-3.1` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,025 | 0,011–0,048 | 1,777 | 0,604–5,319 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,014 | 0,006–0,024 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+pyannote-community-1`

Parte del habla puntuada de cada persona que `none+pyannote-community-1` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,025 | 0,011–0,048 | 1,777 | 0,604–5,319 | pasa el umbral (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,014 | 0,006–0,024 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## `missed:none+sortformer-streaming-v2`

Parte del habla puntuada de cada persona que `none+sortformer-streaming-v2` marcó como silencio (nadie habla).

| Grupo | n | % del total | Media | IC 95 % | Cociente | IC del cociente | Estado |
|---|---|---|---|---|---|---|---|
| F | 4 | 50,000 | 0,150 | 0,123–0,177 | 1,237 | 0,967–1,542 | dentro (el IC cruza el umbral) |
| M | 4 | 50,000 | 0,121 | 0,106–0,141 | 1,000 | — | referencia |

- Cociente = media del grupo / media del grupo de referencia (el de menor media). Umbral 1,25 (nuestro, provisional): por encima se marca.
- Grupos con menos de 3 personas o menos del 2 % del total: se dan n, valor e intervalo, sin cociente.
- Cada persona cuenta una vez (media de sus medidas). IC: bootstrap de percentiles sobre personas, 2000 remuestreos, semilla 0. Con pocas personas por grupo los intervalos son sólo orientativos; un grupo con menos de dos valores distintos no tiene intervalo (—).
- 8 personas.

## Notas

- ES2004b y ES2004c tienen las mismas cuatro personas: cada persona se promedia sobre sus reuniones y cuenta una vez.
- 3 reuniones, 8 personas: los intervalos son anchos y dicen más que los cocientes. No es un contraste de hipótesis.
