# Auditoría de población y campaña D4096

Fecha: 2026-10-29. Esta auditoría usa exclusivamente el split público `train` de
`NAIL-HNU/eAP-dataset` en la revisión
`a814a946e60d37f7242358af5f344542177e50c5`. Conserva el split candidato 24/8/8.
No se ha abierto ninguna etiqueta de test, ni se ha reinterpretado calibración como
auditoría independiente.

## Población completa

Los 40 Parquet de etiquetas coinciden en tamaño y SHA-256 con el inventario fijado.
Los 136 shards RGB requeridos existen y coinciden en tamaño con ese inventario. Se
decodificó además un PNG por secuencia: los 40 miden 1280×720. El chequeo de tamaño
del TAR no equivale a recomputar su SHA-256; por eso el entrenador debe conservar el
`member_sha256` que ya produce el lector para cada frame realmente usado.

| Clase | Raw train24 | Evaluable v2 | Entrenable 640×384 | IDs `(secuencia, track_id)` entrenables |
|---|---:|---:|---:|---:|
| car | 304.200 | 194.772 | 181.954 | 2.235 |
| truck | 107.686 | 38.133 | 34.417 | 612 |
| motorcycle | 22.732 | 11.218 | 10.338 | 682 |
| pedestrian | 7.808 | 3.986 | 3.856 | 255 |
| bus | 5.137 | 2.216 | 1.958 | 35 |
| tricycle | 2.268 | 1.295 | 1.200 | 51 |
| bicycle | 834 | 738 | 655 | 69 |
| other_vehicle | 237 | 61 | 53 | 6 |

`Evaluable v2` procede del GT independiente del entrenador. `Entrenable` reproduce el
perfil actual tras el resize exacto a 640×384, incluida la transformación de centro de
píxel de Pillow. Los IDs no demuestran objetos físicos ni adquisiciones independientes.
La clase `other_vehicle` sigue limitada a seis IDs en train.

## D4096

La selección contiene 4.096 frames únicos: 171 de las primeras 16 secuencias train y
170 de las ocho restantes. Incluye al menos un frame para cada identidad entrenable de
`bicycle`, `tricycle` y `other_vehicle`; el resto se elige con cobertura temporal
uniforme. No admite frames con más de 50 targets entrenables.

| Clase | Anotaciones entrenables D4096 | IDs entrenables D4096 |
|---|---:|---:|
| car | 10.452 | 1.548 |
| truck | 1.994 | 388 |
| motorcycle | 693 | 362 |
| pedestrian | 245 | 130 |
| bus | 110 | 24 |
| tricycle | 97 | 51 |
| bicycle | 162 | 69 |
| other_vehicle | 7 | 6 |

Esto satisface cobertura funcional de ocho clases, pero siete anotaciones de seis IDs
no acreditan robustez de `other_vehicle`.

La selección monitor contiene 1.024 frames, 128 por cada una de las ocho secuencias de
selección, espaciados por tiempo sin consultar etiquetas. El artefacto separado
`full_selection_v1/full_selection_tokens.json` contiene los 23.958 frames completos de
esas secuencias. El monitor sirve para screening; la evaluación completa se exige en
los checkpoints fijos y al cerrar S1. Ambos conjuntos son desarrollo expuesto.

## Memoria y lectura

El auditor procesa una secuencia por vez; no materializa juntas las matrices de
calibración de los 118.247 frames. El `PreparedCache` histórico de 32 entradas necesita
como mínimo unos 94,4 MiB solo para tensores RGB FP32 de 640×384, además de targets y
overhead. La caché de etiquetas de dos secuencias puede releer Parquet con muestreo
aleatorio entre 24 secuencias. Antes del run largo se debe perfilar y, si domina I/O,
precargar solo las etiquetas de los 4.096 tokens o usar bloques por secuencia sin
cambiar el orden estadístico preregistrado.

El perfil label-only confirmó el cuello. Con el orden D4096 barajado por semilla 7,
la caché de dos secuencias tardó 58,94 s para solo 128 frames y 881 anotaciones. Una
caché acotada a las 24 secuencias train procesó los 4.096 frames y 26.223 anotaciones
raw en 12,57 s, con pico Python medido por `tracemalloc` de 47,94 MiB. La campaña debe
fijar `max_label_sequences=24`; esto conserva orden y preparación y evita releer los
mismos Parquet.

El perfil integral final `read_profile_v3_final` repitió los mismos 128 tokens con
`EAPRGBDataset.__getitem__`, apertura TAR, decodificación PNG y
`prepare_sample(640×384, include_velocity=True)`. La traza tiene SHA-256
`4fadf1cc62d5e3c45d27346a0ee5f484abaa2610e89a12fcc6d2ea8fbdfbc82c`.
Los hashes por frame y el agregado
`9d009e0bdbf3f602feee50d59a25fb8d0ffbc93b36f1b9a9dfd71060e5021bce`
son idénticos entre caché 2 y 24, incluidos 881 objetos raw y 441 targets de
velocidad válidos.

| Caché de etiquetas | Primera pasada | Repetición | Pico working set |
|---:|---:|---:|---:|
| 2 secuencias | 22,76 s | 23,56 s | 928,80 MiB |
| 24 secuencias | 12,38 s | 10,00 s | 842,48 MiB |

La caché 24 reduce el tiempo 45,61 % en frío y 57,57 % al repetir la misma traza,
sin diferencia de datos. El resultado procede del snapshot
`read_profile_source_709c60574a8e1588`, agregado
`709c60574a8e15882dee191e013cde13f3770954fabbc34bbd5a257fc90f21d6`.

## Límites

- La agrupación entre adquisiciones relacionadas no está disponible; 24/8/4/4 no se adopta.
- La geometría evaluable sigue siendo el perfil local v2, no un scorer oficial.
- La resolución se comprobó en una muestra por secuencia, no en cada PNG.
- El inventario declara SHA-256 de TAR; esta ejecución verifica tamaño y miembros muestreados.
- TRAIN40 ya estuvo expuesto a las 40 secuencias; este split no crea independencia TTC.

Los artefactos generados quedan bajo `output/campaign_data/`. `receipt.json` distingue
el SHA-256 semántico del JSON canónico y el SHA-256 de los bytes escritos.

La revisión final es `d4096_v4_final`. El manifiesto tiene SHA-256 de archivo
`4af615689bab28016d28881a9339be6c58fe0f35f5d234c0db090eafeff9fb45` y hash
canónico `34adf75a1bfc04e298053b63ddd769718418804a37dffab86fff7773dceae346`.
Se ejecutó desde `source_snapshot_f446055be4fb9de3`, cuyo agregado es
`f446055be4fb9de36db75d43cc1064469e8b45b90f08effdd6309e2b71473554`.
`frame_population.jsonl` contiene 118.247 filas, ocupa 52.108.171 bytes y tiene
SHA-256 `46110db969f98093aac73a997ef7347da91a734e8c0bdf0ac87095fe332c96ab`.
