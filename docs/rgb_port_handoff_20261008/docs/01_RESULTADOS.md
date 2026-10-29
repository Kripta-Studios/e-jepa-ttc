# Lectura de la evidencia nueva

## Estado real

`docs/sota_campaign_20261008/NEXT_DECISION.json` dice COMPLETE para la entrega de evidencia, no para la reproducción oficial ni para R1. La contabilidad de esta campaña registra cero actualizaciones de optimizador. A5/C2F/PAIR/H8 siguen siendo los productores TRAIN40, sin que la ampliación de evaluación haya entrenado una versión RGB propia.

La instantánea R1 incluida en este ZIP conserva 499/768 pares; el mensaje del usuario comunica 505/768. Son observaciones distintas, no dos resultados científicos contradictorios. Consultar el recibo vivo antes de ejecutar. No marcar R1 completo por deducción.

## Dev32 ampliado

3.947 consultas, 3.640 targets elegibles, 32 secuencias ya observadas. H8 seed7/13/23: MAE 1,361284 / 1,394970 / 1,456492 s; RTE 53,700867 / 54,521035 / 59,145012 %. Garl EO: MAE 4,559712, RTE 161,120694 %. Garl full: MAE 94,509602, mediana 0,661416 s. H8 tiene medianas 0,699246–0,721849 s.

El caso full `CCRm-low-0-overlap-0:110` predice 335.544,3125 s frente a 1,813350 s. Aporta 97,5372 % del error absoluto total. Las alturas 112,66834259 y 112,66837311 hacen casi cero el denominador. FP64 sobre esas alturas no recupera una expansión correcta: cambia el número extremo, no la información estimada.

El análisis post hoc común ±60 s YA ESTÁ HECHO: Garl full MAE 1,643287 s y RTE 59,687684 %; EO 2,570345 s y 96,194769 %. Conserva el resultado nativo. La sensibilidad no es una reproducción oficial ni reutiliza los IC nativos como IC de predicciones recortadas.

## FCWD

630 consultas, 597 elegibles, tres secuencias. H8 MAE 1,777725 / 1,597361 / 1,569221 frente a 2,146173 s. Garl gana en estimación puntual de RTE: 26,887265 % frente a 30,196258 / 27,406551 / 27,482217 %. Los IC de las diferencias de RTE atraviesan cero. Tres secuencias no son 597 escenas independientes.

Nuevo recálculo descriptivo, SIN entrenar ni cambiar población:

| TTC verdadero | N | MAE H8 seed7/13/23 | MAE Garl EO |
|---|---:|---|---:|
| 1–3 s | 201 | 0,759307 / 0,690811 / 0,695349 | 0,327243 |
| 3–6 s | 189 | 0,561227 / 0,540048 / 0,558826 | 0,953610 |
| 6–10 s | 108 | 1,065599 / 0,943160 / 0,989312 | 1,897139 |
| >10 s | 99 | 6,944693 / 6,170112 / 5,905008 | 8,387534 |

En 1–3 s los sesgos H8 son positivos, 0,633–0,728 s; Garl 0,166 s. Esto localiza un problema relevante de sobreestimación, no demuestra su causa. Las tres H8 reducen MAE por secuencia, pero no están dominando todas las franjas. El error relativo pondera un mismo error absoluto más cuando TTC es pequeño.

En Dev32 hay 119 targets <=1 s: H8 MAE 2,345 / 2,705 / 3,415 s. Es una prioridad diagnóstica, no un permiso para entrenar con sus etiquetas. Todos los targets elegibles de ambas cohortes son positivos; estos conjuntos no acreditan buen comportamiento frente a objetivos negativos/alejamiento.

## Coste

Medianas de inferencia: H8 con tres cabezas 129,295 ms, Garl EO 7,596 ms, full 38,574 ms. Son ocho consultas, repeticiones sobre host Windows WDDM, modelos/contextos distintos y caché caliente. No dividir 129 ms entre tres: encoders y contexto se comparten entre cabezas. Medir una cabeza canónica aparte.

El E2E de EO incorpora preparación compartida H8; no usarlo como coste mínimo propio de Garl. El E2E incluye más overhead que la suma simple de las medianas de preparación/GPU. R1 responde otra pregunta y no es un resultado completo todavía.

## Baselines geométricas

CMax inicial 0/32 fue un adaptador con warp incorrecto. Su corrección local logró 15/32; STRTTC local 11/32. No llamar fracasos del paper a fallos del adaptador, ni ordenar MAE condicionales de poblaciones de éxito distintas.

## Alcance de mi verificación

ZIP: 2.680.967 bytes; SHA256 `b24b1f68be8e2846805d2b7f3f7cab2bd385864faac7cc662a1ade47940f3917`; 362 miembros, 357 entradas del manifiesto principal verificadas. El inventario de evidencia enumera 346 archivos, una cuenta distinta. Dos copias runtime adicionales coinciden con sus copias ya manifestadas.

Reejecuté los tres scorers: los CSV METHOD_METRICS, PAIRED, PER_SEQUENCE y SCORED_ROWS coinciden byte a byte. METADATA y REPORT JSON conservan los mismos valores, pero en Linux tienen LF en vez de CRLF; el verificador estricto falla por ese detalle y los hashes JSON derivados. Es portabilidad de serialización, no diferencia numérica ni motivo para descartar la campaña.

No ejecuté los 205 tests del repositorio, no repetí inferencia de sus pesos y no terminé el clon (DNS). El SHA de blob Git coincide para README, NEXT_DECISION y la sensibilidad común con los tres archivos recuperados por GitHub conectado. Ver recibos y scripts en evidence/.
