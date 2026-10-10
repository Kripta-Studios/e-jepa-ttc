# 03 — Resultados reconciliados y alcance de cada comparación

Las cifras redondeadas de este informe se apoyan en los CSV enlazados. Las [tablas regeneradas](evidence/recomputed/TABLES.md) conservan mayor precisión y el [replay](replay_public.py) comprueba FCWD por consulta. No se mezclan medias de pilotos con resultados de poblaciones completas.

## eAP TRAIN40: overall disponible, pero sobre entrenamiento

Fuente nueva publicada: [METRICS.csv](evidence/train40/METRICS.csv), [RESULT.json](evidence/train40/RESULT.json) y [script original](evidence/train40/calculate.py). Son 88.744 consultas en 40 secuencias, cero updates y cero GPU para este recálculo. Se verificaron índice, fragmentos de predicción, binding de modelos y scorer por SHA-256.

| Método | mean_MiD | overall_MiD | MiDc | MiDs | MiDl | MiDn |
|---|---:|---:|---:|---:|---:|---:|
| H8 semilla 7 | 50,714 | 73,027 | 101,566 | 44,163 | 46,122 | 43,825 |
| H8 semilla 13 | 50,012 | 72,545 | 100,227 | 45,739 | 45,183 | 41,914 |
| H8 semilla 23 | 48,747 | 72,700 | 102,147 | 44,416 | 42,488 | 40,532 |
| H8 mediana de tres | 49,537 | 72,539 | 101,119 | 44,565 | 44,441 | 41,663 |
| Garl event-only | 55,702 | 93,682 | 143,001 | 43,221 | 47,153 | 45,002 |

La reducción relativa de overall de H8 es aproximadamente 22,57%. Garl sigue mejor en banda s. H8 se entrenó con estos datos y no se ha verificado un manifest completo de las secuencias usadas por el checkpoint público Garl. La tabla permite detectar sesgos y contrastar cálculo de métricas; no demuestra superioridad externa. Los tres resultados H8 comparten productores y no son tres semillas independientes del entrenamiento completo.

## Dev32 y FCWD: predicción nativa

Fuente: [MID_OFFICIAL_SCORER.csv](../sota_evidence_20261010/evidence/garl/MID_OFFICIAL_SCORER.csv), [informe Garl](../sota_evidence_20261010/evidence/garl/REPORT.md) e [informe MiD](../sota_evidence_20261010/evidence/garl/MID_REPORT.md).

| Población | Método | mean_MiD | MiDc | MiDs | MiDl |
|---|---|---:|---:|---:|---:|
| Dev32 | H8 | 138,377 | 213,223 | 55,043 | 32,382 |
| Dev32 | Direct | 137,323* | 209,002 | 57,124 | 31,872 |
| Dev32 | Garl event-only | 165,984 | 249,314 | 74,536 | 49,071 |
| Dev32 | Garl RGB+eventos | 136,336 | 197,576 | 65,891 | 61,405 |
| FCWD | H8 | 79,952 | 177,938 | 25,835 | 17,638 |
| FCWD | Direct | 84,458 | 183,276 | 24,634 | 19,280 |
| FCWD | Garl event-only | 61,817 | 95,057 | 36,698 | 34,212 |

\* Direct tiene un MiD inválido en Dev32: `strict_mean_MiD` no disponible. No hay MiDn ni overall en ninguna de estas cohortes. No existe fila válida Garl full FCWD por falta de mapping espacial certificado.

Las diferencias emparejadas H8−comparador, con bootstrap por secuencia, son:

| Comparación | Diferencia mean_MiD | IC95 exploratorio |
|---|---:|---|
| Dev32 frente a Garl eventos | −27,607 | [−43,689; −11,634] |
| Dev32 frente a Garl RGB+eventos | +2,041 | [−14,335; +18,800] |
| FCWD frente a Garl eventos | +18,135 | [+15,166; +23,171] |

Fuente: [MID_PAIRED_SEQUENCE_BOOTSTRAP.csv](../sota_evidence_20261010/evidence/garl/MID_PAIRED_SEQUENCE_BOOTSTRAP.csv). FCWD solo aporta tres grupos. El intervalo Dev32 full incluye cero; no se afirma victoria ni equivalencia estadística.

## Por qué MAE y MiD dan una lectura diferente

Dev32: H8 MAE=1,3766 s; Direct=1,0729 s; Garl eventos≈4,5597 s. Garl full tiene MAE≈94,5096 s por predicciones extremas, aunque su mediana del error absoluto es≈0,6614 s y su MiD es competitivo. Con el mismo clipping ±60 s, la sensibilidad Garl full da MAE≈1,6433 s y Garl eventos≈2,5703 s. Deben mostrarse ambas políticas.

FCWD: H8 MAE=1,6249 s frente a Garl=2,1462 s, pero H8 pierde en MiD. Las consultas de TTC largo perjudican mucho el MAE de Garl; las de TTC crucial perjudican el MiD de H8. Ninguna métrica puede sustituirse por otra para seleccionar el relato favorable.

La cabeza Direct reduce MAE Dev32 un 22,07%, pero empeora FCWD un 9,00%. En las 119 consultas Dev32 con GT positivo≤1 s, H8 falla el criterio de aviso urgente en 117 y Direct en 119. FCWD no tiene ejemplos de ese rango≤1 s en esta población: sus buenos FR no aportan evidencia de comportamiento en él.

## Truncación y estudiante sobre características guardadas

Fuente: [external/METRICS.csv](../sota_evidence_20261010/evidence/streaming/external/METRICS.csv). Se reutilizan características; esto no mide latencia desde eventos crudos.

| Variante | Dev32 mean_MiD | Dev32 MAE s | FCWD mean_MiD | FCWD MAE s |
|---|---:|---:|---:|---:|
| H8 | 138,377 | 1,377 | 79,952 | 1,625 |
| H4 truncado | 137,879 | 1,310 | 83,225 | 1,778 |
| H2 truncado | 141,278 | 1,368 | 87,069 | 1,908 |
| H8 A5 student | 150,040 | 1,676 | 73,736 | 1,453 |
| Student INT8 CPU | 149,641 | 1,658 | 73,829 | 1,473 |

H4 no es una mejora general: gana algo en Dev32 y pierde en FCWD. H2 pierde en ambos mean_MiD frente a H8 y no supera a Garl FCWD. El estudiante cambia el compromiso: mejora FCWD, empeora Dev32 y sigue sin alcanzar Garl FCWD. INT8 no produjo una ventaja de latencia suficiente en el piloto CPU para promoverlo.

El estudiante aprende a imitar características: usa componentes A5, predice componentes C2F/PAIR y reconstruye diferencias temporales. Su validación por secuencias corresponde al ajuste del estudiante; el profesor ya había visto esas secuencias TRAIN40. No es un holdout externo del sistema completo.

## FCWD completo desde eventos crudos: resultado de streaming

Fuente: [fcwd_stream_cpu/METRICS.csv](../sota_evidence_20261010/evidence/streaming/fcwd_stream_cpu/METRICS.csv). 630 consultas, 597 elegibles, tres secuencias, inferencia CPU cronológica, etiquetas incorporadas después de predecir.

| Variante | mean_MiD | MiDc | MAE s | Sesgo s |
|---|---:|---:|---:|---:|
| H8 referencia | 79,9518 | 177,9379 | 1,6249 | −0,8656 |
| Garl eventos | 61,8165 | 95,0568 | 2,1462 | −1,4873 |
| H8 warp | 77,4534 | 167,4851 | 1,6659 | −0,8909 |
| H8 warp+student | 74,5607 | 162,9689 | 1,5058 | −0,6425 |

La referencia CPU pasa la comprobación frente a las predicciones congeladas, con diferencia máxima TTC≈0,000928879 s. No confundir el estudiante sin warp de la tabla anterior (73,736) con warp+student (74,561). La precisión de este replay no acredita la latencia GPU completa de FCWD.

## test12 y afirmaciones pendientes

El [recibo test12](../sota_evidence_20261010/evidence/test12/CAMPAIGN_STATUS.json) tiene estado `PACKAGED_NOT_SUBMITTED`, 6.762 entradas y `official_test_score_available=false`. Hay H8 y Garl event-only completos; no se dispone de etiquetas para calcular su MiD. El ZIP contiene exactamente `submission.json`, con los sample_token requeridos. La comparación full test12 requiere además datos RGB que no estaban disponibles en la campaña documentada.

No se ha demostrado: victoria en test12, superioridad en FCWD MiD, tres réplicas independientes, mejora final V13, tiempo real de un sistema completo en vehículo o SOTA en toda la literatura. Sí se dispone de una base pública para estudiar cada uno de esos límites sin rehacer la conversación.
