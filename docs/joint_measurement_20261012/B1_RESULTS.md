# B1: correspondencias conocidas y TTC posterior

Estado: **NO_CANDIDATE_B1**. No implica promoción del sistema ni SOTA.

Ambos brazos parten del mismo C1 seed7 entrenado sólo en P. En la primera etapa se entrena únicamente el encoder durante2.000 updates: correspondencia de texturas P con transformaciones conocidas frente al objetivo TTC real. Después se entrenan todos los parámetros con TTC real durante10.000 updates en cada brazo, reiniciando Adam/scheduler/RNG en ambos al cambiar de etapa. Total nuevo:24.000 updates científicos. Los12.000 del padre se reutilizan; no se vuelven a contar. El control iguala updates, no FLOPs ni segundos GPU.

El padre C1 ya recibió supervisión TTC; sólo la fase auxiliar de correspondencia prescinde de esas etiquetas. No es preentrenamiento independiente desde cero. No se sintetizan etiquetas TTC ni se simula la física de una cámara. El régimen de correspondencia utiliza el último tensor event, cuatro warps fijos, volúmenes nativos32/16 y ambos sentidos. Las cuatro etapas completas se congelaron antes de cualquier inferencia H de esta comparación.

## Cohortes completas

| Población/modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Signo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H/A5-P2-FROZEN | 18422 | 112.285 | 157.267 | 68.257 | 66.231 | 65.520 | 5.637 | 5.694% |
| H/B1-CORRESPONDENCE-STAGE2-S7 | 18422 | 81.442 | 100.215 | 55.356 | 63.160 | 84.120 | 5.479 | 5.417% |
| H/B1-TTC-CONTROL-STAGE2-S7 | 18422 | 85.977 | 106.713 | 55.284 | 63.716 | 96.633 | 6.114 | 7.247% |
| P/A5-P2-FROZEN | 53890 | 100.874 | 141.813 | 62.952 | 55.295 | 55.522 | 4.751 | 3.333% |
| P/B1-CORRESPONDENCE-STAGE2-S7 | 53890 | 63.267 | 65.625 | 51.966 | 57.738 | 90.913 | 5.905 | 6.003% |
| P/B1-TTC-CONTROL-STAGE2-S7 | 53890 | 58.479 | 56.729 | 48.528 | 57.163 | 98.394 | 6.503 | 7.699% |

## Decisiones separadas

Frente al control con igual número de updates, la fase auxiliar reduce el agregado H un 5.27% y MiDc un 6.09%. Las protecciones emparejadas de bandas, signo y cobertura se cumplen. Es una señal del régimen de aprendizaje de correspondencia, con una sola semilla y sobre H de desarrollo; no demuestra una causa única ni una mejora universal.

Frente a A5-P2, el agregado H se reduce un 27.47% y MiDc un 36.28%, pero MiDn aumenta un 28.39%. El signo pasa la protección registrada; el rechazo procede de la banda negativa. No debe describirse como rechazo por signo. Se conserva NO_CANDIDATE_B1 y no se abre V para buscar un rescate.

El control TTC ajusta mejor el agregado P y transfiere peor a H. La correspondencia conocida mejora durante la fase auxiliar y pierde parte de esa mejora durante el ajuste TTC posterior, aunque conserva ventaja sobre el control. Estas observaciones justifican estudiar la interacción entre representación y objetivo; no prueban por sí solas que congelar el encoder o cambiar la pérdida vaya a resolverla.

Señal de mecanismo H frente al control TTC: **True**. Razón agregado: 0.947264. Protecciones: `{'bands': True, 'coverage': True, 'sign': True}`. Rutas de mejora: `{'aggregate': True, 'crucial': True}`.

La admisión de desarrollo conserva las protecciones originales frente a A5-P2 y añade el coste preparado. No es una relajación de los gates de promoción del sistema. Ambos brazos y sus resultados negativos se conservan.

- B1-CORRESPONDENCE-STAGE2-S7: `{'checks': {'bands': False, 'coverage': True, 'sign': True}, 'cost_route': 'requires separate paired latency evidence', 'cost_route_pass': False, 'cost_scope': 'prepared event tensor resident on GPU through signed TTC output', 'eligible': False, 'overall_ratio': 0.7253169164282731, 'paired_cost_ratio': 0.6005661578762, 'ratios': {'crucial': 0.6372319533751838, 'large': 0.9536288218329025, 'negative': 1.2838894533054643, 'small': 0.8109992209572899}, 'signals': {'aggregate': True, 'crucial': True}, 'status_scope': 'DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION'}`.
- B1-TTC-CONTROL-STAGE2-S7: `{'checks': {'bands': False, 'coverage': True, 'sign': False}, 'cost_route': 'requires separate paired latency evidence', 'cost_route_pass': False, 'cost_scope': 'prepared event tensor resident on GPU through signed TTC output', 'eligible': False, 'overall_ratio': 0.7656966975068711, 'paired_cost_ratio': 0.5962706969092262, 'ratios': {'crucial': 0.6785476642931006, 'large': 0.9620254707262393, 'negative': 1.4748744477263909, 'small': 0.8099409923910157}, 'signals': {'aggregate': True, 'crucial': True}, 'status_scope': 'DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION'}`.

## Variabilidad por grupos

- H/A5-P2-FROZEN: IC95 agregado CORRESPONDENCE − referencia: `[-38.53191875145595, -22.905136832741512]`.
- H/B1-TTC-CONTROL-STAGE2-S7: IC95 agregado CORRESPONDENCE − referencia: `[-7.044695022713018, -1.800553081664231]`.
- P/A5-P2-FROZEN: IC95 agregado CORRESPONDENCE − referencia: `[-44.401989949721234, -30.967636456919372]`.
- P/B1-TTC-CONTROL-STAGE2-S7: IC95 agregado CORRESPONDENCE − referencia: `[3.583671593853023, 6.080951127137035]`.

Son2.000 resamples de grupos completos, no ventanas IID. Esta comparación tiene una semilla y no establece variabilidad entre inicializaciones.

## Correspondencia conocida en P

Mismas24 consultas y escalas0,9/1,1 que el diagnóstico previo. Exactitud estricta de desplazamiento no nulo, promediada sobre consulta/escala definida. El soporte de caja es sólo de diagnóstico. No mide TTC ni movimiento real.

| Etapa/modelo | Soporte | Casos definidos | Exactitud | Null permutado |
|---|---|---:|---:|---:|
| B1-CORRESPONDENCE-STAGE1-S7 | interior | 48 | 0.829594021042188 | 0.05168269380616645 |
| B1-CORRESPONDENCE-STAGE1-S7 | weak_bbox | 48 | 0.8338740592201551 | 0.048756786447484046 |
| B1-CORRESPONDENCE-STAGE2-S7 | interior | 48 | 0.7673611119389534 | 0.04740918924411138 |
| B1-CORRESPONDENCE-STAGE2-S7 | weak_bbox | 48 | 0.775609036286672 | 0.04650740449627241 |
| B1-TTC-CONTROL-STAGE1-S7 | interior | 48 | 0.7114049152781566 | 0.04139957377143825 |
| B1-TTC-CONTROL-STAGE1-S7 | weak_bbox | 48 | 0.7255280874669552 | 0.04052371644259741 |
| B1-TTC-CONTROL-STAGE2-S7 | interior | 48 | 0.7072649511198202 | 0.04233440287255993 |
| B1-TTC-CONTROL-STAGE2-S7 | weak_bbox | 48 | 0.7197671923786402 | 0.040313509467523545 |
| PAIR-CORR-GEOMETRY-S7 | interior | 48 | 0.708333333954215 | 0.04233440292106631 |
| PAIR-CORR-GEOMETRY-S7 | weak_bbox | 48 | 0.7211289672801892 | 0.040987449300397806 |

## Coste y verificación

| Ruta | Mediana preparada ms | p95 ms |
|---|---:|---:|
| A5-P2-FROZEN | 176.046 | 198.237 |
| B1-CORRESPONDENCE-STAGE2-S7 | 105.727 | 119.374 |
| B1-TTC-CONTROL-STAGE2-S7 | 104.971 | 120.239 |

El coste empieza con tensores preparados en GPU y termina en TTC firmado; excluye preparación sensorial, carga y transferencias. No es E2E. Se ejecutan las tres rutas en orden emparejado sobre128 consultas P fijas.

La verificación independiente regenera768.000 IDs de muestra nuevos entre las cuatro etapas e inspecciona los pasos Adam, scheduler, cambio efectivo del encoder y conservación de las cabezas durante la primera etapa. Las cuatro tablas P/H se recalculan con NumPy y ambos endpoints se exportan con pruebas CPU sobre ejemplos P. No se infiere una reproducción sensorial completa a partir de estos exports.

Registro científico SHA-256: `8c9d4c3ee43e244c5777657e4ecd14c9bfdfe7e5128ec40d8de7bfc8b2ef8479`.
Evidencia completa SHA-256: `d98e2e260515e4bcc63bafaece87c4ed94d6b9cf8598848ccfca04fcc974d0d5`.

Brazos retenidos: `B1-CORRESPONDENCE-STAGE2-S7` y `B1-TTC-CONTROL-STAGE2-S7`.
