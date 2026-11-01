# PHASE-TTC: diagnóstico temporal de productores y fusión

Este piloto ejecuta los nueve componentes originales congelados sobre dos consultas de desarrollo fijadas antes de la prueba. No utiliza targets, no actualiza pesos y no evalúa la precisión del router adaptado que acompaña a la entrega actual.

## Intervención y controles

Se conservan las imágenes RGB. En eventos se mantienen los canales 0–10 de la representación y se transforma la tasa del canal 11 según la duración real de cada ventana. Todos los relojes se transforman alrededor de la consulta con factores 0,5 y 2; la fase canónica sigue siendo de 100 ms. Se aplica el presupuesto causal de 650 ms sin incorporar observaciones nuevas. Esto transforma representaciones ya preparadas: no reproduce una nueva selección de ventanas de 100 ms sobre un stream crudo acelerado. El redondeo de eventos sobre límites semiabiertos puede cambiar su pertenencia; las pruebas sintéticas conservan ese contraejemplo.

El control a=1 exige igualdad exacta de entradas, salidas de productores y contextos/cabeza, con los mismos lotes de la extracción original. Si cambia el conjunto retenido, se vuelve a ejecutar un control limpio con esas identidades y tamaños de lote. Los rechazos posteriores del productor se registran aparte.

Tiempo medido del piloto, incluida carga/extracción/replay: 14.312 s. Actualizaciones de optimizador: **0**.

## Resultados por consulta

Un cociente a·T_transformado/T_control igual a 1 representa la respuesta temporal ideal para esta intervención. No expresa error respecto al TTC real. Los cocientes de contextos/cabeza solo aíslan la intervención sobre las mismas observaciones admitidas cuando la columna correspondiente lo confirma.

| Consulta | Factor | Retenidas/admitidas | Mismas admitidas que control | TTC F0 (s) | Cocientes F0 / contexto RGB / contexto eventos |
|---|---:|---:|---|---:|---|
| road | 0.5 | 3/3 | True | 3.71044 | 0.673557 / 0.710814 / 0.67379 |
| road | 2.0 | 9/9 | True | 1.64733 | 1.18051 / 1.18532 / 1.11519 |
| slider500 | 0.5 | 4/4 | True | 3.49882 | 0.65464 / 1.23982 / 0.795923 |
| slider500 | 2.0 | 11/11 | True | 1.67381 | 1.21967 / 1.24209 / 1.09634 |

## Respuesta por bloque

Los intervalos siguientes abarcan las observaciones con cocientes definidos. Se conserva el número de cocientes finitos sobre el total. Las salidas individuales, máscaras, soporte, saturación y rechazos permanecen en la evidencia JSON/NPZ.

| Consulta | Factor | Bloque | q_transformado/(a·q_control) | a·T_transformado/T_control | Cociente por par corregido por dt real | Máximo cambio absoluto del log-ratio |
|---|---:|---|---|---|---|---:|
| road | 0.5 | E_A5_MATCHED | 0.990451–0.990451 (1/1) | 1.00964–1.00964 (1/1) | 0.984843–1.00329 (2/2) | 0.000416625 (1/1) |
| road | 0.5 | E_C2F_MATCHED | 1.0027–1.0027 (1/1) | 0.99731–0.99731 (1/1) | 1.00077–1.00351 (2/2) | 0.000109449 (1/1) |
| road | 0.5 | PAIR_E_MATCHED | sin cociente válido (0/0) | 0.539426–0.539426 (1/1) | sin cociente válido (0/0) | no definido/no aplica |
| road | 0.5 | PAIR_R | sin cociente válido (0/0) | 0.501443–0.504778 (2/2) | sin cociente válido (0/0) | no definido/no aplica |
| road | 0.5 | R_A5 | 1–1 (2/2) | 1–1 (2/2) | 1–1 (4/4) | 0 (2/2) |
| road | 0.5 | R_C2F | 1–1 (2/2) | 1–1 (2/2) | 1–1 (4/4) | 0 (2/2) |
| road | 2.0 | E_A5_MATCHED | 0.978369–1.01031 (4/4) | 0.989797–1.02211 (4/4) | 0.964812–1.0197 (8/8) | 0.000935376 (4/4) |
| road | 2.0 | E_C2F_MATCHED | 0.986578–0.998716 (4/4) | 1.00129–1.0136 (4/4) | 0.985699–1.00386 (8/8) | 0.000484191 (4/4) |
| road | 2.0 | PAIR_E_MATCHED | sin cociente válido (0/0) | 1.79703–1.87368 (4/4) | sin cociente válido (0/0) | no definido/no aplica |
| road | 2.0 | PAIR_R | sin cociente válido (0/0) | 1.89671–1.92994 (5/5) | sin cociente válido (0/0) | no definido/no aplica |
| road | 2.0 | R_A5 | 0.99999–1.00001 (5/5) | 0.999988–1.00001 (5/5) | 1–1 (10/10) | 0 (5/5) |
| road | 2.0 | R_C2F | 0.99999–1.00001 (5/5) | 0.999989–1.00001 (5/5) | 1–1 (10/10) | 0 (5/5) |
| slider500 | 0.5 | E_A5_MATCHED | 1.02031–1.02031 (1/1) | 0.980097–0.980097 (1/1) | 1.0186–1.02346 (2/2) | 0.000802182 (1/1) |
| slider500 | 0.5 | E_C2F_MATCHED | 1.01615–1.01615 (1/1) | 0.984103–0.984103 (1/1) | 1.01463–1.01918 (2/2) | 0.000644535 (1/1) |
| slider500 | 0.5 | PAIR_E_MATCHED | sin cociente válido (0/0) | 0.530395–0.530395 (1/1) | sin cociente válido (0/0) | no definido/no aplica |
| slider500 | 0.5 | PAIR_R | sin cociente válido (0/0) | 0.51347–0.513912 (3/3) | sin cociente válido (0/0) | no definido/no aplica |
| slider500 | 0.5 | R_A5 | 1–1 (3/3) | 1–1 (3/3) | 1–1 (6/6) | 0 (3/3) |
| slider500 | 0.5 | R_C2F | 1–1 (3/3) | 1–1 (3/3) | 1–1 (6/6) | 0 (3/3) |
| slider500 | 2.0 | E_A5_MATCHED | 0.978592–0.998548 (4/4) | 1.00145–1.02188 (4/4) | 0.976354–1.00442 (8/8) | 0.000809263 (4/4) |
| slider500 | 2.0 | E_C2F_MATCHED | 0.984622–0.999865 (4/4) | 1.00014–1.01562 (4/4) | 0.973608–1.00331 (8/8) | 0.000701433 (4/4) |
| slider500 | 2.0 | PAIR_E_MATCHED | sin cociente válido (0/0) | 1.76415–1.83906 (4/4) | sin cociente válido (0/0) | no definido/no aplica |
| slider500 | 2.0 | PAIR_R | sin cociente válido (0/0) | 1.79953–1.87139 (7/7) | sin cociente válido (0/0) | no definido/no aplica |
| slider500 | 2.0 | R_A5 | 0.999991–1.00001 (7/7) | 0.99999–1.00001 (7/7) | 1–1 (14/14) | 0 (7/7) |
| slider500 | 2.0 | R_C2F | 0.999991–1.00001 (7/7) | 0.999991–1.00001 (7/7) | 1–1 (14/14) | 0 (7/7) |

## Alcance de la evidencia

El cociente corregido por dt es (q_transformado·dt_transformado)/(q_control·dt_control) por par. Utiliza los deltas ejecutados y separa su cuantización del efecto de cambiar la representación. Las salidas PAIR no disponen de ese diagnóstico analítico por par.

Dos consultas no caracterizan una población ni demuestran el origen del error de Slider. La falta de equivariancia puede localizar una sensibilidad, pero corregirla solo se justifica como mejora tras una evaluación agrupada de precisión y cobertura. Una modificación de productores exige recalcular los estados afectados para comprobar eAP; las cachés del modelo previo no certifican su conservación.

No se promueve un checkpoint ni se afirma SOTA. Este documento se regenera con `scripts/report_phase_producer_timewarp.py`; los tensores y pesos permanecen fuera de Git. Los hashes de los archivos publicados pueden diferir de los originales solo por normalización CRLF→LF; los recibos mantienen los hashes de ejecución.

## Intento anterior conservado

El intento `pilot_v1` terminó con `ValueError('Selected Slider identity differs from original pilot')`. Se conservan su recibo de fallo y su configuración; no se presentan como una ejecución completa. El piloto posterior usa una carpeta nueva.
