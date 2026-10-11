# Qué se consiguió y qué sigue fallando

La transformación permite ejecutar RGB y fusión V13 en FCWD sin inventar una
calibración extrínseca. Se terminaron 630 predicciones; 597 tienen TTC público
dentro de su soporte. En paralelo terminó Dev32: 1.024 predicciones, 946 con
etiqueta. Los checkpoints, normalizadores y entrenamiento permanecieron intactos.

Las cifras se regeneran desde `SCORED_PREDICTIONS.csv` con
`operational.external_transfer_report`. `DIAGNOSTICS.json` conserva resultados
por secuencia, cobertura, sesgo, signos y saturación. `SUMMARY_COMPARISONS.json`
contiene las operaciones de comparación y concentración de error.

## FCWD: mejora al añadir RGB, con límites claros

| Ruta | Cobertura etiquetada | MAE s | RTE % | MiD crucial |
|---|---:|---:|---:|---:|
| V13 E_CTX | 597/597 | 1,967 | 26,340 | 150,093 |
| V13 F_TRUE, con fallback explícito | 597/597 | 1,430 | 18,049 | 102,323 |
| V12 H8 seed 7 | 597/597 | 1,778 | 30,196 | 188,137 |
| V12 H8 seed 13 | 597/597 | 1,597 | 27,407 | 177,123 |
| V12 H8 seed 23 | 597/597 | 1,569 | 27,482 | 176,972 |
| Garl event-only publicado | 597/597 | 2,146 | 26,887 | 95,057 |

F_TRUE reduce MAE un 33,38% frente a Garl event-only. **La comparación añade RGB
a nuestro sistema**: no prueba superioridad frente a Garl multimodal. Garl aún
obtiene mejor MiD crucial. No hay TTC negativos en este conjunto y, por tanto,
no se puede completar el `overall_MiD` oficial de cuatro bandas.

Entre las 472 consultas etiquetadas con registro RGB aceptado, F_TRUE alcanza
0,700 s y Garl event-only 1,352 s. En esas mismas filas R_CTX obtiene 0,938 s.
Esta vista está condicionada a disponibilidad RGB; no debe reemplazar la tabla
de cobertura completa.

El registro rechazó 135 consultas, de las que 125 tienen etiqueta. Esas 125
son el **20,94%** de la población etiquetada, pero aportan **61,28% del error
absoluto total de F_TRUE**. La ruta de fallback E_CTX tiene ahí MAE de 4,184 s y
sesgo de −3,769 s. No se debe bajar el umbral del registro buscando mejorar esta
métrica: la siguiente investigación necesita verificar geometría y soporte de
imagen/eventos independientemente del TTC.

FCWD1 concentra el peor comportamiento: MAE F_TRUE 2,310 s, frente a 1,114 s
en FCWD2 y 0,844 s en FCWD3. El sesgo global F_TRUE es −1,034 s: en promedio
subestima. Esto localiza el problema, pero no demuestra si procede del productor,
la ROI, la normalización o el corrector residual.

## Dev32: mejora multimodal, con dos secuencias problemáticas

Tablas completas: `../dev32_transfer_20261012/RESULTS.md`.

| Ruta | MAE s | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|
| V13 E_CTX | 1,855 | 283,351 | 59,198 | 32,954 |
| V13 R_CTX | 1,258 | 182,444 | 59,129 | 36,222 |
| V13 F_TRUE | 1,368 | 181,582 | 56,217 | 35,636 |
| Garl event-only | 8,048 | 287,536 | 83,127 | 46,893 |
| Garl RGB+eventos | 1,868 | 237,242 | 64,023 | 61,505 |

F_TRUE reduce MAE un 26,75% frente al Garl multimodal reproducido y mejora MiD
en las tres bandas positivas. R_CTX conserva mejor MAE que la fusión. V13 eventos
no domina a V12: por ejemplo, H8 seed 7 obtiene MAE 1,321 s en esta misma cohorte.

Dos secuencias concentran **40,72% del error absoluto F_TRUE en solo 61 de 946
consultas etiquetadas**:

- `CCRs-1-medium-100-overlap-100`: 30 consultas, MAE 12,346 s, cuatro signos
  negativos con TTC real positivo y tres predicciones en el límite de ±60 s.
- `CCRs-1-low-100-overlap-100`: 31 consultas, MAE 5,058 s y ocho signos negativos.

La media global del sesgo F_TRUE es solo +0,068 s, pero oculta errores de signo
y cancelación entre sobreestimación y subestimación. Las secuencias laterales
también presentan sobreestimación: `CCRs-side-medium`, sesgo +3,284 s.

## Interpretación y siguiente diagnóstico

Estos resultados apoyan que la información RGB ayuda a V13 fuera de su dataset
de entrenamiento. No sostienen una afirmación SOTA: son conjuntos ya expuestos,
hay protocolos de contexto/preprocesado diferentes, faltan bandas TTC y no se
ha evaluado el test privado oficial.

El siguiente diagnóstico útil es una traza por consulta de alturas, expansión,
productores A5/C2F, PAIR, mediana de expertos, residual de contexto/fusión y salida
final, acompañada de ROI, timestamps y confianza. Debe contrastarse con controles
TRAIN seleccionados antes del análisis. No se ha cambiado la receta ni se han
seleccionado pesos con estos resultados. V14 permanece independiente.

## Reparación de la evaluación y evidencia

Ambos scorers encontraron celdas TTC vacías en los comparadores. Se corrigió la
decodificación posterior al sellado: vacío→NaN; se conservan las filas en CSV y
solo se excluyen de métricas que necesitan una etiqueta. Son 33 en FCWD y 78 en
Dev32. No son errores de inferencia ni se convierten en cero.

- `operational/fcwd_rgb_score_fix.py`: compatibilidad del scorer FCWD.
- `operational/dev32_missing_truth_score.py`: conserva las 1.024 consultas Dev32.
- `operational/fcwd_rgb_resume.py`: campaña FCWD con el lector corregido.
- `operational/external_missing_truth_queue.py`: reanudación V13 que enruta solo
  el scorer corregido; entrenamientos e inferencia conservan sus comandos.
- `INDEPENDENT_METRIC_AUDIT.json`: recálculo independiente de MiD y conteos de
  fases inválidas, comparado con los resultados de ambos scorers.

Raíz local de campaña FCWD:
`E:/EJEPA_results/fcwd_rgb_calibration_20261012/campaign`.

Raíz local V13: repositorio `e-jepa-ttc-v13-rgb-port`, run
`artifacts/rgb_port_20261008`; predicciones Dev32 en `transfer/dev32`, métricas en
`transfer/dev32_score`, costes en `profiles/ROUTE_COSTS.json`.
