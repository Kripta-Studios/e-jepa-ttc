# Evaluación externa V13 sobre EvTTC Dev32

Terminada: **1.024 predicciones en 32 secuencias**, con **946 etiquetas TTC**
disponibles. Las 78 consultas sin etiqueta se conservan. RGB está disponible
en las 1.024 consultas; ninguna se descarta por resultado del modelo.

- [RESULTS.md](RESULTS.md): MAE, RTE, sesgo y MiD por banda en filas comunes.
- `DIAGNOSTICS.json`: métricas por secuencia y conteos de signos/saturación.
- `SCORED_PREDICTIONS.csv`: predicciones individuales y comparadores públicos.
- [CONCLUSIONS.md](../fcwd_rgb_registration_20261012/CONCLUSIONS.md): diagnóstico
  conjunto FCWD/Dev32, resultados negativos y limitaciones.

Es transferencia exploratoria sobre una cohorte expuesta. No equivale al test
privado GarlTTC ni autoriza afirmar SOTA. No contiene ejemplos TTC negativos para
completar la fórmula oficial de cuatro bandas.

Inferencia propia: productores y cabezas V13 finales, sin updates, contexto
causal y normalizadores H congelados. Comparadores Garl: predicciones históricas
del checkpoint público con su preprocesado publicado. No se han igualado por
artificio las recetas de entrenamiento ni todas las duraciones de contexto.

El fallo posterior a inferencia era leer celdas GT vacías como `float('')`.
`operational.dev32_missing_truth_score` mantiene el sellado y selecciona para
las métricas solo etiquetas finitas, sin eliminar predicciones fallidas.
`operational.external_transfer_report` valida las filas contra el CSV sellado
y recalcula MiD independientemente.

Ruta de ejecución: `artifacts/rgb_port_20261008/transfer/dev32_score/RESULT.json`.
Reanudar la cola con `python -m operational.external_missing_truth_queue resume
--run artifacts/rgb_port_20261008`.
