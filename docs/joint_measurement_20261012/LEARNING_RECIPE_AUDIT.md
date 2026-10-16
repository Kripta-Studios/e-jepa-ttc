# V18: pérdida, ponderación y errores de signo

La receta de aprendizaje es una hipótesis pendiente de contraste causal. El deterioro de signo y TTC negativo aparece también en P y en ambas modalidades; una explicación exclusivamente basada en transferencia P/H resulta insuficiente.

## Alcance y reproducción

Ejecutar `python -B -m operational.joint_measurement.loss_audit` con `PYTHONPATH=src;.`. Se leen siete endpoints P ya congelados (53.890 consultas cada uno); no se abre H/V ni se realiza inferencia de modelos, backward, CUDA o updates. Los hashes de predicciones se contrastan con sus recibos COMPLETE. Se verifica igualdad de poblaciones y una reconstrucción independiente de las masas. La reconstrucción NumPy de la pérdida se contrasta con las funciones de producción.

Fuentes numéricas: [RESULT.json](evidence_loss_recipe_20261016/RESULT.json) y [BAND_CONTRIBUTIONS.csv](evidence_loss_recipe_20261016/BAND_CONTRIBUTIONS.csv). Se conservan también en `artifacts/joint_measurement_20261012/diagnostics/loss_recipe_P`. Son pérdidas del endpoint sobre todo P, reconstruidas en float64 desde salidas guardadas; no son curvas históricas ni mediciones de gradientes de parámetros.

## Qué optimiza la receta

`src/e_jepa_ttc/joint_measurement/loss.py` asigna igual masa a cada grupo y masas nominales 0,5/0,3/0,1/0,1 a las bandas (0,1;3], (3;6], (6;10] y (-10;0). Redistribuye por bandas presentes. En esta población las masas efectivas coinciden con las nominales. El muestreo uniforme con los pesos globales implementados en `operational/joint_measurement/train.py` estima ese objetivo.

Por consulta: `|fase_emitida - fase_GT| / 0,03 + 0,1 * |fase_raw - fase_GT| / 0,03`. No hay término separado de clasificación de signo. La fase sí conserva el signo: un cruce de cero tiene coste, pero no una penalización categórica adicional. La conversión firmada y el límite de 60 segundos también intervienen; cerca de fase cero, diferencias pequeñas pueden producir grandes diferencias en segundos. No se ha aislado su efecto causal.

## Distribución observada en la banda negativa de P

| Modelo | Masa % | Signo incorrecto dentro de negativos % | Fracción de todos los errores de signo % | Contribución a pérdida total % |
|---|---:|---:|---:|---:|
| PAIR-DIRECT-S7 | 10.00 | 38.76 | 96.11 | 22.61 |
| PAIR-CORR-S7 | 10.00 | 41.13 | 97.28 | 23.21 |
| PAIR-CORR-FROZEN-S7 | 10.00 | 36.99 | 96.75 | 22.49 |
| PAIR-CORR-GEOMETRY-S7 | 10.00 | 20.99 | 94.82 | 15.37 |
| PAIR-CORR-WIDE-S7 | 10.00 | 51.34 | 95.35 | 23.30 |
| RGB-PAIR-CORR-FT-S7 | 10.00 | 17.39 | 97.64 | 19.83 |
| RGB-PAIR-CORR-FROZEN-S7 | 10.00 | 15.62 | 92.20 | 19.06 |

Una masa del 10% no implica que el 10% de la pérdida o del gradiente proceda de esa banda. Los errores grandes elevan su contribución. Estas cifras no demuestran que el optimizador ignore el alejamiento ni que una BCE de signo lo vaya a resolver; documentan el equilibrio alcanzado con esta receta.

## Selección y límites de la inferencia

CORR mejora agregado y banda crucial frente a A5-P2, pero empeora bandas 3–6, 6–10, negativa y signo en H. C1 mejora todas las bandas MiD de H frente a CORR libre y reduce sus errores de signo, pero aumenta MAE y cola absoluta. No hay dominancia global. Véase [C1_RESULTS.md](C1_RESULTS.md) y [RGB_RESULTS.md](RGB_RESULTS.md). C1 sí fue utilizado y replicado como productor para fusión; su rechazo standalone no eliminó esa señal.

La arquitectura nativa usa otra pérdida. Comparar sus resultados con PAIR no aísla el encoder ni demuestra que la pérdida nueva cause las regresiones. B1 contrasta aprendizaje de correspondencias frente a TTC adicional con cómputo emparejado; tampoco aísla la hipótesis de pérdida.

## Contraste que falta

Mantener arquitectura, inicialización, población, muestreo y cómputo, y cambiar un único factor: ponderación por bandas o término de signo. Definir coeficientes y decisiones antes de nuevos updates y comparar P/H sin rescates en V. Medir además contribuciones y conflictos de gradiente por banda, con un conjunto P fijado prospectivamente.

Este diagnóstico no registra ni ejecuta esa nueva ablación y no modifica B1 en curso. El programa adaptativo original restringe las cuatro familias y no incluye un sweep de pérdidas; ampliar recursos no convierte por sí solo una búsqueda de coeficientes en una prueba registrada. Se conserva la hipótesis como trabajo científico pendiente, sin reclasificar los endpoints históricos ni alterar los gates.
