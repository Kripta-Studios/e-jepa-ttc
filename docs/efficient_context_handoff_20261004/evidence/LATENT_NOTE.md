# Diagnóstico LATENT posterior, sin entrenamiento

Los nueve endpoints LATENT/ZERO, sus predicciones y el informe histórico se conservan. Se verificaron las identidades TRAIN/OLD, normalizadores y productores en los tres folds. Las descomposiciones son descriptivas post hoc; no son gates de aceptación.

## Distribuciones y productores

Se incluyen 870 filas por coordenada/fold/rol en [LATENT_DISTRIBUTIONS.csv](supplement/LATENT_DISTRIBUTIONS.csv): momentos, extremos, cuantiles 1/99, escala TRAIN y conteos |z|>10. El universo es el de observaciones únicas consumidas por la historia H16 de normalización, sin duplicar queries. En las coordenadas latentes, el máximo |z| observado es 175.06543591309136; hay 0 filas coordenada/rol/fold con std raw <1e-8. Las escalas casi nulas no se corrigen.

[LATENT_PRODUCERS.json](supplement/LATENT_PRODUCERS.json) documenta las familias A5 inner/outer independientes. La dimensión 128 no garantiza coordenadas semánticamente alineadas; una cabeza comparte posiciones numéricas entre esos productores. Se compararon únicamente observaciones actuales ya cacheadas de igual query/ROI entre folds, con selección determinista por identidad y agrupación por par de familias. [LATENT_SIMILARITY.csv](supplement/LATENT_SIMILARITY.csv) contiene 27 comparaciones de estructuras de similitud coseno. Rango de MAE de Gram: [0.0127461757890747, 0.0593924112113349]. No se ajustó un alineador, normalizador ni transformación con OLD_DEV. La heterogeneidad observada no demuestra la causa del fallo LATENT.

## Residuales, todas las secuencias

Se separa sobreestimación/infraestimación de TTC sólo cuando target y predicción son positivos. Los errores de signo se cuentan por separado. Un escape perjudicial es un punto fuera del hull de fases de expertos cuya pérdida supera la mediana actual. El 76,3% histórico fuera del hull no describía exclusivamente sobreestimación.

| Brazo | Secuencia | Queries cruciales | Sobreestimación | Infraestimación | Signo erróneo | Escape perjudicial |
|---|---|---:|---:|---:|---:|---:|
| LATENT-D1-H1-C160@7 | 2cyv0Oedzg | 160 | 96 | 59 | 5 | 38 |
| LATENT-D1-H1-C160@7 | 5ilM1PX2vz | 150 | 87 | 63 | 0 | 46 |
| LATENT-D1-H1-C160@7 | 6h5yRW2LGc | 288 | 141 | 128 | 19 | 74 |
| LATENT-D1-H1-C160@7 | OBneIVg4Cw | 325 | 162 | 144 | 19 | 71 |
| LATENT-D1-H1-C160@7 | OYgB6RGWcq | 95 | 58 | 37 | 0 | 23 |
| LATENT-D1-H1-C160@7 | WbCh1DRerJ | 255 | 105 | 144 | 6 | 86 |
| LATENT-D1-H1-C160@7 | mHGFBekt7X | 126 | 35 | 90 | 1 | 64 |
| LATENT-D1-H1-C160@7 | qGsgzl4Q8B | 145 | 88 | 57 | 0 | 38 |
| LATENT-D1-H1-C160@7 | t79dBxj1WS | 152 | 29 | 123 | 0 | 67 |
| LATENT-D1-H8-C160@7 | 2cyv0Oedzg | 160 | 76 | 79 | 5 | 57 |
| LATENT-D1-H8-C160@7 | 5ilM1PX2vz | 150 | 79 | 71 | 0 | 50 |
| LATENT-D1-H8-C160@7 | 6h5yRW2LGc | 288 | 129 | 148 | 11 | 102 |
| LATENT-D1-H8-C160@7 | OBneIVg4Cw | 325 | 164 | 150 | 11 | 71 |
| LATENT-D1-H8-C160@7 | OYgB6RGWcq | 95 | 53 | 42 | 0 | 24 |
| LATENT-D1-H8-C160@7 | WbCh1DRerJ | 255 | 102 | 152 | 1 | 95 |
| LATENT-D1-H8-C160@7 | mHGFBekt7X | 126 | 17 | 109 | 0 | 76 |
| LATENT-D1-H8-C160@7 | qGsgzl4Q8B | 145 | 86 | 59 | 0 | 16 |
| LATENT-D1-H8-C160@7 | t79dBxj1WS | 152 | 27 | 125 | 0 | 72 |
| LATENT_ZERO-D1-H8-C160@7 | 2cyv0Oedzg | 160 | 133 | 23 | 4 | 17 |
| LATENT_ZERO-D1-H8-C160@7 | 5ilM1PX2vz | 150 | 110 | 40 | 0 | 16 |
| LATENT_ZERO-D1-H8-C160@7 | 6h5yRW2LGc | 288 | 178 | 110 | 0 | 61 |
| LATENT_ZERO-D1-H8-C160@7 | OBneIVg4Cw | 325 | 240 | 80 | 5 | 43 |
| LATENT_ZERO-D1-H8-C160@7 | OYgB6RGWcq | 95 | 42 | 53 | 0 | 33 |
| LATENT_ZERO-D1-H8-C160@7 | WbCh1DRerJ | 255 | 187 | 67 | 1 | 45 |
| LATENT_ZERO-D1-H8-C160@7 | mHGFBekt7X | 126 | 76 | 50 | 0 | 25 |
| LATENT_ZERO-D1-H8-C160@7 | qGsgzl4Q8B | 145 | 90 | 55 | 0 | 20 |
| LATENT_ZERO-D1-H8-C160@7 | t79dBxj1WS | 152 | 106 | 46 | 0 | 30 |

`mHGFBekt7X` permanece incluida; véanse también sus MiD y contribuciones ponderadas en METRICS.csv y RESIDUAL_DECOMPOSITION.csv, sin eliminarla del global. La cobertura q10–q90 del H8 registrado es 0.75948820809533946, inferior al nominal 80%. Se conservan 2 cold starts, 8 historias parciales adicionales, 0 cambios rápidos bajo el umbral phase/s=1 y 1 transición de signo muestreada. No se cambió ese umbral ni se infirió latencia hasta detección.
