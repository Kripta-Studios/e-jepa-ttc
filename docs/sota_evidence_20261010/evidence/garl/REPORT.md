# Comparación con los checkpoints GarlTTC descargados

Resultados recalculados de predicciones congeladas con SHA-256 verificado. Dev32 es exploratorio y ya expuesto; FCWD tiene solo tres secuencias. No es una puntuación oficial de test12 ni demuestra SOTA.

El [informe MiD](MID_REPORT.md) añade el scorer oficial, FR, bandas TTC, casos extremos y bootstrap pareado por secuencia. El orden por MAE no equivale al orden por MiD.

| Datos | Método | n | Cobertura | MAE (s) | Mediana AE (s) | p95 AE (s) |
|---|---|---:|---:|---:|---:|---:|
| DEV32 | H8_median3 | 3640 | 1.000 | 1.377 | 0.705 | 4.036 |
| DEV32 | Direct_median3 | 3640 | 1.000 | 1.073 | 0.651 | 3.296 |
| DEV32 | public_Garl_event_lhr | 3640 | 1.000 | 4.560 | 0.958 | 9.484 |
| DEV32 | public_Garl_rgb_event_full | 3640 | 1.000 | 94.510 | 0.661 | 5.037 |
| FCWD | H8_median3 | 597 | 1.000 | 1.625 | 0.601 | 6.706 |
| FCWD | Direct_median3 | 597 | 1.000 | 1.771 | 0.591 | 7.114 |
| FCWD | public_Garl_event_lhr | 597 | 1.000 | 2.146 | 0.602 | 8.831 |
| FCWD | public_Garl_rgb_event_full | 597 | 0.000 | N/D | N/D | N/D |

Sensibilidad separada: mismo límite ±60 s para todos (Dev32).

| Método | MAE (s) |
|---|---:|
| H8_median3 | 1.377 |
| Direct_median3 | 1.073 |
| public_Garl_event_lhr | 2.570 |
| public_Garl_rgb_event_full | 1.643 |

Coste medido anteriormente: ruta cronológica con caché cruda de 512 MiB, 40 consultas y 200 observaciones por sistema tras excluir warmups.

| Sistema | CPU media (ms) | GPU media (ms) | Total medio (ms) | p95 total (ms) |
|---|---:|---:|---:|---:|
| h8_fast_three | 989.1 | 645.3 | 1635.2 | 3354.3 |
| garl_event_only | 343.4 | 49.7 | 393.8 | 817.9 |
| garl_full | 1549.4 | 151.1 | 1701.3 | 2011.5 |

Aviso urgente: GT positivo ≤1 s y predicción positiva ≤1 s. Una salida no finita o de signo incorrecto cuenta como fallo.

| Método | Urgentes Dev32 | Fallos | Tasa de fallo (%) |
|---|---:|---:|---:|
| H8_median3 | 119 | 117 | 98.32 |
| Direct_median3 | 119 | 119 | 100.00 |
| public_Garl_event_lhr | 119 | 119 | 100.00 |
| public_Garl_rgb_event_full | 119 | 116 | 97.48 |

Condiciones y lectura:

- H8 usa tres cabezas sobre productores compartidos y más contexto temporal. Direct es una variante experimental; H8 sigue siendo el candidato fijado.
- Garl conserva su conversión nativa de alturas a TTC, sin clipping. H8 tiene un límite nativo ±60 s. La sensibilidad con ±60 s para todos está separada en los CSV.
- No se eliminan errores extremos ni se descartan predicciones faltantes. Las métricas completas quedan N/D si falta alguna predicción de la cohorte elegible.
- ERROR_BANDS.csv muestra sesgo y subestimación por rango. LARGEST_ERRORS.csv permite identificar las consultas extremas. El bootstrap usa secuencias enteras.
- FCWD RGB+eventos no es puntuable: falta una transformación geométrica verificada.
- LATENCY.csv procede de las mediciones locales anteriores, sin warmups, separando las rutas de caché. H8 incluye sus tres cabezas. No equivale al FPS del paper.
- Test12 contiene etiquetas privadas: comparar entre predicciones no mide exactitud. La puntuación oficial sigue pendiente de CodaBench.
