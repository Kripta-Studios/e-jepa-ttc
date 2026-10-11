# EvTTC Dev32: resultados finales V13

1024 consultas; 1024 con RGB disponible. Cero updates.

Resultados exploratorios sobre conjuntos expuestos. No son una clasificación oficial ni evidencia SOTA.

MiD es la media agrupada de errores de fase por banda; no es la media por secuencia de V13. Se muestran los casos de fase no válida y no se imputa una banda ausente.

## all_queries

| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_H1_MATCHED | 946/946 | 2.575 | 118.811 | 0.763 | 305.354 | 73.613 | 33.794 | — | 0/0/0/0 |
| E_CTX_MATCHED | 946/946 | 1.855 | 86.765 | 0.740 | 283.351 | 59.198 | 32.954 | — | 0/0/0/0 |
| R_H1 | 946/946 | 1.288 | 52.009 | 0.334 | 197.843 | 60.264 | 33.790 | — | 0/0/0/0 |
| R_CTX | 946/946 | 1.258 | 43.771 | 0.240 | 182.444 | 59.129 | 36.222 | — | 0/0/0/0 |
| F_TRUE | 946/946 | 1.368 | 50.680 | 0.068 | 181.582 | 56.217 | 35.636 | — | 0/0/0/0 |
| F_ZERO | 946/946 | 1.937 | 93.248 | 0.953 | 299.456 | 60.309 | 31.035 | — | 0/0/0/0 |
| H8_seed7 | 946/946 | 1.321 | 62.351 | 0.577 | 253.657 | 54.064 | 34.055 | — | 0/0/0/0 |
| H8_seed13 | 946/946 | 1.353 | 62.923 | 0.534 | 245.320 | 55.627 | 32.914 | — | 0/0/0/0 |
| H8_seed23 | 946/946 | 1.409 | 68.866 | 0.729 | 250.098 | 56.160 | 31.475 | — | 0/0/0/0 |
| public_Garl_event_lhr | 946/946 | 8.048 | 293.472 | 5.476 | 287.536 | 83.127 | 46.893 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 946/946 | 1.868 | 76.769 | -0.255 | 237.242 | 64.023 | 61.505 | — | 0/0/0/0 |

## same_RGB_available_queries

| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_H1_MATCHED | 946/946 | 2.575 | 118.811 | 0.763 | 305.354 | 73.613 | 33.794 | — | 0/0/0/0 |
| E_CTX_MATCHED | 946/946 | 1.855 | 86.765 | 0.740 | 283.351 | 59.198 | 32.954 | — | 0/0/0/0 |
| R_H1 | 946/946 | 1.288 | 52.009 | 0.334 | 197.843 | 60.264 | 33.790 | — | 0/0/0/0 |
| R_CTX | 946/946 | 1.258 | 43.771 | 0.240 | 182.444 | 59.129 | 36.222 | — | 0/0/0/0 |
| F_TRUE | 946/946 | 1.368 | 50.680 | 0.068 | 181.582 | 56.217 | 35.636 | — | 0/0/0/0 |
| F_ZERO | 946/946 | 1.937 | 93.248 | 0.953 | 299.456 | 60.309 | 31.035 | — | 0/0/0/0 |
| H8_seed7 | 946/946 | 1.321 | 62.351 | 0.577 | 253.657 | 54.064 | 34.055 | — | 0/0/0/0 |
| H8_seed13 | 946/946 | 1.353 | 62.923 | 0.534 | 245.320 | 55.627 | 32.914 | — | 0/0/0/0 |
| H8_seed23 | 946/946 | 1.409 | 68.866 | 0.729 | 250.098 | 56.160 | 31.475 | — | 0/0/0/0 |
| public_Garl_event_lhr | 946/946 | 8.048 | 293.472 | 5.476 | 287.536 | 83.127 | 46.893 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 946/946 | 1.868 | 76.769 | -0.255 | 237.242 | 64.023 | 61.505 | — | 0/0/0/0 |

## RGB_unavailable_queries

| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_H1_MATCHED | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| E_CTX_MATCHED | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| R_H1 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| R_CTX | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| F_TRUE | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| F_ZERO | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| H8_seed7 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| H8_seed13 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| H8_seed23 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| public_Garl_event_lhr | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |

## Diagnóstico por secuencia

Las métricas, sesgos, signos y predicciones en el límite de ±60 s están en `DIAGNOSTICS.json`.
El subconjunto RGB se fija por entradas y calidad del registro, nunca por el error TTC. Comparar modelos dentro de una misma tabla; las cabezas RGB sin fallback tienen menor cobertura en la tabla completa.

Regeneración: `python -m operational.external_transfer_report --seal SEAL --scored SCORED_PREDICTIONS.csv --output DIRECTORIO --title TITULO`.
