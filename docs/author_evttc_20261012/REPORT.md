# EvTTC: reconstrucción de los intervalos indicados por los autores

No es una reproducción certificada de la Tabla VI: faltan IDs oficiales de consultas, confirmación del checkpoint exacto y detalles completos del mapping.

Candidatos: 49. Tripletas admitidas: 43.

Métrica primaria RTE %, predicciones nativas sin nuevo clipping. MiD es diagnóstico adicional. V13/V12 mantienen contexto más largo que Garl; las consultas y etiquetas sí son comunes.

## own_available

| Modelo | Pares / GT | RTE % | MAE s | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|---:|---:|
| E_H1_MATCHED | 43/43 | 209.761 | 3.958 | 299.790 | — | — |
| E_CTX_MATCHED | 43/43 | 71.932 | 1.303 | 229.905 | — | — |
| R_H1 | 43/43 | 24.605 | 0.483 | 117.929 | — | — |
| R_CTX | 43/43 | 16.403 | 0.286 | 132.311 | — | — |
| F_TRUE | 43/43 | 12.010 | 0.217 | 59.199 | — | — |
| F_ZERO | 43/43 | 84.914 | 1.546 | 250.107 | — | — |
| H8_seed7 | 43/43 | 49.280 | 0.879 | 185.828 | — | — |
| H8_seed13 | 43/43 | 45.323 | 0.811 | 176.255 | — | — |
| H8_seed23 | 43/43 | 47.865 | 0.854 | 182.592 | — | — |
| H8_median3 | 43/43 | 47.379 | 0.845 | 181.786 | — | — |
| Garl_event | 43/43 | 64.522 | 1.188 | 136.787 | — | — |
| Garl_full | 43/43 | 13.674 | 0.258 | 81.641 | — | — |

## common_all_models

| Modelo | Pares / GT | RTE % | MAE s | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|---:|---:|
| E_H1_MATCHED | 43/43 | 209.761 | 3.958 | 299.790 | — | — |
| E_CTX_MATCHED | 43/43 | 71.932 | 1.303 | 229.905 | — | — |
| R_H1 | 43/43 | 24.605 | 0.483 | 117.929 | — | — |
| R_CTX | 43/43 | 16.403 | 0.286 | 132.311 | — | — |
| F_TRUE | 43/43 | 12.010 | 0.217 | 59.199 | — | — |
| F_ZERO | 43/43 | 84.914 | 1.546 | 250.107 | — | — |
| H8_seed7 | 43/43 | 49.280 | 0.879 | 185.828 | — | — |
| H8_seed13 | 43/43 | 45.323 | 0.811 | 176.255 | — | — |
| H8_seed23 | 43/43 | 47.865 | 0.854 | 182.592 | — | — |
| H8_median3 | 43/43 | 47.379 | 0.845 | 181.786 | — | — |
| Garl_event | 43/43 | 64.522 | 1.188 | 136.787 | — | — |
| Garl_full | 43/43 | 13.674 | 0.258 | 81.641 | — | — |

## sequence_CCRm-medium

| Modelo | Pares / GT | RTE % | MAE s | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|---:|---:|
| E_H1_MATCHED | 6/6 | 61.094 | 1.277 | 176.028 | — | — |
| E_CTX_MATCHED | 6/6 | 37.825 | 0.788 | 132.221 | — | — |
| R_H1 | 6/6 | 27.458 | 0.567 | 132.488 | — | — |
| R_CTX | 6/6 | 11.286 | 0.238 | 72.147 | — | — |
| F_TRUE | 6/6 | 4.404 | 0.091 | 21.313 | — | — |
| F_ZERO | 6/6 | 47.072 | 0.981 | 153.938 | — | — |
| H8_seed7 | 6/6 | 16.987 | 0.357 | 69.044 | — | — |
| H8_seed13 | 6/6 | 15.205 | 0.318 | 69.177 | — | — |
| H8_seed23 | 6/6 | 14.146 | 0.296 | 63.386 | — | — |
| H8_median3 | 6/6 | 14.312 | 0.300 | 63.757 | — | — |
| Garl_event | 6/6 | 21.672 | 0.449 | 89.974 | — | — |
| Garl_full | 6/6 | 21.124 | 0.436 | 140.504 | — | — |

## sequence_CCRs2-high

| Modelo | Pares / GT | RTE % | MAE s | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|---:|---:|
| E_H1_MATCHED | 17/17 | 207.748 | 3.578 | 339.933 | — | — |
| E_CTX_MATCHED | 17/17 | 83.223 | 1.408 | 264.572 | — | — |
| R_H1 | 17/17 | 18.923 | 0.332 | 107.178 | — | — |
| R_CTX | 17/17 | 19.039 | 0.311 | 167.279 | — | — |
| F_TRUE | 17/17 | 12.048 | 0.209 | 59.318 | — | — |
| F_ZERO | 17/17 | 100.206 | 1.704 | 288.553 | — | — |
| H8_seed7 | 17/17 | 55.982 | 0.929 | 216.521 | — | — |
| H8_seed13 | 17/17 | 50.192 | 0.837 | 201.170 | — | — |
| H8_seed23 | 17/17 | 54.049 | 0.900 | 213.323 | — | — |
| H8_median3 | 17/17 | 53.320 | 0.887 | 211.263 | — | — |
| Garl_event | 17/17 | 113.200 | 2.019 | 170.728 | — | — |
| Garl_full | 17/17 | 13.401 | 0.229 | 79.677 | — | — |

## sequence_CCRs2-medium

| Modelo | Pares / GT | RTE % | MAE s | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|---:|---:|
| E_H1_MATCHED | 20/20 | 256.072 | 5.085 | 302.796 | — | — |
| E_CTX_MATCHED | 20/20 | 72.568 | 1.367 | 229.742 | — | — |
| R_H1 | 20/20 | 28.579 | 0.586 | 122.701 | — | — |
| R_CTX | 20/20 | 15.698 | 0.280 | 120.637 | — | — |
| F_TRUE | 20/20 | 14.260 | 0.262 | 70.465 | — | — |
| F_ZERO | 20/20 | 83.268 | 1.580 | 246.279 | — | — |
| H8_seed7 | 20/20 | 53.272 | 0.993 | 194.774 | — | — |
| H8_seed13 | 20/20 | 50.219 | 0.937 | 187.200 | — | — |
| H8_seed23 | 20/20 | 52.725 | 0.983 | 192.232 | — | — |
| H8_median3 | 20/20 | 52.249 | 0.973 | 192.140 | — | — |
| Garl_event | 20/20 | 36.001 | 0.703 | 121.981 | — | — |
| Garl_full | 20/20 | 11.672 | 0.229 | 65.651 | — | — |

## Promedio macro de RTE entre los tres escenarios

| Modelo | RTE macro % |
|---|---:|
| E_H1_MATCHED | 174.971 |
| E_CTX_MATCHED | 64.538 |
| R_H1 | 24.987 |
| R_CTX | 15.341 |
| F_TRUE | 10.237 |
| F_ZERO | 76.848 |
| H8_seed7 | 42.080 |
| H8_seed13 | 38.539 |
| H8_seed23 | 40.306 |
| H8_median3 | 39.960 |
| Garl_event | 56.958 |
| Garl_full | 15.399 |

Los tiempos por consulta están en los fragmentos: CPU/FP32 con carga concurrente. El temporizador V13 incluye las seis cabezas; no compararlo con un forward Garl o los milisegundos del paper como si tuvieran el mismo perímetro.

Tabla VI publicada: RTE Garl 8,31 / 10,56 / 12,93 %, promedio 10,60 %. Estas cifras pertenecen al paper y no a esta ejecución. Fuente: https://arxiv.org/html/2603.16303v1#S7.T6
