# 04 — Resultados regenerados desde evidencia

Generado por `replay.py`. No incluye nuevas inferencias ni updates.

## Endpoints V13

| Fit | Estado | Updates |
|---|---|---:|
| E_A5_MATCHED | COMPLETE | 30330 |
| E_C2F_MATCHED | COMPLETE | 30330 |
| PAIR_E_MATCHED | COMPLETE | 6840 |
| PAIR_R | COMPLETE | 6840 |
| R_A5 | COMPLETE | 30330 |
| R_C2F | COMPLETE | 30330 |
| E_CTX_MATCHED | COMPLETE | 2500 |
| E_H1_MATCHED | COMPLETE | 2500 |
| F_TRUE | COMPLETE | 2500 |
| F_ZERO | COMPLETE | 2500 |
| R_CTX | COMPLETE | 2500 |
| R_H1 | COMPLETE | 2500 |

## Validación V: macro por grupo y banda

No es overall_MiD pooled oficial. 16.432 consultas de ocho grupos.

| Modelo | group_macro_bucket_MiD | MAE s | Fallos |
|---|---:|---:|---:|
| E_CTX_MATCHED | 106.082 | 2.497 | 0 |
| E_H1_MATCHED | 113.450 | 2.848 | 0 |
| F_TRUE | 90.650 | 2.003 | 0 |
| F_ZERO | 107.128 | 2.472 | 0 |
| R_CTX | 93.505 | 1.884 | 0 |
| R_H1 | 96.821 | 2.119 | 0 |

| Gate V | Aprobado | IC95 diferencia A−B |
|---|---|---|
| E_CTX_MATCHED=E_H1_MATCHED | True | [-9.424, -5.025] |
| F_TRUE=E_CTX_MATCHED | True | [-22.693, -8.252] |
| F_TRUE=F_ZERO | True | [-23.422, -9.769] |
| R_CTX=R_H1 | False | [-5.557, -1.091] |

## V12 eAP TRAIN40: diagnóstico sobre entrenamiento

88.744 consultas. No es holdout ni puntuación CodaBench.

| Método | overall_MiD | mean_MiD | MiDc | MiDs | MiDl | MiDn |
|---|---:|---:|---:|---:|---:|---:|
| H8_seed7 | 73.027 | 50.714 | 101.566 | 44.163 | 46.122 | 43.825 |
| H8_seed13 | 72.545 | 50.012 | 100.227 | 45.739 | 45.183 | 41.914 |
| H8_seed23 | 72.700 | 48.747 | 102.147 | 44.416 | 42.488 | 40.532 |
| H8_median3 | 72.539 | 49.537 | 101.119 | 44.565 | 44.441 | 41.663 |
| public_Garl_event_lhr | 93.682 | 55.702 | 143.001 | 43.221 | 47.153 | 45.002 |

## FCWD

630 predicciones. Sin banda negativa: overall_MiD oficial no disponible.

### all_queries

| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_CTX_MATCHED | 597/597 | 1.967 | 26.340 | -1.188 | 150.093 | 24.035 | 20.397 | — | 0/0/0/0 |
| E_H1_MATCHED | 597/597 | 2.405 | 32.536 | -0.953 | 152.679 | 33.536 | 19.719 | — | 0/0/0/0 |
| F_TRUE | 597/597 | 1.430 | 18.049 | -1.034 | 102.323 | 31.683 | 16.445 | — | 0/0/0/0 |
| F_ZERO | 597/597 | 2.130 | 30.614 | -1.294 | 182.486 | 28.087 | 19.343 | — | 0/0/0/0 |
| H8_seed13 | 597/597 | 1.597 | 27.407 | -0.864 | 177.123 | 26.649 | 16.777 | — | 0/0/0/0 |
| H8_seed23 | 597/597 | 1.569 | 27.482 | -0.785 | 176.972 | 26.093 | 17.641 | — | 0/0/0/0 |
| H8_seed7 | 597/597 | 1.778 | 30.196 | -0.968 | 188.137 | 26.350 | 19.343 | — | 0/0/0/0 |
| R_CTX | 472/597 | 0.938 | 18.411 | -0.839 | 97.156 | 71.390 | 24.447 | — | 6/19/46/0 |
| R_H1 | 472/597 | 1.291 | 28.228 | -0.696 | 159.716 | 71.096 | 33.134 | — | 6/19/46/0 |
| public_Garl_event_lhr | 597/597 | 2.146 | 26.887 | -1.487 | 95.057 | 36.698 | 34.212 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/597 | — | — | — | — | — | — | — | 201/189/108/0 |

### same_RGB_available_queries

| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_CTX_MATCHED | 472/472 | 1.380 | 24.819 | -0.504 | 147.176 | 24.176 | 22.131 | — | 0/0/0/0 |
| E_H1_MATCHED | 472/472 | 1.251 | 27.302 | -0.158 | 149.563 | 33.423 | 19.498 | — | 0/0/0/0 |
| F_TRUE | 472/472 | 0.700 | 14.331 | -0.310 | 97.936 | 32.679 | 15.248 | — | 0/0/0/0 |
| F_ZERO | 472/472 | 1.585 | 30.223 | -0.639 | 180.565 | 28.681 | 20.295 | — | 0/0/0/0 |
| H8_seed13 | 472/472 | 1.047 | 26.548 | -0.274 | 174.495 | 25.936 | 17.358 | — | 0/0/0/0 |
| H8_seed23 | 472/472 | 1.032 | 26.648 | -0.202 | 174.241 | 25.294 | 18.221 | — | 0/0/0/0 |
| H8_seed7 | 472/472 | 1.139 | 28.900 | -0.271 | 185.542 | 25.373 | 19.616 | — | 0/0/0/0 |
| R_CTX | 472/472 | 0.938 | 18.411 | -0.839 | 97.156 | 71.390 | 24.447 | — | 0/0/0/0 |
| R_H1 | 472/472 | 1.291 | 28.228 | -0.696 | 159.716 | 71.096 | 33.134 | — | 0/0/0/0 |
| public_Garl_event_lhr | 472/472 | 1.352 | 23.192 | -0.616 | 93.913 | 36.543 | 33.823 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/472 | — | — | — | — | — | — | — | 195/170/62/0 |

### RGB_unavailable_queries

| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_CTX_MATCHED | 125/125 | 4.184 | 32.087 | -3.769 | 244.910 | 22.771 | 18.059 | — | 0/0/0/0 |
| E_H1_MATCHED | 125/125 | 6.760 | 52.301 | -3.954 | 253.938 | 34.549 | 20.017 | — | 0/0/0/0 |
| F_TRUE | 125/125 | 4.184 | 32.087 | -3.769 | 244.910 | 22.771 | 18.059 | — | 0/0/0/0 |
| F_ZERO | 125/125 | 4.184 | 32.087 | -3.769 | 244.910 | 22.771 | 18.059 | — | 0/0/0/0 |
| H8_seed13 | 125/125 | 3.675 | 30.647 | -3.090 | 262.522 | 33.022 | 15.993 | — | 0/0/0/0 |
| H8_seed23 | 125/125 | 3.596 | 30.632 | -2.986 | 265.731 | 33.245 | 16.860 | — | 0/0/0/0 |
| H8_seed7 | 125/125 | 4.188 | 35.092 | -3.600 | 272.492 | 35.091 | 18.974 | — | 0/0/0/0 |
| R_CTX | 0/125 | — | — | — | — | — | — | — | 6/19/46/0 |
| R_H1 | 0/125 | — | — | — | — | — | — | — | 6/19/46/0 |
| public_Garl_event_lhr | 125/125 | 5.144 | 40.840 | -4.779 | 132.245 | 38.088 | 34.736 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/125 | — | — | — | — | — | — | — | 6/19/46/0 |

### F_TRUE por secuencia

| Secuencia | GT | MAE s | Sesgo s | Signos erróneos GT positivo | Salidas abs≥59,999 s |
|---|---:|---:|---:|---:|---:|
| FCWD1 | 201 | 2.310 | -1.750 | 4 | 1 |
| FCWD2 | 204 | 1.114 | -0.836 | 1 | 0 |
| FCWD3 | 192 | 0.844 | -0.494 | 0 | 0 |

## Dev32

1024 predicciones. Sin banda negativa: overall_MiD oficial no disponible.

### all_queries

| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_CTX_MATCHED | 946/946 | 1.855 | 86.765 | 0.740 | 283.351 | 59.198 | 32.954 | — | 0/0/0/0 |
| E_H1_MATCHED | 946/946 | 2.575 | 118.811 | 0.763 | 305.354 | 73.613 | 33.794 | — | 0/0/0/0 |
| F_TRUE | 946/946 | 1.368 | 50.680 | 0.068 | 181.582 | 56.217 | 35.636 | — | 0/0/0/0 |
| F_ZERO | 946/946 | 1.937 | 93.248 | 0.953 | 299.456 | 60.309 | 31.035 | — | 0/0/0/0 |
| H8_seed13 | 946/946 | 1.353 | 62.923 | 0.534 | 245.320 | 55.627 | 32.914 | — | 0/0/0/0 |
| H8_seed23 | 946/946 | 1.409 | 68.866 | 0.729 | 250.098 | 56.160 | 31.475 | — | 0/0/0/0 |
| H8_seed7 | 946/946 | 1.321 | 62.351 | 0.577 | 253.657 | 54.064 | 34.055 | — | 0/0/0/0 |
| R_CTX | 946/946 | 1.258 | 43.771 | 0.240 | 182.444 | 59.129 | 36.222 | — | 0/0/0/0 |
| R_H1 | 946/946 | 1.288 | 52.009 | 0.334 | 197.843 | 60.264 | 33.790 | — | 0/0/0/0 |
| public_Garl_event_lhr | 946/946 | 8.048 | 293.472 | 5.476 | 287.536 | 83.127 | 46.893 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 946/946 | 1.868 | 76.769 | -0.255 | 237.242 | 64.023 | 61.505 | — | 0/0/0/0 |

### same_RGB_available_queries

| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_CTX_MATCHED | 946/946 | 1.855 | 86.765 | 0.740 | 283.351 | 59.198 | 32.954 | — | 0/0/0/0 |
| E_H1_MATCHED | 946/946 | 2.575 | 118.811 | 0.763 | 305.354 | 73.613 | 33.794 | — | 0/0/0/0 |
| F_TRUE | 946/946 | 1.368 | 50.680 | 0.068 | 181.582 | 56.217 | 35.636 | — | 0/0/0/0 |
| F_ZERO | 946/946 | 1.937 | 93.248 | 0.953 | 299.456 | 60.309 | 31.035 | — | 0/0/0/0 |
| H8_seed13 | 946/946 | 1.353 | 62.923 | 0.534 | 245.320 | 55.627 | 32.914 | — | 0/0/0/0 |
| H8_seed23 | 946/946 | 1.409 | 68.866 | 0.729 | 250.098 | 56.160 | 31.475 | — | 0/0/0/0 |
| H8_seed7 | 946/946 | 1.321 | 62.351 | 0.577 | 253.657 | 54.064 | 34.055 | — | 0/0/0/0 |
| R_CTX | 946/946 | 1.258 | 43.771 | 0.240 | 182.444 | 59.129 | 36.222 | — | 0/0/0/0 |
| R_H1 | 946/946 | 1.288 | 52.009 | 0.334 | 197.843 | 60.264 | 33.790 | — | 0/0/0/0 |
| public_Garl_event_lhr | 946/946 | 8.048 | 293.472 | 5.476 | 287.536 | 83.127 | 46.893 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 946/946 | 1.868 | 76.769 | -0.255 | 237.242 | 64.023 | 61.505 | — | 0/0/0/0 |

### RGB_unavailable_queries

| Modelo | Pares / GT | MAE s | RTE % | Sesgo s | MiDc | MiDs | MiDl | MiDn | Inválidos c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_CTX_MATCHED | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| E_H1_MATCHED | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| F_TRUE | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| F_ZERO | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| H8_seed13 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| H8_seed23 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| H8_seed7 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| R_CTX | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| R_H1 | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| public_Garl_event_lhr | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/0 | — | — | — | — | — | — | — | 0/0/0/0 |

### F_TRUE por secuencia

| Secuencia | GT | MAE s | Sesgo s | Signos erróneos GT positivo | Salidas abs≥59,999 s |
|---|---:|---:|---:|---:|---:|
| CCRm-low-0-overlap-0 | 32 | 0.454 | -0.208 | 0 | 0 |
| CCRm-low-100-overlap-100 | 26 | 0.769 | -0.753 | 0 | 0 |
| CCRm-low-50-overlap-50 | 30 | 0.877 | -0.866 | 0 | 0 |
| CCRm-medium-0-overlap-0 | 26 | 1.352 | -0.778 | 0 | 0 |
| CCRm-medium-100-overlap-100 | 28 | 1.284 | -1.011 | 0 | 0 |
| CCRm-medium-50-overlap-50 | 31 | 1.471 | -1.423 | 0 | 0 |
| CCRs-1-high-0-overlap-0 | 30 | 0.290 | 0.242 | 0 | 0 |
| CCRs-1-high-100-overlap-100 | 30 | 0.494 | 0.450 | 0 | 0 |
| CCRs-1-high-50-overlap-50 | 30 | 0.392 | 0.212 | 0 | 0 |
| CCRs-1-low-0-overlap-0 | 26 | 0.856 | -0.671 | 0 | 0 |
| CCRs-1-low-100-overlap-100 | 31 | 5.058 | -2.742 | 8 | 0 |
| CCRs-1-low-50-overlap-50 | 32 | 0.251 | -0.133 | 0 | 0 |
| CCRs-1-medium-0-overlap-0 | 28 | 0.307 | -0.013 | 0 | 0 |
| CCRs-1-medium-100-overlap-100 | 30 | 12.346 | -2.737 | 4 | 3 |
| CCRs-1-medium-50-overlap-50 | 31 | 0.293 | -0.116 | 0 | 0 |
| CCRs-2-high-100-overlap-100 | 30 | 0.327 | 0.070 | 0 | 0 |
| CCRs-2-low-100-overlap-100 | 32 | 0.739 | -0.715 | 0 | 0 |
| CCRs-2-medium-100-overlap-100 | 30 | 0.315 | -0.173 | 0 | 0 |
| CCRs-3-low-100-overlap-100 | 25 | 0.429 | -0.047 | 0 | 0 |
| CCRs-3-medium-100-overlap-100 | 31 | 0.599 | 0.153 | 0 | 0 |
| CCRs-side-high | 32 | 0.890 | 0.848 | 0 | 0 |
| CCRs-side-low | 32 | 1.966 | 1.936 | 0 | 0 |
| CCRs-side-medium | 32 | 3.445 | 3.284 | 0 | 1 |
| CPLA-high | 30 | 0.817 | 0.817 | 0 | 0 |
| CPLA-low | 31 | 0.821 | 0.646 | 0 | 0 |
| CPLA-medium | 30 | 0.742 | 0.742 | 0 | 0 |
| CPNA-high | 30 | 0.725 | 0.666 | 0 | 0 |
| CPNA-low | 24 | 1.137 | 0.894 | 0 | 0 |
| CPNA-medium | 26 | 1.040 | 0.963 | 0 | 0 |
| CPNAO-high | 28 | 1.653 | 1.570 | 0 | 0 |
| CPNAO-low | 31 | 0.806 | 0.399 | 0 | 0 |
| CPNAO-medium | 31 | 0.429 | 0.429 | 0 | 0 |

## Costes V13: host con carga CPU concurrente

Rutas distintas; no sumar medianas. No es el piloto streaming V12.

| Ruta | Mediana caliente ms | p95 ms |
|---|---:|---:|
| E_A5_PREPARED_EVENT_TO_TTC | 63.153 | 69.727 |
| E_A5_RAW_EVENT_H5_TO_TTC | 320.156 | 746.911 |
| E_C2F_PREPARED_EVENT_TO_TTC | 102.178 | 107.657 |
| E_C2F_RAW_EVENT_H5_TO_TTC | 341.992 | 641.721 |
| E_CTX_MATCHED_NORMALIZED_PHASE17_TO_TTC | 10.600 | 13.280 |
| FULL_OWN_RGBTAR_TO_R_CTX_TTC | 738.139 | 788.186 |
| FULL_RAW_EVENT_H5_RGBTAR_TO_F_TRUE_TTC | 2581.495 | 3832.962 |
| FULL_RAW_EVENT_H5_RGBTAR_TO_F_ZERO_TTC | 2377.433 | 3691.316 |
| FULL_RAW_EVENT_H5_TO_E_CTX_TTC | 1656.111 | 3092.507 |
| PREPARED_EVENT_RGBTAR_TO_F_TRUE_TTC | 1343.966 | 1455.960 |
| PREPARED_EVENT_RGBTAR_TO_F_ZERO_TTC | 1370.786 | 1465.560 |
| PREPARED_EVENT_TO_E_CTX_TTC | 598.054 | 635.656 |
| RAW_EVENT_H5_TO_VOXELS_CPU | 257.908 | 639.539 |
| R_A5_RGBTAR_TO_TTC | 57.271 | 219.459 |
| R_C2F_RGBTAR_TO_TTC | 96.383 | 109.120 |
| R_CTX_NORMALIZED_PHASE17_TO_TTC | 10.294 | 12.196 |

Fuentes y hashes: `SOURCE_MANIFEST.json`. Las comprobaciones por secuencia incluyen también las vistas no impresas de los demás modelos.
