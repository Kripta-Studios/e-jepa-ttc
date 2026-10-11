# FCWD: resultados finales V13 con registro RGB

630 consultas; 495 con RGB disponible. Cero updates.

Resultados exploratorios sobre conjuntos expuestos. No son una clasificación oficial ni evidencia SOTA.

MiD es la media agrupada de errores de fase por banda; no es la media por secuencia de V13. Se muestran los casos de fase no válida y no se imputa una banda ausente.

## all_queries

| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_H1_MATCHED | 597/597 | 2.405 | 32.536 | -0.953 | 152.679 | 33.536 | 19.719 | — | 0/0/0/0 |
| E_CTX_MATCHED | 597/597 | 1.967 | 26.340 | -1.188 | 150.093 | 24.035 | 20.397 | — | 0/0/0/0 |
| R_H1 | 472/597 | 1.291 | 28.228 | -0.696 | 159.716 | 71.096 | 33.134 | — | 6/19/46/0 |
| R_CTX | 472/597 | 0.938 | 18.411 | -0.839 | 97.156 | 71.390 | 24.447 | — | 6/19/46/0 |
| F_TRUE | 597/597 | 1.430 | 18.049 | -1.034 | 102.323 | 31.683 | 16.445 | — | 0/0/0/0 |
| F_ZERO | 597/597 | 2.130 | 30.614 | -1.294 | 182.486 | 28.087 | 19.343 | — | 0/0/0/0 |
| H8_seed7 | 597/597 | 1.778 | 30.196 | -0.968 | 188.137 | 26.350 | 19.343 | — | 0/0/0/0 |
| H8_seed13 | 597/597 | 1.597 | 27.407 | -0.864 | 177.123 | 26.649 | 16.777 | — | 0/0/0/0 |
| H8_seed23 | 597/597 | 1.569 | 27.482 | -0.785 | 176.972 | 26.093 | 17.641 | — | 0/0/0/0 |
| public_Garl_event_lhr | 597/597 | 2.146 | 26.887 | -1.487 | 95.057 | 36.698 | 34.212 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/597 | — | — | — | — | — | — | — | 201/189/108/0 |

## same_RGB_available_queries

| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_H1_MATCHED | 472/472 | 1.251 | 27.302 | -0.158 | 149.563 | 33.423 | 19.498 | — | 0/0/0/0 |
| E_CTX_MATCHED | 472/472 | 1.380 | 24.819 | -0.504 | 147.176 | 24.176 | 22.131 | — | 0/0/0/0 |
| R_H1 | 472/472 | 1.291 | 28.228 | -0.696 | 159.716 | 71.096 | 33.134 | — | 0/0/0/0 |
| R_CTX | 472/472 | 0.938 | 18.411 | -0.839 | 97.156 | 71.390 | 24.447 | — | 0/0/0/0 |
| F_TRUE | 472/472 | 0.700 | 14.331 | -0.310 | 97.936 | 32.679 | 15.248 | — | 0/0/0/0 |
| F_ZERO | 472/472 | 1.585 | 30.223 | -0.639 | 180.565 | 28.681 | 20.295 | — | 0/0/0/0 |
| H8_seed7 | 472/472 | 1.139 | 28.900 | -0.271 | 185.542 | 25.373 | 19.616 | — | 0/0/0/0 |
| H8_seed13 | 472/472 | 1.047 | 26.548 | -0.274 | 174.495 | 25.936 | 17.358 | — | 0/0/0/0 |
| H8_seed23 | 472/472 | 1.032 | 26.648 | -0.202 | 174.241 | 25.294 | 18.221 | — | 0/0/0/0 |
| public_Garl_event_lhr | 472/472 | 1.352 | 23.192 | -0.616 | 93.913 | 36.543 | 33.823 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/472 | — | — | — | — | — | — | — | 195/170/62/0 |

## RGB_unavailable_queries

| Modelo | Pares finitos / etiquetas | MAE s | RTE % | Sesgo medio s | MiDc | MiDs | MiDl | MiDn | Fases inválidas c/s/l/n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E_H1_MATCHED | 125/125 | 6.760 | 52.301 | -3.954 | 253.938 | 34.549 | 20.017 | — | 0/0/0/0 |
| E_CTX_MATCHED | 125/125 | 4.184 | 32.087 | -3.769 | 244.910 | 22.771 | 18.059 | — | 0/0/0/0 |
| R_H1 | 0/125 | — | — | — | — | — | — | — | 6/19/46/0 |
| R_CTX | 0/125 | — | — | — | — | — | — | — | 6/19/46/0 |
| F_TRUE | 125/125 | 4.184 | 32.087 | -3.769 | 244.910 | 22.771 | 18.059 | — | 0/0/0/0 |
| F_ZERO | 125/125 | 4.184 | 32.087 | -3.769 | 244.910 | 22.771 | 18.059 | — | 0/0/0/0 |
| H8_seed7 | 125/125 | 4.188 | 35.092 | -3.600 | 272.492 | 35.091 | 18.974 | — | 0/0/0/0 |
| H8_seed13 | 125/125 | 3.675 | 30.647 | -3.090 | 262.522 | 33.022 | 15.993 | — | 0/0/0/0 |
| H8_seed23 | 125/125 | 3.596 | 30.632 | -2.986 | 265.731 | 33.245 | 16.860 | — | 0/0/0/0 |
| public_Garl_event_lhr | 125/125 | 5.144 | 40.840 | -4.779 | 132.245 | 38.088 | 34.736 | — | 0/0/0/0 |
| public_Garl_rgb_event_full | 0/125 | — | — | — | — | — | — | — | 6/19/46/0 |

## Diagnóstico por secuencia

Las métricas, sesgos, signos y predicciones en el límite de ±60 s están en `DIAGNOSTICS.json`.
El subconjunto RGB se fija por entradas y calidad del registro, nunca por el error TTC. Comparar modelos dentro de una misma tabla; las cabezas RGB sin fallback tienen menor cobertura en la tabla completa.

Regeneración: `python -m operational.external_transfer_report --seal SEAL --scored SCORED_PREDICTIONS.csv --output DIRECTORIO --title TITULO`.
