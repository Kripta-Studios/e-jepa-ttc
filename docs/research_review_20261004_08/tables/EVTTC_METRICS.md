# EvTTC: métricas regeneradas

Mismo soporte finito: 946 consultas, 32 secuencias. No es el benchmark oficial.

| Modelo | N | MAE s | Mediana AE s | RMSE s | Sesgo s |
| --- | --- | --- | --- | --- | --- |
| H8_seed7 | 946 | 1.320841 | 0.766047 | 3.027209 | 0.577278 |
| H8_seed13 | 946 | 1.352809 | 0.731413 | 3.206100 | 0.533595 |
| H8_seed23 | 946 | 1.408714 | 0.738464 | 3.392155 | 0.728923 |
| public_Garl_event_lhr | 946 | 8.048109 | 0.996929 | 159.246808 | 5.475819 |
| public_Garl_rgb_event_full | 946 | 1.868276 | 0.691512 | 7.875241 | -0.254908 |

## Diferencia macro por secuencia respecto a Garl RGB+eventos

Bootstrap por secuencia, 2.000 remuestreos, semilla 20261008. Negativo favorece al modelo de la fila. No confundir con diferencia de MAE agrupados.

| Modelo | Delta s | IC 95% s |
| --- | --- | --- |
| H8_seed7 | -0.547665 | [-1.284459, -0.033821] |
| H8_seed13 | -0.514758 | [-1.209972, -0.005673] |
| H8_seed23 | -0.463380 | [-1.136674, 0.043723] |
| public_Garl_event_lhr | 5.811809 | [0.254385, 15.783927] |
