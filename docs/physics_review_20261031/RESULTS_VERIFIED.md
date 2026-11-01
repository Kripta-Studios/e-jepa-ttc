# Resultados verificados y procedencia

Generado por verify_evidence.py. No se entrenan modelos ni se ejecuta inferencia de sensores.

## Artículo

13 hashes públicos, 8 resúmenes de transferencia y 12 campos del scorer comprobados por el verificador original.

Overall MiD recalculado: 35.801451580824. Ranking: snapshot histórico, no consulta actual al leaderboard.

## Transferencia Extended-new

468 consultas; 32 secuencias. Cálculo nuevo desde CSV congelado.

| Modelo | MAE s | Mediana s | p95 s | Máximo s | RTE % | Errores >10 s |
|---|---:|---:|---:|---:|---:|---:|
| phase | 1.022677289 | 0.463565130 | 3.243010006 | 32.505470595 | 41.226716 | 5 |
| rgb | 1.015377481 | 0.407897322 | 3.798780919 | 16.644879661 | 38.555541 | 2 |
| event | 1.451422478 | 0.856600033 | 4.397003271 | 35.954097891 | 65.695499 | 4 |
| garl | 3.257595943 | 0.673840014 | 5.252375502 | 624.072595031 | 140.068866 | 13 |

## Intervalos históricos PHASE menos RGB

No se han vuelto a generar las 10.000 réplicas; se transcribe el análisis archivado con hash.

| Estadístico | Delta s | IC95% inferior | Superior |
|---|---:|---:|---:|
| media | 0.007299808 | -0.110305331 | 0.128971192 |
| mediana | 0.055667808 | -0.038113062 | 0.145225079 |
| p95 | -0.555770913 | -1.125724196 | 0.241009745 |

## V8: escalares archivados, no reejecución

| Ruta | MiD |
|---|---:|
| analytic_only | 373.176409681 |
| analytic_residual | 373.596062227 |
| analytic_residual_transport | 163.231022279 |
| analytic_residual_transport_history | 155.823806866 |
| analytic_transport | 373.176409681 |

## Piloto JEPA: análisis archivado

MAE de cambio de log-altura, sin unidades; no MAE de TTC.

| Brazo | MAE |
|---|---:|
| B_static | 0.037632390 |
| C_one_step | 0.035640042 |
| C_shuffle | 0.035061886 |
| random_init | 0.037196409 |

La puerta principal falló. Comparaciones, límites y hashes completos: [verification.json](verification.json).
