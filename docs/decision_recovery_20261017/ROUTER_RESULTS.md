# V19: selección de expertos sobre H

Estado: **NO_PRIMARY_ROUTER_CANDIDATE**. H sigue siendo desarrollo expuesto; no hay promoción ni SOTA.

Ridge fijo 0,01, umbral 0, normalizador y ajuste excluyen el grupo evaluado. Los pesos siguen el scorer por banda/grupo. Todos los pares registrados y el control permutado se conservan.

| Pareja | Modelo | Agregado MiD | MiDc | MiDn | MAE s | Error signo |
|---|---|---:|---:|---:|---:|---:|
| A5__FULL | A5 | 112.285 | 157.267 | 65.520 | 5.637 | 5.694% |
| A5__FULL | FULL | 78.920 | 95.846 | 87.087 | 5.446 | 5.607% |
| A5__FULL | R0 | 78.078 | 94.870 | 80.129 | 3.929 | 5.477% |
| A5__FULL | SHUFFLED | 82.628 | 102.129 | 85.007 | 5.649 | 5.569% |
| NATIVE__CORR | A5 | 112.285 | 157.267 | 65.520 | 5.637 | 5.694% |
| NATIVE__CORR | CORR | 103.510 | 115.556 | 156.002 | 5.496 | 13.804% |
| NATIVE__CORR | NATIVE | 104.053 | 145.117 | 61.259 | 4.646 | 5.434% |
| NATIVE__CORR | R0 | 93.028 | 104.274 | 107.195 | 3.447 | 7.366% |
| NATIVE__CORR | SHUFFLED | 103.956 | 144.977 | 61.259 | 4.650 | 5.423% |
| NATIVE__FULL | A5 | 112.285 | 157.267 | 65.520 | 5.637 | 5.694% |
| NATIVE__FULL | FULL | 78.920 | 95.846 | 87.087 | 5.446 | 5.607% |
| NATIVE__FULL | NATIVE | 104.053 | 145.117 | 61.259 | 4.646 | 5.434% |
| NATIVE__FULL | R0 | 77.629 | 93.793 | 79.071 | 3.935 | 5.618% |
| NATIVE__FULL | SHUFFLED | 96.623 | 131.027 | 73.599 | 4.904 | 5.694% |

Los intervalos de los JSON remuestrean grupos sobre predicciones OOF fijas. No repiten el ajuste ni estiman incertidumbre de selección de arquitectura. El export ALL_H es un modelo de desarrollo, no evidencia OOF sobre H.

Fuente: artifacts/decision_recovery_20261017/router/{pareja}/RESULT.json y DECISION.json. Se requiere ejecutar ambos expertos; no hay coste sensorial completo medido para estos routers.
