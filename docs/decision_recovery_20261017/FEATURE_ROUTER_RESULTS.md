# V19: router R1 con descriptores congelados

Estado: **DEVELOPMENT_LOGO_COMPLETE_NOT_PROMOTION**. La extensión R1 termina con further_evaluation_enabled=false: no supera los controles registrados. H es desarrollo reutilizado; no hay promoción ni SOTA.

Se añaden 194 descriptores causales de la torre espacial FULL a las ocho variables de salida de R0: 202 variables en total. Ridge 0,01, umbral 0 y empate a NATIVE, sin barrido. La selección de variables se registró antes de examinar los resultados H del probe de signo. No se actualiza el encoder.

La evaluación conserva las 18.422 consultas y los ocho grupos H de R0. Cada fold excluye su grupo de ajuste y normalización. Los resultados siguientes son OOF y tienen cobertura completa; las predicciones emitidas son exactamente las de NATIVE o FULL.

| Sistema | Agregado MiD | MiDc | MiDs | MiDl | MiDn | MAE s | Error signo |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5 | 112.285 | 157.267 | 68.257 | 66.231 | 65.520 | 5.637 | 5.694% |
| NATIVE | 104.053 | 145.117 | 62.013 | 67.647 | 61.259 | 4.646 | 5.434% |
| FULL | 78.920 | 95.846 | 53.485 | 62.426 | 87.087 | 5.446 | 5.607% |
| R0 | 77.629 | 93.793 | 53.859 | 66.680 | 79.071 | 3.935 | 5.618% |
| R1 | 78.680 | 96.389 | 53.610 | 64.930 | 79.091 | 3.909 | 5.548% |
| SHUFFLED_R1 | 94.856 | 127.804 | 56.876 | 63.764 | 75.144 | 5.066 | 5.488% |

MiDc/MiDs/MiDl/MiDn son medias por grupo dentro de cada banda; el agregado usa masas 0,5/0,3/0,1/0,1. No equivale al promedio pooled oficial.

R1 mejora ligeramente el agregado de FULL, el MAE y el signo global, pero empeora frente a R0 el agregado y la banda crítica. Frente a A5, MiDn queda en 1,207 veces la referencia, por encima del máximo 1,10. El negativo también incumple la protección numérica frente a NATIVE. El control permutado conserva sus resultados aunque obtenga menor error global de signo que R1.

| Condición registrada de avance | Cumple |
|---|---|
| Gate de desarrollo A5 | No |
| Cobertura completa | Sí |
| Todas las bandas <= 1,10 veces A5 | No |
| Error signo <= A5 + 0,005 | Sí |
| Protecciones originales frente a NATIVE | No |
| Agregado menor que NATIVE y FULL | Sí |
| Agregado menor que control permutado | Sí |
| Agregado menor que R0 guardado | No |

Los intervalos descriptivos del cambio de agregado R1 menos referencia son: frente a FULL [−1,118; 0,520], frente a R0 [0,584; 1,590] y frente al control permutado [−22,869; −10,897]. Se remuestrean grupos 2.000 veces sobre predicciones OOF fijas, sin reajustar el router ni repetir la selección de arquitectura. Estos intervalos no constituyen confirmación independiente.

La exportación ALL_H se ajusta sobre todo H y es exclusivamente de desarrollo; no sustituye estas predicciones OOF. P/H tienen grupos de entrenamiento específico separados, pero H ya influyó en decisiones históricas de arquitectura y experimentación. La procedencia del teacher genérico se conserva sin certificar exhaustivamente su pertenencia externa.

Se necesitan ambos expertos. Este ajuste CPU no mide el coste sensorial completo del router. V permanece cerrado. El rechazo no habilita otra extensión de variables ni un barrido bajo este protocolo.

Fuentes: artifacts/decision_recovery_20261017/feature_router_r1/{REGISTRATION,RESULT,COMPLETE,MATERIALIZATION}.json; descriptores de sign_probe_v2/features/H. El recibo de materialización admite 18.422 filas sin targets en el caché; la extracción H informa paridad EXACT de raw phase y TTC, con cero cambios de signo.

SHA256 de RESULT: d9309f7aa9763b66b192c26fec2d88c1a9e96870a870703ce61ecb639876897c.
SHA256 del registro: 41b51c0f489fcd802ce1fbab97353af030cb7890f3be24c23a9cd5c204947902.

Verificación documental: hashes de todos los archivos listados en COMPLETE comprobados; tablas transcritas desde RESULT. Esta documentación no vuelve a ajustar ni inferir.
