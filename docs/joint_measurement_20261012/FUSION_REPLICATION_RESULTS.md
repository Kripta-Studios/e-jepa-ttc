# Réplicas de la fusión C1 + R_CTX

Se publican las tres semillas registradas (7, 13, 23) y sus controles ZERO. La réplica mide la variabilidad de la adaptación C1 y de las nuevas cabezas. A5 de inicialización y R_CTX histórico son compartidos: no son tres pretrainings independientes. No se elige una semilla favorable.

Por semilla adicional: 12.000 updates P del encoder y lectura C1, extracción nueva de H y normalizador event propio, y 2.500 updates H por cada cabeza TRUE/ZERO. Los seis endpoints nuevos se congelan antes de cualquier inferencia V de las réplicas. R_CTX ya está supervisado en H; la nueva cabeza es ajuste secuencial, no OOF. V sigue siendo desarrollo expuesto.

## Cohorte V completa

| Modelo | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Error signo |
|---|---:|---:|---:|---:|---:|---:|---:|
| FUSION-TRUE-S7 | 88.270 | 127.661 | 51.025 | 42.969 | 48.355 | 1.817 | 2.611% |
| FUSION-ZERO-S7 | 92.638 | 136.989 | 50.225 | 42.182 | 48.580 | 1.842 | 2.684% |
| FUSION-TRUE-S13 | 86.630 | 125.976 | 49.005 | 41.425 | 47.981 | 1.807 | 2.593% |
| FUSION-ZERO-S13 | 92.455 | 136.500 | 50.332 | 42.403 | 48.651 | 1.837 | 2.684% |
| FUSION-TRUE-S23 | 86.961 | 126.069 | 49.994 | 40.765 | 48.520 | 1.818 | 2.593% |
| FUSION-ZERO-S23 | 92.315 | 136.314 | 50.242 | 42.310 | 48.546 | 1.836 | 2.684% |
| Media TRUE | 87.287 | 126.569 | 50.008 | 41.719 | 48.285 | 1.814 | 2.599% |
| Media ZERO | 92.469 | 136.601 | 50.266 | 42.299 | 48.592 | 1.838 | 2.684% |
| R_CTX | 93.505 | 138.634 | 50.841 | 41.204 | 48.154 | 1.884 | 2.684% |
| F_TRUE | 90.650 | 131.501 | 52.740 | 42.788 | 47.986 | 2.003 | 2.550% |

## Variabilidad y gates originales

- TRUE: desviación entre semillas (ddof=1), agregado 0.8673; MiDc 0.9472; MAE 0.0059 s.
- ZERO: desviación entre semillas (ddof=1), agregado 0.1623; MiDc 0.3491; MAE 0.0031 s.

Las medias son de métricas escalares, no un ensemble de TTC. La protección p95 usa la media de los p95 por semilla, tal como se registró antes de los resultados. Se conservan también los gates de cada semilla.

| Contraste | Comprobaciones fallidas |
|---|---|
| mean_TRUE_vs_F_TRUE | crucial_10pct |
| mean_TRUE_vs_R_CTX | crucial_10pct |
| mean_TRUE_vs_ZERO | crucial_10pct |
| seed13_TRUE_vs_F_TRUE | crucial_10pct |
| seed13_TRUE_vs_R_CTX | crucial_10pct |
| seed13_TRUE_vs_ZERO | crucial_10pct |
| seed23_TRUE_vs_F_TRUE | crucial_10pct |
| seed23_TRUE_vs_R_CTX | crucial_10pct |
| seed23_TRUE_vs_ZERO | crucial_10pct |
| seed7_TRUE_vs_F_TRUE | crucial_10pct |
| seed7_TRUE_vs_R_CTX | crucial_10pct |
| seed7_TRUE_vs_ZERO | crucial_10pct |

## Incertidumbre por grupos

- TRUE − F_TRUE, IC95 del cambio agregado: [-6.7639, -0.1881].
- TRUE − R_CTX, IC95 del cambio agregado: [-8.7704, -3.8972].
- TRUE − ZERO, IC95 del cambio agregado: [-7.3454, -3.0317].

Son 2.000 resamples de grupos completos, después de promediar pérdidas MiD por grupo entre las tres semillas fijas. No se tratan ventanas correlacionadas como independientes. Este bootstrap no estima por sí solo la variabilidad de entrenamiento: ésta se presenta por separado.

## Portabilidad y límites

Las tablas H se conservan como diagnóstico del ajuste, no generalización. Los exports de cabeza incorporan sus normalizadores y fallback event dinámico, pero necesitan features de sus productores. Los exports C1 empiezan en los dos tensores sensoriales preparados. Ninguno de esos artefactos aislados equivale al pipeline RGB/event completo.

Esta tabla no concede promoción ni confirma SOTA. La decisión del sistema requiere además el comparador V16 conservado, coste completo y las condiciones del protocolo. Una cifra interna no es una evaluación oficial.

Fuentes: `replication/fusion_C1/evaluation/RESULT.json`, `predictions/{H,V}`, `ENDPOINTS.json`, `exports` y `verification/predictions`. Regenerar con `python -B -m operational.joint_measurement.fusion_replication_report`.
