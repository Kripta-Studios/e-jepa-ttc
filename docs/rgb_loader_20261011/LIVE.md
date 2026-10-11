# Entrenamiento reanudado: observación en vivo

Tabla regenerada desde `LIVE_RESULT.json` por `publish_live.py` sin argumentos.

| Ventana | Duración (s) | Updates | Updates/min |
|---|---:|---:|---:|
| Anterior | 37.55 | 11 | 17.58 |
| Optimizada | 376.35 | 176 | 28.06 |

Última muestra: 2026-10-11T02:07:55.001054+02:00; **1677/30.330**, duraderos 1600; recuperación acumulada 0.

Extrapolación para terminar sólo R_A5 a este ritmo: **17.0 horas**.

Estas ventanas tienen distinta duración, distintos batches y carga del equipo variable. El cociente de sus ritmos es descriptivo; no demuestra una aceleración controlada de todo el entrenamiento.

Se restauró el hash exacto del checkpoint 1.500. El recibo del siguiente checkpoint está incluido en el JSON. Se mantienen BF16, objetivo, sampler y estado del optimizador.

El perfil restante señala preparación de imágenes RGB y trabajo del modelo. Los porcentajes de pilas del hilo principal no son tiempos CUDA y los inclusivos se solapan.
