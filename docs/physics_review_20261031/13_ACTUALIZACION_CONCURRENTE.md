# Actualización recibida al cerrar la revisión

El inventario principal se conserva en `f8131a031f61d9270323fe3961dad88016b582cd`. Mientras se preparaba este dossier, otra ejecución añadió tres commits hasta `e750e38130ae02ba2a362bee3270b4fc3cb8d063`. Esta actualización evita presentar como pendiente un screen que terminó durante el análisis. No se atribuye su ejecución a esta revisión.

| Commit | Cambio y evidencia |
|---|---|
| `6bc855a3` | Publica resultados de las 149 consultas, mejora de F0 y regresión del router |
| `4d9b5b7a` | Amplía extracción con captura compacta, reutilización sellada y población de 2.608 consultas |
| `e750e381` | Prepara evaluación poblacional, inventario de historias eAP y registro de continuación |

Fuentes leídas: [informe del screen](../research_log_20261031_canonical_screen149/README.md), [resultados estructurados](../research_log_20261031_canonical_screen149/evidence/report.json), [continuación](../research_log_20261031_canonical_screen149/CONTINUATION.md) e [inventario de recibos eAP](../research_log_20261031_canonical_screen149/evidence/eap_history_inventory.json). Las cifras siguientes son resultados registrados por esa ejecución; aquí no se repitió su inferencia ni se recalcularon sus métricas desde targets.

## Resultado que modifica la recomendación

| Ruta | Dominio | Disponibles / población, sin cambios | MAE macro original → canónico (s) | Error relativo macro original → canónico (%) |
|---|---|---:|---:|---:|
| F0 | Carretera | 72/72 | 0,748853 → 0,748672 | 31,370683 → 31,358111 |
| F0 | Slider-500 | 76/77 | 1,178156 → 1,120809 | 83,885894 → 82,442047 |
| Router | Carretera | 72/72 | 0,338041 → 0,338051 | 10,295071 → 10,295517 |
| Router | Slider-500 | 77/77 | 0,205066 → 0,198997 | 13,578564 → 13,426316 |

F0 pasa la regla fijada de cobertura y no empeoramiento de las cuatro métricas, con alguna mejora estricta. El router no la pasa: la regresión en carretera es pequeña, pero el criterio no admitía tolerancia elegida después de observar resultados. No cambiarlo retrospectivamente. Tampoco interpretar la magnitud pequeña como daño estadísticamente demostrado.

La mejora de carretera de F0 es mínima; la de Slider es más visible. Esto apoya ampliar el contraste de F0, no una afirmación general de superioridad ni un cambio de entrega. Los grupos tienen exposición histórica. Se mantuvieron las ausencias y no se registraron cambios de signo. El informe declara reproducción exacta de los arrays originales y extracción/replay de 999,372 s.

El registro de continuación comunica una extracción de 2.608 consultas en curso, reutilizando 149 y añadiendo 2.459. No se toma ese registro de progreso como resultado terminado. Esta revisión no inició, duplicó, interrumpió ni modificó el proceso. La próxima decisión depende de sus recibos de finalización y evaluación, no de que exista una carpeta de salida.

## La conservación de eAP sigue abierta

Los recibos inventariados de `E_CTX_MATCHED` y `R_CTX` corresponden a `FIVE_FOLD_PHYSICAL_OOF`, con padres `OOF_AUTHORITY` y sin checkpoints de despliegue declarados. No pueden mezclarse con PAIR133 de productores de despliegue para certificar conservación de toda la cadena. El inventario leyó recibos, no arrays NPZ, y no ejecutó productores.

La recomendación actual pasa de «terminar screen149» a **evaluar la ampliación de F0 y reconstruir historias eAP con genealogía compatible**. La observabilidad dirigida a q sigue siendo el siguiente diagnóstico; no hay razón nueva para lanzar conjuntamente reloj, sensor y confianza ni para reabrir JEPA genérico.
