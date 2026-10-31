# R2-rare: cierre recuperado y decisión de escala

Fecha: 2026-10-31. Tablas regeneradas desde evidencia JSON.

R2-rare termina 8.192 updates y 32.768 exposiciones sobre 4.096 frames únicos. La AP completa final es 0.11565681, frente a 0.11645298 de R2 (-0.00079617 absoluto). El screening prerregistrado no se supera; se conserva **R2** como receta de trabajo para la ampliación.

## Comparación con presupuesto y población fijados

Mismos D4096, R50, ocho clases, batch cuatro, FP32, 50 queries y 11 grupos. El cambio declarado es el calendario de muestreo raro. No es fine-tuning del R2 final: la intervención empezó desde la misma inicialización del screening.

| Update | AP R2 completa | AP R2-rare completa | Diferencia |
|---:|---:|---:|---:|
| 4096 | 0.08690119 | 0.10391695 | +0.01701576 |
| 8192 | 0.11645298 | 0.11565681 | -0.00079617 |

Selección completa: 23,958 frames. Se verificaron hashes de archivos y contenido de todos los GT; las poblaciones son idénticas. Esta selección está expuesta a desarrollo, no es test ni scorer oficial.

| Clase | GT evaluable | AP R2 | AP R2-rare | Diferencia |
|---|---:|---:|---:|---:|
| car | 59074 | 0.53165455 | 0.50728895 | -0.02436560 |
| bicycle | 687 | 0.01363593 | 0.02011027 | +0.00647435 |
| truck | 14083 | 0.19315043 | 0.17133467 | -0.02181576 |
| pedestrian | 7460 | 0.01441345 | 0.01969303 | +0.00527958 |
| motorcycle | 22187 | 0.10524124 | 0.09934138 | -0.00589986 |
| bus | 1916 | 0.06019388 | 0.09606558 | +0.03587170 |
| tricycle | 1828 | 0.01333437 | 0.01138991 | -0.00194447 |
| other_vehicle | 73 | 0.00000000 | 0.00003066 | +0.00003066 |

| Distancia XY | Recall sin exigir clase R2 | R2-rare |
|---:|---:|---:|
| 0.5 m | 0.280538 | 0.245452 |
| 1.0 m | 0.462118 | 0.436929 |
| 2.0 m | 0.662886 | 0.649635 |
| 4.0 m | 0.846013 | 0.849322 |

| Clase rara, umbral 2 m | TP R2 | FN R2 | TP R2-rare | FN R2-rare |
|---|---:|---:|---:|---:|
| bicycle | 51 | 636 | 99 | 588 |
| other_vehicle | 0 | 73 | 5 | 68 |

Los TP/FN usan el pool del protocolo y exigen clase correcta. No representan un umbral de score operacional calibrado. El recall agnóstico mide otro aspecto de las propuestas y no sustituye la AP.

Criterios fijados antes del resultado:

- Mejora la AP macro completa: **FAIL**.
- Mejora el promedio AP de bicicleta y other_vehicle: **PASS**.
- Ninguna otra clase pierde más de 0,01 AP absoluto: **FAIL**.

La regla es operacional, no un contraste estadístico. Una única trayectoria no demuestra una contribución causal reproducible ni liderazgo SOTA. Las semillas adicionales continúan aplazadas.

## Curva de desarrollo

| Update | AP R2, panel 1.024 | AP R2-rare, panel 1.024 |
|---:|---:|---:|
| 1024 | 0.06220262 | 0.05400467 |
| 2048 | 0.07811640 | 0.06602242 |
| 3072 | 0.10292892 | 0.10507807 |
| 4096 | 0.09026599 | 0.11336509 |
| 5120 | 0.10384537 | 0.11326751 |
| 6144 | 0.11027994 | 0.11751462 |
| 7168 | 0.11491516 | 0.12012420 |
| 8192 | 0.11681927 | 0.12636143 |

La comparación principal usa checkpoints finales al mismo presupuesto. Los checkpoints seleccionados por el panel corto se conservan por separado.

## Recuperación y límite de reproducibilidad

El checkpoint recuperado estaba en update 3.840. Se repitieron los 256 updates registrados después de ese checkpoint y se continuó hasta 8.192. Se verificaron configuración, productor congelado, estado de optimizador, scheduler, sampler, RNG y calendario consumido. El ledger final cuenta una sola vez cada update comprometido; no suma la repetición a las exposiciones.

La repetición no es idéntica bit a bit: en el panel 4.096, AP 0.10081029 antes del corte y 0.11336509 tras reanudar. El primer loss reanudado coincide exactamente y la siguiente diferencia es pequeña, pero crece durante los 256 updates. Lotes y LR coinciden. El productor usa grid_sample y una prueba estricta en el pod confirma que su backward CUDA no dispone de implementación determinista. Esto identifica una fuente posible, sin demostrar que sea la causa exclusiva. Se documenta la trayectoria reanudada y no se mezclan sus resultados con los monitores no comprometidos anteriores al corte.

## Decisión sobre Dfull

**Sí se justifica ampliar al train completo legible, partiendo de R2.** La justificación es la cobertura de datos que falta en D4096; no una promesa de que entrenar más resolverá todas las clases. No es necesario lanzar más semillas del screening antes de esa fase de datos. La ejecución Dfull se documenta por separado para no confundir su progreso con este cierre.

La auditoría local admite **70,279 frames de 24 secuencias train**, tras excluir tres medios defectuosos. Se mantienen separadas selección y calibración. Los tres defectuosos no tienen anotaciones y no eliminan identidades entrenables.

| Clase | Anotaciones D4096 | Dfull legible | IDs D4096 | IDs Dfull |
|---|---:|---:|---:|---:|
| bicycle | 162 | 655 | 69 | 69 |
| bus | 110 | 1958 | 24 | 35 |
| car | 10455 | 181954 | 1548 | 2235 |
| motorcycle | 693 | 10338 | 362 | 682 |
| other_vehicle | 7 | 53 | 6 | 6 |
| pedestrian | 245 | 3856 | 130 | 255 |
| tricycle | 97 | 1200 | 51 | 51 |
| truck | 1994 | 34417 | 388 | 612 |

IDs significa (secuencia, track_id), no independencia física o de adquisición. Bicicleta y other_vehicle no ganan identidades; sus nuevas vistas y exposiciones no equivalen a nueva diversidad. El límite de soporte raro sigue vigente.

El lanzamiento Dfull requiere manifest y configuración versionados con las exclusiones, perfil de memoria acotada y presupuesto de validación explícito. Sus imágenes preparadas FP32 ocuparían unos 193 GiB. La continuación usa una representación compacta reversible y precarga CPU, verificadas contra los tensores originales; la velocidad debe medirse en esa ejecución.

## Alcance de entrega

Se cierra este experimento RGB de detección. Dfull sería otra etapa evaluada, y no convierte automáticamente el detector en un modelo de producción. Siguen pendientes la validación final por dominio/clase, rendimiento operativo y evaluación integral de detección, asociación y PHASE. La independencia respecto a TRAIN40 y las comparaciones temporales/eventos del plan requieren su propia evidencia; este resultado no acredita nueva precisión TTC.

Regeneración: `python scripts/report_detection_rare_result.py --analysis <analysis.json> --readiness <dfull_readiness.json> --output <informe.md>`.
