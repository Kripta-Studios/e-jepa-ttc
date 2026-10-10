# 06 — V13: auditoría, traslado de mejoras y continuidad

Todas las referencias V13 de este informe están fijadas al commit [`da068ebdeb2d967ee10b2afa348e1317b8720ad0`](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0). No buscar estos módulos solo en la rama V12. Las métricas V12 no son resultados V13.

## Auditoría previa de RGB-PORT

La decisión y las pruebas originales están en [ADR-0002](https://github.com/Kripta-Studios/e-jepa-ttc/blob/da068ebdeb2d967ee10b2afa348e1317b8720ad0/docs/decisions/ADR-0002-rgb-port-audit-corrections.md) y [auditoría publicada](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0/docs/rgb_port_audit_20261010). Es una fase histórica: sus referencias a entrenamientos pausados no sustituyen el estado posterior de reanudación.

| Corrección | Código V13 | Motivo y comprobación |
|---|---|---|
| Router C2F por modalidad | `src/e_jepa_ttc/models/causal_scale_ttc.py`, alrededor de línea 1082 | RGB ya no interpreta verde/azul como estadísticas de eventos: luminancia y gradiente espacial con log1p. Rama eventos conserva matemática |
| Batch efectivo T2/T3 | `operational/rgb_port_revision/loss.py`; integración en `rgb_port/train_producers.py` | Reducción por elementos válidos de cada término, top-k global del batch efectivo, un backward y un update; sin padding de frames |
| GroupRowCache | `rgb_port_revision/cache.py` | Retiene filas pendientes del grupo de 256 en vez de shards enteros; orden y tensores preservados |
| Intervalos fase→TTC | `rgb_port_revision/outputs.py`; `rgb_port/predict_heads.py` | Invierte extremos dentro de rama; si toca/cruza cero, TTC no disponible con estado explícito; conserva intervalo fase |
| Transferencia Dev32 | `rgb_port_revision/transfer_inputs.py`; `rgb_port/transfer.py` | Historia construida desde timestamps reales, no desde distancia entre consultas de evaluación |
| Rutas de coste | `rgb_port/cost_routes.py`, `rgb_port_revision/raw_events.py`, `rgb_port/profile.py`, `rgb_port/report.py` | HDF5→voxels y HDF5→productores→fusión con paridad previa; corrige lectura de warm_ms.median/p95 |
| Migración de código | `rgb_port_revision/migration.py`; `AUDIT_CODE_MIGRATION.json` | Admite exclusivamente hashes históricos→revisión auditada; conserva originales y manifests |

Directorio de [código de revisión](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0/operational/rgb_port_revision). `typing.cast` solo corrige tipos estáticos: no convierte tensores BF16 a FP32.

La caché nueva redujo lecturas de shards 131→21 y tiempo CPU 85,74→51,01 s sobre 256 ejemplos, con igualdad de tensores/metadatos y máximo retenido de 469.701.456 bytes dentro del límite A5 de 512 MiB. C2F conserva 1 GiB. Los hits cambian de unidad (filas frente a shards). Esta prueba secuencial no demostraba el rendimiento de entrenamiento concurrente.

La validación T2/T3 incluyó un batch real con 3 ejemplos T2 y 29 T3, sin update. La transferencia eventos usa cuatro observaciones a 100 ms, cada una con tres ventanas, dentro de 650 ms. RGB usa frames reales, tolerancia de selección 1 ms, deltas medidos, tripletas y T2 solo en arranque real. La ROI histórica sigue dependiendo de la disponible en la consulta; no se elimina esa limitación con una corrección de timestamps.

`ttc_interval_low/high/status` son límites transformados. Los alias q10_ttc/q90_ttc no garantizan cuantiles marginales TTC calibrados. El coste GPU incluye pico asignado/reservado; RSS CPU antes/después no es pico de RAM.

La primera auditoría acreditó 134 casos únicos aprobados, pérdidas/gradientes de eventos exactamente iguales en CPU, restauración de pesos/AdamW/scheduler/cursor/RNG y cero updates. Los conteos posteriores pertenecen a suites ampliadas: no sumarlos como si fueran pruebas disjuntas.

## Port nativo de las mejoras V12

El [informe de port](https://github.com/Kripta-Studios/e-jepa-ttc/blob/da068ebdeb2d967ee10b2afa348e1317b8720ad0/docs/rgb_port_streaming_20261010/README.md) y [`operational/rgb_port_streaming`](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0/operational/rgb_port_streaming) detallan la adaptación:

- `inputs.py`: `EventInputs` conecta PacketRing y preparación incremental; `RGBInputs` conserva frames uint8 acotados y aplica crop nativo con timestamps reales.
- `runtime.py`: `NativeStream` reutiliza exactamente por defecto, agrupa misses según T2/T3 real y conserva identidad de clocks/deltas. `NativeExperts` usa copias privadas de los productores.
- `transport.py`: correlación vectorizada y aislamiento de globals/objeto de código para evitar colisiones de compilación entre instancias.
- `contracts.py` e `infer.py`: admiten `known_mask` real [B] sin alterar su semántica; el validador antiguo esperaba [B,T].
- `endpoints.py`: exige padres V13 finales y normalizador nativo del rol H. No sustituye pesos V12 ni parciales si faltan endpoints.
- `distill.py`: piloto de imitación con identidad de modalidad/padres/PHASE17 y partición por secuencia. Rechaza el esquema V12. No hay estudiante V13 real final entrenado en esta evidencia.

Ocho posiciones máximas de cabeza no autorizan a violar 650 ms. En el protocolo Dev32 V13, observaciones de eventos separadas 100 ms con tripletas permiten cuatro observaciones reales. No se comprime artificialmente la cadencia para rellenar ocho. H8/H4/H2 son opciones de inferencia que requieren su propia evaluación.

## Probe GPU y límites

El [GPU_PROBE.json](https://github.com/Kripta-Studios/e-jepa-ttc/blob/da068ebdeb2d967ee10b2afa348e1317b8720ad0/docs/rgb_port_streaming_20261010/evidence/GPU_PROBE.json) utiliza checkpoints parciales A5=26.338 y C2F=17.679, entradas sintéticas batch=2 y PAIR inicializado de forma determinista. Diez repeticiones:

| Ruta | Mediana núcleo ms | Máxima diferencia absoluta PHASE17 |
|---|---:|---:|
| Nativo FP32 | 86,072 | 0 |
| Vectorizado FP32 | 71,422 | 0,00003052 |
| Vectorizado + graphs FP32 | 17,490 | 0,00003052 |
| Vectorizado + encoder BF16 | 58,003 | 0,860603 |

Graphs tuvo primera llamada de 9,623 s. BF16 selectivo sigue experimental por su diferencia de características; hace falta medir TTC. No es E2E, no son datos reales, no es una comparación Garl y no demuestra precisión V13. Los primeros intentos fallaron por NumPy dentro del grafo y por identidad de código compartida; los logs se conservaron y la ruta posterior pasó.

## Por qué se pudo reanudar sin empezar de cero

La continuidad se implementa en [`operational/rgb_port_continuity`](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0/operational/rgb_port_continuity). Pausar y volver a guardar el mismo update podía reserializar el checkpoint, cambiando sus bytes y dejando recibos auxiliares apuntando al hash anterior. No significaba necesariamente que se hubieran ejecutado updates nuevos.

`checkpoint.py` verifica igualdad completa del estado —pesos, optimizador, scheduler, cursor y RNG Python/NumPy/CPU/CUDA/sampler— antes de reutilizar un checkpoint en el mismo update. Si algo cambia, falla antes de escribir. En una actualización nueva conserva la cadena original de validadores. Las congelaciones históricas no se sobrescriben silenciosamente.

Para A5 en update 26.338 se archivaron seis archivos auxiliares obsoletos, conservando checkpoint, puntero, recibo nativo, journal y pruebas pendientes. Un intento inicial con cinco archivos se revirtió porque `PIPELINE_RUNTIME.json` también estaba obsoleto; la segunda transacción pasó. [Evidencia de recuperación](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0/docs/rgb_port_streaming_20261010/evidence/a5_pause_recovery). No se declara paridad byte a byte con una serialización antigua no disponible.

La admisión Git exigió demostrar descendencia del commit original y que las fuentes admitidas coincidieran exactamente con el commit publicado. `CONTINUITY_GIT_ADMISSION.json` vincula un HEAD concreto. Una futura publicación V13 exige conservar/renovar esa admisión para otra reanudación. Por eso este handoff se publica en V12 y solo enlaza V13.

Las pruebas ampliadas cubren 235 casos únicos aprobados y una prueba técnica opcional omitida; Ruff/Pyright aprobados. Los warmups de restauración verificaron estado sin consumir updates y después hubo nuevos checkpoints durables. Esto demuestra continuidad efectiva, no solo capacidad teórica de cargar un archivo.

## Estado de entrenamiento y pendientes

La reanudación partió de A5=26.338 y C2F=17.679; endpoint previsto=30.330 cada uno. El [snapshot publicado](evidence/V13_STATUS_SNAPSHOT.json) contiene la hora UTC de lectura y los journals originales; es una instantánea, no un monitor. Los contadores son de optimización, y `durable_updates` identifica el último estado persistido. Pueden diferir mientras corre el entrenamiento.

Los entrenamientos RGB R_A5/R_C2F no deben darse por completados por el avance de los dos productores de eventos. Faltan endpoints finales, etapas posteriores, extracción H, cabezas/contexto/fusión y las evaluaciones que requieran esos activos. Las aproximaciones warp, estudiante y precisión selectiva no se activaron en la receta matched de entrenamiento.

No es necesario reiniciar por las correcciones de inferencia auditadas. Cambiar ahora la loss, las muestras o la geometría de entrenamiento sería otro experimento y exigiría documentar una nueva identidad. La precisión BF16, orden del sampler, splits y datos E: continúan bajo sus contratos existentes.
