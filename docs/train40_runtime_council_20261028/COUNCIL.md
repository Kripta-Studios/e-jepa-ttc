# Consejo TRAIN40 — cierre de ronda solicitado, 28/10/2026

El usuario ha solicitado cerrar esta ronda. Root pausará el objetivo al completar
la publicación; la meta técnica sigue pendiente: 50 actualizaciones completas por segundo y menos de
20 ms por actualización, primero con un objeto y después con dos. La extensión
a veinte objetos queda para una fase posterior. Ningún ensayo de esta ronda
admite todavía ese objetivo. No se han reentrenado modelos ni autorizado cambios
de pesos, normalizadores o tolerancias de paridad.

La ronda llega a un replay de **250/250 consultas con paridad**, sin descartes
ni errores, durante cinco segundos ofertados. Continúa fallando el límite de
20 ms y el gate de ausencia de trabajo previo. No equivale a los 60 segundos
del ensayo sostenido ni a adquisición de un sensor físico.

Una extensión posterior con caché de crops perdió cobertura. Sus TTC
producidos conservan paridad, pero hay respuestas fallidas y descartes; el
resultado negativo se mantiene separado del ensayo anterior. Antes de
atribuir la regresión a una optimización se requiere un control comparable
con la misma fuente y versión. La traza GC propuesta observa correlaciones
sin cambiar políticas ni descontar pausas.

La base recibida fue `2521de1a`. El consejo repartió la investigación entre
numerics, RGB, eventos y medición, con revisiones cruzadas. Root reservó la GPU
en exclusiva y suspendió los tests CPU durante las mediciones. Las decisiones
de implementación y las hipótesis rechazadas permanecen vinculadas a recibos;
el cierre y las instrucciones de reanudación están en [HANDOFF.md](HANDOFF.md).
El snapshot `council_v1` conserva intacto su estado histórico anterior al cierre.

## Resultados y decisiones

Las cifras de latencia, cobertura, paridad y preservación se leen en
[RESULTS_GENERATED.md](RESULTS_GENERATED.md), generado desde JSON, y sus hashes
están en [RESULTS_SOURCES.json](RESULTS_SOURCES.json). Las poblaciones de distintos
relojes o distintos ensayos no se suman. Las siguientes decisiones se apoyan en
los recibos enlazados, sin convertir un microbenchmark en admisión del servicio.

| Decisión | Evidencia y alcance |
| --- | --- |
| Mantener PIL con preparación de ROI primero como perfil por defecto | Se corrigió el flag de imágenes compartidas del runner para respetar el perfil admitido. Uint8 y preparadores alternativos siguen siendo opt-in. |
| No promover native-log ni native-match | [Ensayo native-log](../../artifacts/train40_runtime_council_20261028/native_log_sensitive_v1/summary.json) y [native-match](../../artifacts/train40_runtime_council_20261028/compiled_pipeline/native_match_sensitive_v1/summary.json) conservan fallos de TTC. El reporter agrega todas las llamadas de paridad por variante; los recibos antiguos quedan etiquetados `last-only`. |
| Conservar el control con lectura RAW de disco | [Baseline paced](../../artifacts/train40_runtime_council_20261028/paced/baseline1_50_v1/summary.json) falló, con adquisición, descartes y una captura fría registrados dentro de sus relojes. |
| Añadir primación sintética de las formas nativas explícitas | Prepara formas con ceros antes del replay; no consume frames futuros ni instala historia del sensor. Startup, formas, capturas y memoria quedan en recibos. No sustituye la paridad sobre datos reales. |
| Evaluar una fuente CPU precargada con frontera explícita | [Paridad RAW real](MEMORY_RAW_PARITY.md) acredita los datos que entrega frente al control de disco; [piloto paced memory](../../artifacts/train40_runtime_council_20261028/paced/memory1_50_v1/summary.json) conserva su fallo de cadencia/cobertura. Precarga CPU no equivale a adquisición de un sensor físico. |
| Mantener opt-in la caché RGB y el lanzamiento temprano de eventos | Tokens RAW inmutables permiten retener identidad causal. [Piloto de overlap](../../artifacts/train40_runtime_council_20261028/multiobject/overlap1_threecases_v1/summary.json) conserva paridad de sus consultas y mide el tramo completo declarado, sin probar un stream sostenido. |
| Conservar el fallo paced con caché y overlap | [Piloto conjunto](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap1_50_v1/summary.json) ejercita MISS y HIT, pero acumula trabajo, pierde un slot e invalida la historia posterior. Un HIT con tiempo de servicio local bajo 20 ms sigue incumpliendo el deadline cuando llega con cola. |
| Admitir el companion de inputs inmutables en su alcance diagnóstico | [Companion FrozenRGBHistory](../../artifacts/train40_runtime_council_20261028/multiobject/immutable_rgb_companion_v1/summary.json) completa las consultas fijas con una y dos ROI, MISS→HIT y paridad TTC. Comprueba argumentos device de productores en MISS y features de HIT contra el MISS canónico; el sellado previo queda explícitamente fuera de su reloj interno. No es un replay paced. |
| Evaluar reutilización exacta de transporte como opción independiente | [Ensayo preparado](../../artifacts/train40_runtime_council_20261028/producers/exact_transport_sensitive_v1/summary.json) verifica cada llamada explícita, TTC y cuatro conjuntos de features exactas. Su ahorro preparado no equivale a ahorro de actualización completa. |
| Extender el piloto paced que ya conserva todos los slots | [Piloto con reverse reuse](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_50_v1/summary.json) completa el prefijo de 50 consultas con paridad, sin descartes ni errores. El hito sigue fallando por deadline, acumulación y duración insuficiente. |
| Cerrar la fuente después de drenar el consumidor | [Control lifecycle v2](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_50_v2/summary.json) conserva cobertura y paridad, con finalize posterior al último TTC. Se evita que el cierre de la sesión compita con las últimas consultas; no se atribuye todo outlier anterior a ese solapamiento. |
| Conservar el ensayo completo de cinco segundos y sus incumplimientos | [250 consultas](../../artifacts/train40_runtime_council_20261028/paced/memory_cache_overlap_reverse1_250_v1/summary.json) tiene paridad y preservación, pero 201 respuestas alcanzan o superan 20 ms. Las tablas regeneradas conservan todos los percentiles. El presupuesto explícito de esta fuente es 3 GiB, distinto del piloto corto de 2 GiB. |
| Separar TTC válidos de acknowledgements fallidos | Tras un hueco RAW, el servicio invalida su historia y responde rápidamente con errores. Esos errores no son TTC: las tablas principales filtran por éxito de todos los objetos y muestran fallos, descartes y ausencia de respuestas aparte. |
| Mantener SHA como identidad runtime mientras se evalúa BLAKE3 | [Micro CPU](../../artifacts/train40_runtime_council_20261028/frame_hash_cpu/idx553_v1/summary.json) compara copia y hash de frames, con paridad de bytes y tokens referenciados. No cambia el hash del runtime ni acredita reducción de latencia TTC. |
| Separar la admisión de crops de su comportamiento paced | [Companion de crops](../../artifacts/train40_runtime_council_20261028/multiobject/crop_rgb_companion_v1/summary.json) verifica MISS→HIT de features→borrado de features con HIT de crops en una y dos ROI. El reporter conserva cada repetición, los píxeles/deltas reales, el borrado forzado y la matriz completa; excluye sus tiempos de comparaciones paced. |
| Conservar la regresión observada con crops | [Replay crops de 250 consultas](../../artifacts/train40_runtime_council_20261028/paced/memory_crops_reverse1_250_v1/summary.json) registra TTC válidos, acknowledgements fallidos y drops por separado. El gate global falla aunque cada TTC producido pase paridad. La evidencia no establece la causa de la regresión. |
| Instrumentar GC antes de atribuirle outliers | [Propuesta GC trace v1](GC_TRACE_V1_PROPOSAL.md) registra callbacks con clocks monotónicos, hilo y generación dentro de una capacidad fija. No fuerza colecciones, altera umbrales ni descuenta pausas; la correlación con relojes RAW/servicio/deadline es posterior. |

El preparador RGB exacto también conserva su [fallo inicial de setup](../../artifacts/train40_runtime_council_20261028/rgb_exact_gpu/idx553_v1/summary.json)
y el [ensayo posterior](../../artifacts/train40_runtime_council_20261028/rgb_exact_gpu/idx553_v2/summary.json).
Ese ensayo compara píxeles y deltas de preparación; excluye modelos y TTC. No
acredita por sí solo la paridad del servicio con caché.

## Fronteras y versiones de protocolo

| Protocolo o esquema | Frontera y limitación |
| --- | --- |
| `train40_paced_raw_replay_v1` | Deadlines absolutos, productor independiente, cola acotada y todos los objetos del update. Conserva descartes, errores, clocks y recibos por objeto. |
| `train40_paced_milestone_gate_v2` | Gates distintos para uno y dos objetos, con 60 s ofertados, cobertura íntegra, todos los deadlines bajo 20 ms, ausencia de acumulación y paridad. Un estado `COMPLETE` del proceso no demuestra este hito. |
| `train40_paced_latency_populations_v1` | Estadísticas puras desde `grouped.updates.passed`, cobertura, llegadas y resultados por objeto. `successful_all_object_ttc` es la población principal; `failed_acknowledgements` se informa separadamente. Campos legacy etiquetados `ALL_RESPONSES_INCLUDING_FAILURES`. Cuantiles lineales, población vacía igual a `null`, drops sin latencia inventada. No modifica gates históricos. |
| `train40_paced_source_lifecycle_v2` | El productor publica fin de ofertas, espera drenaje o aborto del consumidor y cierra en su hilo propietario. El wall total conserva cierre/join; se distingue la última respuesta válida. |
| `train40_paced_wall_intervals_v2` | Corrige un defecto preexistente de startup: un fallo de warmup con lead futuro podía producir duración negativa. Sin primera oferta, el intervalo paced es `null`; el wall real desde entrada incluye startup. No se reescriben recibos históricos. |
| `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR` | RAW privado residente antes del reloj; selección, conversión de ABI, copia/sellado de nuevas llegadas y trabajo del servicio dentro del deadline. No se preparan tokens, crops ni features futuros. [Propuesta y cotas](PRELOADED_RAW_CPU_ARRIVALS_PROPOSAL.md). |
| `PREPARED_GPU_PRODUCERS_PAIR_PHASE17_GUARDS_NO_RAW_H2D_CONTEXT` | Productores, transporte y guards desde inputs preparados en GPU; excluye RAW, H2D y contexto. Paridad fuera del reloj de productores, conservando cada repetición registrada. |
| `PREPARED_ONE_OBJECT_EXACT_TRANSPORT_ALL_EXPLICIT_CALLS_NOT_COMPLETE_UPDATE` | Todas las llamadas iniciales, warmup y medidas tienen controles TTC y features. Mantiene batches nativos de un objeto; no admite dos objetos ni el reloj completo. |
| `IMMUTABLE_RGB_CACHE_INPUT_ADMISSION_FIXED_QUERY_REPEATS_NO_ARRIVAL_TIMING` | Historia RAW original independiente frente a tokens factory, argumentos reales device, deltas y features MISS/HIT. Las repeticiones mantienen el anchor; los tiempos instrumentados no se comparan con los paced. |
| `IMMUTABLE_RGB_CROP_AND_FEATURE_CACHE_ADMISSION_FORCED_MISS_NO_ARRIVAL_TIMING` | Añade un tercer paso que vacía sólo features y recupera crops inmutables. Comprueba argumentos reales del productor frente al RAW original, sin cambiar el anchor ni convertir el tiempo interno en latencia de llegada. |
| `train40_gc_trace_v1`, propuesto | Observación opt-in y acotada durante el call paced; callbacks y solapamientos conservan los clocks existentes, eventos descartados e intervalos incompletos. Sin cambios de política, causalidad inferida ni admisión de tiempo real. |
| `stress_replay_v1`, propuesto | Reutiliza episodios cortos con timestamps originales. Cada transición/reset y su prefijo causal se contabilizan dentro del update. Es carga repetida, no una secuencia natural nueva de 60 s. [Diseño pendiente de implementar](STRESS_REPLAY_V1_PROPOSAL.md). |

`deadline_response_ms` comienza en la llegada programada, antes de adquirir o
sellar el payload, y termina con los resultados de host. Incluye retraso de
adquisición y cola. `raw_availability_response_ms` empieza después de adquirir
el payload; `service_span_ms` empieza al tomarlo el consumidor. Se publican
como relojes distintos. La admisión no descuenta jitter ni convierte la media
de latencia en una frecuencia sostenida.

El bloque histórico `capacity_checks` mantiene su objetivo de veinte objetos.
Para la prioridad actual se debe leer `milestone.target_object_count` y su
`milestone_checks.required_object_count`, que derivan el gate separado de uno
o dos objetos. Que falle `twenty_distinct_objects_per_update` en un piloto de
un objeto no es un defecto de ese piloto; sus incumplimientos de deadline,
cobertura y duración se evalúan igualmente y permanecen registrados.

## Datos, cobertura y preservación

La [auditoría de medición](MEASUREMENT_AUDIT.md) y su
[inventario de 32 fuentes](../../artifacts/train40_runtime_council_20261028/measurement_audit_v1/raw_sequence_inventory.json)
establecen que ninguna fuente local alcanza por sí sola los 60 s naturales
del protocolo. Existe un [manifiesto causal CPLA-low](../../artifacts/train40_runtime_council_20261028/measurement_audit_v1/causal_pilot_manifest.json)
con 250 posiciones aceptadas y su [recibo regenerable](../../artifacts/train40_runtime_council_20261028/measurement_audit_v1/manifest_generation_receipt.json).
El prefijo de 50 posiciones permite pilotos completos cortos, manteniendo los
slots fallidos en el denominador.

El companion RAW comparó historia inicial y cada posición entre la fuente de
memoria y `DiskRawSource`, incluidos eventos, dtypes, píxeles, timestamps,
tripletas, cajas y calibración. Conserva fuentes exactas y hashes en su
[binding](../../artifacts/train40_runtime_council_20261028/memory_raw_parity_v1/binding.json).
Los HDF5 se verificaron por tamaño/mtime, sin atribuirles un hash completo que
no se calculó. Los recibos de preservación identifican qué fuentes y metadata
fueron comprobadas; el companion no cargó pesos ni ejecutó TTC.

En el ensayo de 250 consultas, 200 deadlines encontraron una actualización
previa todavía pendiente y 50 no encontraron ninguna; el máximo observado fue
uno. Esto caracteriza la acumulación durante esos cinco segundos, sin probar
estabilidad fuera de ellos. La fuente declara una cota numérica de
2.611.872.720 bytes bajo 3 GiB; excluye overhead Python, HDF5 interno, modelos,
GPU y recibos. El último TTC llegó a 5,0036785 s y el wall con cierre/join fue
5,0493202 s, ambos referidos al primer deadline.

## Verificación CPU

La corrección de poblaciones pasó los 52 casos de
[test_train40_runtime_paced.py](../../tests/unit/test_train40_runtime_paced.py)
y [test_train40_council_report.py](../../tests/unit/test_train40_council_report.py),
Ruff, Pyright y `git diff --check`. Incluye fallos rápidos tras TTC lentos,
población sin éxitos, cobertura contradictoria, drops/missing sin imputación,
lectura histórica sin mutar sus JSON y ejecución CLI aislada sin importar
Torch, PIL ni NumPy. La verificación fue CPU; no certifica kernels ni cadencia.
El [recibo QA](../../artifacts/train40_runtime_council_20261028/measurement_qa_v1/qa.json)
conserva comandos, versiones de fuentes por SHA, códigos de salida y hashes
de los logs de cada comprobación.

Comando de verificación de estos cambios:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -m pytest `
  tests/unit/test_train40_runtime_paced.py `
  tests/unit/test_train40_council_report.py -q
```

El archiver admite recibos compactos de hasta 8 MB, preflight, comandos, QA,
manifiestos y snapshots de fuentes/metadata. Comprueba hashes y rechaza enlaces
que escapen de la raíz. No copia pesos, RAW, caches de compilación o dependencias.
Cada paquete de evidencia se crea en un destino nuevo; los artefactos ejecutados
se conservan intactos, incluidos sus fallos.

## Trabajo pendiente

1. Reducir la latencia sobre las 250 consultas con las siguientes optimizaciones,
   manteniendo un control comparable al ensayo de cinco segundos ya conservado
   y toda la población de TTC válidos.
2. Conservar en cada extensión la evidencia del companion de inputs y su
   identidad de código/modelos; repetirlo si cambian crop, caché o ABI. Mantener
   la paridad de todos los TTC, identidad causal y preservación en el paced.
3. Resolver el coste que todavía supera 20 ms y la acumulación. Cualquier
   subdivisión del ingest debe medir deuda y trabajo completo de cada período.
4. Ejecutar un protocolo sostenido explícito cuando el piloto pase: conseguir
   datos continuos suficientes o implementar y etiquetar `stress_replay_v1`.
5. Tras admitir un objeto, repetir cobertura y todas las salidas con dos objetos;
   luego diseñar el escalado a veinte sin extrapolar desde throughput por ROI.
6. Regenerar el informe y empaquetar evidencia compacta tras los siguientes
   ensayos al reanudar. Esta ronda termina por petición del usuario; la pausa
   administrativa no constituye admisión del objetivo técnico.

El diagnóstico final de subpaquetes de 5 ms conserva el fallo de presupuesto
v1 y el pase numérico v2, sin medir latencia ni integrar una fuente paced.
Ver [SUBPACKET_NUMERICS.md](SUBPACKET_NUMERICS.md) y las tablas regeneradas.
El [perfil final](DEFAULT_PROFILE.md) deja crops, wiring BLAKE3 y 5 ms fuera
del default. Las propuestas GC/V2 quedan congeladas sin implementación adicional.
