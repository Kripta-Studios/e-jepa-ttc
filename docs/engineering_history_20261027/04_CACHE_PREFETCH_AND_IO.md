# Caché, prefetch e I/O: de NPZ comprimido a staging y RAW verificable

## 1. Alcance

Este documento reconstruye el plano de datos que alimentó las campañas: lectura de NPZ comprimidos, cachés en proceso, prefetch acotado, transferencia a volúmenes de red, staging local y control de page cache. Las cifras se presentan con el alcance de su prueba. En particular:

- una mejora en un benchmark pareado del lector no equivale automáticamente a una mejora de entrenamiento;
- bytes aconsejados con `POSIX_FADV_DONTNEED` no son bytes de RAM necesariamente liberados;
- una copia completada y hasheada demuestra integridad, no throughput científico;
- no se usaron `drop_caches` global ni controles que desalojaran indiscriminadamente datos de otros procesos.

## 2. Anatomía del cuello de entrada original

En la medición conjunta, el forward preparado pasó de unos 158 ms eager a unos 9 ms capturado, mientras la carga y transferencia de un batch B4 seguían en 170–179 ms. Esta separación mostró que acelerar el modelo no bastaba: el siguiente batch podía tardar más que el forward.

Un perfil CPU sobre las primeras 128 consultas atribuyó por batch B4:

- 110,627 ms a descompresión de miembros ZIP;
- 50,857 ms a arrays internos NPZ;
- 180,288 ms al batch original completo.

El formato tenía NPZ externos almacenados, pero miembros internos comprimidos con DEFLATE. La presión era, por tanto, una combinación de lectura, descompresión y construcción de arrays. Las pruebas privadas dieron 78,96 ms/B4 con cuatro lectores y 71,80 ms/B4 con ocho; una variante de decodificación desde memoria dio 78,22 y 54,82 ms/B4. Son pruebas concurrentes del lector, no una campaña científica completa. Fuente: [`INPUT_PIPELINE_20261016.md`](../joint_measurement_20261012/INPUT_PIPELINE_20261016.md).

## 3. Prefetch B1: un rechazo y una admisión

### 3.1 Prefetch v1 rechazado por el gate de rendimiento

La primera variante mantuvo exactitud, pero no alcanzó el umbral fijado de `ratio <= 0.8`. Dos pares midieron:

- 4,324→3,587 s, ratio 0,8294;
- 4,315→3,489 s, ratio 0,8086.

Aunque reducía tiempo, fue rechazada. Conservar este resultado evita reescribir la historia como una sucesión de mejoras siempre positivas.

### 3.2 Prefetch v2 admitido

La segunda variante, ligada por SHA `bf8ad4…`, usó un productor y ocho decodificadores. Mantuvo como máximo dos packs decodificados, aproximadamente 805 MB, además del pack comprimido y buffers de trabajo. Se ligó a hashes, CRC, tokens, tipos y reloj.

En dos repeticiones pareadas de 512 consultas:

| Métrica | Original | Prefetch v2 | Resultado |
|---|---:|---:|---:|
| Espera por batch, repetición 1 | 25,84 ms | 5,59 ms | menor espera visible |
| Espera por batch, repetición 2 | 25,52 ms | 5,21 ms | menor espera visible |
| Total, repetición 1 | 3,899 s | 2,173 s | ratio 0,5573 |
| Total, repetición 2 | 3,939 s | 2,053 s | ratio 0,5213 |

Eso equivale a 44,3–47,9% menos tiempo en el benchmark y 1,79–1,92 veces el throughput del lector. El primer batch seguía costando 494–530 ms, un recordatorio de que el prefetch amortiza después del arranque. Se comprobaron 516 entradas exactas y 132 salidas exactas, sin actualizaciones de optimizador. El RSS total del proceso fue 4,37 GB; no debe interpretarse como memoria exclusiva del reader.

## 4. Caché y prefetch durante entrenamiento

A1 entrenó con:

- prefetch de profundidad 1;
- LRU de 8 GiB;
- memoria pinned y copias non-blocking;
- CUDA Graph del modelo;
- datos comprimidos cuyo universo completo de píxeles rondaba 79 GiB.

La LRU cubría, como orden de magnitud, sólo alrededor del 10% del universo. Además, el reinicio del hijo cada 900 s descartaba caché y grafos. El perfil del hilo principal mostró espera de siguiente batch en 54,43%, por lo que el problema no era falta de una API CUDA más rápida, sino que la producción CPU no permanecía suficientemente adelantada.

En el refit posterior, un prefetch de profundidad 2 ocultó razonablemente la preparación:

| Fit | Preparación por batch | Espera directa del prefetch | Espera del decoder |
|---|---:|---:|---:|
| R_A5 | 415,010 ms | 4,531 ms | 111,412 ms |
| R_C2F | 425,212 ms | 3,771 ms | 95,151 ms |

La espera directa de pocos milisegundos no justificaba por sí sola aumentar threads o profundidad. Más concurrencia podía elevar RSS, page cache y contención sin reducir el camino crítico. Fuente: [`CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md`](../final_refit_20261017/CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md).

## 5. Exploración de una caché descomprimida

[`UNCOMPRESSED_INPUT.md`](../joint_measurement_20261012/UNCOMPRESSED_INPUT.md) documenta un probe acotado de 128 consultas. La suma de tiempos de la variante descomprimida respecto al original dio ratio 0,064516, y los ocho pares quedaron por debajo de 0,75. La prueba no usó CUDA ni realizó updates.

Se diseñó una caché completa para 53.890 entradas, con manifest, hashes, mmap y ocho packs, pero el propio plan exigía una comparación CUDA y una instalación sellada antes de producción. Por tanto, el resultado correcto es: la representación descomprimida era prometedora en CPU, no que toda la campaña científica entrenara desde esa caché ni que obtuviera un speedup del mismo factor.

## 6. Prefetch acotado en Runpod

La migración incorporó [`bounded_prefetch.py`](../../operational/runpod_migration/bounded_prefetch.py) y [`runtime_prefetch.py`](../../operational/runpod_migration/runtime_prefetch.py). El diseño añadió límites explícitos:

- profundidad configurable 1/2;
- presupuesto compartido por defecto de 512 MiB;
- máximo por batch de 192 MiB;
- máximo de decoder de 128 MiB;
- rechazo si la suma supera el límite;
- cursor y orden ligados al checkpoint;
- propagación fail-closed de abortos y errores;
- decoder RGB paralelo con cuatro workers donde aplicaba.

`ByteBoundedFoldPrefetch` controla bytes, no sólo número de elementos. Esto fue necesario porque un “batch” puede variar mucho en tamaño y un límite por cantidad no protege el cgroup. Los límites son parte del contrato operacional y no una alteración de batch, seed, scheduler o precisión.

## 7. Volumen de red, transferencia y staging local

### 7.1 Topología real

En Runpod, `/workspace` era MooseFS/FUSE sobre un volumen de red. El cgroup real era 60.999.999.488 bytes, aunque el anuncio comercial mostraba más RAM. Se trabajó con presupuesto piloto de 46 GB y reserva adicional de 2 GiB. Los artefactos no deben describir `/workspace` como disco local.

### 7.2 Transferencia técnica inicial

El tar piloto pesó 10.569.820.160 bytes, contenía 7.093 archivos y tenía SHA `a3dadc…`. [`NETWORK_TRANSFER_PROGRESS.json`](../../artifacts/runpod_migration_20261026/NETWORK_TRANSFER_PROGRESS.json) registró 10.560.208.896 de 10.569.820.160 bytes en 1.751,23 s, 5,7508 MiB/s. La finalización completa ocurrió a 1.763,51 s.

La primera observación del staging a `/opt` fue 198 MB en 125 s, 1,51 MiB/s, pero era una lectura parcial no durable. No se presenta como velocidad final. El objetivo del staging era ejecutar desde almacenamiento local inmutable y evitar releer continuamente código e inputs pequeños a través de FUSE. El preflight cargó 21,34 GB de memoria, de los cuales 20,75 GB eran cache, y observó 18,23 GB ocupados en el volumen. Fuente: [`NETWORK_TECHNICAL_OVERLAY_V3.md`](../runpod_migration_20261026/NETWORK_TECHNICAL_OVERLAY_V3.md).

### 7.3 Bundle de productores EVENT

El bundle EVENT completo tenía 45.593.222.091 bytes. Una captura intermedia, [`EVENT_TRANSFER_CURRENT.json`](../../artifacts/runpod_migration_20261026/EVENT_TRANSFER_CURRENT.json), registró 11.778.603.517 bytes. El protocolo evolucionó en scripts versionados y mantuvo seals por archivo y reanudación; una copia parcial nunca se promovía a bundle listo. No se atribuye aquí una tasa final que no esté respaldada por un recibo estable.

La compresión adicional no era una optimización asumible para NPZ ya comprimidos. Las muestras operativas dieron ratios cercanos a 1, por lo que el diseño favoreció paralelismo de streams y reanudación sobre recomprimir el payload.

## 8. Descarga RAW de 449 GB sin llenar el cgroup

La restauración RAW necesitaba transportar archivos individuales de hasta decenas de GiB sin retenerlos completos en page cache. [`runpod_event_hf_restore_v2.py`](../../scripts/runpod_event_hf_restore_v2.py) implementa un protocolo secuencial por `Range` HTTP:

1. validar repo, commit, ruta, tamaño y SHA esperados;
2. descargar un bloque de 64 MiB;
3. escribir y hacer `flush`;
4. ejecutar `fdatasync`/`fsync`;
5. actualizar el hash del prefijo;
6. aplicar `POSIX_FADV_DONTNEED` sólo al rango ya durable;
7. publicar atómicamente el recibo de reanudación;
8. pedir el siguiente bloque.

El servidor debía responder `206` con `Content-Range` y total exactos. Un partial sólo descontaba espacio reservado si su recibo de reanudación, identidad, offset, hash de prefijo y tamaño eran coherentes. Al terminar, se verificaban tamaño y SHA antes del rename y `fsync` del directorio. La reserva de cuota era 50 GiB más el archivo más grande en vuelo, sobre snapshot fresco de una cuota de 600 GB.

[`event_page_cache.py`](../../operational/runpod_migration/event_page_cache.py) contiene el helper durable de hash y eviction advisory. No llama a `drop_caches`; actúa únicamente sobre el fichero que acaba de escribir o verificar.

El recibo final [`POD2_RAW_ROOT_CHECK.json`](../../artifacts/runpod_migration_20261026/POD2_RAW_ROOT_CHECK.json) demuestra:

- estado `VERIFIED`;
- 41 archivos, 40 secuencias;
- 449.003.803.674 bytes;
- 41 hashes verificados;
- 6.711 llamadas de advice;
- 449.003.803.674 bytes aconsejados;
- `fsync` anterior al advice;
- ausencia de drop global;
- `data/train.parquet` presente con 3.945.777 bytes;
- SHA del recibo `464e50…`.

El contador de bytes aconsejados coincide con el tamaño lógico verificado. No prueba que el kernel liberara instantáneamente todos esos bytes, porque páginas mapeadas o activas pueden permanecer residentes.

## 9. Control de page cache durante entrenamiento

[`runpod_event_cache_controller_v1.py`](../../scripts/runpod_event_cache_controller_v1.py) se diseñó para recuperar margen sin tocar tensores vivos ni datos mutables. Su alcance queda restringido a NPZ inmutables bajo prefijos exactos `producer_inputs/` y `teacher/`, ligados a manifest y seal. Antes de aconsejar una entrada verifica:

- ruta bajo el root admitido;
- ausencia de symlink mediante `O_NOFOLLOW`;
- dispositivo, inode, modo, tamaño y metadatos sellados;
- apertura sólo lectura;
- identidad del manifest.

El controlador soporta cgroup v1/v2, observa `current`, `anon`, `file` y headroom, y usa por defecto un objetivo de 8 GiB. No cambia el guard científico de 2 GiB, no escribe inputs, no ejecuta `drop_caches` y no usa `memory.force_empty`.

La validación viva RGB, [`RGB_CACHE_V2_LIVE_VALIDATION.json`](../../artifacts/runpod_migration_20261026/RGB_CACHE_V2_LIVE_VALIDATION.json), observó:

- memoria actual entre 45,4 y 50,9 GB de unos 61 GB;
- cache entre 34,1 y 39,0 GB;
- RSS entre 10,2 y 11,4 GB;
- tareas activas;
- estado del controlador `IDLE_HEADROOM_OK`.

En esas muestras no aconsejó bytes porque ya existía margen. Eso valida que el controlador no desaloja por inercia cuando no hace falta; no mide su velocidad de reclaim.

Dos auditorías posteriores muestran por qué se separó RSS de cache:

- muestra A: 32,53 GB actuales, 26,59 GB cache, 5,57 GB RSS;
- muestra B: 47,42 GB actuales, 40,14 GB cache, 6,91 GB RSS.

El controlador RGB v2 mantuvo objetivo de 8 GiB y seleccionó de forma acotada 55 tar con 27,366 GB lógicos; la evidencia conserva `training_modified=false` y SHA de fuente `a362b160…`. De nuevo, selección lógica no equivale a bytes físicos desalojados.

## 10. Integridad y comportamiento fail-closed

El plano de I/O fue endurecido alrededor de cinco invariantes:

1. **Nada parcialmente copiado se vuelve autoridad.** El seal sólo se publica después de tamaño y hash completos.
2. **La reanudación está ligada a identidad.** Repo, commit, ruta, offset y hash de prefijo impiden continuar sobre otro objeto.
3. **El eviction advisory ocurre después de durabilidad.** Nunca se aconseja descartar una página sucia antes de `fsync`.
4. **No se toca caché ajena.** Sólo ficheros propios, inmutables, cerrados y admitidos; nunca `drop_caches` global.
5. **Los límites son por bytes.** Prefetch y restores consideran tamaño real, no sólo número de batches o archivos.

Los journals de checkpoint seguían haciendo `fsync` por update. La auditoría midió el checkpoint cada 100 updates en aproximadamente 0,08% del tiempo observado, por lo que no hubo base para relajar la durabilidad a cambio de rendimiento.

## 11. Resultados negativos y afirmaciones que no deben hacerse

- Prefetch v1 fue exacto pero falló el gate de 20% de mejora.
- La caché descomprimida fue rápida en un probe CPU acotado; no se puede convertir su ratio 0,064516 en speedup de entrenamiento.
- Profundidad 2 ya redujo la espera directa a 3,8–4,5 ms en el refit; no hay evidencia de que más profundidad ayude.
- Un volumen de red que copia a 5,7508 MiB/s en una transferencia no fija el throughput de todos los archivos ni del entrenamiento.
- `FADV_DONTNEED` es un consejo al kernel, no una garantía instantánea de headroom.
- Las cifras de page cache no son fugas de memoria del modelo; deben separarse de RSS/anon.
- El éxito de la descarga RAW prueba identidad y cobertura, no que todas las etapas downstream estén libres de problemas de routing.

## 12. Índice de evidencia principal

- [`INPUT_PIPELINE_20261016.md`](../joint_measurement_20261012/INPUT_PIPELINE_20261016.md): perfiles del lector, prefetch v1/v2 y paridad.
- [`A1_TRAINING_BOTTLENECK_20261016.md`](../joint_measurement_20261012/A1_TRAINING_BOTTLENECK_20261016.md): LRU, pinned memory, prefetch activo y espera de batch.
- [`UNCOMPRESSED_INPUT.md`](../joint_measurement_20261012/UNCOMPRESSED_INPUT.md): probe de representación descomprimida y límites de alcance.
- [`CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md`](../final_refit_20261017/CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md): tiempos de preparación, decoder y espera directa.
- [`bounded_prefetch.py`](../../operational/runpod_migration/bounded_prefetch.py) y [`runtime_prefetch.py`](../../operational/runpod_migration/runtime_prefetch.py): límites por bytes y orden reanudable.
- [`NETWORK_TECHNICAL_OVERLAY_V3.md`](../runpod_migration_20261026/NETWORK_TECHNICAL_OVERLAY_V3.md): MooseFS, cgroup, transferencia y staging.
- [`runpod_event_hf_restore_v2.py`](../../scripts/runpod_event_hf_restore_v2.py) y [`event_page_cache.py`](../../operational/runpod_migration/event_page_cache.py): descarga durable por rangos y advice localizado.
- [`POD2_RAW_ROOT_CHECK.json`](../../artifacts/runpod_migration_20261026/POD2_RAW_ROOT_CHECK.json): cobertura RAW final y estadísticas de advice.
- [`runpod_event_cache_controller_v1.py`](../../scripts/runpod_event_cache_controller_v1.py): controlador de page cache sellado y fail-closed.
- [`RGB_CACHE_V2_LIVE_VALIDATION.json`](../../artifacts/runpod_migration_20261026/RGB_CACHE_V2_LIVE_VALIDATION.json): evidencia viva de memoria, cache y liveness.
