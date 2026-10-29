# 05 — Concurrencia, recursos y recuperación durable

## Alcance y convención de evidencia

Este capítulo documenta la evolución operativa que llevó desde la ejecución local en Windows hasta la ejecución concurrente en Linux/Runpod. No es una afirmación de mejora universal del modelo ni una reconstrucción basada únicamente en memoria. Se distinguen tres estados:

- **Ejecutado y observado**: existe un recibo, journal, muestra o informe reproducible en el repositorio.
- **Implementado y probado**: existe código y cobertura local, pero la evidencia citada no demuestra por sí sola una campaña completa.
- **Propuesto o descartado**: diseño considerado, prueba negativa o mecanismo que no se promovió.

La historia de Git anterior a la migración queda anclada, entre otros, por `771c96a2` (decisiones post-V18), `a811cc3a` (entrega V18 verificada), `c3f1440` (empaquetado de evidencia) y `fb41681` (cierre de decisiones y coste de fusión). Los overlays operativos del 26–27 de octubre se preservaron como fuentes y recibos locales; no se presentan aquí como si todos hubieran formado parte de esos commits.

## 1. El problema local no era solamente «falta de RAM»

### 1.1 Dos reservas diferentes en Windows

El runtime local terminó tratando por separado:

1. memoria física disponible;
2. margen de memoria comprometida de Windows.

La autorización final fijó ambos suelos en **2 GiB**. El guard acepta exactamente 2 GiB y rechaza un byte menos; no se redujo el intervalo científico de checkpoint ni se eliminó la validación de inventario o disco. La decisión y sus pruebas están descritas en [RAM_RESERVE_2GIB.md](../final_refit_20261017/RAM_RESERVE_2GIB.md) y [COMMIT_RESERVE_2GIB.md](../final_refit_20261017/COMMIT_RESERVE_2GIB.md).

Esto corrigió una ambigüedad importante: `available RAM` no equivale al margen de commit. Una máquina puede conservar páginas recuperables y, al mismo tiempo, acercarse al límite de commit. También puede tener memoria física aparentemente ocupada por caché/standby sin que toda esa ocupación sea RSS anónima de los entrenadores.

**Ejecutado:** hubo pausas reales cuando el margen de commit cayó por debajo del umbral. Un ejemplo anterior registró aproximadamente 2,96 GB físicos disponibles pero sólo 0,91 GB de commit, y el controlador rehusó continuar. Ese fallo se conserva en [RESUME_PROOF.json](../simplex_t_shared_gpu_route_20261004/partial_723/RESUME_PROOF.json).

### 1.2 Page cache, standby y RSS no son intercambiables

Durante la ejecución local y la migración se comprobó que sumar RSS de procesos no explica por sí solo la carga de memoria:

- en Windows, el working set, las listas standby y el commit responden a contabilidades distintas;
- en Linux/cgroups, `memory.current` incluye RSS/anon, file cache y otras categorías cargadas al cgroup;
- un guard que sólo mira RSS puede permitir una tormenta de page cache;
- un guard que trata toda cache como memoria anónima puede pausar trabajo recuperable innecesariamente.

La solución no fue desactivar el guard. Se mantuvo la reserva de 2 GiB y se introdujeron observaciones separadas de RSS/anon y file cache. En Runpod, la auditoría posterior a reclaim registró aproximadamente 6,9 GB de RSS/anon y 40,0 GB de file cache dentro de 47,3 GB cargados, con 13,7 GB de headroom efectivo. Véase [POD1_MEMORY_AUDIT_AFTER_RECLAIM_B.json](../../artifacts/runpod_migration_20261026/POD1_MEMORY_AUDIT_AFTER_RECLAIM_B.json).

### 1.3 Recuperación selectiva de cache

Para datos públicos inmutables y manifest-bound se usó `POSIX_FADV_DONTNEED` sobre descriptores `O_RDONLY|O_NOFOLLOW`. No se usó `drop_caches`, no se cambió el guard científico y no se modificaron TAR/NPZ. El recibo citado documenta:

- 55 TAR seleccionados por dos manifests;
- 27.366.062.080 bytes elegibles;
- objetivo de 8 GiB de headroom;
- `global_cache_controls_used=false`;
- `scientific_files_modified=false`;
- `training_guard_changed=false`.

El primer controlador sondeaba cada 15 s. **Fallo ejecutado:** una recarga de decenas de GB podía ocurrir entre dos sondeos y provocar una pausa antes del siguiente reclaim. Se reemplazó por un overlay de sondeo de 1 s, conservando el mismo alcance y las mismas invariantes. No se atribuye a este cambio una aceleración científica; su función fue evitar pausas debidas a cache recuperable.

## 2. Temperatura, energía y Modern Standby

La evidencia local mostró condiciones térmicas y de energía reales, pero no autoriza una causalidad única:

- una ventana concurrente registró `SW Thermal Slowdown=Active` en 13 puntos y temperaturas de 86–88 °C; véase [CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md](../final_refit_20261017/CONCURRENT_RUNTIME_BOTTLENECK_AUDIT.md);
- otra medición de 15 minutos registró media 83,46 °C, p95 86 °C y máximo 89 °C, pero no justificó convertir esa observación en una ETA universal; véase [RUNTIME_15MIN_REPORT.md](../final_refit_20261017/RUNTIME_15MIN_REPORT.md);
- la caída de R_C2F coincidió temporalmente con entradas de Modern Standby, pero persistió después de la salida. El informe conserva además frecuencia efectiva reducida y flags térmicos, y concluye explícitamente que no se demostró una única causa física; véase [RC2F_SLOWDOWN_20261025.md](../final_refit_20261017/RC2F_SLOWDOWN_20261025.md).

**Propuesto, no demostrado:** atribuir toda variación de throughput a temperatura, standby o firmware. Las pruebas QoS tampoco reprodujeron de forma estable la gran ganancia inicial; la política se restauró. La lección operativa fue registrar temperatura, potencia, frecuencia y estado de energía junto a throughput, no seleccionar retrospectivamente la explicación más conveniente.

## 3. Checkpoints cada 100 updates como frontera transaccional

### 3.1 Contrato

Los productores R_A5 y R_C2F mantuvieron un checkpoint periódico cada **100 optimizer updates**, además de checkpoint al endpoint o pausa limpia. El intervalo no se redujo para mejorar artificialmente la apariencia de recuperación. El diseño está detallado en [CHECKPOINT_RECOVERY_DESIGN.md](../final_refit_20261017/CHECKPOINT_RECOVERY_DESIGN.md).

Una frontera durable no se define por la mera existencia de `checkpoint_last.pt`. Se exige coherencia entre:

- `CHECKPOINT_RECEIPT.json`;
- `CHECKPOINT_POINTER.json`;
- una versión inmutable dentro de `checkpoint_versions/`;
- `UPDATE_JOURNAL.json`;
- hash del checkpoint y de la versión;
- identidad científica y binding de ejecución;
- `completed_updates == durable_updates`;
- `pending_update_upper == 0`;
- `recovery_upper == 0` cuando se declara una frontera limpia;
- estado de RNG, sampler, scheduler, optimizer y acumulación.

La implementación de comprobación puede seguirse en [resource_monitor.py](../../operational/final_refit/resource_monitor.py), [auto_resume.py](../../operational/final_refit/auto_resume.py) y [train_producers.py](../../operational/rgb_port/train_producers.py).

### 3.2 Qué significa una interrupción

El contador de trabajo ejecutado y el contador durable se mantienen separados. Si una interrupción sucede entre dos fronteras, el journal conserva una cota superior explícita; no se adjudican al checkpoint updates de otro proceso ni se presenta como durable un batch sólo iniciado. Los fault probes de checkpoint se encuentran en [checkpoint_fault_probe.py](../../operational/final_refit/checkpoint_fault_probe.py).

**Ejecutado:** R_C2F fue pausado limpiamente en el update 1234 con `recovery_upper=0`, mientras R_A5 continuó. Después ambos se reanudaron desde sus fronteras durables. La secuencia y sus límites están documentados en [RUNTIME_V4_APPLIED.md](../final_refit_20261017/RUNTIME_V4_APPLIED.md).

## 4. Concurrencia R_A5/R_C2F

### 4.1 Hipótesis y límites

R_A5 y R_C2F comparten familia de arquitectura y recursos, pero entrenan objetivos/grupos diferentes. Ejecutarlos juntos podía aumentar el throughput agregado del host si la GPU estaba subutilizada, a costa de menor throughput individual, más presión de CPU/RAM/cache y mayor sensibilidad térmica.

El contrato de comparación prohibió dividir por la suma de las duraciones de workers concurrentes. Para una ventana común `W`:

```text
throughput agregado = sum(delta_updates_worker) * 60 / W
media por worker    = throughput agregado / número de workers
```

### 4.2 Ventana controlada en Runpod

La evidencia más limpia es [DUAL_ANALYSIS.json](../../artifacts/runpod_migration_20261026/DUAL_ANALYSIS.json):

| Ventana | Duración | Workers | Updates totales | Agregado | Media/worker |
|---|---:|---:|---:|---:|---:|
| baseline | 360,47 s | 1 | 1.142 | 190,08/min | 190,08/min |
| dual | 360,00 s | 2 | 2.209 | 368,17/min | 184,08/min |

La relación observada fue 1,937× en throughput agregado y 0,968× por worker. Esto demuestra que, **en ese pod, ventana y overlay**, dos procesos utilizaron mejor la capacidad agregada. No demuestra 1,937× para otro hardware, otra fase o una ejecución completa.

Los recursos de la misma ventana quedaron en [DUAL_RESOURCES.json](../../artifacts/runpod_migration_20261026/DUAL_RESOURCES.json): GPU media 40,47 %, p95 83 %, máximo 98 %; VRAM 5.847 MiB; potencia media 129,79 W; temperatura máxima 46 °C; RAM del pod alrededor de 29,7 GiB, de los que aproximadamente 20,1 GiB eran cache y 9,15 GiB RSS según esa contabilidad.

### 4.3 Resultados negativos y mediciones no comparables

No toda ventana concurrente fue favorable. En Windows, los ritmos de R_A5 y R_C2F variaron con warmup, lotes, temperatura y carga de I/O. [OOF_RUNTIME_OPTIMIZATION_20261021.md](../final_refit_20261017/OOF_RUNTIME_OPTIMIZATION_20261021.md) conserva un pico inicial cercano a 169 updates/min que regresó a unos 26 tras aproximadamente 100 updates. Se rechazó usar el pico como ETA.

Del mismo modo, [RUNTIME_V4_APPLIED.md](../final_refit_20261017/RUNTIME_V4_APPLIED.md) registra ventanas donde ambos ritmos fueron inferiores a una referencia anterior. Esa observación no invalida la ventana Runpod: demuestra que el efecto depende del régimen y que una ventana corta no reemplaza una campaña completa.

## 5. Watchdogs: reanudar sin fabricar ciencia

Los watchdogs operativos nunca ejecutan optimizer updates. Su responsabilidad es:

- adoptar sólo PID, tiempo de nacimiento y cmdline admitidos;
- distinguir proceso vivo, zombie, PID reutilizado y transición `exec` autorizada;
- reanudar únicamente después de una pausa automática y una frontera durable;
- no borrar STOP manual;
- respetar deadline y backoff;
- publicar estado y razón exacta;
- mantener un máximo de reintentos consecutivos sin progreso.

Los overlays incrementales V2–V9 se conservan en [scripts/runpod_event_resource_watchdog_v2.py](../../scripts/runpod_event_resource_watchdog_v2.py) a [scripts/runpod_event_resource_watchdog_v9.py](../../scripts/runpod_event_resource_watchdog_v9.py). Entre los fallos corregidos:

1. un `PROGRESS.json` heredado podía pertenecer al PID anterior;
2. un `EVENT_PIPELINE_RECEIPT` antiguo podía bloquear una pausa HEAD nueva o, inversamente, hacer que evidencia HEAD vieja pareciera pre-head;
3. una transición waiter→queue por `exec` debía conservar PID/start ticks y cambiar sólo a otro argv admitido;
4. el contador de reintentos downstream debía resetearse únicamente cuando el `FragmentStore` avanzara con partes contiguas y verificadas;
5. una tarea aplazada por capacidad debía permanecer `WAITING_CAPACITY`, no convertirse en un falso fallo científico.

[OVERLAP_VERIFIED_STATUS.json](../../artifacts/runpod_event_migration_20261026/resume_20261027/OVERLAP_VERIFIED_STATUS.json) muestra el resultado de este enfoque: tareas producer COMPLETE con cloud admission, downstream activos, anchors de fragmentos ligados a hashes y `optimizer_updates_performed_by_watchdog=0`.

## 6. Conclusiones de ingeniería

1. La reserva de 2 GiB fue una política doble —física y commit—, no un número único de RSS.
2. La page cache debe medirse y, si se recupera, limitarse a ficheros inmutables y manifest-bound.
3. Un checkpoint robusto es una transacción con journal, pointer, versión y hashes; un fichero `.pt` aislado no basta.
4. La concurrencia se evalúa con tiempo de pared compartido y métricas agregadas y por worker.
5. Temperatura, standby y cache explican riesgos operativos, pero requieren evidencia para afirmar causalidad.
6. Los watchdogs sólo gestionan procesos y fronteras ya producidas; no alteran pesos, optimizador ni receta científica.
