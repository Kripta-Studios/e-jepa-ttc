# 06 — Migración a Runpod y colas autónomas

## 1. Objetivo y límites

La migración perseguía tres objetivos operativos:

1. trasladar artefactos y estados durables sin reescribir la autoridad científica;
2. usar GPU, RAM y almacenamiento cloud dentro del presupuesto y deadlines autorizados;
3. permitir que productores, downstream e inferencia final avanzaran de forma autónoma y recuperable.

No se autorizó cambiar seed, precisión BF16, arquitectura, pérdidas, splits, endpoint de updates ni identidad de checkpoints. Los overlays se diseñaron como capas operativas hash-bound. La autoridad histórica se conservó incluso cuando el límite real del kernel era mayor.

## 2. Del bundle local al runtime cloud

### 2.1 Binding de migración

El primer binding técnico está en [MIGRATION_BINDING.json](../../artifacts/runpod_migration_20261026/MIGRATION_BINDING.json). Fija:

- Python 3.11, seed 7 y BF16;
- checkpoint periódico de 100 updates;
- hashes de bundle, autoridad y fuentes runtime;
- mapeo explícito de raíces Windows a Linux;
- cuota de workspace declarada;
- presupuesto histórico de memoria de 46.000.000.000 bytes;
- `scientific_files_rewritten=false`.

El entorno observado fue Python 3.11.15, PyTorch 2.11.0+cu128, CUDA runtime 12.8 y RTX 4090; véase [ENVIRONMENT_V3.json](../../artifacts/runpod_migration_20261026/ENVIRONMENT_V3.json).

El volumen `/workspace` se identificó como MooseFS/FUSE. Pasó una prueba de escritura temporal, `fsync`, rename atómico y `fsync` de directorio, pero el recibo no afirma durabilidad frente a corte eléctrico ni supervivencia al stop del pod. Esa limitación permanece explícita en el binding.

### 2.2 46 GB históricos frente a 61 GB efectivos

El valor de 46 GB permaneció dentro de la identidad histórica. Un overlay operativo independiente consultó el límite efectivo real recorriendo ancestros cgroup v1/v2 y usó el mínimo válido. En el pod observado el kernel exponía 60.999.999.488 bytes. El monitor rechazaba valores ilimitados o inválidos y preservaba la reserva de 2 GiB.

Esta separación fue deliberada:

- **identidad histórica:** lo que la receta/binding original declaró;
- **admisión operativa:** lo que el kernel realmente permitía en ese pod;
- **sin cambio científico:** el límite operativo no modifica datos, pesos ni optimizer.

## 3. Transferencia: red, disco local y publicación atómica

### 3.1 Bundle inicial

La transferencia inicial movió 10.569.820.160 bytes en 1.763,5 s hacia el bundle cloud. El recibo completo está en [NETWORK_TRANSFER_COMPLETE.json](../../artifacts/runpod_migration_20261026/NETWORK_TRANSFER_COMPLETE.json). No se reproduce aquí el endpoint de red.

Las primeras pruebas mostraron que el cliente SCP de Windows era mucho más lento que el OpenSSH incluido con Git. Los uploaders posteriores fijaron explícitamente el binario y limitaron los flujos concurrentes. No se convirtió esa diferencia observada en una afirmación general sobre SCP.

### 3.2 Protocolo de fichero

Para bundle, RGB sparse TAR y raw EVENT se usó el mismo patrón lógico:

1. fichero local READY sólo después de `fsync` y rename desde `.pending`;
2. transferencia a nombre temporal remoto;
3. SHA-256 y tamaño contra manifest;
4. `fsync` del fichero y del directorio;
5. publicación por rename atómico;
6. sidecar/receipt durable;
7. reanudación por offset o prefijo validado;
8. rechazo de colisiones de path con hash diferente.

El uploader no interpretaba un tamaño reservado como bytes realmente transferidos. La telemetría separó committed, partial e inflight. Un fallo real por payload JSON grande embebido en la línea SSH se corrigió enviando el inventario por `stdin`, sin ignorar códigos de salida genéricos.

### 3.3 Disco local y Network Volume

Los roles se separaron:

- `/workspace`: Network Volume durable, sujeto a cuota y latencia de metadata;
- `/opt`: almacenamiento local rápido pero efímero, usado para bundles sellados y materialización de trabajo;
- manifests/receipts en `/workspace`: autoridad de qué bytes locales eran admisibles.

Una carrera real apareció cuando `du -sb /workspace` encontró paths eliminados concurrentemente y devolvió error. La corrección reintentó sólo ENOENT de forma acotada, conservó TTL/reserva y siguió rechazando otros errores. No se reutilizó inventario obsoleto indefinidamente.

## 4. Dos pods, responsabilidades separadas

La topología final separó funciones:

- **Pod 1 / RGB:** productores y downstream RGB, materialización sparse RGB, scopes físicos y retorno de artefactos.
- **Pod 2 / EVENT:** productores EVENT, colas PAIR_E/E_CTX, raw EVENT y, posteriormente, inferencia Test12.

La separación no implicó dos autoridades científicas. Cada endpoint y cache se vinculó a manifests, admissions y parent checkpoint hashes. La relocación física del segundo pod verificó 9.852 ficheros del bundle, 13 del suplemento y 7 de autoridad, sin modificar ficheros científicos; véase [PHYSICAL_RELOCATION.json](../../artifacts/runpod_event_migration_20261026/resume_20261027/PHYSICAL_RELOCATION.json).

## 5. Colas de productores y downstream

### 5.1 Identidad de proceso

Cada worker persistente se registra con:

- PID;
- tiempo de nacimiento/start ticks;
- cmdline exacta o hash;
- tarea/rotación;
- frontera durable;
- contador de reintentos y backoff.

Un PID aislado no es identidad suficiente porque puede reutilizarse tras reboot o salida. El inventario con `psutil` también cierra la ventana entre `Popen` y escritura del registry: antes de lanzar se vuelve a enumerar el sistema y se adopta cualquier proceso exacto ya presente.

### 5.2 Dependencias y estados de espera

Las colas distinguen:

- `WAITING_DEPENDENCY`: falta un endpoint;
- `WAITING_INPUT_COVERAGE`: falta cobertura raw/media;
- `WAITING_CAPACITY`: tarea durable aplazada por límite de concurrencia;
- `RESTARTABLE`: pausa automática con frontera durable;
- `BLOCKED`: evidencia inválida, STOP manual, identidad de proceso incorrecta o fallo científico;
- `COMPLETE`: endpoint y admission verificadas.

[EVENT_QUEUE_RAW_VERIFY.json](../../artifacts/runpod_migration_20261026/EVENT_QUEUE_RAW_VERIFY.json) conserva una fase temprana donde las tres rotaciones esperaban E_A5 mientras el raw transfer continuaba. Esto fue una espera legítima, no un endpoint completo.

### 5.3 Cloud admission y ventana de publicación

Se detectó una ventana donde `ENDPOINT_ADMISSION.json` ya existía pero `SCIENTIFIC_CLOUD_ADMISSION.json` aún no. El wrapper de bootstrap espera hasta el deadline sólo en esa condición; cuando aparece cloud admission ejecuta el verificador original. Un fichero malformed no se convierte en espera y falla cerrado. Véase [runpod_event_queue_bootstrap_v6.py](../../scripts/runpod_event_queue_bootstrap_v6.py).

### 5.4 Capacidad aplazada

R3_E_A5 se mantuvo durable en `WAITING_CAPACITY` mientras cuatro productores ocupaban la capacidad autorizada. El overlay V9 sólo liberó la tarea cuando uno de los candidatos exactos alcanzó COMPLETE con cloud admission, volvió a validar checkpoint/journal/pointer/hash, deadline y ausencia de STOP, y después usó el mecanismo normal de restart. [OVERLAP_VERIFIED_STATUS.json](../../artifacts/runpod_event_migration_20261026/resume_20261027/OVERLAP_VERIFIED_STATUS.json) registra `capacity_released_by=R4_E_A5` y el receipt de liberación.

## 6. Recursos cloud y cgroups

### 6.1 Lectura efectiva

Los monitores no confiaron sólo en la RAM anunciada por el proveedor. Leyeron:

- cgroup v1: `memory.usage_in_bytes` / `memory.limit_in_bytes`;
- cgroup v2: `memory.current` / `memory.max`;
- memoria disponible del host;
- mínimo efectivo entre límites aplicables.

Las colas pausan en fronteras seguras cuando el headroom cae por debajo de 2 GiB. La pausa no borra checkpoints ni cuenta updates pendientes como durables.

### 6.2 Page cache en Pod 1

La carga simultánea de TAR podía llevar el file cache a decenas de GB aunque el RSS fuera moderado. [POD1_MEMORY_AUDIT_AFTER_RECLAIM_A.json](../../artifacts/runpod_migration_20261026/POD1_MEMORY_AUDIT_AFTER_RECLAIM_A.json) y [POD1_MEMORY_AUDIT_AFTER_RECLAIM_B.json](../../artifacts/runpod_migration_20261026/POD1_MEMORY_AUDIT_AFTER_RECLAIM_B.json) separan ambos componentes.

**Fallo ejecutado:** un poll de 15 s permitió un burst de cache antes del siguiente ciclo. El wrapper de 1 s redujo esa ventana sin tocar el guard. No se hizo reclaim de ficheros parciales, sucios o no manifest-bound.

### 6.3 Page cache en Pod 2

El controlador EVENT raw aplicó el mismo principio a inputs sellados. [RAW_CACHE_RECOVERY_VERIFIED.json](../../artifacts/runpod_event_migration_20261026/resume_20261027/RAW_CACHE_RECOVERY_VERIFIED.json) registra cgroup v1, límite efectivo de 60.999.999.488 bytes, unos 38,0 GB de file cache, 12,2 GB anon/RSS y aproximadamente 9,96 GB de headroom después de recuperación. También registra progreso simultáneo de las tres rotaciones; no es sólo una captura de memoria.

## 7. Fallos reales durante bootstrap y migración

Los siguientes fallos se ejecutaron y se corrigieron; no son escenarios hipotéticos:

1. imports dependientes de `sys.path` en el bundle cloud;
2. `canonical_sha256` importado sólo en una rama y usado en otra;
3. `_write_once` comparando JSON con campos operativos variables;
4. autoridad dinámica faltante (`INPUTS.json`);
5. configuración `efficient_context_v1.json` ausente;
6. routing de directorio incapaz de resolver H8 repartido entre bundle y suplemento;
7. monkeypatch de `Path` cuyo `isinstance(PosixPath, Path)` devolvía falso;
8. callback V5 que no aceptaba kwargs añadidos por V6;
9. comprobación legacy RGB que exigía exactamente un flag de sample aunque un TAR antiguo tenía hashes/shape de todos los miembros;
10. inventario `du` interrumpido por ENOENT concurrente;
11. cache que llenó cgroup entre polls;
12. registro inmediato tras `Popen` antes de que fork/exec publicara la cmdline final.

Las correcciones fueron overlays aditivos y hash-bound. No se reescribieron markers científicos antiguos para «hacerlos pasar».

## 8. Retorno y recuperación

Los retornos cloud no se consideraron completos por la mera presencia de un fichero. Los receipts verifican:

- paths contenidos;
- tamaños y SHA-256;
- checkpoint y journal;
- parent lineage;
- optimizer updates no inferidos;
- ausencia de partials/pending;
- archive local antes de declarar publicable.

[PODS_RECOVERY_VERIFIED_20261026T210622Z.json](../../artifacts/runpod_migration_20261026/PODS_RECOVERY_VERIFIED_20261026T210622Z.json) documenta una ventana de 95,45 s después de reparar blockers: cero tareas bloqueadas, R2 R_CTX avanzó 600 updates en Pod 1 y cinco productores EVENT avanzaron en Pod 2. El mismo recibo limita la conclusión: podían reaparecer pausas de recursos, no todos los fits estaban completos todavía y no se afirmaba un nuevo submission oficial.

## 9. Cola final Test12

La última cola autónoma se implementó en [runpod_test12_sequence_queue.py](../../scripts/runpod_test12_sequence_queue.py). Sus contratos son:

- exactamente 12 secuencias derivadas del inventario congelado, no de una lista inventada;
- máximo dos workers;
- inputs EVENT `COMPLETE` y RGB `READY` antes de lanzar;
- adopción por PID, birth time y cmdline exacta;
- adapter versionado por SHA para que un cambio futuro no altere procesos ya activos;
- OMP/MKL/OpenBLAS limitados a dos threads;
- reserva de 2 GiB leída directamente desde cgroup v1/v2 y host;
- STOP propio del controlador y deadline autorizado;
- fragmentos por token reanudables e inmutables;
- `complete.json` no libera el slot hasta que exista `consumed PASS` ligado por hash;
- `ResourcePause` es reintentable; un `ValueError` científico es fatal y persistente;
- `package` se ejecuta una sola vez después de 12 completions admitidas;
- un `PREDICTION_RECEIPT` parcial mientras el hijo package sigue vivo es espera, no error ni COMPLETE;
- ZIP, sidecar, prediction receipt y hashes de las 12 completions se verifican antes del receipt final de la cola.

### 9.1 Carreras corregidas

**Complete→consumed.** Una caída después de `complete.json` pero antes del receipt `consumed` podía dejar una secuencia aparentemente terminada pero no evictable. El adapter nuevo revalida admission, features, todos los fragmentos y diagnósticos, y reconstruye sólo el sidecar faltante sin repetir inferencia.

**Fork→exec.** El primer controlador intentó leer cmdline inmediatamente después de `Popen`; un hijo que pausó por recursos podía salir antes del registro y derribar al coordinador. La versión final espera hasta 2 s por el `exec` exacto. Si el hijo sale, clasifica su log: pausa de recursos reintentable o fallo fatal durable. No deja un proceso huérfano sin registrar.

**Prediction receipt→ZIP receipt.** El package escribe varios artefactos en secuencia. La cola no inspecciona como terminal un estado parcial mientras el hijo está vivo; si sólo existe prediction receipt y no hay hijo, relanza `package` idempotentemente.

## 10. Qué quedó demostrado y qué no

### Demostrado por recibos

- migración byte-verificada y relocación física sin modificar ciencia;
- checkpoints y journals reanudables;
- concurrencia agregada superior en la ventana Runpod dual observada;
- recuperación selectiva de page cache sin `drop_caches`;
- watchdogs con cero optimizer updates;
- progreso posterior a pausas y relocación;
- cola Test12 reanudable y empaquetado verificado.

### No demostrado / no afirmado

- durabilidad del Network Volume ante corte eléctrico;
- throughput universal de la RTX 4090 o de dos workers;
- causalidad térmica única en Windows;
- que toda pausa futura sea evitable;
- que un artefacto empaquetado haya sido enviado a un servidor oficial;
- que un overlay operativo cambie o mejore la ciencia del modelo.

La regla de cierre fue siempre la misma: una cola puede automatizar trabajo ya autorizado, pero no puede convertir ausencia de evidencia en éxito ni ampliar la receta científica.
