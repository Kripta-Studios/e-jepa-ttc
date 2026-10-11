# V13: diagnóstico y optimización del productor RGB A5

Fecha de la intervención: 11 de octubre de 2026. Código de la implementación:
commit `3f889ff` de la rama `scientific-recovery-v13-rgb-port`.

La investigación encontró trabajo repetido en CPU que dejaba al entrenamiento
esperando datos y controles del sistema. Se implementó una caché DINO dimensionada
para el conjunto de shards usado por el sampler, una construcción de batch que
evita tensores descartados y un control de recursos que consulta la memoria sólo
de los procesos pertinentes. La comparación CPU tiene paridad exacta y reduce el
tiempo de preparación. **El entrenamiento ya se reanudó y guardó un checkpoint
nuevo.** La observación del entrenamiento completo y su ETA están en
[LIVE.md](LIVE.md); los resultados de componentes están en [TABLES.md](TABLES.md).

## Qué se estaba ejecutando

El productor `R_A5` acababa de empezar después de finalizar los productores de
eventos. Sus entradas RGB proceden de una caché preparada con imágenes `uint8`
reales y longitudes temporales T2/T3; el profesor DINO procede de un sidecar
congelado. Esta intervención no vuelve a ejecutar el profesor ni modifica sus
targets.

El entrenamiento se pausó en una frontera segura y guardó el checkpoint completo
del update **1.500**, con cursor **48.000** de la primera época. Se conservaron
checkpoint, puntero, recibo y journal de origen para comparación. El benchmark CPU
consumió una copia de sus siguientes 512 índices, sin avanzar el cursor de la
campaña ni ejecutar updates de optimizador.

La capa anterior `operational/rgb_port_loader_v3/producer.py` aplicaba su caché y
prefetch exclusivamente a los productores de eventos. Los productores RGB seguían
la ruta histórica; las mejoras anunciadas para eventos no implicaban que RGB
dispusiera del mismo cargador.

## Diagnóstico y correspondencia con el código

### 1. Recargas del sidecar DINO

En [RGBProducerSource._dino_for_indices](../../src/e_jepa_ttc/rgb_port/data.py),
cada muestra recorría el manifiesto de shards buscando su `source_ordinal`. El
cargador retenía sólo cuatro shards y, tras expulsarlos, volvía a abrir y
descomprimir el NPZ para otra muestra del mismo grupo. El sampler canónico de
[train_producers.py](../../operational/rgb_port/train_producers.py) baraja grupos
de 256 filas y las filas dentro de cada grupo; no se cambió para optimizar I/O.

Una inspección de los ordinales de la población P mostró que los grupos necesitan
más shards DINO que los cuatro retenidos. Esa observación motivó la hipótesis;
la prueba ABBA demuestra el efecto: el número de aperturas DINO cae manteniendo
exactamente los mismos batches. Los resultados de lecturas están en las tablas,
y la ruta RGB mantiene el mismo número de aperturas.

[loader.py](../../operational/rgb_port_rgb_loader/loader.py) construye un índice
del manifiesto y conserva hasta 32 shards, con límite adicional de 64 MiB de
arrays retenidos. Si un shard individual supera ese límite se puede servir,
pero no se almacena en la caché. El límite de caché no representa el pico total
del proceso: existe almacenamiento transitorio para la lectura y para las salidas.

Se mantienen la carga completa del NPZ, los tipos originales y las comprobaciones
de ordinal y token de cada fila. Si el manifiesto contiene intervalos solapados,
se conserva la selección del primer intervalo de la implementación original.
La ruta monolítica del sidecar sigue delegando en su lector original. El adaptador
se instala sobre la instancia original del source y restaura su método al salir.

### 2. Tensores apilados que el productor descartaba

La construcción genérica `collate_rgb_port()` apila campos destinados también a
otras rutas. `RGBProducerSource.batch()` sólo consume un subconjunto. Entre los
campos descartados hay dos representaciones `uint8` de las mismas imágenes,
máscaras, relojes auxiliares y metadatos de objetivos que el batch final no expone.

[collate.py](../../operational/rgb_port_rgb_loader/collate.py) apila únicamente
los seis campos tensoriales que usa el productor, conserva las listas de identidad
y adjunta el mismo DINO. Mantiene la secuencia `as_tensor → stack → float`, el
orden de acceso a la fuente y el objeto `ObjectEventV4Batch` completo.

La separación de T2/T3 sigue a cargo de
[EffectiveSource](../../operational/rgb_port_revision/loss.py). No se rellenan
frames ni se cambia el número de ejemplos del batch efectivo. La pérdida conserva
sus reducciones y la única frontera de actualización del optimizador.

### 3. Control de recursos caro en Windows

El control completo histórico pedía información de memoria a todos los procesos
antes de filtrar cuáles pertenecían al proyecto. El muestreo de pilas mostró
coste en las consultas de `psutil` además de las lecturas NPZ.

[guards.py](../../operational/rgb_port_rgb_loader/guards.py) consulta primero PID
y comando, aplica los mismos marcadores del proyecto y exclusión de `pytest`, y
después obtiene RSS sólo de los procesos seleccionados. Conserva las reglas de
errores de acceso y desaparición de procesos. No almacena una lista antigua de
PIDs ni cambia la frecuencia de la comprobación.

El experimento de control es independiente del ABBA de batches. Compara tres
parejas de llamadas; las parejas identifican los mismos procesos. Entre rondas
arrancaron otros procesos de la intervención, por lo que no se reclama igualdad
de todas las instantáneas a lo largo del experimento. Las decisiones observadas
coinciden, y las pruebas unitarias comprueban umbrales y errores con estados
deterministas. El experimento conserva la política histórica; la integración
aplica después la reserva de 1 GiB ya autorizada por el usuario.

## Qué muestra el perfil y qué no muestra

[DIAGNOSTICS.json](DIAGNOSTICS.json) conserva un resumen público de las pilas y
una ventana corta de progreso previa a la pausa. En el hilo principal aparecen
lectura y descompresión de NPZ, recuperación del sidecar DINO y consultas de
procesos. Los porcentajes inclusivos se solapan: no deben sumarse como categorías
independientes. Tampoco son tiempos de kernels CUDA.

No se atribuyó automáticamente el problema a la GPU, a falta de trabajadores o
a saturación del USB. La caché RGB de 256 MiB se mantuvo y el benchmark muestra
las mismas aperturas RGB. Las mediciones identifican dos costes CPU concretos;
una desconexión física de E: sigue siendo un problema de I/O que el software debe
recuperar mediante las rutas existentes.

## Condiciones exactas de la comparación CPU

El runner [benchmark.py](../../operational/rgb_port_rgb_loader/benchmark.py):

1. Oculta CUDA dentro de su proceso y carga el checkpoint sólo en CPU.
2. Comprueba el SHA256 del checkpoint indicado por el puntero duradero.
3. Extrae los siguientes 512 índices del cursor, sin regenerar el sampler.
4. Construye `make_bound_rgb_producer_source()` y el `EffectiveSource` de la
   revisión admitida; exige identidad idéntica a la del checkpoint.
5. Ejecuta original/optimizado/optimizado/original con dos threads de PyTorch,
   coherentes con `OMP_NUM_THREADS=2` y `MKL_NUM_THREADS=2` de la cola original.
6. Reconstruye la fuente entre ensayos para empezar con cachés lógicas vacías.
7. Cronometra `EffectiveSource.batch()` y cuenta `np.load` por tipo de archivo.
8. Calcula por separado un digest de todos los campos, metadatos, dtypes, formas
   y bytes; exige igualdad en los cuatro ensayos y conservación de los estados
   aleatorios de Python, NumPy y PyTorch CPU.
9. Verifica que el puntero de checkpoint no cambió y que CUDA no se inicializó.

El tiempo de preparación excluye el cálculo del digest. El tiempo total del
runner sí incluye hashing y otra instrumentación. No se vació la caché de
archivos de Windows; el orden ABBA reduce un sesgo simple de calentamiento,
pero dos repeticiones por condición no caracterizan toda la variación del disco.
La mejora agrupa caché DINO, búsqueda indexada y collation: no es una ablation
que separe la contribución temporal individual de cada cambio.

Antes del ensayo válido hubo un arranque abortado por `PYTHONPATH` apuntando al
repositorio base y se corrigió una instrumentación que resolvía repetidamente
rutas ya absolutas. Esos intentos no produjeron resultados de comparación ni
updates. El ABBA publicado usa la versión corregida del runner; sus hashes de
código están en el JSON.

## Integración, continuidad y recuperación

[producer.py](../../operational/rgb_port_rgb_loader/producer.py) añade el adaptador
al runtime admitido. Los datos preparados y resultados voluminosos permanecen
en la unidad externa; la intervención no migra datasets al SSD interno.

Se conservan BF16, pesos, AdamW, scheduler, normalización, pérdidas, sampler,
selección de ejemplos y endpoint científico. Se mantienen los validadores de las
capas anteriores y se agrega `RGB_LOADER_FREEZE.json` mediante
[contracts.py](../../operational/rgb_port_rgb_loader/contracts.py). La nueva
capa exige QA aprobada y paridad real exacta antes de admitir su código.

La collation instalada en la instancia sustituye al método de clase; el wrapper
reintroduce explícitamente `retry_input_read` para conservar los reintentos y
la pausa segura ante errores transitorios. La cola sigue usando su supervisor
de recuperación; no se crean productores duplicados.

Cada checkpoint nuevo registra `RGB_LOADER_RUNTIME.json` y un recibo en
`fits/<fit>/rgb_loader_checkpoints/`, con hash de origen, updates, tiempo desde
restauración, tiempo de preparación, contadores de caché y reserva de memoria.
No hay prefetch en esta intervención: el valor registrado es cero. Tampoco se
añaden procesos de decodificación ni se activa CUDA Graphs para RGB.

## QA y evidencia pública

[QA.json](QA.json) registra la suite focalizada, Ruff y Pyright, junto a los hashes
de las evidencias originales. Incluye nombres de casos, no salidas con rutas de
usuario. La suite comprueba paridad de batches T2/T3, límites de caché, corrupción,
restauración de métodos, recuperación I/O, hooks de checkpoint, política de
recursos, routing de cola, pérdida efectiva y continuidad de la revisión previa.

Los JSON públicos omiten rutas locales absolutas, tokens del dataset e índices
individuales innecesarios. Conservan hashes de evidencia original, hashes de los
batches, identidad y metadatos, tiempos por batch, estados RNG y contadores para
que las tablas sean auditables. Sin los datasets y checkpoint privados se pueden
regenerar las tablas; repetir la lectura CPU completa requiere esos artefactos.

```powershell
python docs/rgb_loader_20261011/regenerate_tables.py
```

Para reconstruir el extracto público desde evidencia local completa:

```powershell
python docs/rgb_loader_20261011/publish_evidence.py --evidence <directorio_evidencia>
python docs/rgb_loader_20261011/regenerate_tables.py
```

Para repetir el ABBA, con entorno del proyecto activo y source correcto:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -B -m operational.rgb_port_rgb_loader.benchmark `
  --run artifacts/rgb_port_20261008 `
  --output <directorio_evidencia>/cpu_abba `
  --samples 512 --torch-threads 2
```

No debe ejecutarse contra una campaña cuyo puntero esté cambiando: el runner
comprueba esa condición y devuelve error si ocurre.

## Estado de la evidencia al publicar esta revisión

La ruta nueva restauró el checkpoint 1.500 con su hash exacto y guardó el siguiente
checkpoint completo en 1.600. [LIVE_RESULT.json](LIVE_RESULT.json) contiene ambos
recibos, las muestras de progreso y memoria y el perfil posterior. Las tablas de
[LIVE.md](LIVE.md) se regeneran ejecutando `python docs/rgb_loader_20261011/publish_live.py`.
La cola sigue ejecutándose con recuperación I/O y la misma secuencia de fases;
la ruta RGB optimizada admite tanto R_A5 como R_C2F, mientras eventos delega en
su cargador anterior.

El perfil posterior reduce la presencia del sidecar DINO y del control de procesos;
la preparación de imágenes RGB y el trabajo del modelo siguen siendo relevantes.
El tiempo de preparación acumulado del recibo de checkpoint permite compararlo
con el tiempo completo transcurrido, sin confundir porcentajes de pilas con
duraciones CUDA. El prefetch RGB permanece en cero: solapar preparación y cómputo
es un siguiente experimento, que requiere comprobar cursor, T2/T3, recuperación y
memoria. Tampoco se extienden los CUDA Graphs de eventos sin validarlos para RGB.

Se observó además un arranque lento por validaciones repetidas de la migración:
22 archivos migrados, sus versiones históricas, cinco freezes y 536 entradas del
SOURCE_FREEZE. En esta ruta RGB el supervisor espera al productor mediante
`process.wait()`: ese coste corresponde al arranque y los cambios de fase, no a
cada batch. Deduplicar comprobaciones dentro de una transacción de arranque queda
como mejora separada; esta intervención conserva todos los validadores.

La reducción de preparación CPU no se aplica directamente a la ETA. También
quedan forward/backward, checkpoints, controles restantes e I/O. No se han medido
aquí precisión, MiD, FCWD, Dev32 o test12, ni se deriva una afirmación SOTA de estos
cambios. La primera conclusión es eficiencia de componentes con datos idénticos;
la siguiente comprobación es velocidad sostenida y estabilidad del entrenamiento.
