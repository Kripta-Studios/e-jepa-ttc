# Caché exacta de features RGB: implementación y prueba de metadatos

Estado: helper implementado y validado en CPU. No integrado todavía en el
servicio ni medido en GPU. No demuestra latencia completa ni paridad TTC nueva.

## Qué se reutiliza

Únicamente las salidas `raw[historia,17]` y `experts[historia,3]` de los productores
RGB y PAIR, con las mismas entradas físicas y ejecución. La cabeza de contexto
y el TTC se recalculan en cada actualización porque sus edades dependen del
timestamp actual. No se guardan TTC ni features event en esta caché.

La identidad por objeto contiene:

- ID de objeto y ROI exacta, serializada mediante `float.hex()`.
- SHA del candidato/modelos y SHA de la configuración/fuentes de ejecución.
- Precisión explícita FP32 con TF32 desactivado.
- Fuente, dataset, frames referenciados, formas y timestamps.
- Grupos RGB T2/T3 completos y `delta_t` derivado de los timestamps.
- Forma real del batch productor y posición del objeto dentro de ese batch.

La clave excluye `query_us` después de verificar causalidad: ese reloj no entra
en los productores RGB y sí debe entrar de nuevo en el contexto. La inclusión
de forma/posición evita convertir un hit de batch20 en una supuesta referencia
bit-exact de batch1, pues los algoritmos y reducciones pueden depender del lote.

## Fuentes y mutabilidad

`bind_rgb_history` se llama una vez por actualización, antes de crear las claves
por ROI. El fallback calcula SHA-256 de cada frame RAW referenciado una vez,
compartido entre todos los objetos; no hashea veinte lotes de crops.

Todos los frames deben ser uint8 HWC y read-only hasta terminar su consumo. Un
sensor con buffers reutilizables debe aportar primero una copia propia congelada.
No se acepta cachear una identidad de contenido por dirección de NumPy ni por
timestamp. El hash debe volver a calcularse cuando no exista una fuente inmutable
atestada. Aun con `writeable=False`, el caller no puede mutar el backing storage
a través de un alias; el ownership del paquete sigue siendo un contrato explícito.

El modo rápido opcional `source_path + expected_source_stat` está limitado a
frames contiguos read-only del lector validado. Comprueba ruta resuelta, tamaño y
mtime antes y después de construir la identidad. Se declara `READONLY_READER_STAT`,
no checksum completo del HDF5. Presupone el mismo contrato de fuente inmutable que
`PersistentSensorInputs._source`; no protege frente a modificaciones adversarias
que restauren el mismo stat. El caller debe invalidar el caché RAW si cambia fuente.

`RawRGBHistory` todavía no transmite ese binding al servicio. El propietario del
servicio conoce la propuesta y lo añadirá en una integración separada después del
piloto actual. No se editó `multiobject.py` ni ninguna fuente de otros agentes.

## API y primera integración

```python
shared = bind_rgb_history(frames, timestamps, triplets, query_us=query_us)
keys = [
    rgb_feature_key(
        object_id, roi, shared,
        model_sha256=model_digest,
        runtime_sha256=runtime_digest,
        batch_shape=(objects * history_length, steps, 3, 128, 128),
        object_slot=position,
    )
    for position, (object_id, roi) in enumerate(object_regions)
]
cached = cache.lookup_many(keys)
if cached is None:
    # Ejecutar el mismo lote RGB completo y validar sus guards.
    cache.store_many(keys, raw_by_object, experts_by_object, guard_passed=True)
else:
    # Saltar productores RGB/PAIR; construir el contexto con el query_us actual.
    ...
```

La política inicial es **todo el lote acierta o se recalcula el lote completo**.
Ejecutar solo los misses cambia batch y requiere otra admisión numérica y claves
de ejecución apropiadas. Los contadores separan coincidencias individuales de
hits servidos, para no anunciar ahorro cuando un único miss obliga a ejecutar R.

`store_many` verifica shapes, FP32, grad desactivado, guard y finitud del lote. La
comprobación de finitud añade una sincronización escalar por lote. Las inserciones
y lecturas clonan tensores para impedir que el productor o consumidor corrompan
entradas futuras. Todo ese trabajo debe quedar dentro del reloj de actualización.

La instancia tiene un único hilo y stream CUDA propietario. El caller debe esperar
a que terminen los productores asíncronos y validar sus tickets antes de insertar.
`retain_objects` elimina objetos desaparecidos. El límite LRU es por número de
objetos y bytes de tensores retenidos; las copias devueltas pertenecen al caller.
Para veinte objetos con cinco observaciones, la caché retiene 8.000 bytes de
features FP32, además de metadatos Python acotados por veinte entradas. No retiene
frames RAW ni modelos.

## Validación ejecutada

32 tests CPU pasan. Incluyen contenido alterado, fuentes modificadas, buffers
mutables, timestamps, cambios T/historia, ROI subpíxel, batch/slot/modelo/runtime,
aislamiento entre veinte IDs, copias de ownership, LRU por bytes, guards, NaN,
dtype/grad y rechazo desde otro hilo. Ruff y Pyright pasan. No se utilizó GPU.

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -m pytest -o 'addopts=' -q tests/unit/test_train40_runtime_feature_cache.py
& '../e-jepa-ttc/.venv/Scripts/ruff.exe' check operational/train40_runtime/feature_cache.py tests/unit/test_train40_runtime_feature_cache.py
& '../e-jepa-ttc/.venv/Scripts/pyright.exe' --venvpath '../e-jepa-ttc' operational/train40_runtime/feature_cache.py
```

Pendiente: tests CUDA de ownership/lifetime, integración end-to-end, paridad contra
original sin caché, coste del fingerprint y medición de bursts sin hits. Conservar
los misses cuando llega un frame/ROI nuevo; no anunciar un promedio favorable
como cumplimiento del máximo de 20 ms.

## Prueba real de metadatos: el 80% era una hipótesis demasiado optimista

`probe_metadata_reuse.py` leyó relojes y anotaciones causales de las 32 secuencias,
sin píxeles, modelos, TTC ni GPU. Generó hasta cien actualizaciones de 20 ms por
secuencia desde su primera ancla registrada; los límites originales produjeron
3.191 instantes programados. Comparó la ROI RGB y la historia completa que entrarían
en los productores, manteniendo fijos modelo, precisión y forma de un objeto.

Recibo: `metadata_reuse_v1.json`.

- Programados: **3.191**; aceptados: **3.129**; excluidos: **62**.
- Exclusiones: 53 anotaciones ambiguas/no primarias y 9 ausentes.
- Misma llamada RGB que en el update inmediatamente anterior: **1.719**.
- Fracción sobre todos los programados: **53,8703%**.
- Fracción condicionada a aceptados: **54,9377%**, declarada separadamente.
- El intervalo mediano de la cámara es aproximadamente **49.999 us (20 Hz)**.
  Los 100 ms son el espaciado de las tripletas del modelo, no la frecuencia de
  llegada de frames. Por tanto, el supuesto de ahorrar cuatro de cada cinco
  actualizaciones no está respaldado por estos metadatos.

Se conserva la población completa de exclusiones en el resumen y se invalida la
continuidad al saltarse un instante. Este resultado es un **proxy de un objeto
primario por secuencia**, no una medición de hits del servicio ni de veinte tracks
reales. Los veinte objetos pueden tener distintas ROI/frames y la política all-hit
podría producir una fracción menor. Incluso con hits frecuentes, las actualizaciones
que reciben nuevos frames continúan requiriendo el lote completo.
