# Paridad RAW real de la fuente CPU precargada

El companion `memory_raw_parity_v1` pasó la comparación exacta de la historia
inicial de 600 ms y las 50 consultas del prefijo `CPLA-low`, sin omitir slots.
Este resultado valida datos CPU entregados por la fuente. No ejecuta modelos,
no compara TTC y no mide capacidad a 50 Hz.

Evidencia:

- [summary.json](../../artifacts/train40_runtime_council_20261028/memory_raw_parity_v1/summary.json)
- [binding.json](../../artifacts/train40_runtime_council_20261028/memory_raw_parity_v1/binding.json)
- [script ejecutado](../../artifacts/train40_runtime_council_20261028/memory_raw_parity_v1/run_raw_parity.py)
- [snapshots de fuentes y metadata](../../artifacts/train40_runtime_council_20261028/memory_raw_parity_v1/sources)

Se compararon `PreloadedRawSource` y `DiskRawSource` sobre los mismos anchors
y archivos físicos. Para cada consulta se exigió igualdad exacta de dtype,
shape y valores de todos los campos de eventos, todos los píxeles RGB RAW,
tripletas y timestamps, cajas, calibración, JSON de metadata y conteo de bytes.
La lectura por disco actuó como control independiente, sin usar los arrays
privados de la fuente precargada.

La fuente mantuvo los eventos en su dtype nativo durante la precarga y produjo
la conversión ABI durante cada adquisición. No había tokens de frames ni
anotaciones parseadas al terminar la precarga. El warmup selló trece frames
anteriores al corte permitido. En total se sellaron 33 frames y los payloads
no copiaron de nuevo sus píxeles antiguos. La comparación incluyó todos los
arrays devueltos; esos conteos no sustituyen las comprobaciones de igualdad.

La cota numérica de la fuente fue 1.651.465.904 bytes, inferior al presupuesto
de 2 GiB. El archivo RAW residente ocupó 332.914.352 bytes. La cota conserva
reservas para ring, payloads y scratch; no es una medida RSS. El control de
disco añadió arrays temporales de comparación declarados aparte en el binding.
No se amplió el límite para conseguir que el ensayo pasara.

Warmup, adquisiciones y cierre ocurrieron en el mismo hilo propietario. El
último payload siguió teniendo píxeles válidos después del cierre de la fuente.
Los 207 archivos vinculados permanecieron intactos; el HDF5 conservó tamaño y
mtime. No se recalculó el hash del HDF5 completo.

El script tardó 9,6003 segundos en realizar la comparación CPU. Esa duración
incluye ambas fuentes y comprobaciones independientes; no es latencia de
inferencia. Ruff y Pyright pasaron. El directorio queda sellado por la política
de escritura exclusiva del script y no debe reutilizarse para otro ensayo.

El [primer piloto paced con esta fuente](../../artifacts/train40_runtime_council_20261028/paced/memory1_50_v1/summary.json)
se ejecutó después de este companion y falló cobertura/cadencia. Sus poblaciones
de TTC válidos, acknowledgements fallidos y descartes se separan en el
[informe regenerado](RESULTS_GENERATED.md). Sigue pendiente un replay completo
admitido, además de la evidencia de los tensores efectivamente consumidos
cuando se active la caché de features. El control original con lectura de
disco sigue conservado con su propio scope y resultados negativos.
