# Arquitectura aditiva y restricciones

Todo el código nuevo vive en `operational/train40_runtime/`. No se editaron los módulos científicos originales para que el oracle siga siendo ejecutable y sus hashes sigan verificando.

## Módulos

| Módulo | Responsabilidad y estado |
|---|---|
| `encoding.py` | LUT float64 de proyección y tiempo, histogramas canónicos y crop RGB antes de copiar todo el frame. Probado contra referencia. |
| `inputs.py` | `PersistentSensorInputs`: HDF5 persistente, caché acotada de RAW, seis ventanas únicas en H4, frames RAW, bbox verificada, pool opcional de 1–8 crops. |
| `integral.py` | `IntegralEventHistory`: ring causal de momentos por buckets de 5 ms, ingestión de 20 ms, ROI actual, `query`/`query_many`. Experimental hasta admisión completa. |
| `producers.py` | Grafos de productores o bloque completo, fase float64, reutilización de encoders, grupos por objeto y tickets asíncronos con guards. |
| `context.py` | Contexto padded8 residente y cabeza final; una pequeña transferencia final a CPU. Grafos opcionales. |
| `predictor.py` | Integración individual. `predict` incluye entradas; `infer_prepared` empieza con representaciones y no debe venderse como latencia total. |
| `measurement.py` | Paridad, cobertura, percentiles, actualizaciones multiobjeto y preservación. |
| `replay_inputs.py` | Nuevo manifiesto causal a pasos de 20 ms; no modifica el original ni lee TTC. Anotaciones con disponibilidad simulada declarada. |
| `transport.py` | Experimento separado de correlación y estadísticos redundantes. No integrado; tiene dos xfail estrictos por una opción no admitida. |

## Entradas y cachés

El adaptador original aplica la **ROI de la consulta actual a todo el historial**. Un embedding anterior no es reutilizable solo por timestamp cuando cambia esa ROI. Las cachés implementadas conservan eventos y frames RAW; el crop y la representación se recalculan para la ROI actual. Cualquier caché futura de features debe incluir identidad de modelo, fuente, ventanas, tiempos, precisión y ROI exacta.

Cuatro tripletas event contienen seis intervalos únicos, no doce. El adaptador evita almacenar los doce slots H16 no usados y retorna `[4,3,12,128,128]`. Las reducciones canónicas siguen usando float64 antes de salida float32. La LUT temporal se limita a ventanas nativas de 100 ms; otras duraciones conservan el camino original.

El caché RAW verifica ruta, tamaño y mtime; el caché bbox liga todos los campos de identidad y stat, y vuelve a comprobar SHA al cambiar. Los lectores interrumpidos no anuncian como completo un sufijo parcial. Los límites son por bytes. RGB se recorta con el mismo PIL bilinear y normaliza cada frame único una vez, sin cambiar orden ni color. `preload_raw_rgb` carga únicamente frames causales; no hace ROI, normalización ni inferencia.

## Ring integral

Se almacenan cuentas enteras y sumas de timestamps relativos en 120 buckets de 5 ms, con integrales espaciales por polaridad. Cinco buckets forman un cuarto temporal de 25 ms; con esos momentos se reconstruye el voxel lineal de cinco bins de cada ventana de 100 ms. Las sumas rectangulares aplican una ROI nueva sin recorrer otra vez millones de eventos.

`initialize(raw,start,end)` exige exactamente 600 ms; `ingest` exige paquetes consecutivos de 20 ms y preserva intervalos semiabiertos. `reset` cambia la fase explícitamente. Los timestamps originales EvTTC fluctúan en microsegundos; **no redondearlos** para caber en el ring. Para las 554 consultas se reinicializa por fase; para un stream nuevo a 20 ms se mantiene una fase fija y hay que validar esa población separadamente.

`query_many` conserva orden y procesa todos los ROI, con workspace por chunks. Veinte objetos y batch ROI cuatro reservan aproximadamente 3,68 GiB en esta versión; un objeto debe configurar `max_objects=1, roi_batch_size=1`. Hay temporales eager pequeños declarados. No confundir memoria reservada del ring con pico completo de todos los modelos.

Las salidas son **prestadas** y se sobrescriben en la siguiente consulta. Mantener todas las operaciones del ring en un stream CUDA. Los fences de staging pinned protegen DMA host; no convierten todo el ring en thread-safe. Consumir o copiar la representación antes de otra consulta.

## Productores y streams

El modo por productor conserva la fase histórica CPU float64. El modo de bloque mueve esa fórmula a GPU float64 y captura productores, PAIR y PHASE17. La optimización JIT posterior al warmup cambió números, por lo que `optimized_execution(False)` se mantiene en captura.

La reutilización de encoders evalúa endpoints únicos y reconstruye el orden original. Para varios objetos, usar explícitamente `history_group_size=4` en eventos y `5` en RGB; no deduplicar atravesando fronteras de objeto. Los guards comprueban igualdad del solapamiento. Cambiar el tamaño del batch de convolución necesita paridad final aunque los endpoints sean iguales.

`expert_block_async` entrega `raw`, `experts` y `guard`. El caller debe esperar a que ambos streams terminen y llamar `validate_async_blocks` antes de publicar TTC. Las entradas inválidas se neutralizan internamente para evitar errores CUDA, pero su guard falso **rechaza** la respuesta. Los tickets conservan vida útil; las salidas se copian y la reutilización de un grafo se ordena con eventos.

La captura fría se serializa antes de lanzar ramas concurrentes. No capturar un nuevo grafo mientras otra rama realiza trabajo no capturado. Los clocks y datos cambian en buffers estáticos; no suponer que un grafo representa toda la cadena. Véase [PyTorch 2.11 CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html).

## Integración que falta

`PersistentTrain40Predictor` sigue siendo individual. El batching de productores por objeto no crea por sí mismo una cabeza temporal batch20 ni un servicio de veinte tracks. Falta conectar `query_many`, crops de todos los objetos, los dos productores agrupados, contexto/TTC por objeto, y una cola medida. Ninguna opción experimental se activa en el candidato preservado.
