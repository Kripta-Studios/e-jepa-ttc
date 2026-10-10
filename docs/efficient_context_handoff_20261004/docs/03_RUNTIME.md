# 3. E1 — acelerar sin cambiar el problema

## Lo que ya existe
`cached_event_reader.py` conserva handles e índice `ms_to_idx`, con 8 MiB de caché
por dataset y una secuencia abierta. `context_raw_union.py` lee la unión temporal
en chunks de 250.000 eventos, retiene sólo el ROI y usa un fallback por ventana si
excede 256 MiB. `query_context_voxel.py` repite filtro espacial y proyección por
ventana. `voxel_grid.py` ya utiliza dos `bincount`, interpolación temporal y una
normalización por cuantil 0,95 sobre voxels ocupados.

El runner mantiene un lote de 16 observaciones incluso para H1/H8. Ejecuta
productores CUDA FP32 y trae varios diagnósticos a CPU. El perfil publicado no
separa qué operación domina dentro de ROI/voxel. Instrumentar ese tramo antes de
escribir un kernel. Reutilizar los 64 IDs ya perfilados, no abrir otra campaña
general e interminable de medición.

## Orden de implementación
**E1a — desglose.** Medir lectura, máscara espacial, proyección, selección temporal,
histogramas, cuantil/normalización, asignaciones/copias, transferencias y productores.
Mantener un timer de solicitud completo; no reconstruir un p95 sumando p95 parciales.

**E1b — cambio principal.** Aplicar filtro y mapeo ROI una sola vez a la unión.
Conservar arrays mapeados x/y/p, el orden original y timestamps int64. Para cada
intervalo usar `searchsorted` y views; ejecutar las mismas reducciones y normalización
por ventana. Preasignar la salida. No redondear límites ni aproximar ROI para fabricar
coincidencias: la deduplicación exacta intraconsulta tiene escaso beneficio medido.

**E1c — búfer de sensores.** Implementar, si el perfil lo justifica, un búfer acotado
para replay cronológico con índice temporal/espacial. Almacenar sensores, no targets
ni respuestas memoizadas por consulta. Claves: stream, intervalos exactos, ROI,
offset, dtype y preprocesamiento. Para features expertas, añadir checkpoints y
normalizador. Un cambio de ROI invalida tokens. No asumir que CNN sobre el frame y
luego crop equivale a crop y luego CNN: padding, stride y GroupNorm pueden cambiarlo.
La ingestión y la memoria tienen coste; no copiar cientos de GB sin autorización.

**E1d — productores.** Ejecutar sólo slots válidos y agrupar transferencias, como
variante numérica separada. Cambiar batch16 puede cambiar kernels y redondeo.
Validar antes de medir. PAIR reutiliza el output A5. Conservar disponible la ruta
histórica, sin editarla para que desaparezcan discrepancias.

**E1e — sólo si hace falta.** Si histogramas dominan, prototipar acumulación
multiintervalo o índices por pixel. La equivalencia algebraica no garantiza igualdad
de redondeo. Normalizar cada ventana, no toda la unión. Evitar cancelación con
timestamps absolutos; restar en enteros antes de convertir cuando esa semántica
sea compatible con la referencia. FP16, INT8 o un nuevo event encoding serían
experimentos diferentes, no optimizaciones exactas.

## Tres regímenes que deben medirse separados
**R0:** solicitud independiente HDF5→TTC con ROI suministrado, comparable al perfilado.

**R1:** replay cronológico con búfer residente. Medir ingestión, índices, retención,
evicciones, descompresión y trabajo amortizado. Usar sólo eventos anteriores al
cutoff. No precalentar con datos futuros ni cobrar cero por la ingestión. Reportar
coste por segundo de stream y por consulta, throughput sostenible y p50/p95. Si no
se acredita disponibilidad real, llamarlo replay, no un sistema online.

**R2:** inputs preparados→productores/cabeza, diagnóstico de kernels.

No comparar R1 optimizado contra R0 histórico como si fueran la misma carga.
Publicar R0 nuevo y, si se implementa R1, una referencia bajo ese mismo régimen.

## Paridad antes de velocidad
No modificar pesos, GT, consultas, splits, métricas, ventanas, ROI o normalización.
E1b y caché idempotente deberían producir el mismo tensor FP32 bit a bit. Si no lo
hacen, investigar. Para E1d, mantener como máximo las tolerancias históricas:
features 1e-4; point_phase 1e-5; TTC 0,01 s; tiempos 1e-7; máscaras exactas.
Reportar máximos, cuantiles y casos individuales. No ampliar tolerancias después
de fallar. Una variante tolerada se denomina NUMERICALLY_VALIDATED, no EXACT.

Pruebas requeridas: límites half-open y empates; varios chunks; ROI vacío, borde y
fuera de imagen; offset; polaridad; overflow; timestamps grandes; fallback por
capacidad; orden de consultas; exclusión de productores; invalidación de caché.
Reproducir los 64 contextos TRAIN admitidos y hasta 128 IDs TRAIN adicionales
seleccionados por hash. No filtrar por GT. Si la ruta es bitexacta, no hace falta
otra selección científica sobre OLD_DEV para demostrar que el output se conserva.

## Medición acotada
Referencia y optimizado, H1/H8/H16 y WIDE cuando exista. Mismos 64 IDs y tres bloques
emparejados, con orden AB/BA fijado por hash. Diez warmups por ruta como máximo antes
del bloque. Un modo application-cold y dos warm, con registro de recursos y GPU
compartida. Guardar cada solicitud atómicamente. No publicar sólo el bloque más rápido.

Objetivos exploratorios de ingeniería: preparación mediana al menos 2× más rápida y
p95 de la ruta al menos 20% menor. No son promesas ni condiciones para permitir
WIDE/Garl. Si no se logran, publicar el desglose y la limitación, sin otra reescritura
abierta para «rescatar» el resultado.
