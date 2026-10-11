# 06 — Qué aceleró, qué no y dónde queda el coste

## No mezclar tres tipos de tiempo

Preparación CPU de batches, entrenamiento forward/backward e inferencia desde
medios crudos tienen perímetros distintos. Los tiempos calientes excluyen
arranque; throughput con prefetch no es latencia individual. La suma de
medianas de etapas no es la mediana del total. Los pilotos V12 y las rutas
nativas V13 tampoco son el mismo benchmark.

## V12: de reconstrucción repetida a streaming

El informe histórico completo y CSV están en
`V12/docs/agent_handoff_20261010/05_LATENCIA_STREAMING.md` y
`V12/docs/sota_evidence_20261010/evidence/streaming/` dentro del ZIP.

La medición original incluía preparación innecesaria de historia H8 para Garl
event-only. Se separaron entradas nativas, productores compartidos y una/tres
cabezas. Después se implementaron paquetes compactos, caché cruda acotada,
reutilización de características, reproyección de voxels históricos, agrupación
de nuevos cálculos y CUDA Graphs. La latencia H8 completa incluye tres cabezas.

Código V12: `operational/streaming_revision/{packets,preparation,state,runtime,kernels}.py`.
El warp conserva anclas originales y no aplica sucesivas interpolaciones al
mismo tensor. Una ROI nueva puede incluir eventos que fueron descartados por
la ROI anterior: la reproyección no puede recuperarlos. Por eso es aproximada.

El piloto final aislado `isolated_fixed` obtuvo H8 referencia 226,717 ms,
warp 119,802 y warp+graphs 82,345 ms de mediana caliente. Garl nativo registró
120,228 ms; selección/ROI exacta 40,022 y graphs 36,806. **Garl optimizado sigue
siendo más rápido.** Es un piloto de 12 consultas de dos secuencias TRAIN40,
con diez medidas calientes; no un replay completo ni latencia de un coche.
H8 warp+graphs tuvo p95 203,490 ms y primera llamada de unos 12,36 segundos.

Dos correcciones importantes respecto a la conversación inicial: los 448 ms
pertenecían a warp+student (MiD≈44,473), mientras warp solo obtenía MiD≈32,672
y 545,560 ms en aquel piloto. Garl a 92,128 ms en otro piloto empleaba ROI exacta,
no reproyección histórica. No se deben combinar precisión de una variante y
tiempo de otra ni ocultar la versión Garl optimizada.

Los paquetes compactos bajaron ingestión CPU 90,194→49,284 ms bajo la misma
carga; hubo igualdad de eventos/voxels en 36/36 comprobaciones. Bloques de 50 ms
dieron 50,296 ms: la fragmentación adicional no mejoró esa ingestión y quedó
experimental. La compactación exacta sí quedó activa.

El estudiante A5 que aproxima características C2F/PAIR mejoró FCWD pero empeoró
Dev32; H4/H2 tampoco ofrecieron dominancia. INT8 dinámico CPU del estudiante
no demostró ahorro suficiente para promoverlo. No es cuantización GPU integral.
Resultados completos y negativos: informe 03 histórico V12.

## V13: reducir preparación durante entrenamiento

La caché event por filas disminuyó decodificaciones conservando sampler y
metadatos. Loader v3 con cuatro hilos totales y dos batches anticipados redujo
CPU A5 14,520→9,977 s y C2F 17,869→11,948 s sobre 512 muestras por ensayo.
La prueba fue de caché caliente y dos grupos, sin updates ni trabajo GPU;
no acredita ese mismo porcentaje en una época completa.

RGB conservaba inicialmente el cargador histórico. La intervención específica
`operational/rgb_port_rgb_loader/` añadió índice/caché DINO, collate de campos
consumidos y consulta de RSS solo de procesos pertinentes. ABBA sobre 512
ejemplos: media fuente 22,2496→7,1170 s (68,01% menos), aperturas DINO 355→31,
aperturas RGB iguales. El control completo de recursos pasó de mediana 5,4671
a 0,2146 s por llamada. Son costes de componentes con paridad exacta.

`operational/rgb_port_rgb_overlap/` añadió cuatro lectores privados de shards
y dos batches efectivos anticipados. Mantiene hashes, CRC, tokens, T2/T3,
orden y cursor. La reserva estimada pendiente es 128 MiB, aparte de caches,
buffers y RSS; no se presenta como consumo total del proceso.

La caché de máscaras geométricas constantes tuvo paridad GPU exacta, pero
no mejoró los tiempos: ABBA aislado 0,574 s referencia frente a 0,923 s
cacheada por batch. Se conserva el ensayo y **COMPUTE_ENABLED=False**.
No se justifica activar una optimización solo porque su idea parezca razonable.

Fuentes y tablas regenerables: `docs/loader_v3_20261010/`,
`docs/rgb_loader_20261011/`, `docs/rgb_overlap_20261011/`. La finalización de
los fits prueba que la ruta se ejecutó hasta su endpoint; no sustituye un
experimento sostenido A/B para atribuir una aceleración global.

## V13 final: rutas de inferencia completas

La tabla 04 se genera desde `docs/external_mask_20261012/COST_SUMMARY.json`.
Incluye preparación cruda, productores y contexto/fusión, y separa entradas
preparadas. Hubo igualdad raw→prepared en ocho consultas antes de medir.
El host **no estaba aislado**: existía entrenamiento CPU concurrente V14 y
actividad del escritorio. Son costes observados, no comparación justa final
de arquitecturas ni máximo rendimiento alcanzable.

E_CTX desde phase17 normalizado ronda 10,6 ms, pero desde eventos HDF5 cuesta
mediana 1.656 ms; F_TRUE desde HDF5+RGB tar ronda 2.581 ms. La diferencia de
perímetro es enorme: no culpar al pequeño corrector de todo el tiempo.
La lectura/voxelización, consultas históricas de productores, decodificación
RGB y transferencias merecen medición separada. Restar medianas de rutas
distintas no estima exactamente el tiempo de una etapa.

No hay en esta entrega una medición aislada completa del registro FCWD nuevo
ni una activación de todos los warps experimentales V12 en los fits V13.
La compatibilidad de código no acredita precisión/latencia de una combinación
que no se ha medido. BF16 de productores durante entrenamiento se conservó;
las campañas externas registran sus propias precisiones, incluida FP32.

## Medición que falta

Replay GPU cronológico completo, mismos queries y hardware, comparadores
Garl nativo y optimizado, warm/cold, p50/p95/p99, resets y cambio de ROI,
varios objetos, consumo de memoria y cobertura. Añadir timers CPU/GPU que
separen descompresión, H2D, productor, PAIR, cabeza y commit de caché.
Si se evalúa uso en vehículo, incluir además sensor, detección/tracking,
sincronización y colas: ni 82 ms de piloto ni 200 FPS del paper cubren todo eso.
