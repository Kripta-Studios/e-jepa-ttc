# 05 — Latencia, reutilización y cuellos de botella

## Qué se midió y qué no

El coste total incluye la preparación de entradas y la ejecución sincronizada del sistema bajo el runner indicado. La latencia H8 incluye **tres cabezas**, salvo una ablation que lo indique expresamente. La suma de medianas por etapa no equivale a la mediana del tiempo total. Las medidas medias por etapa y los percentiles de consultas deben permanecer separados.

HDF5 offline, replay residente de eventos y un sensor físico son tres escenarios distintos. El replay residente excluye lectura de disco para ambos modelos e incluye ingestión de paquetes; conserva resets ante saltos temporales. No mide un sensor físico continuo, detección/tracking de ROI, espera de sincronización ni planificación de un coche. El sistema operativo WDDM y el escritorio siguen presentes. Los resultados tampoco son comparables directamente con los 200 FPS declarados para el forward del paper Garl sin igualar hardware y perímetro de medida.

## Campaña original y corrección de equidad

La [tabla original](../sota_campaign_20261008/tables/system_cost.csv) del 8 de octubre registraba aproximadamente:

| Sistema histórico | CPU preparación p50 ms | GPU p50 ms | Total p50 ms |
|---|---:|---:|---:|
| H8, tres cabezas | 838,426 | 129,295 | 1.374,746 |
| Garl event-only con preparación compartida | 838,426 | 7,596 | 981,995 |
| Garl RGB+eventos | 485,553 | 38,574 | 614,569 |

Era una medida de ocho consultas con batch externo uno, FP32/TF32 desactivado y cachés calientes. La preparación event-only compartida construía historia H8 que Garl no necesitaba. La revisión separó entradas nativas. Eliminar trabajo inútil del wrapper mejora el sistema, pero no representa una aceleración de la arquitectura neuronal por ese mismo porcentaje.

El [protocolo de revisión](../ttc_revision_20261009/PROTOCOL.md) compara rutas dedicadas, compactas, una/tres cabezas, preparación independiente y cronológica. Registra pausa segura de los entrenamientos concurrentes durante medidas aisladas, orden aleatorio por consulta, dos warmups y cinco repeticiones. El buffer 512 MiB es una ablation distinta del inicial 128 MiB. El prefetch offline mejora throughput y no se vende como latencia individual.

## Evolución de los pilotos de streaming

Los pilotos siguientes usan doce consultas de dos secuencias TRAIN40 vistas por el profesor. Las medianas calientes excluyen la primera consulta de cada secuencia: solo diez observaciones de latencia. El conjunto sirve para ingeniería, no para declarar superioridad externa.

| Piloto | H8 referencia ms | H8 warp ms | Variante adicional | Garl ms |
|---|---:|---:|---|---:|
| gpu_warp_pilot | 1.245,991 | 545,560 | warp+student 448,265 | 264,600 |
| isolated_roi_graph | 347,289 | 177,076 | warp+student 141,332 | 92,128 |
| isolated_fixed | 226,717 | 119,802 | warp+graphs 82,345; student 79,642 | nativo 120,228; ROI exacta 40,022; graphs 36,806 |

Fuentes: [informe completo](../sota_evidence_20261010/evidence/streaming/REPORT.md) y los `SUMMARY.csv` en cada directorio de [evidencia streaming](../sota_evidence_20261010/evidence/streaming).

Dos correcciones de interpretación son necesarias:

- El par **MiD 32,67 y 448 ms no corresponde a una sola variante**. En el primer piloto warp tiene MiD≈32,672 y 545,560 ms; warp+student tiene MiD≈44,473 y 448,265 ms.
- Garl a 92,128 ms no usa reproyección histórica. Sus optimizaciones de selección/ROI son exactas. En el piloto final, comparar H8 82,345 con Garl nativo 120,228 es una comparación de implementaciones válida si se etiqueta, pero debe acompañarse de Garl optimizado 36,806. No ocultar ese baseline para afirmar victoria de velocidad.

En `isolated_fixed`, H8 warp+graphs reduce la mediana respecto a H8 referencia un 63,7%, mantiene MiD≈32,672 y tiene p95≈203,490 ms. Su primera llamada tarda≈12.362,295 ms. La media de todas las consultas del CSV incluye arranque/compilación y es muy superior a la mediana caliente. Garl graphs también tiene arranque,≈1.389,929 ms. Un sistema desplegado necesitaría preparación previa y una política de recuperación cuando cambien las formas.

## Dónde permanece el coste

Fuente: [STAGES.csv](../sota_evidence_20261010/evidence/streaming/STAGES.csv); medias de diez consultas posteriores al arranque del piloto final.

| Etapa | H8 referencia ms | H8 warp ms | H8 warp+graphs ms |
|---|---:|---:|---:|
| Productores | 62,936 | 57,496 | 19,240 |
| Cabezas y commit de estado | 11,425 | 10,785 | 31,594 |
| Lectura de preparación | 40,383 | 19,013 | 17,969 |
| Crop/remapeo | 7,520 | 2,927 | 2,865 |
| Codificación/caché de voxels | 74,326 | 18,657 | 18,372 |
| Lookup/reproyección | 0,037 | 3,323 | 3,135 |
| Total medio de consultas calientes | 217,654 | 131,375 | 112,253 |

La ingestión media registrada es 17,576 ms en estas rutas. No sumar indiscriminadamente columnas: consultar el runner para el anidamiento exacto de timers. Tampoco combinar esta media 112,253 con la mediana 82,345 como si fueran el mismo estadístico.

La reproyección reduce sobre todo lectura y voxelización. Graphs reduce lanzamiento y ejecución de productores, pero desplaza el peso relativo hacia preparación, copias, sincronización, cabezas y gestión de estado. Los 31,594 ms de «head_and_commit» no son prueba de que el MLP por sí solo cueste eso: hacen falta timers más finos antes de rediseñar la cabeza.

## Mecanismos implementados

### Paquetes compactos y caché cruda

[`packets.py`](../../operational/streaming_revision/packets.py) implementa `PacketRing`, búsqueda de límites sin promover arrays completos de timestamps y offsets uint32 por paquete. Coordenadas int16 cuando son representables, con fallback exacto int32; retención temporal y presupuesto de memoria acotados. El formato habitual reduce aproximadamente 17 a 9 bytes por evento, según los tipos aplicables.

La prueba CPU [packet_blocks/cpu_trial](../sota_evidence_20261010/evidence/streaming/packet_blocks/cpu_trial) compara bajo la misma carga concurrente: ingestión 90,194→49,284 ms con aritmética compacta, igualdad exacta de eventos y voxels en 36/36 comprobaciones. La compactación está activa por defecto. Dividir además en bloques de 50 ms da 50,296 ms: no mejora esa ingestión, aunque favorece algunas lecturas. Por eso el tamaño de bloque queda como opción, no como supuesto ahorro universal.

### Voxels históricos y ROI

[`preparation.py`](../../operational/streaming_revision/preparation.py), `warp_voxel` e `IncrementalPreparer`, permite construir solo las ventanas nuevas y reproyectar voxels ya construidos. Reproyecta los diez planos espaciales y conserva dos planos escalares; rellena con cero fuera del soporte disponible. Usa anclas originales para evitar acumular repetidos warps.

Una ROI nueva puede contener eventos descartados por la ROI antigua. Ninguna interpolación recupera esos eventos. El error de aproximación debe medirse por TTC, banda, cambio de ROI y tiempo; paridad de un warp consigo mismo no equivale a paridad con reconstrucción cruda.

### Características y productores

[`state.py`](../../operational/streaming_revision/state.py) fija políticas de identidad, causalidad, edad y distancia de ROI. La configuración de pilotos usa límites como IoU mínimo 0,65, cambio de escala logarítmico 0,35, tolerancia temporal 2,5 ms y edad 500 ms, con caché acotada. Son hiperparámetros de ingeniería, no constantes físicas. Revisar la configuración efectiva de cada resultado.

[`runtime.py`](../../operational/streaming_revision/runtime.py) agrupa cálculos nuevos. [`kernels.py`](../../operational/streaming_revision/kernels.py) vectoriza correlaciones y envuelve graphs con salidas independientes. Graphs requiere estabilidad de formas/memoria; no elimina preparación CPU ni admite gratuitamente todas las longitudes de historia.

### Precisión y destilación

La precisión reducida se aplica selectivamente al encoder en experimentos, manteniendo geometría de inferencia según el contrato de la ruta. La receta BF16 de entrenamiento V13 no se cambia por estas opciones. INT8 probado es cuantización dinámica CPU de capas Linear del estudiante, no cuantización integral del modelo ni un kernel GPU INT8.

Reducir productores mediante estudiante aporta un ahorro mayor que comprimir una cabeza pequeña, pero el estudiante empeora Dev32. La promoción debe depender de un presupuesto conjunto de precisión y latencia fijado de antemano, no de elegir a posteriori el dataset favorable.

## Resultados fallidos y trabajo pendiente

Se conservaron un fallo de operación in-place del estudiante previo a completar la prueba, un piloto graphs con falta de memoria y otro interrumpido por presupuesto de 300 s. Los directorios `live_graph_pilot` y `live_graph_pilot_v2` no deben tratarse como evaluaciones completas. Las pruebas posteriores corrigieron rutas concretas; no borraron esos recibos.

Falta un replay GPU completo Dev32/FCWD con modelos finales, más consultas, p95/p99, arranque, resets, varias ROI por frame y contabilidad de memoria. También falta una medida con detección/tracking real y hardware objetivo. Una mediana inferior a 100 ms no certifica uso seguro en un vehículo, ni el resultado más lento prueba que toda optimización sea imposible. La evidencia permite priorizar preparación y estado después de los productores, con un baseline Garl optimizado contemporáneo.
