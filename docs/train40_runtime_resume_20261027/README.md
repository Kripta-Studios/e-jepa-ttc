# TRAIN40: reanudación de optimización

Estado: **EN CURSO**, reanudado por petición expresa del usuario tras el cierre
`9f62aebc`. La meta permanece en 50 actualizaciones por segundo, menos de 20 ms
hasta el último TTC de hasta veinte objetos, pesos y normalizadores originales,
paridad `atol=rtol=1e-5`, cobertura y ausencia de acumulación de trabajo.

**Prioridad actual, indicada por el usuario el 28 de octubre:** demostrar primero
menos de 20 ms completos y 50 Hz sostenidos con **un objeto**, y después con
**dos objetos**. Veinte objetos queda como objetivo posterior. Cada hito debe
registrar su población; cumplir uno o dos no se presentará como cumplir veinte.

Este documento continúa el [handoff anterior](../train40_runtime_handoff_20261027/README.md).
Su estado pausado describe aquel cierre y no se reescribe. Ninguna mejora en un
helper ni una prueba con una sola ROI acredita la meta completa.

## Organización y medición

La validación inicial ejecuta el commit publicado en el worktree detached
`../e-jepa-ttc-train40-baseline-9f62aebc`. Los checkpoints, manifiesto y resultados
originales se abren en sus ubicaciones existentes, sin modificar sus bytes.
El freeze anterior contiene seis fuentes absolutas. Un wrapper separado verifica
el hash esperado de ambas copias y registra su mapa de reubicación antes de llamar
al guard original con rutas relativas. El primer intento sin esa adaptación se
conserva como fallo de infraestructura anterior a inferencia.

Solo un proceso puede medir la GPU a la vez. Las líneas de investigación tienen
propiedad de archivos independiente. Los tiempos de la validación inicial se
declaran diagnósticos bajo desarrollo CPU concurrente; no son una comparación de
rendimiento controlada. Los ensayos posteriores deben registrar código congelado,
hardware, precisión, población, calentamiento y fronteras del cronómetro.

| Frente | Entrega |
|---|---|
| Paridad | Las 554 consultas, tres TTC, tensores event y preservación |
| Productores | Aislar deriva por batch; layout y algoritmos de convolución |
| Transporte | Redundancias y compilación de bloques completos |
| Representación | Prefijos temporales con épocas acotadas y ROI independientes |
| Normalización | Selección exacta del percentil, sin ordenar todos los valores |
| Servicio | Entrada compartida y contexto/TTC separados por objeto |
| Coordinación | Llegadas independientes, cola acotada, evidencia y admisión |
| Consejo | Revisión numérica, límites físicos y fuentes técnicas primarias |

## Protocolo de carga sostenida

`operational/train40_runtime/paced.py` libera observaciones RAW desde un hilo
independiente y mantiene `t0+k*20 ms`. La inferencia se ejecuta en el hilo llamante,
que debe esperar hasta disponer de todos los TTC en host. Registra por separado
deadline, disponibilidad real, comienzo y finalización; retiene errores de fuente,
fallos de inferencia y descartes por cola llena. Una bandera de éxito sin las
predicciones finitas requeridas no constituye una finalización válida.

El helper declara siempre replay/capacidad, nunca sensores o veinte tracks reales.
Un éxito de capacidad no sustituye el gate de paridad ni demuestra el alcance
físico. Los payloads deben conservar sus bytes hasta el consumo; la integración
debe acotar bytes por paquete además del número de elementos de la cola.

## Evidencia durante la investigación

Los recibos nuevos se guardan bajo `artifacts/train40_runtime_resume_20261027/`,
en carpetas nuevas y separadas. Esas carpetas están ignoradas por Git. Al consolidar
un resultado se conservarán recibos compactos publicables, fuentes ejecutadas y
comandos; no checkpoints, medios RAW, ruedas ni cachés del compilador.

La validación completa del cierre anterior conserva las 554 consultas, pero
rechaza cinco salidas E_CTX. La investigación aisló una dependencia numérica del
batch del encoder: desactivar la reutilización de endpoints de eventos corrige
los cinco casos sensibles; desactivarla también para RGB introduce otro fallo.
La variante conserva la reutilización RGB y desactiva la de eventos. La nueva
admisión `corrected_all554_v1` pasa las tres salidas y las entradas en las 554
consultas, con E_CTX bitexact. No demuestra el plazo ni una frecuencia sostenida.

El piloto de veinte regiones trasladadas ya pasa las tres salidas TTC con RGB
procesado en grupos de un objeto, manteniendo el batch del encoder. Eso acredita
paridad en ese piloto, no veinte tracks anotados ni cobertura de toda la cohorte.
Su latencia completa sigue muy por encima de la meta. No se amplía la tolerancia.

Las cifras, poblaciones, estados negativos y fronteras de medición se regeneran
en [RESULTS_GENERATED.md](RESULTS_GENERATED.md), con hashes en
[RESULTS_SOURCES.json](RESULTS_SOURCES.json). Mientras la investigación está activa,
esas referencias apuntan a los artefactos locales; un paquete de evidencia
publicable debe copiarse y verificar sus hashes antes de afirmar portabilidad.

## Próximas admisiones

1. Transporte compilado dentro del productor original: demostrar primero las
   tres salidas TTC en casos sensibles y después toda la cohorte. El éxito del
   microbenchmark de transporte no admite el modelo completo.
2. Caché RGB: separar MISS, HIT y cambios de ROI/historial. Un HIT exige identidad
   completa y procedencia del MISS; contexto y TTC usan siempre el reloj actual.
3. Transferencia uint8: mantener los píxeles PIL exactos, normalizar mediante una
   LUT FP32 exacta y medir también empaquetado CPU y última salida disponible.
4. Selector de prefijos compilado: preservar exactamente conteos y momentos
   enteros antes de conservar la interpolación y normalización originales.
5. Servicio sostenido: observaciones nuevas a 50 Hz, hasta veinte ROI distintas,
   cola acotada y todos los fallos/descartes visibles. No dividir la latencia de
   una actualización entre su número de objetos.
