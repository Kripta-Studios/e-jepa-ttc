# TRAIN40: reanudación de optimización

Estado: **EN CURSO**, reanudado por petición expresa del usuario tras el cierre
`9f62aebc`. La meta permanece en 50 actualizaciones por segundo, menos de 20 ms
hasta el último TTC de hasta veinte objetos, pesos y normalizadores originales,
paridad `atol=rtol=1e-5`, cobertura y ausencia de acumulación de trabajo.

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

La validación inicial encontró un fallo de E_CTX en la consulta índice 137 que
los tres pilotos anteriores no cubrían. No se amplía la tolerancia. Las cifras
definitivas y la población de fallos se tomarán del resumen del ensayo completo.
