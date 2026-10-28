# Hallazgos y criterios para continuar

Investigación pausada a petición del usuario. El objetivo de menos de 20 ms por actualización completa
de hasta veinte objetos, a 50 Hz sostenidos, **no está demostrado**. Las cifras
regenerables y sus poblaciones están en [RESULTS_GENERATED.md](RESULTS_GENERATED.md).
Este documento explica decisiones; no sustituye los recibos JSON ni sus hashes.

El usuario ha priorizado alcanzar ese plazo y frecuencia primero con un objeto
y después con dos. Las nuevas pruebas e integraciones se centran en esos hitos;
veinte objetos queda para una fase posterior. Se mantiene el mismo gate numérico
y la misma frontera de actualización completa, sin rebajar condiciones para
declarar éxito.

## Paridad y tamaño del batch

La admisión de las 554 consultas del cierre anterior encontró cinco rechazos
E_CTX, aunque PHASE y R_CTX pasaban toda la población. Reutilizar endpoints
redujo el batch del encoder event y cambió su aritmética de convolución. Las
entradas canónicas y el contexto original permiten aislar ese efecto.

Desactivar la reutilización event corrige los cinco casos sensibles. Desactivar
también la reutilización RGB introduce otra discrepancia. La ruta candidata
actual usa E OFF / R ON. La nueva admisión completa `corrected_all554_v1`, desde
el worktree aislado `../e-jepa-ttc-train40-admission-corrected` basado en `8c5226dd`,
pasa las tres salidas TTC en las 554 consultas, con E_CTX bitexact. Todas las
entradas pasan el gate y los 606 archivos vinculados permanecen intactos. Este
resultado admite la paridad de esa ruta; sus latencias aún no cumplen la meta.

El piloto multiobjeto requirió además procesar RGB en chunks de un objeto para
conservar su batch numérico. Cada objeto mantiene su ROI, entradas, características,
historial, reloj y TTC. Las regiones adicionales son traslaciones artificiales
de una ROI sobre un historial real compartido; no son veinte tracks anotados.

## CUDA Graphs y ejecución paralela

Las primeras capturas compartían el stream de captura implícito, aunque el
calentamiento usaba streams distintos. Se observó un fallo grande e intermitente
al introducir una nueva forma y reproducir ramas en paralelo. La corrección
captura cada productor en su propio stream, conserva su vida útil y mantiene
guards y propiedad de las salidas. La regresión ejercita la transición de uno
a dos objetos y cambios de entradas, además de las primeras reproducciones.

La presencia de CUDA Graphs en un productor no significa que la preparación,
las transferencias o el servicio entero estén capturados. Cada benchmark declara
sus fronteras y qué trabajo de primación sintética excluye.

## Transporte compilado

Compilar el bloque completo de transporte reduce su coste en microbenchmarks.
La integración mantiene los encoders y pesos originales, y usa un operador
opaco de dispatcher para que el trazado TorchScript no entre en Dynamo. El
runtime conserva la propiedad del ejecutor; los grafos graban sus kernels.

La primera prueba integrada pasó las tres salidas TTC de una consulta y redujo
el tiempo preparado. El ensayo conservó un fallo de auditoría del IR, corregido
sin cambiar la aritmética. La prueba ampliada posterior rechazó dos de los siete
casos sensibles: solo pasar PHASE no admite el candidato si E_CTX o R_CTX falla.

Las variantes siguientes deben separar FMA y redondeo de división. En el entorno
instalado, `use_fast_math=False` no desactiva FMA; `force_same_precision` tampoco
promete igualdad con ATen. `emulate_precision_casts=True` cambia la emisión de
contracción, y `eager_numerics.division_rounding=True` controla divisiones FP32.
Esos controles requieren nuevas mediciones. Cambiar el árbol de reducción puede
seguir alterando el resultado aunque se desactive FMA.

El gate de admisión conserva las tres salidas TTC con `atol=rtol=1e-5`, cobertura,
guards e integridad de archivos. La comparación de features intermedias se
registra aparte: no se sustituye el gate final por uno más débil ni se introduce
un requisito nuevo después de observar resultados.

## Prefijos y normalización exacta

Los prefijos temporales de conteos y momentos evitan reconstruir los buckets
del historial para cada ROI. El query graph permite preparar todas las regiones
con menos lanzamientos. La selección del percentil por top-k conserva su rango,
interpolación y máscara: no es una aproximación del percentil.

Estas mejoras aún dejan la representación event de veinte ROI por encima del
presupuesto completo. El siguiente selector compilado intenta evitar direcciones
y gathers intermedios, conservando exactamente los enteros antes de ejecutar la
interpolación y normalización originales. Se mide memoria allocated y reserved;
un pequeño incremento de allocated no describe por sí solo el pool del grafo.

## RGB y reutilización causal

La caché exacta guarda características, nunca el TTC final dependiente del reloj.
Sus claves incluyen fuente, frames, ROI, precisión, modelo, runtime y batch. Un
HIT debe probar su procedencia desde un MISS cuyos píxeles fueron comparados al
original. El benchmark de caché exige que haya ocurrido realmente al menos un HIT.
Repetir entradas idénticas acredita ese mecanismo, pero no su tasa de acierto
durante un stream con objetos en movimiento.

El análisis de metadatos de replay encontró que los cambios de ROI invalidan casi
toda la reutilización event entre las cinco fases de 20 ms. Reutilizar por tiempo
sin comprobar la proyección de ROI cambiaría el modelo. RGB presenta más
oportunidades, pero los misses siguen contando en los percentiles y plazos.

La ruta uint8 mantiene los crops PIL, transfiere frames recortados únicos y usa
una LUT FP32 de normalización exacta. Multiplicar por un recíproco redondeado no
reproduce necesariamente la división original entre 255. El informe distingue
bytes H2D propuestos de transferencias realmente ejecutadas en GPU. La siguiente
investigación prepara recorte PIL entero en GPU con RAW residente y capacidad
acotada; todavía requiere validación de píxeles y del servicio completo.

## Resultados negativos y límites de las mediciones

Se conservan los intentos fallidos: baseline completo, captura paralela inicial,
batch RGB grande, channels-last, encoder compilado e integración de transporte
que no pasa todas las salidas. Channels-last fue más lento en el contraste
medido y falló paridad; no se promueve. El rechazo intermedio del encoder no
demuestra por sí solo un rechazo TTC, que aún no se ha medido para esa variante.

No se convierten recíprocos de una media en frecuencias sostenidas. El runner
paced conserva las llegadas independientes, cola acotada, errores, descartes y
tiempos hasta el último TTC. El coste posterior a la última observación y el
trabajo causal adelantado necesitan ambos medición: adelantar trabajo no permite
ocultar una GPU que acumula tareas entre actualizaciones.

Los artefactos locales están bajo `artifacts/train40_runtime_resume_20261027/`.
Los scripts `benchmark_train40_*` fijan configuración y escriben recibos nuevos;
`report_train40_resume.py` regenera las tablas y puede empaquetar evidencia
compacta sin pesos, RAW ni cachés del compilador. Los archivos del candidato
congelado y las predicciones originales permanecen de solo lectura.
