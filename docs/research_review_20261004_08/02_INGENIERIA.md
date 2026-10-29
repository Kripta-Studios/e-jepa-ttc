# Ingeniería de rendimiento y recuperación

La mejora más defendible es haber completado el trabajo manteniendo identidad de datos, checkpoints y contabilidad. Hubo aceleraciones observadas, pero también propuestas lentas, ventanas no comparables y fallos externos. La [auditoría técnica](evidence/engineering_audit.json) conserva las cifras y hashes de los recibos; la [tabla de ventanas](tables/PERFORMANCE.md) evita seleccionar únicamente los mejores momentos.

## Dónde se consumía el tiempo

La ruta original combinaba lectura/descompresión NPZ, preparación de tensores, consultas del sistema, transferencias CPU→GPU y cómputo. A5 comenzó alrededor de 37–38 updates/min. No existía un único cuello de botella fijo: al reducir cómputo o lanzamientos de kernels, quedaba visible más espera de datos. Durante H8, la cantidad de eventos y la región de secuencia cambiaban mucho el coste de cada consulta.

Un muestreo de 30 segundos identificó descompresión NPZ, relaciones de procesos y consultas de memoria entre quienes retenían el GIL. Sus porcentajes describen muestras con GIL retenido, no fracciones de tiempo total. No permiten calcular una aceleración por la ley de Amdahl sin otra medición. El artículo aportado sobre escalado de grandes modelos fue útil por su método de perfilar, optimizar y volver a medir; sus resultados multi-GPU y MFU no se trasladan a un portátil con una GPU.

## Caché y preparación en procesos

Una caché grande no ayuda si expulsa repetidamente datos de uso inmediato al bajar la memoria disponible. Se coordinó presión de memoria y se limitaron buffers en vuelo. Se separaron preparación y monitorización en procesos persistentes con intérpretes independientes, reduciendo competencia por el GIL del entrenador. El intercambio mediante memoria compartida acotó copias y serialización. Se mantuvieron orden de consumo, semillas y puntos de recuperación.

La caché adaptativa fue una corrección de comportamiento bajo presión, pero **no demostró aceleración**: una ventana candidata midió 26,31 updates/min frente a 36,75 de su referencia. No se oculta ese resultado. Tampoco se contabiliza como implementada y validada una conversión general NPZ→NPY/mmap de todo el dataset: se estudió como alternativa, con coste adicional de disco, sin atribuirle resultados que no se ejecutaron.

El pipeline por procesos alcanzó 101,59 updates/min durante 400 updates, desde una referencia de 45,66. Más tarde una ventana de la misma familia cayó a 36,37. Cuatro hilos dieron 81,71 en otra ventana breve; no fue un control aleatorizado. La confirmación de fast-collate sobre 2.000 updates dio 67,47; su subconjunto posterior de 1.000 dio 82,63. Son observaciones de carga real, no una garantía de multiplicador constante para todo el entrenamiento.

## Consultas del sistema

Se optimizó el escaneo identificando primero procesos relevantes y se trasladó monitorización fuera del camino principal. El microbenchmark pasó de 362,74 a 17,16 ms por escaneo: 21,14× para esa operación aislada. La ventana del entrenador asociada midió 32,89 updates/min. No sería correcto decir que el entrenamiento se aceleró 21 veces.

La propuesta de consultar procesos sólo cada 1.000 updates habría dejado intervalos largos sin observar presión del host. Se prefirió un monitor independiente y un estado reciente pequeño, manteniendo las comprobaciones de pausa, contabilidad y recuperación. La estimación previa de ahorro por reducir frecuencia era una extrapolación; no se presenta como ahorro final medido.

## Lanzamientos GPU y paridad

Se retiraron sincronizaciones escalares innecesarias, se solaparon transferencias y se admitieron rutas CUDA Graphs. La admisión A5 redujo lanzamientos de 5.877 a 1.270, un 78,39%, con salidas/pérdidas exactas, 59 tensores de gradiente examinados y RNG exacto. La admisión no hizo updates de optimizador. La primera repetición compilada fue más lenta; el coste de preparación no se debe borrar del análisis.

La ventana posterior con graphs dio 41,78 updates/min frente a 32,89, una razón observada de 1,27. El cómputo medio bajó de unos 1.242 a 380 ms, mientras la espera de datos creció. La carga del host cambió entre ventanas, así que el dato respalda utilidad práctica, no una atribución causal aislada de todo el 27%.

Después se extendió la técnica a C2F y al transporte H8. En el tramo final H8 quedaron registrados 366.036 replays de transporte y cero fallos de graph. No se afirma una aceleración numérica de C2F sin un control comparable en las evidencias de este informe. Las rutas compiladas conservan estados canónicos para guardado; una estructura interna del compilador no sustituye el checkpoint científico.

## H8: memoria y coste real de preparación

La extracción completó 88.744 filas. En el sufijo final de 30.503, la preparación raw acumuló 3,347 segundos de trabajo de workers por fila; la extracción, 0,170; las guardas, 0,010. El primer valor suma trabajo paralelo y **no es latencia de pared**. Cuatro workers, ocho arenas y un máximo de ocho elementos en vuelo permitieron continuar con memoria acotada. El ritmo de pared de ese sufijo fue 68,65 filas/min, incluyendo el intervalo observado.

Los límites antiguos de RSS agregado pausaron con unos 4,48 y 3,04 GB disponibles en el host. La ruta final usó límite de árbol de 23 GB, reserva dura del host de 2 GiB e histéresis de recuperación de 3 GiB. No se desactivó indiscriminadamente la seguridad de memoria. El controlador final registró cero espera y cero throttling; la ausencia de nuevas pausas no permite aislar la contribución de ese mecanismo frente a los otros cambios.

El cierre de canales de monitor produjo algunos mensajes `handle is closed` después de terminar. El recibo H8 anterior al cierre dice COMPLETE con monitor sano; esos mensajes no prueban fallo de extracción. Se distingue cierre de telemetría de pérdida de datos.

## Checkpoints, disco y concurrencia

Se conservan pesos completos, estado del optimizador y estados necesarios para reproducir la continuación, junto con ledger, hash de configuración, normalización y cursor. Las escrituras son atómicas y el progreso durable se distingue del progreso ejecutado desde el último guardado. Los ensayos de fallos incluyeron productores, cabezas y features. Los recibos de paridad y de recuperación son condiciones diferentes: cargar un archivo no basta para probar continuación equivalente.

La caída de E: interrumpió la prueba fast-4 en progreso comprometido 27.430 frente a durable 27.400. Se preservó la exposición de 30 updates de replay y se invalidó la ventana para elegir número de hilos. El total de recuperación TRAIN40 conservador terminó en 91. La contabilidad no elimina esos costes por haber recuperado el resultado científico.

El piloto paralelo v4 alcanzó aproximadamente 93,25 updates/min combinados durante su segmento concurrente de 1.500 updates; el resto fue serial y su tiempo incluye restore. Su cuota completa de 2.000 no es una medida continua. La carrera de inventario al terminar el compañero se corrigió y se conservaron checkpoints antes de seguir.

## Temperatura, QoS y ETA

Hay evidencia de limitación térmica anterior y de condiciones difíciles de refrigeración, pero una captura con flags térmicos instantáneos desactivados no permite culpar a temperatura de cualquier ralentización posterior. Una prueba QoS inicial pareció mejorar mucho; la repetición dio sólo 1,034× y se restauró la política. No se publica como mejora sostenida.

Un ETA fiable necesita unidades homogéneas, ventana reciente y coste de reinicio/preparación. Los ritmos de entrenamiento son updates/min; los de features son filas/min; R1 cuenta pares de medición. Mezclarlos, usar caché caliente como si cubriera todo el dataset o proyectar una secuencia poco densa sobre otra muy densa dio estimaciones demasiado optimistas. En el cierre ya no queda entrenamiento TRAIN40: sólo la medición R1 abierta requiere ETA, y no se extrapola mientras espera o reconstruye historia tras una pausa.
