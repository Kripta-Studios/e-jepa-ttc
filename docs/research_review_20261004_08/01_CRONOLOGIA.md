# Cronología y evolución del protocolo

El periodo cubre del 4 al 8 de octubre de 2026, ambos incluidos. El [historial completo](tables/COMMITS.md) contiene 77 commits anteriores a la redacción de estos informes: 14, 27, 25, 7 y 4 por día, respectivamente. Los commits acreditan cambios de código; los recibos y resultados acreditan las ejecuciones.

## 4 de octubre: cierre histórico y nueva campaña E0–E3

Se terminó de recuperar la medición raw/ROI de la release histórica: de 1.114 fragmentos ya conservados se pasó a 1.728, completando los 614 pendientes tras recuperar el disco. Ese cierre pertenece a la release base. Al iniciar la nueva campaña se reutilizaron sus T6, réplicas H16, 24 cabezas y mediciones; no se repitieron ni se añadieron a la contabilidad de updates nuevos.

La autorización inicial definió cuatro ramas: E0 de estado y control analítico, E1 de preparación con pesos congelados, E2 H8-WIDE con slots fijos y réplicas condicionadas, y E3 de comparadores Garl admisibles. Se implementó una cola con límites, contratos, inventario, checkpoints completos y separación de outputs. Los fallos de inventario del bundle, resolución de rutas y replay de cabezas se corrigieron conservando el historial.

Se restauró material TRAIN de Hugging Face tras la pérdida del disco. Las raíces eAP y GarlTTC separan medios y anotaciones. Se ejecutaron las nueve cabezas WIDE, incluidas las réplicas condicionadas: 22.500 updates en total. No se repitió el entrenamiento histórico de H8/H16. El resultado replicado favorece numéricamente a WIDE, pero no permite declararlo superior con el intervalo jerárquico utilizado.

## 5 de octubre: recuperación, límites y decisión TRAIN40

Se aumentó la RAM autorizada, primero a 12 GB y después a 16 GB, y se incorporaron lectores, cachés acotadas y prefetch. Se midieron propuestas antes de admitirlas. Una ruta de dispatch GPU falló y se conservó como intento rechazado; no se presentó como optimización validada. Se endurecieron la pausa en fronteras seguras, el diario de updates, la identificación del único entrenador y las pruebas de fallos en cabezas y generación de features.

La cola nativa Garl acumuló 17.948 updates conservados. Su coste y el plazo solicitado motivaron una revisión de estrategia. Primero se autorizó una pausa recuperable y un piloto TRAIN40 sin entrenamiento; después el usuario ordenó expresamente entrenar nuestros encoders, productores y cabezas sobre las 40 secuencias, y comparar con los checkpoints públicos Garl. Esa orden cambió el alcance inicial de E0–E3. No se interpretó el techo físico como permiso general para brazos adicionales.

Se investigó la viabilidad de los pesos públicos y de las particiones. Una lista pública TRAIN40 identifica secuencias, pero no certifica por sí sola qué datos o selecciones vio cada checkpoint. Tampoco convierte una lista upstream de 46 secuencias en seis holdouts limpios. Test12 no se trató como evaluación local disponible con ground truth. Se redactaron dudas para los autores, sin enviar correos desde el agente.

La opción Runpod se discutió y después el usuario la descartó expresamente. La ejecución continuó en el hardware local. La raíz original E0–E3 conserva los productores incompletos y la razón de su pausa; la nueva raíz TRAIN40 no sobrescribe esos checkpoints.

## 6 de octubre: entrenamiento y coordinación CPU/GPU

Se ejecutó la cola TRAIN40 con endpoints fijos para A5, C2F, PAIR y tres cabezas H8. Se sellaron entradas y normalización, se protegieron escrituras atómicas y se añadió preparación anticipada. El perfil señaló coste tanto en descompresión/collation como en consultas de procesos y miles de lanzamientos de operaciones pequeñas.

Las modificaciones incluyeron solapar transferencias de entrada, retirar sincronizaciones escalares evitables, coordinar expulsiones de caché, abaratar el escaneo de procesos y aislar preparación/monitorización en procesos persistentes. Se admitió CUDA Graphs para A5 después de comprobar salidas, pérdidas, gradientes y RNG. Se ensayaron distintos números de hilos y formas de ensamblar batches sin cambiar el orden científico de muestras.

Los 101,6 updates/min de una ventana inicial no se mantuvieron. Una confirmación de 2.000 updates de fast-collate midió 67,47 updates/min. No se extrapoló el mejor pico como ETA permanente. La desconexión de E: invalidó una prueba fast-4; se restauró desde el punto durable con una exposición de replay de 30 updates, sin perder progreso científico tras la recuperación.

El usuario autorizó probar A5 y C2F simultáneamente, modificando la restricción inicial de un único entrenador para ese ensayo. El piloto terminó su cuota de 2.000 updates combinados en dos segmentos: 1.500 concurrentes y 500 seriales, tras una carrera al salir un proceso compañero. Después se admitió producción concurrente con las salvaguardas correspondientes. El piloto segmentado no acredita una aceleración concurrente continua de 2.000 updates.

## 7 de octubre: C2F, features H8 y presión de memoria

Se llevó la ruta de CUDA Graphs a C2F y se añadió supervisión horaria recuperable. La preparación de features H8 pasó a buffers compartidos, monitor independiente, unión acotada de eventos y transporte con replay. Se corrigió la recuperación directa desde el cursor durable y la observación de procesos durante inicializaciones largas.

H8 recorrió 88.744 consultas. Las regiones iniciales llegaron a ritmos cercanos a 175–199 filas/min; las regiones posteriores, más costosas, fueron sensiblemente más lentas. No son unidades de trabajo equivalentes. El cuello de botella final estaba principalmente en preparación raw/ROI, no en las cabezas pequeñas.

Hubo pausas conservadoras por límites agregados de 16 y 20 GB. La autorización posterior elevó el límite a 23 GB, con cuatro workers y reservas de RAM del host. Se sustituyó el tratamiento rígido de presión por control de flujo y recuperación, sin eliminar las protecciones necesarias para no agotar el sistema. El tramo final completó 30.503 filas a 68,65 filas/min. Los recibos finales no registran esperas de ese controlador: su instalación no prueba que por sí misma causara una aceleración.

Se investigaron temperaturas y políticas de energía. Un contador acumulado prueba limitación térmica anterior, pero no identifica la causa de cada descenso posterior de ritmo. Las pruebas QoS no reprodujeron la gran mejora inicial y se restauró la política. Se respetó la orden de no seguir tocando Control Center.

## 8 de octubre: cierre TRAIN40 y evaluación externa

La cola de entrenamiento terminó a las **00:44:13 de Madrid** (el recibo usa 22:44:13 UTC del día 7). Los endpoints suman 114.204 updates científicos nuevos. Se generaron predicciones TRAIN40, diagnóstico de colas, costes y 44 filas de análisis estratificado. Las cifras sobre TRAIN40 se etiquetaron como ajuste sobre entrenamiento.

Se implementó R1 para comparar preparación causal de eventos con una ruta residente de capacidad fija, pesos congelados y fragmentos recuperables. La paridad CPU completó 256 casos exactos. Su medición GPU seguía parcial en el corte: 486 de 768 pares guardados. Las pausas para las comparaciones EvTTC se registraron y se solicitó reanudación automática; el replay de historia al reanudar añade coste antes de guardar nuevos pares.

Para EvTTC se verificó la identidad local de 32 HDF5, aproximadamente 184,2 GiB, y se construyó un manifiesto fijo de 1.024 consultas. Se corrigieron ambigüedad y antigüedad de cajas antes de leer targets para scoring. El adaptador de eventos y el evaluador congelado produjeron resultados H8 y Garl event-only, completos a las 17:51. Se sellaron y respaldaron sus predicciones.

Por orden posterior se añadió Garl RGB+eventos, reutilizando exactamente las predicciones previas. Una revisión previa corrigió la selección de imágenes RGB: los timestamps de las cajas delimitan disponibilidad de ROI, no el fotograma que debe alimentar al modelo. La versión corregida admitió las 1.024 consultas, pasó la prueba CPU/GPU y terminó a las 18:35. Los dos bundles permanecen separados y verificados. Esta extensión se diseñó después de ver los resultados event-only, por lo que no se describe como comparación completamente ciega.

El último trabajo del periodo es esta entrega: reconstrucción de evidencia, informes, regeneración de tablas y publicación de la rama en GitHub. La autorización explícita de push sustituye la prohibición anterior; no autoriza merge a main ni submissions.
