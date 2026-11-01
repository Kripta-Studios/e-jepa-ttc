# Latencia local del original y la corrección de cadencia

Ejecutar solo tras cerrar la comparación nativa emparejada completa y verificar
sus recibos. El fallo de paridad histórica permanece; estos tiempos no son los
del servidor del resultado oficial. No usar targets nativos ni seleccionar
consultas por su predicción o coste observado.

Selección fija: primeras diez consultas físicas, ordenadas por timestamp/token,
de cada secuencia del inventario completo. Conservar todas sus historias de
entrada. Registrar secuencias, tracks, observaciones nuevas y población exacta.
No afirmar que este prefijo cubre toda la distribución de servicio.

Mismo hardware, FP32, lote uno, un hilo Torch, TF32 desactivado, benchmark
desactivado y algoritmos deterministas. Una carga común de los pesos; medirla
por separado. Calentar ambos modelos con las dos primeras consultas y reiniciar
sus cachés antes del ensayo. Los pesos están calientes; el primer acceso a un
track no equivale a proceso/GPU completamente frío.

Cada brazo tiene cachés de observaciones y lector propios. Reiniciar lectores
y cachés al cambiar de secuencia. Alternar el orden original/candidato por
consulta y por secuencia, para no favorecer siempre al segundo brazo con la
caché del sistema operativo. No vaciar cachés del sistema ni alterar otros
procesos. Registrar que el IO compartido no queda experimentalmente aislado.

Primera pasada sin hooks de medida: latencia host completa de predict, con
sincronización CUDA antes/después, desde lectura/preparación de las entradas
disponibles hasta salida host. Incluye verificación habitual del tensor de
eventos y decodificación RGB. Excluye importación, admisión/hash previo de shards,
carga de pesos y captura física anterior del sensor. Verificar predicción exacta
contra el replay local completo de cada brazo.

Segunda pasada instrumentada independiente: medir preparación/caché, productores
y resto de contextos/fusión/salida. Sus sincronizaciones añaden coste; no usarla
para la primaria de latencia ni sumarla a la primera. Exigir las mismas salidas.
Guardar todas las mediciones y p50/p95/p99, además de memoria pico CUDA/RAM.

Informar el cociente p95 candidato/original y el límite descriptivo inicial 1,10;
no supone promoción ni prueba estadística de no inferioridad. Separar primeras
consultas de track y las posteriores. Simular cola por secuencia con los tiempos
de llegada reales y tiempos de servicio medidos; declarar que es una simulación
de servidor único, no una demostración de streaming a 50 Hz. Contar retrasos de
salida superiores a 20 ms como diagnóstico, no umbral de seguridad.
