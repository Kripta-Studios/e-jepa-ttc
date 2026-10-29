P3 bloqueado por una fuente externa ausente: inferencia con autorización explícita de GPU compartida.

Se conservaron H8, los 24 endpoints y los 60.000 updates científicos anteriores. Esta continuación ejecutó cero actualizaciones de optimizador.

Se preservan 1114 de 1.728 solicitudes previstas: 41 consultas TRAIN tienen las 27 mediciones completas. Se validaron primero los 64 contextos crudos y las tres familias de productores contra el extractor canónico. Los brazos reducidos no llaman a expertos excluidos.

Dependencia pendiente: [WinError 3] El sistema no puede encontrar la ruta especificada: 'E:\\eAP_dataset\\data\\train\\Z30i8WuL2N\\events.h5'. Los fragmentos confirmados se reutilizan al reanudar; no se repiten los entrenamientos ni se cambian consultas.

| Ruta | MiD OLD_DEV seed7 | p95 caliente 1 (ms) | p95 caliente 2 (ms) | p95 HDF5 frío (ms) |
|---|---:|---:|---:|---:|
| H1_SEED7 | 132.831619 | 3084.605 | 3023.812 | 3353.193 |
| H8_SEED7 | 121.646332 | 10101.055 | 8941.477 | 9568.702 |
| H16_SEED7 | 118.865739 | 18101.137 | 15801.887 | 15677.648 |
| FULL_C0 | 121.404706 | 9258.812 | 10211.306 | 8865.762 |
| A5_ONLY_C0 | 127.785376 | 9096.821 | 8902.122 | 8965.624 |
| C2F_ONLY_C0 | 126.704974 | 8557.175 | 9484.959 | 11642.354 |
| A5_PAIR_C0 | 124.668732 | 7816.698 | 9666.627 | 9424.270 |
| SET_AGE_C0 | 122.442604 | 7979.192 | 9109.960 | 9145.242 |
| SET_NOTIME_C0 | 123.686600 | 7867.713 | 8601.395 | 11814.924 |

Los CSV preservan precisión completa. El total se midió directamente, desde la petición con ROI suministrado hasta TTC; no se sumaron p95 de etapas.

STAGE_SHARES.csv desglosa las fracciones de la suma de tiempos medidos, por ruta y bloque, en las mismas consultas completas. No son cocientes de p95 ni una estimación del coste fuera de esta población. Familias internas observadas en ese subconjunto: 0, 1.

GPU compartida: las diferencias observadas incluyen contención y variación del host. No prueban un ahorro causal o una latencia aislada. Frío significa cierre de las cachés HDF5 de la aplicación, con modelos residentes y caché del sistema operativo intacta. La carga de modelos se publica separadamente.

El protocolo fija 64 consultas TRAIN de fold0 y tres familias de productores internos. La tabla usa las 41 consultas que tienen las 27 mediciones; si faltan consultas, esos cuantiles son parciales y no representan la población completa. La precisión referenciada corresponde a seed7, nueve secuencias OLD_DEV y tres folds. No son una nueva evaluación ni una nueva selección de checkpoints. Los bytes CUDA son snapshots posteriores al forward con los tres productores residentes; no son picos por solicitud ni prueban una reducción de memoria del sistema.

Se preserva el intento inicial fallido de admisión numérica y sus recibos. Activar algoritmos deterministas alteraba el runtime histórico de los productores; se restauró su configuración original antes de medir, sin ampliar tolerancias. Los 64 contextos admitidos coinciden exactamente. Dos solicitudes de diagnóstico se contabilizan aparte de los intentos del driver.

Se mantiene el batch16 original de productores con padding. H1/H8/H16 preparan sólo su historia necesaria, pero este experimento no cambia ni optimiza los encoders.

Los criterios de precisión anteriores siguen vigentes: los brazos reducidos y agregadores no pasan el cribado registrado. Una latencia menor no cambia esa decisión. H16 conserva una mejora estable entre seeds de cabeza y la incertidumbre entre escenas; no sustituye automáticamente al H8 histórico.

En la réplica H16, la media de pérdidas de las tres seeds mejora 2,542666 MiD frente a H8 (intervalo jerárquico [-5,213398; -0,111241]). Las dos seeds nuevas mejoran 2,423702 MiD, pero su intervalo jerárquico [-5,244259; +0,041091] incluye cero. Seed7 ya era exploratoria. La réplica corresponde a las cabezas; las unidades independientes siguen siendo nueve secuencias. El contexto pasado aporta información, pero el mecanismo cronológico no está demostrado.

No se midieron detección, tracking online, sensor-to-AEB ni calibración de incertidumbre. La paridad numérica y las fuentes reutilizadas no constituyen una réplica nueva de los expertos.

Propuesta única posterior, sin ejecutar: optimizar la preparación retrospectiva raw/ROI con pesos congelados y paridad exacta, bajo un protocolo independiente que fije consultas TRAIN, memoria y p50/p95 antes de medir. Una caché sólo será válida para la misma observación y ROI acreditados. Ningún entrenamiento ni holdout se abre con esta entrega.
