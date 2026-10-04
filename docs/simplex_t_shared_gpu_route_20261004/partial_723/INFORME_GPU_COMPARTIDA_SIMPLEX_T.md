P3 incompleto por recursos: inferencia con autorización explícita de GPU compartida.

Se conservaron H8, los 24 endpoints y los 60.000 updates científicos anteriores. Esta continuación ejecutó cero actualizaciones de optimizador.

Se preservan 723 de 1.728 solicitudes previstas: 26 consultas TRAIN tienen las 27 mediciones completas. Se validaron primero los 64 contextos crudos y las tres familias de productores contra el extractor canónico. Los brazos reducidos no llaman a expertos excluidos.

| Ruta | p95 caliente 1 (ms) | p95 caliente 2 (ms) | p95 HDF5 frío (ms) |
|---|---:|---:|---:|
| H1_SEED7 | 2957.132 | 3004.112 | 3063.310 |
| H8_SEED7 | 11119.230 | 10014.514 | 11698.414 |
| H16_SEED7 | 21706.977 | 19977.805 | 21215.474 |
| FULL_C0 | 10967.939 | 10493.350 | 11618.893 |
| A5_ONLY_C0 | 10570.349 | 10367.352 | 11311.792 |
| C2F_ONLY_C0 | 11209.905 | 10976.241 | 13609.119 |
| A5_PAIR_C0 | 10425.267 | 10966.547 | 10653.420 |
| SET_AGE_C0 | 10297.579 | 10487.116 | 11097.948 |
| SET_NOTIME_C0 | 10434.790 | 9685.206 | 10793.796 |

Los CSV preservan precisión completa. El total se midió directamente, desde la petición con ROI suministrado hasta TTC; no se sumaron p95 de etapas.

GPU compartida: las diferencias observadas incluyen contención y variación del host. No prueban un ahorro causal o una latencia aislada. Frío significa cierre de las cachés HDF5 de la aplicación, con modelos residentes y caché del sistema operativo intacta. La carga de modelos se publica separadamente.

El protocolo fija 64 consultas TRAIN de fold0 y tres familias de productores internos. La tabla usa las 26 consultas que tienen las 27 mediciones; si faltan consultas, esos cuantiles son parciales y no representan la población completa. La precisión referenciada corresponde a seed7, nueve secuencias OLD_DEV y tres folds. No son una nueva evaluación ni una nueva selección de checkpoints. Los bytes CUDA son snapshots posteriores al forward con los tres productores residentes; no son picos por solicitud ni prueban una reducción de memoria del sistema.

Se preserva el intento inicial fallido de admisión numérica y sus recibos. Activar algoritmos deterministas alteraba el runtime histórico de los productores; se restauró su configuración original antes de medir, sin ampliar tolerancias. Los 64 contextos admitidos coinciden exactamente. Dos solicitudes de diagnóstico se contabilizan aparte de los intentos del driver.

Se mantiene el batch16 original de productores con padding. H1/H8/H16 preparan sólo su historia necesaria, pero este experimento no cambia ni optimiza los encoders.

Los criterios de precisión anteriores siguen vigentes: los brazos reducidos y agregadores no pasan el cribado registrado. Una latencia menor no cambia esa decisión. H16 conserva una mejora estable entre seeds de cabeza y la incertidumbre entre escenas; no sustituye automáticamente al H8 histórico.

No se midieron detección, tracking online, sensor-to-AEB ni calibración de incertidumbre. La paridad numérica y las fuentes reutilizadas no constituyen una réplica nueva de los expertos.

Propuesta única posterior, sin ejecutar: optimizar la preparación retrospectiva raw/ROI con pesos congelados y paridad exacta, bajo un protocolo independiente que fije consultas TRAIN, memoria y p50/p95 antes de medir. Una caché sólo será válida para la misma observación y ROI acreditados. Ningún entrenamiento ni holdout se abre con esta entrega.
