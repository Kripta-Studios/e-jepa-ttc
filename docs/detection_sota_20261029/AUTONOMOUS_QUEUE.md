# Cola local sin supervisión continua de agentes

El supervisor `scripts/run_detection_autonomous_queue.py` espera al R0 original
ya activo y después ejecuta, en serie, R1 y R2. Cada run tiene 8192 updates,
evaluaciones de selección cada 1024 y selección completa en 4096 y 8192.
R1 añade warmup/cosine; R2 añade el LR del backbone a 0,1×. No incorpora los
lectores o perfiles numéricos que siguen sin admisión.

El candidato de screening se elige por `best_selection_metric` sobre el monitor
fijado. Los empates favorecen R0, después R1 y después R2. Se ejecutan las semillas
13 y 23 del candidato y de R0; si gana R0 se evita duplicarlo. Con la semilla 7
ya disponible, la confirmación compara las medias de las tres semillas. Esto es
una confirmación local RGB, no el modelo final ni una afirmación SOTA.

La cola tiene como máximo seis entrenamientos nuevos. Reutiliza el verificador
congelado para comprobar resúmenes, hashes de checkpoints y la población GT de
ambas evaluaciones completas. Nunca usa test para seleccionar. No modifica ni
reinicia el R0 que ya estaba ejecutándose. No hace reintentos automáticos tras
un fallo de entrenamiento o verificación.

En el directorio de la cola:

- `plan.json`: trabajos y reglas sellados por hash.
- `status.json`: estado actual y trabajos completados.
- `receipts/`: PID, salida, errores y recibo de cada proceso hijo.
- `screening_decision.json`: decisión de screening cuando exista.
- `rgb_confirmation.json`: comparación final del alcance encolado.
- `supervisor.stderr.log`: errores del supervisor.

El supervisor espera hasta 24 horas si faltan los 8 GiB de RAM/VRAM requeridos o
hay otro proceso de cómputo CUDA. Mantiene una petición de actividad de Windows
para evitar suspensión por inactividad mientras está vivo; no mantiene encendida
la pantalla ni modifica la configuración de energía. Solo usa procesos locales,
sin LLM, API externa ni coste en tokens durante la ejecución.

Para impedir que arranque el siguiente trabajo, crear un archivo vacío
`STOP_AFTER_CURRENT` en el directorio de la cola. No elimina checkpoints ni mata
el entrenamiento activo. Un reinicio o fallo del supervisor requiere revisar el
estado antes de relanzarlo: el archivo `OWNER.json` impide duplicados accidentales.

Las fases R3/R4, ampliación de datos, ranking/geometría, temporalidad, eventos,
refit independiente de PHASE y confirmación final no están encoladas. Su estado
pendiente está en `plan.json`; aún requieren implementación o admisión. El
supervisor termina como `COMPLETED_QUEUED_SCOPE`, sin declarar S0–S6 completas.

Validación inicial del supervisor: diez pruebas focalizadas, Ruff y Pyright.
Los recibos de ejecuciones futuras son la evidencia de ejecución; preparar la
cola no acredita que esos entrenamientos hayan terminado.
