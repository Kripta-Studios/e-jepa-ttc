# Reserva de memoria comprometida: 1 GiB

Por petición explícita del usuario del 10 de octubre, la continuación utiliza una reserva de commit de **1 GiB**. La revisión anterior (`ca2f6fa`) usaba 3 GiB. El archivo de paginación de Windows no se modifica y los datos permanecen en E:.

`operational/rgb_port_loader_v3/memory.py` aplica el cambio sobre los resultados de los guardianes originales. Antes comprueba que su decisión coincide con la política histórica: cualquier otra divergencia falla explícitamente. Se conservan RAM disponible mínima de 2 GiB, espacio libre tras reserva de checkpoint de 10.000.000.000 bytes y RSS agregado máximo de 23.000.000.000 bytes.

Las pruebas cubren los límites 0,99 / 1 / 2,95 / 3 GiB, los restantes límites y el rechazo de cambios inesperados de política. Suite focalizada: **38 casos aprobados**, Ruff aprobado y Pyright sin errores. Evidencia: `MEMORY_QA.json`.

## Resultado previo con reserva de 3 GiB

El ensayo CPU ABBA está en `README.md` y `CPU_COMPARISON.json`: igualdad exacta de batches, con menor tiempo de preparación. Durante continuación GPU, C2F se pausó en 19.096 updates al detectar 3.171.852.288 bytes libres de commit (aproximadamente 2,95 GiB). Su checkpoint se guardó correctamente. Esto impide afirmar todavía aceleración sostenida de ambos entrenamientos.

Tras la petición de cambiar el umbral se pausaron ambos de forma controlada: A5 en 27.777 updates, C2F en 19.120. No se reinician pesos, optimizador, scheduler ni sampler. El nuevo umbral permite seguir entre 1 y 3 GiB de margen; no crea memoria adicional.

Los recibos `fits/<fit>/LOADER_V3_RUNTIME.json` registran el umbral aplicado y hasta 120 muestras recientes de RAM y commit tomadas cada cinco segundos. La telemetría del reintento se guarda en `E:/EJEPA_results/v13_loader_v3_20261010/reserve1_transition/` y debe evaluarse por separado del ensayo CPU.
