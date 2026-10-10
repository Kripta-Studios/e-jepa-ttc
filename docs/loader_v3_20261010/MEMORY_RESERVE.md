# Reserva de memoria comprometida: 1 GiB

Por petición explícita del usuario del 10 de octubre, la continuación utiliza una reserva de commit de **1 GiB**. La revisión anterior (`ca2f6fa`) usaba 3 GiB. El archivo de paginación de Windows no se modifica y los datos permanecen en E:.

`operational/rgb_port_loader_v3/memory.py` aplica el cambio sobre los resultados de los guardianes originales. Antes comprueba que su decisión coincide con la política histórica: cualquier otra divergencia falla explícitamente. Se conservan RAM disponible mínima de 2 GiB, espacio libre tras reserva de checkpoint de 10.000.000.000 bytes y RSS agregado máximo de 23.000.000.000 bytes.

Las pruebas cubren los límites 0,99 / 1 / 2,95 / 3 GiB, los restantes límites y el rechazo de cambios inesperados de política. Suite focalizada: **38 casos aprobados**, Ruff aprobado y Pyright sin errores. Evidencia: `MEMORY_QA.json`.

## Resultado previo con reserva de 3 GiB

El ensayo CPU ABBA está en `README.md` y `CPU_COMPARISON.json`: igualdad exacta de batches, con menor tiempo de preparación. Durante continuación GPU, C2F se pausó en 19.096 updates al detectar 3.171.852.288 bytes libres de commit (aproximadamente 2,95 GiB). Su checkpoint se guardó correctamente. Esto impide afirmar todavía aceleración sostenida de ambos entrenamientos.

Tras la petición de cambiar el umbral se pausaron ambos de forma controlada: A5 en 27.777 updates, C2F en 19.120. No se reinician pesos, optimizador, scheduler ni sampler. El nuevo umbral permite seguir entre 1 y 3 GiB de margen; no crea memoria adicional.

Los recibos `fits/<fit>/LOADER_V3_RUNTIME.json` registran el umbral aplicado y hasta 120 muestras recientes de RAM y commit tomadas cada cinco segundos. La telemetría del reintento se guarda en `E:/EJEPA_results/v13_loader_v3_20261010/reserve1_transition/` y debe evaluarse por separado del ensayo CPU.

## Observación con reserva de 1 GiB

Dos intervalos consecutivos: cinco y tres minutos de entrenamiento concurrente, después del calentamiento CUDA. Ambos procesos conservaron sus PID y avanzaron sin una nueva pausa automática. Esto es evidencia acotada, no una garantía de estabilidad indefinida.

| Intervalo | Duración estable (s) | Margen mínimo (GiB) | Margen final (GiB) | RAM libre mínima (GiB) |
|---|---:|---:|---:|---:|
| first_5min | 301.9 | 2.371 | 2.995 | 5.255 |
| extra_3min | 182.6 | 1.417 | 2.570 | 4.124 |

La presión fluctuó; no se interpreta un pico aislado como una fuga. Los últimos minutos y la memoria privada de cada productor están desglosados en `MEMORY_OBSERVATION.json`; las muestras completas, incluidos los picos, están en `MEMORY_TIMESERIES.json`. El umbral anterior de 3 GiB habría detenido el entrenamiento durante parte de estos intervalos.

| Intervalo / productor | Memoria privada inicial (GiB, mediana) | Final (GiB, mediana) |
|---|---:|---:|
| first_5min / E_A5_MATCHED | 6.522 | 6.231 |
| first_5min / E_C2F_MATCHED | 6.193 | 6.820 |
| extra_3min / E_A5_MATCHED | 6.293 | 6.276 |
| extra_3min / E_C2F_MATCHED | 6.654 | 6.712 |

No se modificó el archivo de paginación de Windows. Se conserva la pausa segura si el margen cae por debajo de 1 GiB o se incumple cualquiera de los demás límites.
