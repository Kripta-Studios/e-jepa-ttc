# Jitter de llegadas a 50 Hz en Windows

El sondeo de 60 segundos encontró una diferencia relevante entre las esperas: `threading.Event.wait` retrasó el comienzo de adquisición falsa hasta 16,65 ms, mientras `time.sleep` tuvo un máximo de 1,27 ms en la misma sesión. Este resultado justifica ensayar una espera de mayor resolución en el runner paced. No demuestra inferencia, cola, TTC, objetos reales ni un plazo de seguridad.

## Evidencia y frontera medida

- Script: `scheduler_jitter_probe.py`.
- Datos completos: `scheduler_jitter_v1.json`, con los 3.000 recibos, hashes del script y `paced.py`, versiones y relojes.
- Seis bloques de diez segundos, 500 llegadas a 50 Hz por bloque, orden Event/sleep/sleep/Event/Event/sleep. Cada método tiene 1.500 llegadas en 30 segundos de exposición; el total es 60 segundos. No afirmar 60 segundos por método.
- Un solo hilo Python, sin importar Torch, sin modelos ni GPU. RAW falso inmutable y control trivial de identidad del payload. No hay productor-consumidor separado, cola ni transferencia de GIL entre hilos en este probe.
- Deadlines absolutos `perf_counter_ns`, separados por 20.000.000 ns; cada espera usa el tiempo restante. No desplaza la siguiente llegada por el retraso de la anterior.
- Carga normal del host más validación GPU concurrente del coordinador. CPU global medida mediante GetSystemTimes: 11,94–22,31% entre bloques. CPU del proceso del probe: 0,125 segundos en total; resolución del contador CPU gruesa, algunos bloques redondean a cero.
- No se modificó prioridad, afinidad, plan de energía, resolución global del timer ni fuentes del runtime.

Todas las cifras de la tabla provienen de `by_method.*.wake_lateness_ms` del JSON; percentiles con interpolación lineal.

| Método | Media (ms) | p95 (ms) | p99 (ms) | Máximo (ms) | >1 ms / 1.500 |
|---|---:|---:|---:|---:|---:|
| Event.wait | 8,203591 | 15,247055 | 16,061170 | 16,649300 | 1.442 |
| time.sleep | 0,339261 | 0,585190 | 0,772634 | 1,266400 | 5 |

Incluyendo la construcción del payload falso, `raw_availability_lateness_ms` da p99 16,070613 ms frente a 0,777328 ms y máximos 16,6520 frente a 1,2695 ms. El control trivial final añade microsegundos; no equivale a trabajo TTC. Ambos métodos registraron cero retrasos de más de 20 ms en estas 1.500 muestras cada uno. Ese máximo observado no es una cota futura.

Comando ejecutado:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -u artifacts/train40_runtime_resume_20261027/compute_council/scheduler_jitter_probe.py
```

La salida es exclusiva; para repetir, crear otra versión del script/salida sin sobrescribir v1.

## Relación con paced.py

`operational/train40_runtime/paced.py:119` calcula un deadline absoluto y llama a `stop.wait(delay_s)`. Después registra `acquisition_started_ns` y ejecuta la fuente RAW. La latencia de adquisición queda incluida en `observation_available_ns`; `deadline_response_ms` mide desde el instante programado, y `raw_availability_response_ms` desde la disponibilidad realmente observada. Es correcto mantener ambos relojes: los 16 ms de espera pueden consumir casi todo el presupuesto deadline→TTC antes de comenzar el trabajo real. No restar esta demora ni modificar los gates para obtener un PASS.

El reloj local `perf_counter` usa QueryPerformanceCounter con resolución declarada de 100 ns. `monotonic` informa GetTickCount64 con resolución 15,625 ms. El runner ya programa con `perf_counter_ns`, así que cambiar únicamente el reloj no corrige el primitivo de espera. La resolución declarada de un reloj tampoco garantiza cuándo Windows vuelve a ejecutar el hilo.

La distribución observada se mantuvo en los tres bloques de Event.wait y los tres de sleep, pero la carga de fondo no se controló experimentalmente. Antes de atribuir todo el efecto a una propiedad universal de Windows, repetir el runner completo con la cola, las lecturas y la inferencia reales; aquí se ha aislado únicamente la espera.

## Fuentes primarias contrastadas

El `threading.py` instalado confirma que Event.wait entra en Condition.wait y espera sobre un lock. En CPython 3.11.15, el timeout del lock Windows se convierte a milisegundos y se redondea hacia arriba. La implementación de condición usa APIs de espera de Windows, no el temporizador de alta resolución de sleep. Véanse [thread_nt.h de CPython 3.11.15](https://github.com/python/cpython/blob/v3.11.15/Python/thread_nt.h) y [condvar.h](https://github.com/python/cpython/blob/v3.11.15/Python/condvar.h). La cuantización del argumento y la planificación del sistema son efectos distintos; el redondeo de un milisegundo por sí solo no explica un máximo de dieciséis.

`time.sleep` de CPython 3.11.15 comprueba soporte de `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`, crea un timer con esa opción cuando está disponible y usa SetWaitableTimerEx. La documentación avisa de que la ejecución puede reanudarse más tarde por planificación. Fuentes: [timemodule.c 3.11.15](https://github.com/python/cpython/blob/v3.11.15/Modules/timemodule.c), [documentación de time.sleep](https://docs.python.org/3.11/library/time.html#time.sleep) y [CreateWaitableTimerExW de Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw). Una resolución de 100 ns no significa jitter de 100 ns ni plazos duros.

## Propuesta acotada para el dueño del runner

Añadir una variante opt-in de espera de alta resolución que conserve deadlines absolutos, todos los recibos, detección de drops, cola acotada y gates originales. Si se implementa con sleep, debe revisarse expresamente la cancelación: sleep no despierta al hacer `stop.set()`, mientras Event.wait sí. Se puede usar espera acotada con reconsulta de stop, documentando la cota de respuesta a cancelación; no introducir spin ni cambiar timers globales.

Comparar después las dos variantes en el productor real, y finalmente en el servicio de uno/dos objetos, con carga igual y reporte separado de jitter de llegada, RAW disponible→TTC y deadline→TTC. Una mejora del primer tramo puede permitir cumplir el presupuesto del runner sin que cambie el coste de inferencia, pero el cumplimiento completo solo lo demuestra ese ensayo final. El sondeo actual no admite ningún objetivo TTC.

## A/B posterior: helper real, hilos y cola, 60 s por estrategia

El coordinador autorizó un segundo ensayo después de que el agente productor añadiera `wait_strategy='event'|'sleep'` a `run_paced_replay`, conservando `event` como default de compatibilidad. La variante sleep vuelve a calcular el tiempo restante contra el deadline absoluto, espera como máximo 10 ms por llamada y consulta el evento de cancelación sin busy-spin. La planificación del SO puede demorar cualquier despertar: 10 ms no es una garantía dura de cancelación.

Este segundo ensayo sí usa el helper real con productor independiente, cola de capacidad dos y consumidor en el hilo principal. Ejecutó 3.000 llegadas a 50 Hz durante 60 s por estrategia, en orden Event y luego sleep. RAW falso con copia inmutable por llegada; un ID `FAKE_CONTROL_1` y resultados constantes etiquetados explícitamente como falsos. No se importó Torch ni se usó GPU. El desarrollo CPU y la admisión GPU del coordinador continuaron durante el ensayo.

Fuentes: `paced_scheduler_ab_v1/summary.json`, `event_receipts.json` y `sleep_receipts.json`; copias exactas del helper, medición y script en `paced_scheduler_ab_v1/sources`. Estado `COMPLETE_FAKE_CONTROL_ONLY`, `sources_unchanged=true`, `torch_imported=false`. SHA de paced ejecutado: `aa166bacfffbc2f5dd2ffb9e0f28bbd2fa421f739cad320f62324562626cf5d3`.

| Frontera medida | Estrategia | Media (ms) | p95 (ms) | p99 (ms) | Máximo (ms) |
|---|---|---:|---:|---:|---:|
| Deadline → RAW falso disponible | Event | 8,222847 | 15,225025 | 16,026630 | 16,8984 |
| Deadline → RAW falso disponible | sleep | 0,351052 | 0,610805 | 0,825125 | 1,5396 |
| Deadline → control final | Event | 8,346423 | 15,356815 | 16,192146 | 17,1889 |
| Deadline → control final | sleep | 0,466894 | 0,770450 | 1,043656 | 8,1876 |
| RAW falso disponible → control final | Event | 0,123576 | 0,208505 | 0,355631 | 0,5891 |
| RAW falso disponible → control final | sleep | 0,115842 | 0,194710 | 0,390228 | 7,9385 |

Ambos: 3.000/3.000 recibos con cobertura, cero drops, cero errores de fuente y máximo cero trabajos anteriores pendientes en los instantes programados. El gate conjunto del helper sigue `false` porque no son veinte objetos; todos los campos de objetivo físico/TTC siguen `false` por tratarse de controles ficticios. Un PASS de cobertura solo valida el arnés de prueba.

CPU global media: 14,995% con Event y 17,409% con sleep. CPU del proceso: 1,390625 s y 1,453125 s, respectivamente, incluyendo la agregación posterior de recibos del helper. Estos tiempos no son coste puro de espera ni de inferencia. La comparación no fue simultánea y la carga externa no se controló; el probe anterior alternado ayuda a contrastar que el efecto no depende solo del orden.

El máximo sleep de 8,1876 ms proviene principalmente de un retraso después de disponer del RAW falso (7,9385 ms), no de la liberación programada. Es evidencia de que el timer de mayor resolución mejora las llegadas pero no elimina planificación/GIL/cola. No atribuir al modelo ese outlier del control, ni deducir un techo de 1 ms para el servicio real.

Comando ejecutado:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -u artifacts/train40_runtime_resume_20261027/compute_council/paced_scheduler_ab_probe.py
```

El experimento permite continuar con la variante sleep en la medición real de uno/dos objetos manteniendo intacto el gate deadline→TTC. Aún falta ese ensayo con inferencia y RAW reales; estos resultados no constituyen una demostración de 50 Hz TTC.
