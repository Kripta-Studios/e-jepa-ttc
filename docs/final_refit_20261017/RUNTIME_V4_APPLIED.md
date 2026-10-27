# Optimización del entrenamiento concurrente — runtime v4

Se han reanudado R_A5 y R_C2F simultáneamente desde sus estados durables, conservando datos TRAIN40, arquitectura, pérdidas, seed 7, BF16, AdamW, scheduler, orden de muestras y endpoint de 49.932 updates por productor. El checkpoint periódico sigue siendo cada 100 updates.

Opciones registradas en la admisión:

| Opción | Activa |
|---|---|
| `cudagraph_local_transport` | No |
| `host_loss_metadata` | Sí |
| `merge_parts_by_t` | No |
| `scalar_transfer` | Sí |
| `shared_resource_snapshot` | Sí |
| `transport_reuse` | No |

El inventario de procesos se obtiene en un publicador compartido. Cada productor comprueba RAM, commit de Windows y disco localmente, y rechaza snapshots caducados, alterados, de otra sesión o de un publicador muerto. La firma, PID y hora de creación se verifican. El inventario no se enumera dentro de cada update; el publicador sigue consumiendo CPU y no elimina ese coste del equipo.

Las validaciones de entrada y cinco conteos derivados de targets se preparan en CPU durante el prefetch, con identificación y fingerprint comprobados antes de transferirlos a CUDA. Los conteos dependientes de las predicciones y las comprobaciones de finitud necesarias permanecen. Las métricas se transfieren juntas a CPU. No se afirma haber eliminado todas las sincronizaciones.

Pruebas focalizadas: **77 aprobadas**. Matriz BF16: **16 casos aprobados**, A5/C2F, épocas 2/4 y T2/T3/mixto/B32. Comparación de salidas, pérdida, gradientes, optimizer y estado; RNG y reanudación exactos. Máscara física CPU/CUDA idéntica en las 88,744 consultas TRAIN. Tolerancias fijadas antes de ejecutar: atol=2e-5, rtol=2e-4. Son pruebas técnicas acotadas, no garantía matemática sobre cualquier ejecución futura.

Los recibos finales de pruebas son UNIT_CORE_VERIFICATION.xml, UNIT_MIGRATION_VERIFICATION.xml, OPTIMIZATION_PROBE_UNIT.xml y UNIT_COMPATIBILITY_VERIFICATION.xml. El total cuenta identificadores únicos de pruebas, sin sumar duplicados entre suites. UNIT_VERIFICATION.xml conserva un fallo intermedio de la migración anterior a las correcciones; no es el recibo de admisión final. La última suite del probe sustituye sus cinco pruebas iniciales y añade una sexta.

La reutilización del transporte quedó rechazada: aunque el forward era idéntico, compartir la rama cambia el orden de acumulación de gradientes BF16. Los cuatro casos observados de A5 con objetivo completo superaron la tolerancia. El código experimental está desactivado, y el ensayo fallido se conserva.

La agrupación de particiones se evaluó por separado:

| Productor | Lote | Particiones originales → agrupadas | Resultado |
|---|---|---|---|
| R_A5 | T2 | 3 → 1 | REJECT_RNG_OR_NUMERICAL_DRIFT |
| R_A5 | MIXED | 6 → 2 | REJECT_RNG_OR_NUMERICAL_DRIFT |
| R_C2F | T2 | 3 → 1 | REJECT_RNG_OR_NUMERICAL_DRIFT |
| R_C2F | MIXED | 6 → 2 | REJECT_RNG_OR_NUMERICAL_DRIFT |

CUDA Graphs parciales: 4 firmas fijas FP32 probadas sobre el muestreador bilineal aislado, sin optimizer updates. Se verificaron salidas, gradientes, RNG y restauración de buffers. Esto no valida el entrenamiento BF16 completo. La advertencia de stream de AccumulateGrad y la ausencia de una prueba integrada del modelo completo impiden admitirlos en producción.

La migración conserva los bytes del checkpoint y nueve grupos de estado, además de curvas, identidad científica y genealogía de runtime. La revisión cubre journal adelantado, update pendiente, cortes durante save, recibos/alias atrasados y retención de versiones current+2. El origen inmutable se conserva aparte.

La primera activación v3 reveló un fallo real del restaurador con el binding previo de A5. C2F ejecutó 24 updates y se detuvo con los 767 guardados; A5 permaneció en 6.184. No se descartaron esos updates. V4 corrige únicamente la compatibilidad del origen y su genealogía. Antes del relanzamiento se restauraron ambos checkpoints reales con los productores y optimizadores efectivos, comprobando nueve grupos de estado exactos, sin optimizer updates. REAL_RESTORE_PREFLIGHT.json conserva esa evidencia. La matriz GPU anterior se reutiliza mediante igualdad AST del cálculo y de save/begin/commit; no se presenta como una matriz nueva.

El primer preflight construía los modelos en CPU: ocho estados coincidían, pero la inicialización diferida de CUDA no reproducía el orden del entrenamiento. Se rechazó ese intento y se repitió con los modelos en CUDA antes del restore, como en producción. CPU_PREFLIGHT_REJECTED.json y el destino archivado conservan el fallo; no hubo updates ni modificación de fuentes activas en ese intento.

| Productor | Update durable de migración | SHA-256 del padre |
|---|---:|---|
| R_A5 | 6184 | `b6e7fb2689b12fee2d4084d2e4b4213a087fb237ba7aa6ae3f2f666c38581497` |
| R_C2F | 767 | `c59f1785a246516ba2e6fafd2f6bf88b1f1b66f89b0d5346242cd9a9b40e39fa` |

Admisión: 120 updates técnicos. Ensayo rechazado: entre 68 y 74 updates técnicos; el límite superior contabiliza la interrupción. Ninguno modifica los checkpoints científicos.

El ensayo inicial del transporte no fijó la política de un hilo; sus tiempos no se usan como medida del runtime actual. La matriz aislada admitida sí fijó la política del entrenamiento.

Medición posterior con ambos productores activos:

| Productor | Updates medidos | Updates/min | s/update | ETA del productor a ritmo observado | Guard medio desde arranque |
|---|---:|---:|---:|---:|---:|
| R_C2F | 771 → 817 | 15.77 | 3.805 | 51.91 h | 20.42 ms |
| R_A5 | 6190 → 6258 | 23.31 | 2.574 | 31.22 h | 18.92 ms |

Ventana: 2026-10-17T16:44:15.309263+00:00 a 2026-10-17T16:47:10.241038+00:00, 175.02 s entre extremos, 36 muestras. GPU media 39.6%, memoria 7338 MiB, potencia 62.6 W, temperatura 86.5 °C. La ETA cubre estos productores; no el sistema final, OOF o export.

Referencia concurrente anterior: A5 30,17 updates/min; C2F 18,51 updates/min, en una ventana de 175,02 s. Ambos ritmos actuales son inferiores: esta medición no demuestra una aceleración global. Las ventanas no son una comparación causal controlada: arranque, carga del equipo y temperatura pueden diferir. No se extrapola la mejora aislada del guard a una aceleración proporcional de todo el entrenamiento.

Una muestra posterior de CPU confirma que retirar el inventario del camino crítico no elimina su consumo. Las cifras siguientes expresan porcentaje de un procesador lógico, no del equipo completo; los intervalos por proceso son próximos, pero no perfectamente simultáneos.

| Proceso del proyecto | CPU sobre un procesador lógico | Intervalo |
|---|---:|---:|
| R_A5 | 142.5% | 9.77 s |
| R_C2F | 114.6% | 9.79 s |
| publisher | 86.0% | 9.43 s |

También hay carga de otros procesos y servicios del equipo. No se modificaron sus prioridades ni se detuvieron. La muestra no demuestra que esa carga cause toda la diferencia de throughput. El coste del publicador sigue siendo una oportunidad de optimización; no se ha validado todavía una eliminación de ese coste. cpu_followup_rejected.json conserva una primera medición con denominador incorrecto, excluida de esta tabla.

Probe del guard con dos clientes: medianas 6.67 ms / 6.86 ms; cero inventarios completos en los clientes. Esta medida aislada tampoco equivale al ritmo de entrenamiento.

Evidencias locales: `artifacts/final_refit_20261017_runtime_optimization_v3/` y `artifacts/final_refit_20261017_runtime_v4/POST_CHANGE_180S/`. Regeneración: ejecutar `render_report.py` de la primera carpeta con el entorno Python del proyecto. No hubo push ni envío a CodaBench.

## Medición posterior: A5 solo por petición del usuario

R_C2F quedó pausado en el update 1234, con checkpoint completo y recovery_upper=0. R_A5 continuó sin reinicio; el guardado periódico permanece cada 100 updates. Durante este ensayo A5 estuvo activo y C2F pausado.

Ventana exacta: 2026-10-17T17:14:13.412039+00:00 a 2026-10-17T17:19:13.459867+00:00, 300.05 segundos. A5 avanzó de 6987 a 7125: 138 updates.

| Régimen | A5 updates/min | A5 s/update |
|---|---:|---:|
| Concurrente, referencia previa de 15 min | 27.00 | 2.222 |
| A5 solo, 5 min | 27.60 | 2.174 |

Cambio observado de ritmo: +2.2%. ETA restante de A5 a ese ritmo: 25.85 horas. No incluye otros productores ni etapas. C2F no tiene ETA de calendario mientras siga pausado.

Comparación secuencial observacional: cambian los lotes y pueden cambiar la temperatura y la carga del equipo. No es una prueba pareada sobre los mismos batches. Las muestras del ensayo solo se guardan en la carpeta SOLO_A5_5MIN correspondiente.

## Reanudación concurrente posterior

El usuario autorizó volver a ejecutar ambos productores en paralelo. Se guardó A5 en 7190 y se restauró C2F desde 1234, bajo un único lanzador y publicador. Arranque UTC: 2026-10-17T17:21:56.9154121Z. Estado registrado: RUNNING_BOTH_ADVANCING. Checkpoints cada 100 updates; recovery_upper=0.

El localizador ACTIVE_RUNTIME.json vuelve a indicar ambos fits activos. La pausa del ensayo solo queda como registro histórico; no se cambió la receta científica.
