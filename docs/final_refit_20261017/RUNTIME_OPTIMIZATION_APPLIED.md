# Optimización de runtime aplicada al refit final

Estado documental: **optimización aplicada, entrenamiento reanudado y medición posterior completada**.

Este documento describe la modificación de runtime aplicada para continuar `R_A5`.
La migración y la continuación se verifican mediante los recibos enlazados; las pruebas
técnicas no se contabilizan como entrenamiento científico. La referencia anterior al cambio es
[`RUNTIME_15MIN_REPORT.md`](RUNTIME_15MIN_REPORT.md).

## Cambios admitidos

Se conservan arquitectura, pérdidas, BF16, batch efectivo 32, semilla, población,
orden del sampler y checkpoint cada 100 updates. Los cambios admitidos son:

1. **Un hilo intra-op de PyTorch.** El runtime fija `torch_intraop_threads=1` y
   conserva `torch_interop_threads=16`. La admisión compara 16 frente a 1 hilo en
   `R_A5` y `R_C2F`, épocas 2 y 4, con igualdad exacta de índices, RNG de CPU y CUDA,
   pérdida, componentes, clipping, gradientes, modelo y optimizer.
2. **Monitor de recursos con inventario cacheado.** RAM física, margen de memoria
   comprometida y disco continúan comprobándose en cada update. El inventario completo
   se ejecuta cada 25 updates. Las líneas de comando se cachean por `(PID, create_time)`;
   la caché interna de `psutil.process_iter()` se invalida antes de cada inventario para
   detectar reutilización de PID. El RSS de procesos del proyecto se vuelve a leer en
   cada inventario. Un proceso protegido queda registrado y su RSS se incorpora como
   cota superior conservadora; un fallo sin cota deniega la continuación.
3. **Reserva de recuperación corregida.** El límite global mantiene el valor histórico
   `11.092`, pero ahora contabiliza `recovery_upper + max(0, completed_updates -
   durable_updates) + pending_update_upper`, además del siguiente update prospectivo.
   Los journals se leen mediante la ruta compartida con reintentos de Windows.

La configuración de hilos sigue la API pública
[`torch.set_num_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html).
La identidad de proceso y la invalidación de caché siguen el contrato documentado de
[`psutil.process_iter()`](https://psutil.readthedocs.io/en/latest/#psutil.process_iter).

## Evidencia técnica

### Admisión eager completa

[`RGB_EAGER_ADMISSION_V2.json`](../../artifacts/final_refit_20261017/runtime_optimization_20261017/RGB_EAGER_ADMISSION_V2.json)
registra `status=PASS`, `runtime_admission=true`, backend
`eager_bf16_no_compile_no_graphs`, 12 casos, 96 optimizer steps técnicos, dos pases
forward-only y cero updates científicos. La política de control, fuentes ejecutadas,
manifiesto docente y descriptor de datos quedan ligados por hash en el recibo.

### Admisión de un hilo

[`RUNTIME_THREAD_ADMISSION.json`](../../artifacts/final_refit_20261017/runtime_optimization_20261017/RUNTIME_THREAD_ADMISSION.json)
registra cuatro casos, 16 optimizer steps técnicos y cero updates científicos. Cada
caso usa un update medido después de un warmup descartable por brazo. La reducción de
tiempo de pared observada fue:

| Caso | 16 hilos | 1 hilo | Reducción |
|---|---:|---:|---:|
| R_A5, época 2 | 288,989 ms | 272,406 ms | 5,738 % |
| R_A5, época 4 | 294,007 ms | 281,801 ms | 4,152 % |
| R_C2F, época 2 | 435,140 ms | 418,375 ms | 3,853 % |
| R_C2F, época 4 | 437,068 ms | 416,435 ms | 4,721 % |

Es una mejora técnica modesta: 3,853–5,738 % en tiempo de pared y 4,007–6,088 % en
updates/s. El 7,46 % del informe anterior corresponde al tiempo muestreado dentro de
las comprobaciones de recursos, no a la ganancia ya demostrada por fijar un hilo. Estas
pruebas acotadas no sustituyen una ventana real del entrenamiento reanudado.

### Monitor de recursos

[`RESOURCE_MONITOR_AB_V2.json`](../../artifacts/final_refit_20261017/RESOURCE_MONITOR_AB_V2.json)
compara cinco inventarios completos sobre el mismo host:

- misma población de PIDs en 5/5 pares;
- diferencia de RSS igual a cero para cada proceso del proyecto en 5/5 pares;
- todos los guards antiguos y nuevos permitieron continuar;
- mediana antigua: 0,444672 s;
- mediana cacheada: 0,382346 s, razón 1,163×;
- guard barato por update: mediana 0,00015995 s en diez llamadas, todas permitidas.

El primer inventario cacheado es más lento porque construye la caché. La comparación
no demuestra todavía cuánto cambia el tiempo total de un update científico.

### Contratos de continuación y migración

[`TECHNICAL_TEST_REPORT.json`](../../artifacts/final_refit_20261017/runtime_optimization_20261017/MIGRATION_STAGING/TECHNICAL_TEST_REPORT.json)
registra 41 tests, cero fallos, cero errores y cero omitidos; Ruff y Pyright también
pasan. El alcance cubre productores, eager probe, continuación, migración y monitor de
recursos, con cero updates científicos.

El origen exacto de la migración es el checkpoint durable **1407** de `R_A5`:

- checkpoint: `checkpoint_001407.pt`;
- SHA-256 del checkpoint completo: `f290d26ca8df094c595e6efc72782416d2a41feca126ac262bf472efdd2b0ca6`;
- `completed_updates=1407`, `durable_updates=1407` y `pending_update_upper=0`;
- `recovery_upper=0`.

La [migración ejecutada](../../artifacts/final_refit_20261017_runtime_v2/MIGRATION_EXECUTION_REPORT.json)
conservó exactamente los nueve estados comprobados: modelo, optimizador, scheduler,
cursor, generador del sampler y RNG de Python, NumPy, CPU y CUDA. No ejecutó updates.
El entrenamiento se reanudó desde 1407 en `artifacts/final_refit_20261017_runtime_v2`,
con el [binding de ejecución](../../artifacts/final_refit_20261017_runtime_v2/fits/R_A5/RUNTIME_RESUME_BINDING.json)
`e2fabd40c2c426ea2d42ab0f1916803621e962d88975ad0f567105bcf1b40e32`.
El primer checkpoint periódico nuevo se verificó en **1500**, con ese mismo binding,
identidad científica original y cero reserva de recuperación consumida.

La [aclaración del alcance de la admisión](../../artifacts/final_refit_20261017_runtime_v2/MIGRATION_SCOPE_ERRATUM.json)
distingue los casos numéricos de batch pequeño de las comprobaciones del adaptador B32.
El arranque inicial fue secuencial. Posteriormente, por instrucción expresa del usuario,
se guardó R_A5 en 4.990 y se cerró esa cola. El script
[RESUME_RUNTIME_V2.ps1](../../artifacts/final_refit_20261017/runtime_optimization_20261017/RESUME_RUNTIME_V2.ps1)
ahora delega en [RESUME_CONCURRENT_RGB.ps1](../../artifacts/final_refit_20261017/runtime_optimization_20261017/RESUME_CONCURRENT_RGB.ps1),
que lanza ambos productores con protección contra duplicados por fit. Se verificó
avance científico simultáneo de ambos; véase el
[recibo de concurrencia](../../artifacts/final_refit_20261017_runtime_v2/CONCURRENT_LAUNCH.json).
Las mediciones siguientes preceden a ese cambio de planificación y no son una ETA
del entrenamiento concurrente. No se ejecutaron comparativas nuevas al lanzarlo.

## Perfil CUDA y decisiones no adoptadas

[`RUNTIME_THREAD_PROBE.json`](../../artifacts/final_refit_20261017/runtime_optimization_20261017/RUNTIME_THREAD_PROBE.json)
sitúa entre los principales costes CUDA a `aten::_index_put_impl_` y al kernel
`indexing_backward_kernel_stride_1`: 135.624,454 µs y 132.111,549 µs acumulados,
respectivamente, con 258 y 240 invocaciones en la captura. Esto identifica el backward
de indexación como un frente de trabajo; no demuestra que pueda sustituirse sin cambiar
gradientes, orden de acumulación o determinismo.
Son niveles anidados de la captura: sus tiempos no deben sumarse.

La precarga anterior sólo acumuló 0,280 s de espera en 500 lotes medidos. Ese resultado
no justifica aumentar RAM, workers o compresión para esta etapa: las cachés y la
decodificación paralela existentes ya mantienen alimentado el entrenamiento. La
optimización aplicada se centra en los hilos CPU y en el inventario de procesos.

No se ha admitido ningún kernel alternativo, CUDA Graph, `torch.compile`, cambio de
optimizer, eliminación de guardas ni reducción de frecuencia de checkpoint. El backend
permanece eager BF16. Cualquier intervención posterior sobre indexación requiere un A/B
con paridad de pérdida, gradientes, optimizer, RNG y estado reanudable antes de tocar el
entrenamiento científico.

## Medición real posterior

El [monitor de cinco minutos](../../artifacts/final_refit_20261017_runtime_v2/RUNTIME_POST_300S/summary.json)
terminó sin detener el entrenamiento. Se tomaron 60 muestras cada cinco segundos durante
300,032 s. Entre la primera y la última muestra transcurrieron 295,068 s y el journal
avanzó de 1.532 a 1.870: **338 updates**, 1,1455 updates/s o **0,873 s/update**.
La media anterior era 1,6455 s/update, pero este cociente de 1,885× no es una estimación
causal del efecto de las optimizaciones: cambian los lotes, el runtime, el perfilado y
las condiciones del equipo. La causa de la variación temporal no quedó aislada.

La tendencia impide presentar esa media como una velocidad sostenida:

| Bloque de la ventana | Updates/s | Segundos/update |
|---|---:|---:|
| Primer minuto | 2,036 | 0,491 |
| Segundo minuto | 1,092 | 0,916 |
| Tercer minuto | 0,891 | 1,122 |
| Cuarto minuto | 0,928 | 1,078 |
| Quinto minuto | 0,891 | 1,123 |

Cada bloque utiliza el intervalo entre sus muestras disponibles, aproximadamente 55 s.
Los últimos dos minutos observados dieron **0,913 updates/s, 1,095 s/update**. La
[comprobación A/B/A posterior](../../artifacts/final_refit_20261017_runtime_v2/RUNTIME_POST_300S/PYSPY_ABA_SUMMARY.json)
dio 0,586 sin profiler, 0,655 con profiler y 0,448 updates/s al retirarlo, en ventanas
de 30 s y sólo 13–19 updates. El resultado es **inconcluso**: no demuestra ralentización
por py-spy ni ausencia de sobrecoste. Ambas capturas utilizaron `--nonblocking`, que,
según la [documentación de py-spy](https://github.com/benfred/py-spy#how-can-i-avoid-pausing-the-python-program),
evita pausar el proceso al leer sus pilas. No se atribuye al profiler la caída observada.
No se calcula una ETA del refit completo con estos intervalos.

Consumo de la ventana, regenerable desde [analysis.json](../../artifacts/final_refit_20261017_runtime_v2/RUNTIME_POST_300S/analysis.json):

| Medida | Resultado |
|---|---:|
| CPU del proceso, media | 1,184 núcleos equivalentes |
| RAM del proceso, pico | 5,896 GiB |
| GPU, utilización media / p95 | 21,67 % / 72,35 % |
| VRAM del dispositivo, pico | 3,984 GiB |
| Potencia GPU, media | 56,27 W |
| Temperatura GPU, media / máxima | 80,37 / 86 °C |
| Espera de precarga, delta de 300 lotes | 0,0037 s |

Los contadores GPU son del dispositivo completo. La recogida de telemetría ocupó
12,746 s dentro de su propio proceso; ese 4,25 % no es una medida de ralentización del
entrenamiento. Desde la migración hasta el checkpoint 2.000, el monitor de recursos
acumuló 30,968 s en 593 updates: **52,2 ms/update**, con 25 inventarios completos.
Las 569 comprobaciones ligeras promediaron 1,659 ms. Estos costes reales, mayores que
los del microbenchmark, también deben conservarse al presupuestar.

La [verificación del runtime activo](../../artifacts/final_refit_20261017_runtime_v2/LIVE_RUNTIME_VERIFICATION.json)
comprobó los hashes de las 28 fuentes ligadas, el checkpoint 1.900 y la cobertura exacta
de las curvas 1–1.900. Después se observó el checkpoint 2.000 con cero recuperación.
El proceso se dejó entrenando, sin profiler y con checkpoint cada 100. La optimización
técnica está admitida; **una ganancia sostenida de extremo a extremo sigue sin demostrarse**.

Recibo consolidado: [conclusion.json](../../artifacts/final_refit_20261017_runtime_v2/RUNTIME_POST_300S/conclusion.json),
SHA-256 `61cffdaa85a9eae3fedefcbd815d44e6a9421707aaf7580caa06c16f65ed002d`.
