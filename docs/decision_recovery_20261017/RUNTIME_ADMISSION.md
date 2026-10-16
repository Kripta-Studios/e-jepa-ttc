# Admisión del runtime V19

Fuentes admitidas en `8b5d8ff`, posteriores a la implementación `8438122`. Se conservan 75 pruebas CPU de integración aprobadas y ocho casos nuevos del guard de procesos; el módulo modificado de entrenamiento pasó sus 24 pruebas. La revisión independiente no dejó bloqueos antes de la ejecución GPU.

## Medición técnica y equivalencia

La QA GPU terminó con PASS: 21 comprobaciones, 24 updates técnicos sintéticos y cero updates científicos. Compara pérdida nativa y pérdida capturada sobre el forward capturado de V18; la admisión histórica de captura del modelo queda enlazada por hash. También compara transferencia antigua y nueva sobre el mismo lote nativo, incluyendo gradientes, pérdida, datos y RNG, sin actualizar pesos.

| Brazo | Ratio tiempo nuevo/anterior | Reducción | Pico de asignación CUDA observado |
|---|---:|---:|---:|
| L0 | 0,909781 | 9,02 % | 787.204.608 bytes |
| LW | 0,885666 | 11,43 % | 936.781.312 bytes |

La política fijada antes de la prueba usa cinco ensayos por brazo, descarta el primero para tiempo estable, alterna el orden, calcula la mediana de los ratios emparejados y exige al menos 2% de mejora con paridad exacta. Ambos brazos admitieron la transferencia B32 completa antes de crear ocho vistas microbatch4. Las masas FP32 también se transfieren juntas; el orden de reducción se conserva.

**El tiempo medido incluye preparación de transferencia y forward/backward; excluye lectura sensorial y optimizador. No es una estimación de aceleración de toda la campaña.** Los picos son asignaciones de PyTorch durante la prueba, no memoria total ocupada por el escritorio ni una garantía sobre el pico del entrenamiento.

La continuidad se comprobó por brazo con cuatro updates continuos, 2+2 updates en la misma sesión y 2+2 restaurando el checkpoint completo. Modelo, Adam, scheduler, sampler, RNG y curvas coinciden exactamente. La sesión persistente conserva una captura del modelo y una de la pérdida.

## Datos y workers

El lector utiliza ocho hilos de decodificación, un único hilo de precarga del siguiente lote y caché LRU de 4 GiB en entrenamiento. En evaluación completa la caché de filas es cero, porque no hay reutilización en una pasada secuencial. Los ZIP FP32 históricos no se reescriben.

Dos observaciones CPU sobre un único pack P midieron 3,01x y 4,55x frente al lector serie con batches B32 idénticos. Se conservan ambos registros, incluidos los tiempos crudos. La caché del sistema operativo no estaba controlada y probablemente estaba caliente. Estas cifras describen el lector, no el entrenamiento completo.

El perfil final V18 ya ocultaba casi toda la espera sensorial mediante precarga. Añadir procesos replicaría caché y transporte entre procesos sin una ganancia demostrada; se mantiene un proceso de entrenamiento con hilos acotados.

## Incidencia conservada

El primer arranque se detuvo antes de reservar GPU o hacer updates: el guard nuevo confundía el servidor exacto `mlflow.exe mcp run` con entrenamiento y trataba como cómputo las aplicaciones gráficas enumeradas por WDDM. Se reutilizaron las excepciones y la detección de propietarios Python de V18, manteniendo los locks de campañas. El registro y log iniciales se archivaron en `training/attempts/000_guard_service_false_positive`; después se registraron las fuentes corregidas antes de ejecutar QA.

## Artefactos

- `artifacts/decision_recovery_20261017/training/REGISTRATION.json`
- `artifacts/decision_recovery_20261017/training/qa/RESULT.json`
- `artifacts/decision_recovery_20261017/training/ADMISSION.json`
- `artifacts/decision_recovery_20261017/performance/SENSOR_CPU_BENCHMARK.json`
- `artifacts/decision_recovery_20261017/performance/SENSOR_CPU_BENCHMARK_OBSERVATION_1.json`
- `artifacts/decision_recovery_20261017/PIPELINE_STATE.json`

El pipeline ejecuta L0 y LW hasta sus endpoints fijos y después evalúa P/H completos, exporta modelos batch1 y publica el informe con la decisión de réplica. Este documento acredita el runtime; no acredita todavía los resultados científicos de los entrenamientos.
