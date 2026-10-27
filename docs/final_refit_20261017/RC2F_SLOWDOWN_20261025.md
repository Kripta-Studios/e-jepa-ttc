# R_C2F OOF rotación 1: ralentización del 25 de octubre de 2026

Actualización operativa del 26 de octubre: se retiró la solicitud de pantalla activa porque impedía su apagado tras bloquear con Win+L. El launcher ahora permite apagar la pantalla por defecto (`AllowScreenOff = $true`). El guard anterior salió y liberó su solicitud; se inició otro que solo solicita mantener el sistema activo, sin interrumpir la cola. Se verificó que Windows conserva 300 segundos para apagar la pantalla con corriente y suspensión del sistema desactivada. La descripción de la prueba con pantalla activa más abajo se conserva como histórico, no como configuración vigente.

**Resultado: no se ha demostrado una aceleración sostenida.** El entrenamiento sigue en ejecución, con la receta original y una protección temporal contra suspensión por inactividad. No se admite una optimización numérica nueva.

## Observaciones

El historial pasa de aproximadamente 75–80 updates/min a 21–22 alrededor de las 08:48–09:00, hora local CET. Windows registra entradas en Modern Standby a las 08:48:20 y 08:51:30, y salida a las 11:21:54. La caída persiste después de esa salida: la coincidencia temporal no prueba que Modern Standby sea su única causa.

La GPU alcanza 84–90 °C y NVIDIA informa repetidamente `SW Thermal Slowdown` activo. El contador de tiempo acumulado de esta limitación también aumenta. En la CPU, los contadores Windows `PercentProcessorPerformance` y `PercentofMaximumFrequency` están alrededor del 23 %, incluso en núcleos ocupados al 100 %. Se confirma la restricción de rendimiento observada, pero no se ha obtenido una temperatura de CPU ni RPM de ventiladores que demuestren su causa física exacta. El usuario confirma Turbo en Chuwi Center. La interfaz de control del escritorio no está disponible para leer sus sensores.

La precarga registra unos 6,7 segundos de espera efectiva por datos en más de cuatro horas. Las esperas de los decodificadores ocurren dentro del trabajador de precarga: no deben sumarse como esperas del hilo principal. La memoria disponible fue aproximadamente 11,3 GiB. No hay evidencia de que ampliar caché o comprimir más los datos resuelva esta caída.

El perfil Python de 30 segundos mantiene como puntos de coste las correspondencias locales, muestreo bilineal determinista, validaciones y resúmenes. Es muestreo de stacks Python, no una partición exhaustiva del tiempo CUDA.

## Mediciones ejecutadas

| Condición | Updates/min | s/update |
|---|---:|---:|
| Prioridad normal (dos bloques) | 20.61 | 2.912 |
| Prioridad elevada (dos bloques) | 19.61 | 3.060 |
| Solicitud de sistema activo | 19.19 | 3.127 |
| Sistema y pantalla activos, 5 min | 18.99 | 3.160 |
| Tras enfriamiento/reinicio, 5 min | 20.43 | 2.937 |

Los bloques de prioridad son una comparación ABBA de 90 segundos por bloque sobre el entrenamiento en marcha. La prioridad original quedó restaurada. Las pruebas de energía y reinicio son secuenciales y usan batches distintos; no son una comparación emparejada de kernels ni prueban equivalencia de tasas. Ninguna recupera sostenidamente el tramo rápido.

La pausa técnica guardó 11985 updates, todos durables, cero pendientes y cero perdidos. Se verificó el SHA-256 `3a297c420c3fdd9e5142ef446d3efc9d40db678774cfb2a963730102c5a91a3d` antes de reanudar. Tras enfriamiento breve, la GPU bajó a 75 °C y la CPU alcanzó un contador de rendimiento del 135 %. Los primeros 13 updates tardaron 7,657 segundos, pero esa recuperación desapareció en menos de un minuto. No se utiliza ese tramo para anunciar un speedup o extrapolar la ETA.

## Cambio operativo conservado

`operational/oof_awake.py` solicita a Windows mantener sistema y pantalla activos únicamente con cola RUNNING y alimentación externa. Se libera al pausar, finalizar, desconectar la alimentación o morir la cola; no modifica planes de energía ni la política de suspensión manual. Usa un lock exclusivo y verifica el PID y la ruta de la cola. Se integra en `scripts/resume_scientific_oof.ps1`; `-AllowScreenOff` omite la solicitud de pantalla. La pausa real verificó la retirada del permiso y la reanudación verificó su adquisición para el nuevo PID de cola.

Ocho pruebas específicas pasan. Ruff pasa y el launcher pasa el parser de PowerShell. Los 246 archivos científicos conservan sus hashes; configuración `3268e2998391b40bb72b07b6fe2bfc873f3bf61f159ca581b28575ba068d767e`. Se mantienen checkpoints cada 100 updates y márgenes de memoria de 2 GiB.

No se activan CUDA Graphs ni aproximaciones: esta intervención no modifica aritmética, pesos, orden de muestras, pérdidas, BF16, semillas ni scheduler. La [documentación de PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) recomienda perfilar y evitar sincronizaciones innecesarias; las variantes anteriores de caché de constantes y validación en prefetch ya dieron resultados negativos y no se repiten como supuestas mejoras.

La API y sus límites de suspensión manual están documentados por [Microsoft](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate). La protección de inactividad es preventiva; no elimina una limitación térmica existente.

## Evidencia

En `artifacts/final_refit_20261019_scientific_oof`: `POWER_EVENTS_20261025.json`, `PERFORMANCE_DIAG_20261025_BASELINE.json`, `RC2F_PROFILE_20261025_BEFORE.json`, `PROCESS_PRIORITY_ABBA_20261025.json`, `AWAKE_SYSTEM_PROBE_20261025.json`, `AWAKE_DISPLAY_PROBE_20261025.json`, `RESTART_PROBE_20261025.json`, `PERFORMANCE_RESTART_20261025.json`, `PERFORMANCE_SOURCE_AUDIT_20261025.json`, `AWAKE_TESTS_20261025.xml` y `PERFORMANCE_DECISION_20261025.json`.

Siguiente diagnóstico: temperatura de CPU y RPM reales de ambos ventiladores durante la limitación, para distinguir refrigeración, controlador/firmware y otras restricciones de plataforma. No se desactivan protecciones térmicas ni se declara un fallo de hardware específico sin esas mediciones.
