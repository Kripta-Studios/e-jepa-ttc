# A1: cuello de botella y reparación de la evidencia de frontera — 2026-10-16

## Perfil real de FULL

Muestreo pasivo de 40 segundos, 50 Hz, PID 6044, sin inyectar operaciones CUDA ni updates. Evidencia local: `artifacts/joint_measurement_20261012/runtime/A1_FULL_profile_20261016/`. Los últimos 100 updates observados cerca del update 700 tardaron 0,5240 s/update:

| Intervalo del host | Tiempo/update | Fracción |
|---|---:|---:|
| Espera de entrada | 1,11 ms | 0,21 % |
| Preparación/transferencia | 70,24 ms | 13,40 % |
| Forward/backward, pérdida y sincronizaciones | 389,19 ms | 74,27 % |
| Optimizador y journal durable | 63,47 ms | 12,11 % |

Estos intervalos incluyen esperas y posible solapamiento; no equivalen a tiempos de kernels. El muestreo externo atribuye el 70,85 % del hilo principal al bloque forward/loss/backward, consistente con el perfil interno. GPU utilization media 19,15 % no demuestra por sí sola falta de trabajo CUDA.

Se verificaron 8 lectores paralelos, caché sensorial de 7.773.719.820 bytes, 211 packs abiertos/verificados y una captura del modelo y otra de la pérdida. LAST-BLOCK conservó las capturas entre cuatro scopes. FULL había ejecutado un scope. No se reconstruyen los procesos cada 15 minutos. La caché mantiene su límite de 8 GiB y la reserva de memoria del host.

La alimentación ya no domina: aumentar compresión no es la prioridad. Quedan transferencias/preparación, sincronizaciones de validación y autograd, y costes de optimizador/journal. No se han eliminado comprobaciones de dominio, finitud ni durabilidad. El ahorro preparado del 40,7 % de la prueba de grafo de pérdida no debe presentarse como ahorro end-to-end.

## Incidente observado a las 22:00

LAST-BLOCK terminó 12.000 updates. FULL guardó 1.425 completos, sin updates pendientes ni recuperación, y falló al validar la frontera del primer scope. La reproducción CPU de `verify_change` falla con FileNotFoundError: el runtime persistente no escribía WEIGHT_CHANGE.json. En LAST-BLOCK existía un testigo antiguo del update 1.086, por lo que ese defecto había quedado oculto.

SHA-256 del checkpoint FULL preservado: `9ec7bf549bb6202b653a7210d0648f30f6aa55f2bb6d8742cb1068c0cdcff359`.

La reparación está en un wrapper independiente, `adaptation_observed_run.py`: compara en CPU el checkpoint durable completo con el padre registrado, comprueba claves/formas/tipos/finitud/identidad/cursor/hash, escribe un testigo actualizado y aplica la política original de pesos congelados y bloque aprendido. Conserva los testigos anteriores. También captura explícitamente stdout/stderr de los hijos. Los módulos científicos y de entrenamiento admitidos permanecen byte a byte iguales; no se cambia pérdida, muestreo, optimizador, scheduler, pesos o RNG.

Validación: seis pruebas CPU del cálculo de diferencias; comprobación real sin updates de LAST-BLOCK 12.000 y FULL 1.425 contra su padre. Ambas cumplen la política. El checkpoint original y el error del coordinador se conservan en `runtime/A1_boundary_evidence_v1/`.

Reanudación: `./scripts/Resume-JointMeasurement.ps1 -Stage A1Training`. Resta verificar en producción la continuidad de estado y el siguiente cierre de scope, antes de declarar reparada su ejecución sostenida.

## Trabajo que sigue fuera de los grafos

Cada update conserva ocho microbatches. `geometry_capture.checked_phase` materializa en CPU la validez del modelo y `captured_phase_loss.checked` la de la pérdida: hay dos barreras explícitas por microbatch, además de la transferencia final de diagnósticos. La pérdida sí está capturada; sus guardas de admisión y su agregación Python siguen fuera. Las masas se transfieren por microbatch y el batch completo se prepara en ocho llamadas `.to(cuda)`. El optimizador, clipping, scheduler y journal también siguen fuera de los grafos.

Son puntos concretos para un siguiente benchmark, no ahorros demostrados. El nuevo stderr conserva un aviso de PyTorch sobre streams de AccumulateGrad: puede introducir sincronización, pero este muestreo no cuantifica su coste ni demuestra que sea la causa principal. No se suprime el aviso.

La prueba de continuidad posterior a la reparación pasó en FULL 1.600: prefijos de curva/tiempos intactos, muestras y estado del sampler correctos, Adam y scheduler avanzados exactamente 175 updates y recuperación cero. Evidencia: `runtime/A1_boundary_evidence_v1/RESUME_CONTINUITY.json`.

## Frontera reparada en producción

A las 22:22:37 FULL cerró un scope con 3.616 updates, publicó el testigo actualizado y siguió en el mismo proceso. En el update 3.800 constaban dos scopes y exactamente una captura de modelo y una de pérdida. Recuperación cero. La reparación queda comprobada más allá del punto que provocó la parada.

La nueva ventana de 100 updates arroja 0,3951 s/update: espera 0,789 ms (0,20 %), preparación/transferencia 39,476 ms (9,99 %), forward/backward y sincronización 294,471 ms (74,53 %), optimizador/journal 60,348 ms (15,27 %). Es otra observación de producción, no una comparación causal controlada con el primer perfil.

Las evidencias pequeñas publicables, incluidos los stacks originales del muestreo, están en [evidence_A1_boundary_20261016](evidence_A1_boundary_20261016/MANIFEST.json). El manifiesto vincula tamaño y SHA-256. El observador quedó armado para las 23:00, PID del coordinador 18080.
