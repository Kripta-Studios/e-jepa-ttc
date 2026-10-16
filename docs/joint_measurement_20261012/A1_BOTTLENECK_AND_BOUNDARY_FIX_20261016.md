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
