# Dfull: investigación de cómputo en la A40

Fecha: 2026-10-31. La campaña principal continúa con su productor v2. Estas pruebas no modificaron sus pesos, configuración ni ledger.

## Método y alcance

Cada variante restaura pesos, optimizador y RNG desde el mismo checkpoint dentro de su prueba. Usa ocho actualizaciones sobre un lote fijo de frames reales; el tiempo es la mediana de las seis últimas. No acredita convergencia ni AP. El sampler de la campaña principal no se modifica. Las pausas de aislamiento conservan el proceso y su caché en memoria, con finally y un watchdog independiente que envían SIGCONT.

Pausa total medida: 66.00 s. Los intentos iniciales v1/v2 fallaron al adaptar la API del optimizador, antes de pausar el entrenamiento. Sus logs permanecen en el pod.

v3 usa checkpoint 1.024; v4/v5 usan 2.048. No se comparan pérdidas entre checkpoints ni entre batches diferentes. La divergencia de trayectorias posteriores al primer paso también aparece entre repeticiones FP32 por el backward CUDA no determinista del productor.

## Etapas

Medianas de tres pasos v3 con sincronización entre etapas; instrumentar estas fronteras añade sobrecoste. Los tiempos no son porcentajes de un perfil sin instrumentación.

| Etapa | ms |
|---|---:|
| forward | 72.17 |
| output_checks | 1.60 |
| criterion | 34.19 |
| loss_checks | 2.18 |
| backward | 116.56 |
| optimizer | 9.63 |

La traza registra 7.290 lanzamientos CUDA y 668 llamadas de sincronización de stream en un paso instrumentado. Esos contadores no significan que todo ese trabajo pueda eliminarse. Hay convoluciones, GEMM, grid_sample/backward y muchos kernels elementales. Los rangos anidados del profiler se solapan: no deben sumarse como fracciones exclusivas del tiempo GPU.

## Variantes medidas

| Variante | Referencia ms/update | Candidato ms/update | Reducción temporal |
|---|---:|---:|---:|
| Un hilo CPU, entrada residente | 249.32 | 248.02 | 0.52% |
| Comprobaciones finitas agrupadas | 249.32 | 248.31 | 0.41% |
| TF32 y comprobaciones agrupadas | 249.32 | 224.49 | 9.96% |
| Un hilo CPU con preparación concurrente | 296.30 | 296.77 | -0.16% |
| BF16 + pérdidas FP32, con preparación | 296.30 | 287.25 | 3.05% |

Batch 8: 390.79 ms/update, 20.47 frames/s. Los controles batch 4 dan 16.17–16.81 frames/s. Para las mismas exposiciones, la reducción de tiempo de cómputo sería 17.9–21.0%, antes de lectura, migración y evaluación.

El campo first_outputs_exact de v5 compara cada variante consigo misma porque el harness reinicia la referencia al cambiar de batch. **No es evidencia de paridad batch 4/8 ni TF32/FP32.** Batch 8 cambia la frecuencia de los pasos AdamW y requiere una receta y calendario de exposiciones nuevos. No basta con cambiar batch_size conservando los 70.279 updates.

## Decisiones

- Caché compacta sin pérdida y precarga: ya activas. En el run real la espera de datos ronda 2 ms por lote. La preparación concurrente eleva el cómputo de aproximadamente 248 a 296 ms en v4; el control con un hilo no lo corrige. Después de llenar la caché desaparece la preparación de primeras visitas.
- Checks agrupados / menos hilos: ganancia pequeña o nula; no justifican reiniciar y volver a llenar la caché de esta ejecución.
- TF32: ganancia moderada, con diferencias de salidas. No equivale al productor FP32 histórico y no se ha demostrado igualdad de calidad.
- BF16 directo: falla en cdist del matching. Convertir outputs a FP32 para pérdidas permite ejecutar, pero la ganancia con precarga es pequeña. La diferencia máxima observada de un elemento de output es 51,10, sin asignarle unidades físicas ni interpretarla como error de profundidad.
- Batch 8: mejor candidato de throughput probado. Es una intervención de entrenamiento, aún sin evidencia de calidad ni calendario migrado; no se aplicó a la campaña principal.
- CUDA Graphs completos: la ruta inspeccionada usa targets variables, matching SciPy en CPU y decisiones host. Necesita separar regiones capturables y comprobaciones; no es un interruptor de configuración. No se ha medido un speedup de CUDA Graphs ni de torch.compile.
- Más entrenamientos en paralelo en la misma A40 no reduce el trabajo de esta trayectoria. No se han lanzado otras semillas.

Las cuatro pasadas siguen siendo el presupuesto registrado, no una prueba de que sean necesarias u óptimas. Reducirlas sería una decisión de presupuesto/calidad distinta de acelerar las mismas actualizaciones.

Documentación oficial consultada para la versión instalada: [CUDA y Graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html), [precisión matmul](https://docs.pytorch.org/docs/2.11/generated/torch.set_float32_matmul_precision.html), [profiler](https://docs.pytorch.org/docs/2.11/profiler.html).

## Fuentes

- `compute_probe_v3.py`: `cf85de6cf6ef3ec8b0104cff08d7d1c6337d7a81ad7de17ec4172caf93f19069`
- `compute_probe_v4.py`: `336a1a49739a79515f13f80e802490f1d4a3f931780fd3bcb7b01b48f5efbed2`
- `compute_probe_v5.py`: `49cc74740aae8fe36e138803636ec120a2254c0a6abfca3862d9346bb355b55f`
- `probe_v3.json`: `39d3a205d53a0a902301fc1d13044c1abd0765642c931fc3850ef71116e82aa3`
- `probe_v4.json`: `f1eb2bba8521382e8e8852e76cb16b04a42a2ecd120d86962b9bfd57b3f8942b`
- `probe_v5.json`: `b35f6239de320887d65c7b4b658391a8b2a6ac422e5eaef5333c4d90246069e0`
- `profiler_cpu.txt`: `333be6f6d3c99fcc18bb385b9e8885c9e61d50c792528ad1651bf703d279893f`
- `profiler_cuda.txt`: `775dbcdf16fe2d6c8cff94deeb878f7d2068f6f62a3613f5ba3d6d95d46a50df`

Regenerar con `python scripts/report_dfull_compute_probe.py --evidence docs/detection_sota_20261029/evidence/dfull_compute_20261031 --output docs/detection_sota_20261029/DFULL_COMPUTE_INVESTIGATION_20261031.md`.
