# Reparación de la mediana determinista de fusión — 2026-10-11

F_TRUE y F_ZERO fallaron a las 21:25–21:26 CEST, antes del primer update:
`median CUDA with indices output does not have a deterministic implementation`.
El origen está en `src/e_jepa_ttc/rgb_port/fusion.py:84`. La cola terminó con
`TERMINAL_WITH_BLOCKERS`; su campo `supervisor_status=RUNNING` quedó obsoleto.
Los diez entrenamientos anteriores ya estaban completos.

## Corrección y alcance

`operational/rgb_port_fusion_fix/fusion.py::deterministic_forward` conserva el
forward original y sustituye exclusivamente la mediana de tres fases finitas por
`sort(dim=-1, stable=True).values[:, 1]`. Los parámetros, state_dict, pérdidas,
datos, semilla, presupuesto y FP32 de las cabezas se conservan. La geometría BF16
de los productores no cambia. No se desactiva el determinismo en entrenamiento.

Las fases de los expertos proceden de una caché congelada: no reciben gradientes.
Si en el futuro se entrenan expertos conjuntamente, los empates pueden seleccionar
otro subgradiente hacia esas entradas; esta revisión no admite ese uso sin validación.

La implementación original permanece congelada. El paquete adicional permite
conservar sus hashes y los checkpoints, sin reescribir manifests históricos.
`contracts.py` comprueba la implementación original, el paquete nuevo, las pruebas,
la evidencia QA y la revisión anterior, tanto en disco como en el commit actual.
`FUSION_DETERMINISTIC_FREEZE.json` registra esa admisión.

`runtime.py` aplica el forward corregido antes de ejecutar la CLI original.
`queue.py` dirige por ese runtime F_TRUE/F_ZERO y las rutas predict_heads, transfer,
profile y cost_routes. Los demás entrenamientos y la reparación anterior de
serialización de observaciones se conservan. Cada fusión guarda un
`fits/<fit>/FUSION_RUNTIME.json` que vincula su admisión histórica al nuevo freeze;
el bundle incluye estos recibos y el código adicional.

Para reanudar esta campaña debe utilizarse:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path + ';.'
python -B -m operational.rgb_port_fusion_fix.queue resume --run artifacts/rgb_port_20261008
```

Invocar directamente la CLI histórica omite este runtime adicional.

## Validación

[EVIDENCE.json](EVIDENCE.json) recoge 45 tests aprobados, Ruff aprobado y Pyright
sin errores ni advertencias. Las regresiones incluyen comparación CPU/CUDA de
salidas, pérdidas y gradientes, permutaciones, empates, extremos, rutas de la cola
y recuperación selectiva. También se ejecutaron las suites existentes de firmas,
características/fusión, entrenamiento de cabezas y predicción.

`audit.py` reproduce la comparación en GPU sobre 128 filas de la caché H real,
seleccionadas con semilla 7. Carga ambos checkpoints y perturba en memoria los pesos
de salida para comprobar gradientes también en las torres. Para ambas fusiones,
salidas, pérdida y gradientes de los 32 tensores de parámetros fueron exactamente
iguales, sin tolerancia. El forward histórico se ejecuta con el flag determinista
desactivado únicamente como referencia de la prueba; la corrección se comprueba
con determinismo estricto. No hay pasos de optimizador en QA.

Los 60 archivos protegidos de los diez fits completos conservaron sus hashes.
La evidencia completa está en `E:/EJEPA_results/fusion_deterministic_20261011`.

```powershell
python -B -m operational.rgb_port_fusion_fix.audit --run artifacts/rgb_port_20261008 --output E:/EJEPA_results/fusion_deterministic_20261011/CUDA_PARITY_REPLAY.json
python -B -m pytest tests/test_rgb_port_fusion_fix.py tests/test_rgb_port_signature_fix.py tests/test_rgb_port_features_fusion.py tests/test_rgb_port_head_pipeline.py tests/test_rgb_port_predict_heads.py
```

## Recuperación

`recovery.py` exige que ambas fusiones hayan fallado con cero updates y que ninguna
tarea figure activa. Reactiva exclusivamente esas dos tareas y sus dependencias
omitidas, conservando intentos, logs, tareas completas y checkpoints de update cero.
Antes de aplicarlo se verifica además que no queden procesos de campaña vivos.
Se conservan el estado y la admisión Git anteriores en el directorio de evidencia.

Esto resuelve un bloqueo de ejecución. No constituye una evaluación de precisión
ni demuestra superioridad frente a GarlTTC; esas mediciones siguen en la cola.
