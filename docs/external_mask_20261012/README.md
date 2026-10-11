# Dev32 y costes: contrato nativo de la máscara del productor

El 11 de octubre, después de completar los doce entrenamientos y la evaluación V,
Dev32 y PROFILE_ROUTES fallaron con `producer known mask differs from support clock`.
El productor emite `known[B]` para la predicción TTC actual, mientras que el soporte
conserva su historia `support[B,T]`. El validador histórico exigía ambas formas iguales.

La corrección ya admitida para extracciones H/V estaba en
`operational/rgb_port_streaming/contracts.py::validate_observation`; esas dos rutas
externas no la instalaban. El nuevo paquete `operational/rgb_port_external_fix`
propaga exactamente ese validador a transfer y cost_routes, conservando también
el runtime de mediana determinista de fusión. No modifica máscaras ni datos:
la expansión solo existe en una copia para validar. Se conservan los controles
de forma, finitud, soporte, tipo booleano y relojes. El código histórico permanece
intacto y las fuentes adicionales se vinculan con `EXTERNAL_MASK_FREEZE.json`.

## Evidencia

`EVIDENCE.json` registra 23 pruebas aprobadas, Ruff aprobado y Pyright sin errores
ni advertencias. Una consulta real de Dev32, sin etiquetas, reproduce el error
original en eventos y RGB. Con la corrección, los vectores phase17 y las fases de
los tres expertos coinciden exactamente con el adaptador NativeObservation que
ya utiliza la ruta nativa. Se cargan los modelos finales; no hay updates de
optimizador. Los 1.412 archivos protegidos de fits, predicciones V y resultados V
conservan sus hashes durante la intervención.

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path + ';.'
python -B -m pytest tests/test_rgb_port_external_fix.py tests/test_rgb_port_cost_routes.py tests/test_rgb_port_transfer.py
python -B -m operational.rgb_port_external_fix.audit --run artifacts/rgb_port_20261008 --manifest ../e-jepa-ttc-v12-efficient-context/artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json --output E:/EJEPA_results/external_mask_20261012/REAL_PARITY_REPLAY.json
python -B -m operational.rgb_port_external_fix.queue resume --run artifacts/rgb_port_20261008
```

## Recuperación y medición

La recuperación reactiva DEV32_PREDICT, DEV32_SCORE y PROFILE_ROUTES. Regenera REPORT
al acabar; el informe anterior se conserva en el directorio de evidencia.
Los modelos completos, V_RESULTS, las predicciones V y la calibración no cambian.
La cola ejecuta costes antes de la transferencia CPU larga; las dependencias,
consultas, dispositivo, repeticiones y definiciones de cada tarea se conservan.
Los logs fallidos permanecen y los intentos nuevos usan el wrapper corregido.

V14 estaba activo en CPU durante esta intervención y no se modificó. Los tiempos
medidos bajo esa carga concurrente no deben presentarse como una medición aislada
del sistema; el contexto de procesos se registra junto a la recuperación.
Los tiempos de cabezas sobre caché siguen separados de las rutas completas.

Evidencia local: `E:/EJEPA_results/external_mask_20261012/`.

## FCWD: límite de datos conservado

Se inspeccionaron los archivos de `E:/FCWD_dataset/inputs/calibration` y los grupos
y metadatos de los tres HDF5, sin leer las etiquetas selladas. Los archivos contienen
intrínsecos en objetos MATLAB cameraParameters y una transformación LiDAR→RGB.
No se encontró una transformación RGB→eventos utilizable. La inspección está en
`FCWD_CALIBRATION_INSPECTION.json`, incluida en EVIDENCE.json.

FCWD RGB/fusión continúa BLOCKED_EXTERNAL. No se identifica una transformación de
otro par de sensores como sustituto, ni se desactiva este requisito para obtener
una cifra. Dev32, costes y validación V pueden progresar independientemente.

Esta corrección resuelve un contrato de ejecución; no modifica el protocolo ni
supone una mejora de precisión o una demostración de SOTA.
