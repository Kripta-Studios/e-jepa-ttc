La campaña de entrenamiento está cerrada (24 endpoints / 60.000 updates).

El perfilado P3 conserva 1.114/1.728 fragmentos; el siguiente ID es `q32_application_hdf5_cold_FULL_C0`. Requiere recuperar la misma fuente `E:\eAP_dataset\data\train`, incluida `Z30i8WuL2N/events.h5`. No sustituir las fuentes ni alterar el protocolo.

Desde el worktree de esta rama, con la unidad E: recuperada y los límites de recursos disponibles:

```powershell
$env:PYTHONUTF8 = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& ..\e-jepa-ttc\.venv\Scripts\python.exe -B -m operational.simplex_t_shared_route.run run
```

El módulo selecciona únicamente la cola fijada en `artifacts/simplex_t/shared_gpu_route_20261004/PROTOCOL.json`, verifica su propietario y conserva los fragmentos existentes. No ejecuta optimizadores. Las revisiones manuales serán aproximadamente cada 40 minutos o al terminar/fallar el proceso; los guardrails y persistencia automáticos siguen activos.
