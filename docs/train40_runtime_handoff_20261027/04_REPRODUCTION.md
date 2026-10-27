# Reproducción y entorno

## Entorno que realmente funcionó

Windows, RTX 5070 Ti Laptop (12 GB), driver 591.86, Python 3.11.15, PyTorch 2.11.0+cu128. La inferencia se midió en FP32, matmul/cuDNN TF32 desactivados, cuDNN benchmark desactivado, dos threads Torch y semilla 7. El escritorio comparte GPU; no se atribuye cada variación al modelo.

El `.venv` de este checkout falla al importar `site`: una ruta editable UTF-8 con `Á` se interpreta como cp1252. `-X utf8` y `PYTHONUTF8=1` no resolvieron ese `.pth`. **No se modificó el entorno recibido.** Se usó el intérprete existente del checkout hermano:

```powershell
$runtimePython = '../e-jepa-ttc/.venv/Scripts/python.exe'
$runtimeRuff = '../e-jepa-ttc/.venv/Scripts/ruff.exe'
$runtimePyright = '../e-jepa-ttc/.venv/Scripts/pyright.exe'
```

Los scripts principales insertan este repo y `src` en `sys.path`. Para scripts inline sin ese bootstrap puede ser necesario `$env:PYTHONPATH='src;.'`. Pyright requiere `--venvpath ../e-jepa-ttc`; pasar solo `--pythonpath` deja que la configuración del repo seleccione el `.venv` incorrecto.

## Tests y controles

```powershell
$runtimeTests = Get-ChildItem tests/unit/test_train40_runtime*.py | ForEach-Object FullName
& $runtimePython -m pytest -o 'addopts=' -q @runtimeTests
& $runtimeRuff check operational/train40_runtime scripts/benchmark_train40_runtime.py scripts/benchmark_train40_integral.py scripts/profile_train40_runtime.py scripts/preserve_train40_runtime_handoff.py
& $runtimePyright --venvpath ../e-jepa-ttc operational/train40_runtime scripts/benchmark_train40_runtime.py scripts/benchmark_train40_integral.py scripts/profile_train40_runtime.py scripts/preserve_train40_runtime_handoff.py
```

Los dos xfail estrictos de transporte son resultados negativos intencionales de la opción experimental; no son una admisión. Los tests CUDA son opt-in y **no deben ejecutarse mientras otro agente usa la GPU**:

```powershell
$env:EJEPA_TRAIN40_CUDA_TESTS='1'
& $runtimePython -m pytest tests/unit/test_train40_runtime_context.py -m cuda_fp32
Remove-Item Env:EJEPA_TRAIN40_CUDA_TESTS
$env:RUN_TRAIN40_CUDA_TESTS='1'
& $runtimePython -m pytest tests/unit/test_train40_runtime_integral.py
Remove-Item Env:RUN_TRAIN40_CUDA_TESTS
```

Estos comandos futuros son instrucciones de reproducción, no pruebas nuevas ejecutadas después de la pausa. Los resultados finales del cierre se conservan en `evidence/closeout/`.

## Validación de identidad sin reentrenar

```powershell
& $runtimePython scripts/final_train40_transfer.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --preflight
```

La comprobación oficial carga pesos y valida la cabeza sobre estados guardados. Su alcance no es todo el sensor→TTC. No modificar `verify_frozen_runtime` para saltarse un fallo de hash.

## Ensayos existentes y siguientes comandos

Elegir siempre un output **nuevo**. Los directorios del piloto original ya existen y no deben sobrescribirse.

```powershell
& $runtimePython scripts/profile_train40_runtime.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json --output artifacts/train40_runtime_resume/oracle_profile

& $runtimePython scripts/benchmark_train40_runtime.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json --reference artifacts/final_train40_transfer_20261027/final_train40/own --output artifacts/train40_runtime_resume/offline --modes oracle full-graphs reuse-graphs --indices 0 200 553 --repeats 2

& $runtimePython scripts/benchmark_train40_integral.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json --reference artifacts/final_train40_transfer_20261027/final_train40/own --output artifacts/train40_runtime_resume/integral_all554 --parallel
```

El último comando **no se ejecutó sobre las 554 consultas en esta sesión**. Al omitir `--indices` usa todas; verifica también cada tensor event contra la referencia. Puede tardar minutos, consumir varios GiB de VRAM y reconstruye historia por cada fase irregular. No describe un stream continuo. `--skip-input-check` existe para diagnóstico, pero no sustituye la primera admisión completa.

`replay_inputs.generate_sensor_clock_replay` genera nuevas anclas de 20 ms sin escribir archivos ni leer TTC. El caller debe guardar el manifiesto y sus exclusiones. La disponibilidad de anotaciones se supone como timestamp de sensor más `annotation_delay_us`; no representa un detector real. Todavía no existe un CLI de servicio continuo multiobjeto completo.

## Compilador aislado

El probe instaló únicamente `triton-windows==3.6.0.post26` bajo `artifacts/train40_runtime_20261027/compiler_probe/deps`. PyTorch y el lockfile no cambiaron. Wheel cp311:

```text
triton_windows-3.6.0.post26-cp311-cp311-win_amd64.whl
SHA256 369a47ed3ca25f6d405387a058b8fbc62e6ff4592bbf87b0c5f6de372ab68931
```

El wheel y `deps/` son locales y no se publican. Metadatos primarios: [PyPI, versión fijada](https://pypi.org/pypi/triton-windows/3.6.0.post26/json), [maintainer Triton Windows](https://github.com/triton-lang/triton-windows/blob/readme/README.md).

```powershell
uv pip install --python ../e-jepa-ttc/.venv/Scripts/python.exe --target artifacts/train40_runtime_20261027/compiler_probe/deps --no-deps artifacts/train40_runtime_20261027/compiler_probe/triton_windows-3.6.0.post26-cp311-cp311-win_amd64.whl
& $runtimePython artifacts/train40_runtime_20261027/compiler_probe/probe_compile.py --batches 8 --radii 1 --repeats 30 --output artifacts/train40_runtime_20261027/compiler_probe/compile_radius1_b8_next.json
```

El script exploratorio preservado configura **solo su proceso**:

- ruta corta 8.3 ASCII para caches, evitando `Á` en el `getenv` C de Triton;
- `CC` al TinyCC del wheel: `deps/triton/runtime/tcc/tcc.exe`;
- `CUDA_PATH` a `deps/triton/backends/nvidia`, bundle CUDA 12.8;
- `sys.path` al directorio `deps` y caches locales;
- `torch.compile(fullgraph=True,dynamic=False)`, `use_fast_math=False`, `force_same_precision=True`, `triton.cudagraphs=False`.

Un primer `uv --target` seleccionó Python 3.13 por defecto y rechazó el wheel cp311. Otro intento falló por Unicode en cache; otro por autodetección de GCC externo/CUDA 12.4 y prefijo UNC. Se conservan los tres recibos, incluido el satisfactorio `compile_radius1_b8_toolchain.json`. CUDA 12.4 no corresponde a esta GPU Blackwell. No instalar herramientas globalmente para resolverlo sin necesidad.

La comparación válida del helper es contra CUDA Graph: aproximadamente 0,288→0,231 ms. Compararlo únicamente contra eager 1,945 ms exageraría el beneficio incremental disponible. La compilación inicial fue aproximadamente 10 s y está separada del runtime.

## Regenerar la documentación numérica

```powershell
& $runtimePython scripts/preserve_train40_runtime_handoff.py --destination docs/train40_runtime_handoff_20261027/evidence --tables-only
```

No abre sensores ni modelos. Para preservar otro paquete, usar `--source` y un destino nuevo. Los snapshots `.py.txt` contienen código exploratorio exacto con rutas locales y no son scripts de producción portables; revisar y restaurar sus rutas antes de ejecutarlos en otro host.
