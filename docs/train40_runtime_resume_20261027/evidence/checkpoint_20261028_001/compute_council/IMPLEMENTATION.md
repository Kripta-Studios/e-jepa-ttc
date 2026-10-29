# Encoder FP32 compilado: preparación del experimento

Implementado por este frente únicamente en:

- `operational/train40_runtime/compiled_encoder.py`
- `scripts/benchmark_train40_encoder.py`
- `tests/unit/test_train40_runtime_compiled_encoder.py`

`CompiledEndpointEncoder` hace una copia privada del encoder, conserva ABI `(foreground, token, dense|None)` y usa un core tensorial con los tres outputs. La selección explícita de Inductor usa `max_autotune=True`, backends Conv `ATEN,TRITON`, `use_fast_math=False`, `force_same_precision=True`, `triton.cudagraphs=False`, formas estáticas. Se rechaza ejecución CUDA si matmul o cuDNN tienen TF32 habilitado. La prueba CPU usa `compile_backend=None`; no se confunde ese wrapper con compilación GPU.

El driver carga los pesos sellados y un query registrado. El modo `--endpoint-layout unique` deriva seis endpoints event o siete RGB; el modo `original` conserva los doce slots event o quince RGB originales. Prepara tamaños nativos y veinte veces esos tamaños, o los tamaños explícitos solicitados. Los grupos artificiales adicionales tienen escalado explícito: sirven para capacidad, no para afirmar veinte tracks reales. Captura dos grafos, base y compilado, alterna su orden por repetición y comprueba inputs originales, escalados y vacíos. Conserva tiempo de compilación y captura aparte. No mide transporte, TTC, entrada o H2D; `final_ttc_admitted=false` permanece explícito. Los outputs no finitos y fuera del gate fijo dan salida distinta de cero con resumen preservado.

Validación CPU ejecutada:

```text
15 passed in 4.18s
Ruff: All checks passed!
Pyright: 0 errors, 0 warnings, 0 informations
```

Tras obtener turno GPU exclusivo se ejecutó E_A5 con los doce slots originales. El resultado es **FAILED_ENCODER_PARITY**: dos valores dense no cumplen atol=rtol=1e-5, aunque las demás salidas y casos pasan. No se integra ni se promueve. Los 594 originales sellados siguen intactos. La comparación de grafos dio p50 1,7988 → 1,3763 ms; es únicamente una observación de rendimiento de una variante rechazada. Véase `GPU_RESULTS.md` para resultados y próximos pasos. Comando ejecutado:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' -u scripts/benchmark_train40_encoder.py --candidate D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json --reference artifacts/final_train40_transfer_20261027/final_train40/own --compiler-root artifacts/train40_runtime_20261027/compiler_probe --output artifacts/train40_runtime_resume_20261027/encoder_ea5_b12_v1 --models E_A5_MATCHED --batches 12 --endpoint-layout original --repeats 20
```

Integración pendiente: el productor actual se traza con TorchScript y contiene guards host en su forward original. Un encoder compilado no puede anidarse en esa traza ni capturarse todo el forward original directamente. El wrapper falla explícitamente si se intenta trazar. Una integración futura debe separar encoder compilado y tail trazado, manteniendo todos los guards originales en la frontera pública. El helper de copias prepara la vista; no declara ese trabajo de integración terminado.
