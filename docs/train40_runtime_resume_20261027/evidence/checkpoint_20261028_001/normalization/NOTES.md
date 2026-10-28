# Normalización exacta por cola superior — 27 octubre 2026

Estado: implementación, pruebas CPU y microbenchmarks GPU eager/graph terminados. Falta integrar y medir la representación completa.

## Fórmula

El normalizador original cuenta `m = sum(voxel > 1e-6)` valores ocupados entre `N` columnas. Ordena toda la fila en orden ascendente y usa los estadísticos en posiciones `N-m+floor(.95*(m-1))` y `N-m+ceil(.95*(m-1))`, con el caso vacío clampeado al último elemento.

Contando desde el máximo, el estadístico inferior está en `m-1-floor(.95*(m-1)) = ceil(.05*(m-1))`. Su máximo para `m<=N` es `ceil((N-1)/20)`. Por tanto basta un top-k descendente de `1+ceil((N-1)/20)` elementos. Para `N=10*128*128=163840`, son 8193 elementos. La implementación conserva literalmente el cálculo float64 del rango y la interpolación original; únicamente cambia la selección de los dos valores.

No se usa `kthvalue` directamente porque la API requiere un `k` escalar mientras cada fila tiene distinta ocupación. Leer esos rangos al host introduciría sincronizaciones y llamadas por ventana. El top-k fijo se ejecuta por lote sin transferencias a CPU y admite captura CUDA.

Los índices entre empates pueden variar, pero solo se consumen los valores. La selección no aproxima el percentil, no cambia la máscara de `1e-6`, no cambia los canales de conteo y no reduce precisión. El helper requiere la misma precondición de voxels finitos; no valida valores mediante un `bool(tensor)` que introduciría sincronización.

## Archivos

- `operational/train40_runtime/normalization.py`: `ExactTopKNormalizer`, buffer de ocupación y cola superior reutilizable.
- `tests/unit/test_train40_runtime_normalization.py`: todas las ocupaciones para tamaños hasta 1001; frontera de rango para 163840; vacíos, ties, umbral, forma nativa, reutilización y contrato de layout.
- `scripts/benchmark_train40_normalization.py`: sort original frente a top-k, orden alternado, entrada idéntica, evidencia de fuentes y resultados negativos conservados.

## Validación CPU ejecutada

- 25 tests pasados y 1 test CUDA omitido explícitamente.
- Ruff y Pyright pasan.
- Microbenchmark CPU con dos threads, un objeto, seis ventanas 128×128, ocupación 50%, tres repeticiones: sort media 72.0945 ms; top-k media 22.1870 ms. Escalas, voxel float64 y salida float32 bit-exactos. No extrapolar esta relación a CUDA.
- Scratch retenido: 1,769,568 bytes frente a 16,711,680 bytes del sort original para seis ventanas. Los temporales internos de PyTorch no están incluidos en ese contador.
- Evidencia: `cpu_smoke_v1/binding.json`, `cases.json`, `summary.json`.

El reloj del microbenchmark incluye restaurar una copia del voxel preparado y normalizarla. Excluye sensores, ingestión, ROI, H2D, canales escalares de conteo, redes y TTC.

## GPU ejecutada con turno exclusivo

Los dos comandos siguientes terminaron con código cero. Dieciséis casos pasan igualdad bit a bit de escalas y salidas float64/float32; fuentes intactas. Las tablas se regeneran en [GPU_RESULTS.md](GPU_RESULTS.md) desde `gpu_eager_v1/summary.json` y `gpu_graph_v1/summary.json`.

En veinte objetos, top-k reduce la normalización de aproximadamente 17–23 ms a 7–8 ms según modo/ocupación. Un objeto eager empeora con top-k; con captura ambos bajan a menos de 1 ms y top-k resulta más rápido en los cuatro casos. Esto no prueba el rendimiento del chunk de cuatro objetos usado por el ring ni una actualización completa.

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_normalization.py --device cuda --objects 1 20 --occupancy 0 .01 .5 1 --repeats 30 --warmup 5 --output artifacts/train40_runtime_resume_20261027/normalization/gpu_eager_v1

& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_normalization.py --device cuda --objects 1 20 --occupancy 0 .01 .5 1 --repeats 30 --warmup 5 --graphs --output artifacts/train40_runtime_resume_20261027/normalization/gpu_graph_v1
```

No incorporar por defecto ni afirmar mejora global hasta medir la representación completa. Esta tarea no ha modificado el ring ni el prefix; la integración opcional pertenece al agente de prefijos. GPU liberada explícitamente después de los dos procesos.

## Fuentes primarias consultadas

- [PyTorch 2.11 topk](https://docs.pytorch.org/docs/2.11/generated/torch.topk.html): selección ordenada y observación sobre índices no estables ante empates.
- [PyTorch 2.11 kthvalue](https://docs.pytorch.org/docs/2.11/generated/torch.kthvalue.html): orden estadístico escalar.
- [Implementación CUDA top-k v2.11.0](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/TensorTopK.cpp): el backend NVIDIA no elige automáticamente full-sort en `should_use_sort`; `sorted=True` ordena la cola seleccionada y puede asignar temporales. Esto explica por qué los bytes retenidos no son el pico total de memoria.
