# E-JEPA-TTC / OGE-JEPA-TTC

Investigación reproducible de Time-to-Collision con cámaras de eventos. La ruta
activa compara un Event-JEPA global auditado, representación densa causal,
geometría object-centric y una réplica local de Garl-TTC sobre EvTTC-32.

Estado al 30 de julio de 2026:

- `BASE` histórico reproducido con predicciones idénticas byte a byte;
- FlowMimic e inverse-TTC global rechazados por resultados negativos;
- Dense Patch, AttnRes, Object-KDA, geometría y Garl implementados;
- screens comparables Core/Garl pendientes de promoción científica;
- eAP train-40 reservado para una fase posterior de pretraining sin TTC;
- Benchmark-10 sellado y no abierto.

Documentación:

- [estado verificable](STATUS.md);
- [plan completo v6](PLAN.md);
- [informe técnico](docs/technical_report.md);
- [informe PDF](docs/e_jepa_ttc_paper.pdf);
- [protocolo de datos](docs/dataset_card.md);
- [model card](docs/model_card.md).

## Resultado de referencia

`B0_HISTORICAL_BASE_EXACT`, seed 7, checkpoint downstream de época 26/30:

| Split | MAE | RMSE | Error relativo medio |
|---|---:|---:|---:|
| validation histórica | 0,322892 s | 0,584432 s | 8,1554 % |

La reproducción está en
`artifacts/audit/oge_sota/historical_base_reproduction.json` y demuestra
paridad exacta con el artefacto original. Este resultado es un ancla histórica:
la comparación de arquitectura usa `A0_MATCHED_GLOBAL`, no reutiliza esta fila
como si hubiera sido entrenada con la matriz nueva.

No existe todavía un claim SOTA ni un resultado oficial de Benchmark-10.

## Instalación

Requisitos principales:

- Windows o Linux;
- Python 3.11;
- PyTorch con CUDA;
- `uv`.

```powershell
uv sync --all-groups --no-editable
uv run --no-sync python -m e_jepa_ttc --help
```

En el host auditado:

```text
GPU       NVIDIA GeForce RTX 5070 Ti Laptop, ~12,8 GB VRAM
RAM       32 GB
CPU       Ryzen 9, 32 threads lógicos
PyTorch   2.11.0+cu128
```

El entrenamiento usa BF16, workers persistentes, memoria fijada, prefetch,
microbatch y acumulación. Usar 32 workers no es recomendable en Windows:
multiplica procesos y RAM sin garantizar más throughput.

## Inventario cerrado de datos

```text
datasets/evttc
    EvTTC-32 etiquetado: desarrollo, grouped CV y entrenamiento final

datasets/evttc_official_benchmark_sealed
    Benchmark-10: una inferencia final después del freeze

E:\eAP_dataset\data\train
    eAP Hugging Face train-40: descarga ya iniciada, sin TTC oficial
```

No se añaden nuevos datasets, no se descarga eAP test y el pseudo-TTC no se
considera ground truth.

## Auditoría exacta de BASE

```powershell
.\.venv\Scripts\python.exe scripts\audit_historical_base.py `
  --checkpoint artifacts\runs\evttc32_article_ablation\base\seed7\ft30\tiny_cnn_best.pt `
  --cache artifacts\features\evttc32_trainval\cache.npz `
  --output artifacts\audit\oge_sota\historical_base_reproduction.json `
  --batch-size 24
```

La auditoría comprueba arquitectura, checkpoint, cache, hashes, métricas y
predicciones.

## Selección rápida de arquitectura

Validar código:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Validate
```

Screen Core:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Screen `
  -Stage Core `
  -Protocol HistoricalBase `
  -Resume
```

Screen Garl con ResNet-50:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Screen `
  -Stage Garl `
  -Protocol HistoricalBase `
  -Resume
```

El perfil Screen utiliza hasta ocho épocas, 304 ventanas train y 80 validation.
Core y Garl escriben resúmenes separados:

```text
artifacts/runs/evttc32_architecture_v4_historical_base_screen/
├── core/fold-0/matrix_summary.json
└── garl/fold-0/matrix_summary.json
```

Un smoke solo comprueba integración. No promueve componentes.

## Grouped CV y semillas

Después del screen se pasan explícitamente solo los candidatos promovidos:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Confirm `
  -Stage Core `
  -Protocol GroupedCV `
  -AllFolds `
  -Seed 7 `
  -Resume `
  -Variants A0_MATCHED_GLOBAL,K1_OBJECT_KDA
```

Solo BASE y un máximo de dos finalistas se repiten con:

```powershell
-AllFolds -AllSeeds
```

No se ejecuta el producto cartesiano de módulos, folds y seeds.

## Arquitecturas bajo gate

- `A0_MATCHED_GLOBAL`: control object-cache global.
- `A1_MATCHED_DENSE_BLOCK`: patches espaciales antes de atención temporal
  block-causal.
- `A2_MATCHED_DENSE_ATTNRES`: recuperación por tarea a través de profundidad.
- `K1_OBJECT_KDA`: memoria delta temporal posterior a la mezcla espacial.
- `A4_GT_GEOMETRY`: oracle de bbox GT con height/area/affine/event contrast.
- `G0`–`G7`: direct, LHR, early/late fusion y foreground de Garl-TTC.

TargetQuery, máscara predicha, refiner, router, residual e incertidumbre no se
promueven hasta que la geometría bbox-GT supere el gate frente a BASE.

## Ego-motion

El HDF5 EvTTC almacena velocidad norte/este/arriba y heading en grados. El
loader convierte esa señal a la cámara de eventos mediante:

```text
navegación → LiDAR → Blackfly izquierda → Prophesee izquierda
```

La de-rotación por yaw no utiliza datos futuros. El warp traslacional usa
velocidad de cámara, brazo rígido, intrínsecos y profundidad causal. Si la
profundidad procede de la distancia oficial EvTTC, el resultado se etiqueta
obligatoriamente como oracle/teacher y no puede usarse como inferencia final.

## Almacenamiento

- caches EvTTC separados por rol y perfil;
- máximo `best`, `last` y `weights_only` por run;
- sin voxel cache global de eAP;
- sin extracción masiva de TAR RGB;
- sin logits SAM full-resolution;
- sin hidden states DINO de todas las capas.

Los datos, caches y checkpoints no se versionan en Git.

## Integridad científica

- splits completos por secuencia;
- selección por validación macro de secuencia;
- sin TTC durante SSL;
- sin tuning sobre Benchmark-10;
- resultados negativos conservados;
- latencia medida con el candidato realmente evaluado;
- ningún claim SOTA sin evaluación oficial reproducible.

Este repositorio es investigación. No es un sistema certificado de seguridad ni
debe controlar un vehículo.

## Licencia

Consulta [LICENSE](LICENSE) y las licencias de EvTTC, eAP, Garl-TTC y los
teachers antes de redistribuir datos, pesos o derivados.
