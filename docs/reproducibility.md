# Reproducibilidad

## Entorno auditado

```text
Python   3.11
PyTorch  2.11.0+cu128
GPU      NVIDIA GeForce RTX 5070 Ti Laptop
VRAM     ~12,8 GB
RAM      32 GB
```

## Registro mínimo por run

```text
git commit
config y hash
manifest y hash
split y hash
seed
sample selection hash
checkpoint hash
mejor época
historial de métricas
latencia y peak VRAM
estado de Benchmark-10
```

Instalación determinista, incluida la alternativa segura para rutas Windows
con caracteres Unicode:

```powershell
uv sync --locked --all-groups --no-editable
uv run --no-sync python -m e_jepa_ttc --help
```

No ejecutar `uv run` sin `--no-sync` dentro de los tests o pipelines: podría
recrear una instalación editable y modificar el entorno durante una corrida.

## Checkpoints

Solo se conservan:

```text
best.pt
last.pt
weights_only.pt
```

`resume.pt` es temporal y se elimina al completar.

## Comandos

### Orquestador completo CARLA → EvTTC

```powershell
# Plan exacto sin entrenamiento.
.\scripts\run_carla_evttc_complete.ps1 -Profile Full -DryRun

# CARLA SSL + validation/test sintéticos + A0 control + transferencia OOF.
.\scripts\run_carla_evttc_complete.ps1 -Profile Full -Resume
```

`--resume`/`-Resume` omite etapas con artefactos completos y restaura CARLA
desde `resume.pt` cuando existe. Cada subprocess escribe un `.log`; el estado,
comandos, hardware y duraciones quedan en
`artifacts/runs/carla_evttc_complete_v1/orchestration_status.json`. Los runs
EvTTC guardan su propio `summary.json`, predicciones OOF y checkpoints. El
pipeline compara solamente `A0_MATCHED_GLOBAL` para medir el efecto de la
inicialización; no reabre la búsqueda arquitectónica.

El resultado final se regenera con
`scripts/compare_evttc_initializations.py`. Ese comparador exige igualdad de
fold/seed, selección de muestras, cache, cabeza común y trainer, y calcula
bootstrap por secuencia. Benchmark-10 permanece sellado.

### EvTTC

Ruta automatizada y cross-platform:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py --help
uv run --no-sync python scripts/run_evttc_final_pipeline.py validate
uv run --no-sync python scripts/run_evttc_final_pipeline.py compare --resume
```

`compare` usa por defecto folds `0..4`, seeds `7/13/21`, A0/A1 y el perfil
frozen `matched`. El agregador exige 15 runs por variante antes del freeze.
Cada pareja fold/seed debe compartir hash de samples, backbone y cabeza común.
También registra victorias pareadas, coste, dispersión entre medias por seed y
bootstrap OOF por secuencia. El resultado cerrado es
`artifacts/metrics/evttc_a0_a1_grouped_cv_5fold_3seed.json`.

Validación:

```powershell
.\scripts\run_evttc_architecture_selection.ps1 -Mode Validate
```

Screen:

```powershell
.\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Screen -Stage Core -Protocol HistoricalBase -Resume
```

Los comandos equivalentes para Garl cambian `-Stage Core` por `-Stage Garl`.

Confirmación grouped-CV de la comparación promovida:

```powershell
.\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Confirm -Stage Core -Protocol GroupedCV `
  -AllFolds -Seed 7 -RandomControl `
  -Variants A0_MATCHED_GLOBAL,A1_MATCHED_DENSE_BLOCK
```

`-RandomControl` es obligatorio mientras no existan checkpoints SSL
preentrenados exclusivamente con el train de cada fold. Impide reutilizar el
checkpoint histórico en folds cuyas secuencias de validación ya aparecieron en
su pretraining.

La ablación bbox-ROI reproducible se lanza de forma aislada para no repetir
brazos cerrados:

```powershell
.\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Confirm -Stage Core -Protocol GroupedCV `
  -AllFolds -Seed 7 -RandomControl -Resume `
  -Workers 8 -ExecutionProfile Matched `
  -Variants R1_MATCHED_BBOX_ROI
```

Sus cinco folds terminaron en el commit `42a90e0`: score 0,59814, error
relativo 30,99 % y MAE 1,0100 s. Frente a A0 seed 7 empeora 2,90 %, 2,74 % y
4,55 %, respectivamente; por el gate secuencial no se ejecutan seeds 13/21.

Entrenamiento del candidato ya fijado y evaluación separada:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py fit-holdout `
  --variant A0_MATCHED_GLOBAL --seeds 7 13 21 --resume

uv run --no-sync python scripts/run_evttc_final_pipeline.py select-final `
  --variant A0_MATCHED_GLOBAL `
  --matched-root artifacts/runs/evttc32_final_family_holdout_matched/core/fold-0/A0_MATCHED_GLOBAL `
  --throughput-root artifacts/runs/evttc32_final_family_holdout/core/fold-0/A0_MATCHED_GLOBAL `
  --output artifacts/metrics/evttc_final_a0_profile_selection.json

uv run --no-sync python scripts/run_evttc_final_pipeline.py evaluate-holdout `
  --checkpoint <best.pt> `
  --cache-manifest artifacts/features/evttc32_final_family_holdout_core/manifest.json `
  --splits validation `
  --output-dir artifacts/metrics/final_validation
```

Para abrir el holdout familiar:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py evaluate-holdout `
  --checkpoint <best.pt> `
  --cache-manifest artifacts/features/evttc32_final_family_holdout_core/manifest.json `
  --selection-manifest artifacts/metrics/evttc_final_a0_profile_selection.json `
  --splits test --allow-diagnostic-test `
  --output-dir artifacts/metrics/final_family_ood
```

Este `test` es el holdout diagnóstico CCRs-2/CCRs-3/CPNAO. No es el
Benchmark-10 sellado. El JSON de salida registra splits, secuencias, hashes,
checkpoint, commit, métricas macro, bootstrap por secuencia y
`diagnostic_test_opened=true`.

## Perfiles de recursos

- `matched`: batch 16 × acumulación 2; reproduce la evidencia.
- `throughput`: batch 32 × acumulación 1; mismo batch efectivo, menor overhead.
- workers: autodetección `min(12, logical_cpus/2)`; ajustar hacia abajo si otro
  proceso intensivo comparte RAM o disco.
- BF16, `pin_memory`, workers persistentes y prefetch permanecen activos.

No se cambia de perfil a mitad de una matriz. Una optimización de ejecución se
aplica a todos los brazos o se reporta como protocolo distinto.

En la ejecución final, throughput redujo el tiempo agregado de tres seeds de
1.653 s a 411 s (4,02×), pero empeoró el score medio de 0,30400 a 0,34139.
Por ello se conserva para iteración y matched para el candidato de precisión.

Tiempos medidos, no estimaciones del artículo:

```text
EvTTC A0 grouped CV, 5 folds × 3 seeds     1,324 h
EvTTC A1 grouped CV, 5 folds × 3 seeds     2,095 h
A0 + A1, 30 runs                              3,419 h
CARLA JEPA full, por época (proyección)       32,5 min
CARLA JEPA full, máximo 30 épocas            16,2 h
CARLA test sintético completo (proyección)     8,5 min
```

La transferencia A0 de una seed sobre cinco folds requiere aproximadamente
0,4–0,7 h; las tres seeds son del orden de 1,3 h si se ejecutan. CARLA puede
terminar antes por early stopping; el intervalo operativo prudente es 8–16 h.

## eAP sin TTC

eAP train-40 está completo: 40 secuencias, 216 archivos y 536,64 GiB. Puede
usarse en una etapa posterior de SSL sin etiquetas. No
puede producir MAE TTC, seleccionar arquitectura ni sustituir EvTTC. El gate
recomendado es 2–4 secuencias, presupuesto fijo y fine-tuning EvTTC idéntico
antes de ampliar a 40.

## Preparación CARLA DVS Looming

El adapter no descarga ni duplica el dataset. Audita la extracción existente,
crea un manifest firmado y separa bloques completos:

```powershell
uv run --no-sync python scripts/prepare_carla_looming.py `
  --root datasets/CARLA_DVS_Looming_Dataset/random_spawn `
  --manifest data/manifests/carla_dvs_looming_v1.json `
  --split data/splits/carla_dvs_looming_blocked_v1.json `
  --context-ms 100 --group-size 25 --folds 5 --seed 42
```

Salida esperada de la versión auditada:

```text
total / válido / inválido   1.406 / 1.395 / 11
train / validation / test   803 / 298 / 294
eventos válidos             7.692.294.635
allow_pickle                false
```

La validación exhaustiva de los 7.692 millones de eventos es opcional porque
lee los 71,64 GiB completos:

```powershell
uv run --no-sync python scripts/prepare_carla_looming.py `
  --full-event-validation
```

Para comprobar el artefacto de distribución sin confiar en el manifest:

```powershell
Get-FileHash `
  datasets/CARLA_DVS_Looming_Dataset/random_spawn.tar.gz `
  -Algorithm MD5
# 21A3E72A1C1D9C441A7426393F4E545F
```

Entrenamiento y evaluación directa:

```powershell
.\.venv\Scripts\python.exe scripts/pretrain_carla_jepa.py `
  --profile full --dry-run

.\.venv\Scripts\python.exe scripts/pretrain_carla_jepa.py `
  --profile full --output artifacts/runs/carla_jepa_full_seed42_v1

# Solo si existe resume.pt de una interrupción:
.\.venv\Scripts\python.exe scripts/pretrain_carla_jepa.py `
  --profile full --output artifacts/runs/carla_jepa_full_seed42_v1 --resume

.\.venv\Scripts\python.exe scripts/evaluate_carla_jepa.py `
  --checkpoint artifacts/runs/carla_jepa_full_seed42_v1/carla_jepa_encoder_best.pt `
  --role test `
  --output artifacts/runs/carla_jepa_full_seed42_v1/test_evaluation.json
```

El perfil auditado consume mmap, respeta el split, no usa TTC, colisión,
`vel` ni `diameter_object` como features y no materializa un cache voxel
global. Usa 12.020/4.457/4.297 pares train/validation/test; BF16, batch 24,
acumulación 2, ocho workers, prefetch 2, AdamW fused, clipping, EMA,
warm-up/cosine y early stopping 8/6 con máximo 30 épocas. Guarda best, last,
resume atómico, `history.jsonl`, `metrics.json` y evaluaciones separadas.

Los probes de hardware mostraron `8,46` observaciones/s con batch 24 y ocho
workers. Batch 16 (`8,20/s`), 32 (`7,83/s`), 48 (`7,63/s`) y 96 (`6,45/s`)
fueron peores; seis workers quedó prácticamente empatado (`8,45/s`) y 12 bajó
a `6,69/s`. El cuello de botella es voxelización/SSD. El smoke bajó
validation loss `0,02563→0,02247`; un test de contrato de 16 pares obtuvo
`0,02195` y cero dimensiones colapsadas. No se presenta como error TTC ni OOD
real.

La reproducción del híbrido bbox causal usa:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_causal_geometry_hybrid.py `
  --manifest data\manifests\evttc_all32_local.yaml `
  --cache-manifest <manifest-del-cache-core> `
  --neural-predictions <validation_predictions.npz> `
  --derivative-window 21 `
  --output artifacts\metrics\causal_geometry_dense_hybrid.json
```

## Resultados

Los JSON bajo `artifacts/runs` son artefactos locales ignorados por Git. Solo
los resúmenes pequeños seleccionados se copian a `artifacts/metrics` cuando el
run está cerrado y auditado.
