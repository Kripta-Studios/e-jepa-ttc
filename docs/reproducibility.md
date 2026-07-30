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

## Checkpoints

Solo se conservan:

```text
best.pt
last.pt
weights_only.pt
```

`resume.pt` es temporal y se elimina al completar.

## Comandos

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
