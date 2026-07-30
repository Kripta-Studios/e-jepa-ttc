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

## eAP sin TTC

eAP train-40 está completo: 40 secuencias, 216 archivos y 536,64 GiB. Puede
usarse en una etapa posterior de SSL sin etiquetas. No
puede producir MAE TTC, seleccionar arquitectura ni sustituir EvTTC. El gate
recomendado es 2–4 secuencias, presupuesto fijo y fine-tuning EvTTC idéntico
antes de ampliar a 40.

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
