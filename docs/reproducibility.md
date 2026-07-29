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

## Resultados

Los JSON bajo `artifacts/runs` son artefactos locales ignorados por Git. Solo
los resúmenes pequeños seleccionados se copian a `artifacts/metrics` cuando el
run está cerrado y auditado.
