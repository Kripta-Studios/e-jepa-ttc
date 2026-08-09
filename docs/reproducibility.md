# Reproducibilidad

Actualizado: 2026-08-10.

## Protocolo sintético Causal Scale v5

El runner usa grupos deterministas disjuntos: train 101, validation 202 y test 303.
`diagnostic` nunca instancia test; `full` exige código/config limpios y lo evalúa una
sola vez. Cada summary conserva historial completo por época, commit/dirty flag,
hashes de configs y checkpoint, snapshot de entorno, splits abiertos y contador de
evaluaciones test. La calibración de varianza se ajusta solo en validation.

```powershell
uv run --no-sync python scripts/train_causal_scale_v5_synthetic.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_synthetic_v5.yaml `
  --output-dir artifacts/debug/causal_scale_v5_synth_diagnostic `
  --stage diagnostic --device auto
```

La comparación compacta se regenera con
`scripts/build_causal_scale_v5_diagnostic_comparison.py`; sus inputs son los summaries
completos, nunca cifras copiadas a mano.

V5/test 303 y V7/test 603 ya fueron consumidos una vez y son evidencia inmutable, no
datos de desarrollo. El artefacto V7 está en
`artifacts/metrics/causal_scale_v7_synthetic_learning_gate_v1.json`; su identidad es
`97e52b2a9d3463d6a2e57d12e9408f80bb6a3b8e0d491beeb3546c2d1586a52b` y registra
el fallo Pearson `.9201432`. Un nuevo intento requiere grupos y seeds nuevos
prerregistrados; volver a ejecutar 303 o 603 no constituye replicación independiente.

V8 usa la configuración
`configs/experiment/e_jepa_garl_event_causal_scale_synthetic_v8.yaml`. El modo
diagnóstico abre solo 701–703 y 801–803. Los tests 901–903 solo pueden abrirse una vez
por grupo en modo `full`, después de publicar el commit exacto y desde estado limpio.

## Entorno auditado

```text
Python   3.11
PyTorch  2.11.0+cu128
GPU      NVIDIA GeForce RTX 5070 Ti Laptop
VRAM     ~12,8 GiB
RAM      32 GiB
```

Instalación:

```powershell
uv sync --locked --all-groups --no-editable
uv run --no-sync python -m e_jepa_ttc --help
```

Usar `--no-sync` dentro de tests/runs evita que `uv` cambie la instalación durante
una ejecución.

## Registro mínimo

Cada run debe incluir:

```text
experiment_id, run_name, git_commit, git_dirty
config_hash, dataset_manifest_hash, split_version, seed
host, Python, Torch, CUDA, GPU
start_time, end_time, status
checkpoint_path/hash, metrics_path
selection metric, sample counts, provenance
```

## Validación del repositorio

```powershell
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest -q
git diff --check
```

La suite completa pasa en un árbol sin `artifacts/runs` ni
`artifacts/features`. Los tests que auditan artefactos reales ignorados se omiten si
estos no existen; los contratos de formato siguen cubiertos con fixtures unitarios.

## Auditoría semántica sin dataset

```powershell
make jepa-shortcut-audit
```

El target ejecuta el benchmark de shortcut fijo, el control frame-varying y su
agregación. No usa eAP/EvTTC ni etiquetas durante el entrenamiento de la
representación. Artefactos versionados y SHA-256:

```text
jepa_semantic_shortcut_benchmark_v1.json
  EDF8EFA639A845D1D228AC43DDF778FF5C0B573DE72FC0A5685E5DEE74C18368
jepa_semantic_shortcut_frame_control_v1.json
  AC496D937B5C3B1BE9B7FD35802C2BD368D0D1D2973C258FEC266614E21E0F4D
jepa_semantic_capacity_audit_v1.json
  393CA65AD8E4C1453F23985085E56EFC48DB89FAF77413334319EE8A3948C36E
```

La decisión reproducida es
`reject_full_r2_prefer_temporal_residual_on_synthetic_gate`. No autoriza un
cambio de producción: exige confirmación sobre filas eAP reales.

## Entrenamiento event-only raw

Dry-run full:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --dry-run
```

Screen:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile screen --stages train `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_screen_v1
```

Full multisemilla:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages train freeze `
  --eap-root 'E:\eAP_dataset' `
  --garlttc-root 'E:\GarlTTC_dataset' `
  --output-root artifacts/runs/e_jepa_garl_event_full_v1 `
  --resume
```

El full exige Git limpio, todas las filas válidas y seeds exactas 7/13/23. Cada
época usa shuffle derivado de `seed + epoch`, por lo que resume no depende del
estado interno previo del DataLoader.

Checkpoints retenidos:

```text
best.pt
last.pt
best_validation_predictions.csv
summary.json
```

## EvTTC predict/score

Ejemplo después de crear un config de inferencia y manifest label-free válidos:

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages evttc-predict evttc-score `
  --output-root artifacts/runs/e_jepa_garl_event_full_v1 `
  --evttc-config configs/local/evttc_table_vi_inference.yaml `
  --evttc-predictions artifacts/official/evttc_table_vi/predictions.json `
  --evttc-targets configs/local/evttc_table_vi_targets.json `
  --evttc-metrics artifacts/official/evttc_table_vi/metrics.json
```

`evttc-predict` rechaza campos privilegiados antes de cargar checkpoint/GPU. El
archivo de targets solo se abre en `evttc-score`.

## Submission

```powershell
uv run --no-sync python scripts/run_e_jepa_garl_final.py `
  --profile full --stages submission-validate `
  --submission artifacts/official/candidate/submission.json `
  --sample-submission configs/local/sample_submission.json `
  --submission-validation artifacts/official/candidate/validation.json
```

La validación es offline. El runner no autentica, no contacta CodaBench y no cuenta
un upload inexistente como evaluación oficial.

## Almacenamiento

- `artifacts/runs` y `artifacts/features`: locales, ignorados y regenerables;
- `artifacts/metrics`: resúmenes compactos seleccionados y versionados;
- datasets: fuera de Git y nunca modificados por los trainers;
- no construir el cache high-resolution full (~455 GiB);
- usar shards de 256–2.048 solo para diagnóstico;
- conservar margen de disco antes de cada etapa.

El 2026-08-02 se eliminaron CARLA, runs, features y caches locales, dejando más
de 315 GiB libres en C:.

## Claim boundary

Un run local no es oficial. Para comparar con Garl-TTC deben coincidir modalidad,
split, métrica, protocolo y presupuesto; el resultado final necesita freeze,
evaluación externa y hashes verificables.

## Causal-scale eAP screen v1

El runner `scripts/train_causal_scale_eap_screen.py` valida antes de GPU el SHA del
manifest, identidad de artifact, counts y secuencias congeladas. Guarda estado
atómico cada época: modelo, optimizer, scheduler, RNG CPU/CUDA/Python/NumPy, estado
del generador del DataLoader, historial, best y paciencia. `--resume` continúa desde
la siguiente época y el límite de 6 h descuenta el tiempo previo.

Antes de usarlo como evidencia debe añadirse un test end-to-end de resume. El output
esperado es `summary.json`, `validation_predictions.csv`, `model_best.pt` y
`state/{last,best}.pt`. El baseline oficial debe recibir exactamente los sample
tokens guardados en el CSV; no se permiten filtros posteriores por error.
