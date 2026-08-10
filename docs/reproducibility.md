# Reproducibilidad

## Artefactos A0 / Garl exact-2048

- A0: `artifacts/runs/causal_scale_eap_screen_v1_seed7/summary.json`.
- Subset: `artifacts/subsets/garl_validation_common_roi_v1/manifest.json`.
- Referencia: `artifacts/runs/garl_official_event_only_same2048/metrics.json`.
- Comparación: `artifacts/metrics/causal_scale_eap_garl_event_only_comparison_v1.json`.
- Cola: `artifacts/metrics/causal_scale_eap_garl_event_only_a0_top10pct_outliers_v1.csv`.
- Matched subset: `artifacts/subsets/garl_event_only_matched_screen_v1/manifest.json`,
  identidad `dd08ecc983f30e38a939204f9a2df09e4966bbe73bd764c972f7726e5d4e34d3`.
- Failure decomposition:
  `artifacts/metrics/causal_scale_eap_a0_failure_decomposition_v1.json`, identidad
  `75918c58cd91258fac5aac11f8d6fca00ce6cf43014e5ee19ab3a30d7c91beb7`.
- Matched official preprocessing cache:
  `artifacts/cache/garl_official_event_only_matched_preprocessing_v1/manifest.json`,
  identidad `92af281030170733411ef9d65b19e88ebc8019c729dd6743e02ae9c40f564b52`.

Regeneración del cache matched (los shards locales no se versionan):

```powershell
uv run --extra geometry python scripts/build_garl_matched_preprocessing_cache.py `
  --release-root E:\Garl-TTC `
  --official-config E:\Garl-TTC\configs\ablation\event_lhr.yaml `
  --subset-manifest artifacts/subsets/garl_event_only_matched_screen_v1/manifest.json `
  --eap-root E:\eAP_dataset `
  --output-dir artifacts/cache/garl_official_event_only_matched_preprocessing_v1 `
  --batch-size 32 --num-workers 16 --shard-size 64 --seed 7
```

Entrenamiento Garl matched exacto, desde cero y en GPU:

```powershell
uv run --extra geometry python scripts/train_garl_matched_from_cache.py `
  --release-root E:\Garl-TTC `
  --cache-manifest artifacts/cache/garl_official_event_only_matched_preprocessing_v1/manifest.json `
  --output-dir artifacts/runs/garl_matched_event_only_cached_seed7 `
  --device cuda --seed 7 --epochs 18 --batch-size 32 `
  --minimum-epochs 8 --early-stopping-patience 5 `
  --maximum-runtime-hours 4.5
```

Si existe `state/last.pt`, repetir el mismo comando con `--resume`; el estado liga
config, cache, release, seed, épocas, batch, selección y guard temporal y rechaza
cambios. Resultado firmado:
`artifacts/runs/garl_matched_event_only_cached_seed7/summary.json`, identidad
`553904c18874b3509e10a71e5b46b33e0f5df6ddb4fec7a7e57b6abc34322937`.
Predicciones exactas: `validation_predictions.parquet`, 2.048 filas, SHA256
`d547d9261a6a772fefa4b46fae44cbe264e21403d15cd6de6e61b5755852cdbf`.

Regeneración de las tablas release y matched firmadas:

```powershell
uv run python scripts/build_causal_scale_eap_garl_comparison.py `
  --causal-predictions artifacts/runs/causal_scale_eap_screen_v1_seed7/validation_predictions.csv `
  --causal-summary artifacts/runs/causal_scale_eap_screen_v1_seed7/summary.json `
  --release-predictions artifacts/runs/garl_official_event_only_same2048/predictions.parquet `
  --release-metrics artifacts/runs/garl_official_event_only_same2048/metrics.json `
  --matched-predictions artifacts/runs/garl_matched_event_only_cached_seed7/validation_predictions.parquet `
  --matched-summary artifacts/runs/garl_matched_event_only_cached_seed7/summary.json `
  --subset-data artifacts/subsets/garl_validation_common_roi_v1/data.parquet `
  --subset-labels artifacts/subsets/garl_validation_common_roi_v1/labels.parquet `
  --subset-manifest artifacts/subsets/garl_validation_common_roi_v1/manifest.json `
  --official-train-assets E:\Garl-TTC\configs\splits\train.txt `
  --official-train-labels E:\GarlTTC_dataset\annotations\train.parquet `
  --official-config E:\Garl-TTC\configs\ablation\event_lhr.yaml `
  --official-checkpoint E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth `
  --output-json artifacts/metrics/causal_scale_eap_garl_event_only_comparison_v1.json `
  --outliers-csv artifacts/metrics/causal_scale_eap_garl_event_only_a0_top10pct_outliers_v1.csv `
  --bootstrap-iterations 10000 --bootstrap-seed 7
```

El comparador verifica igualdad exacta de tokens, secuencia y target y remuestrea
las tres secuencias completas, nunca ventanas.

Preregistro y ejecución A1 geometry-only:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_geometry_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_a1_geometry_v1_seed7 `
  --device cuda
```

Si existe un `state/last.pt` compatible, añadir `--resume`; nunca borrar un estado
válido. La config tiene SHA256
`bc3fe3daabb8f205b1dda81f6da442c2d7452253330960d0c3ff65af7795ba28`.
El runner exige Git limpio, cache firmado 2.048/2.048, 9/3 secuencias disjuntas,
344.591 parámetros y fuentes test selladas. A1 no llama a `weak_box_masks`.

Resultado A1: best epoch 18/18, MiD global `346.1117485`, macro `346.8294571`,
failure `9.9609375%`; summary identity
`b8eca64e1f4c89fd224fd95031ab9bb8271b4d7c4189311238cde85a893026c3`.
Predicciones SHA256 `70b30028e24124e7015ccbd39abc61523518fedf193fe073bfb657c9cb4f30d7`;
checkpoint SHA256 `29ed410b39372e67cac87e5fb0e4be2b659f1a923ea1ebfff3f49e364e139e43`.

Regeneración de la comparación firmada A1/Garl (el label explícito evita
confundir A1 con A0):

```powershell
uv run python scripts/build_causal_scale_eap_garl_comparison.py `
  --causal-predictions artifacts/runs/causal_scale_eap_screen_a1_geometry_v1_seed7/validation_predictions.csv `
  --causal-summary artifacts/runs/causal_scale_eap_screen_a1_geometry_v1_seed7/summary.json `
  --release-predictions artifacts/runs/garl_official_event_only_same2048/predictions.parquet `
  --release-metrics artifacts/runs/garl_official_event_only_same2048/metrics.json `
  --matched-predictions artifacts/runs/garl_matched_event_only_cached_seed7/validation_predictions.parquet `
  --matched-summary artifacts/runs/garl_matched_event_only_cached_seed7/summary.json `
  --subset-data artifacts/subsets/garl_validation_common_roi_v1/data.parquet `
  --subset-labels artifacts/subsets/garl_validation_common_roi_v1/labels.parquet `
  --subset-manifest artifacts/subsets/garl_validation_common_roi_v1/manifest.json `
  --official-train-assets E:\Garl-TTC\configs\splits\train.txt `
  --official-train-labels E:\GarlTTC_dataset\annotations\train.parquet `
  --official-config E:\Garl-TTC\configs\ablation\event_lhr.yaml `
  --official-checkpoint E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth `
  --output-json artifacts/metrics/causal_scale_eap_garl_event_only_a1_geometry_comparison_v1.json `
  --outliers-csv artifacts/metrics/causal_scale_eap_garl_event_only_a1_geometry_top10pct_outliers_v1.csv `
  --bootstrap-iterations 10000 --bootstrap-seed 7 `
  --candidate-label causal_scale_a1_geometry
```

La salida verifica 2.048 tokens/targets/secuencias exactos y tiene identidad
`471fa106f4137f71ecfa4165abec696e5f83644830ded14a82abff8fb7ba485d`.

Auditoría A1 geometry/observability por endpoint (siempre GPU):

```powershell
uv run python scripts/analyze_causal_scale_eap_geometry_observability.py `
  --checkpoint artifacts/runs/causal_scale_eap_screen_a1_geometry_v1_seed7/model_best.pt `
  --cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --summary artifacts/runs/causal_scale_eap_screen_a1_geometry_v1_seed7/summary.json `
  --output-json artifacts/metrics/causal_scale_eap_a1_geometry_observability_v1.json `
  --device cuda --batch-size 64
```

Resultado: 2.048 filas, tres secuencias y 108 tracks, `12.678 s`, 161,54
muestras/s y `934.62 MiB` peak VRAM. Identidad firmada
`737a3663c13dc083b918e0101f4954bcfc22b23257255e0d183f8e09f0aa635d`.

Preregistro y ejecución A1-FR:

```powershell
uv run python scripts/train_causal_scale_eap_screen.py `
  --config configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a1_fullres_v1.yaml `
  --output-dir artifacts/runs/causal_scale_eap_screen_a1_fullres_v1_seed7 `
  --device cuda
```

Si existe `state/last.pt`, añadir `--resume`. No borrar un run compatible. Config
SHA256 `7ceb114963e8aad8f4c7edeb70344759543d3ac58abc6a47b862d3acf772c42e`;
model SHA256 `97232184d7fb00520136319f5e902c726e26766ddaae236459b6d42d9596d39a`.

Regeneración del diagnóstico:

```powershell
uv run python scripts/analyze_causal_scale_eap_failure.py `
  --checkpoint artifacts/runs/causal_scale_eap_screen_v1_seed7/model_best.pt `
  --cache-manifest artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json `
  --summary artifacts/runs/causal_scale_eap_screen_v1_seed7/summary.json `
  --output-json artifacts/metrics/causal_scale_eap_a0_failure_decomposition_v1.json `
  --device cuda --batch-size 32
```

El `--dataset-root` del evaluador debe ser `E:\eAP_dataset`, que resuelve
`events_path`; `E:\GarlTTC_dataset` contiene los parquets, no los medios.

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

`tests/integration/test_causal_scale_eap_resume.py` prueba equivalencia exacta de
resume y rechazo de contratos distintos. El output esperado es `summary.json`,
`validation_predictions.csv`, `model_best.pt` y
`state/{last,best}.pt`. El baseline oficial debe recibir exactamente los sample
tokens guardados en el CSV; no se permiten filtros posteriores por error.

`scripts/build_garl_validation_subset_from_predictions.py` materializa ese conjunto
exacto de manera atómica y firma fuentes, outputs, secuencias, counts, buckets y hash
canónico de tokens. Falla ante duplicados, ausencia, join desigual, target diferente o
roundtrip que cambie el orden.
