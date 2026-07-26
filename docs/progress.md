# Progress Log

## 2026-06-30

Read `README_FIRST.md`, `DATASETS.md`, and `AGENTS.md`.

Decisions:

- Follow the milestone order from `AGENTS.md`.
- Start with M0/M1 and the data-facing part of M2/M3.
- Do not train or report performance metrics until the loader, splits, and representations are
  tested.
- Treat the local EvTTC data as a mini smoke subset, not as a final evaluation corpus.
- Ignore `.bag` and `.mp4` files for the first MVP; use HDF5, `ttc.csv`, and labels.

Known limitations:

- The `ttc.csv` column names are inferred from local rows and must be confirmed with official
  documentation before final experiments.
- The local subset has only one scenario family, so the default split is by full sequence rather
  than by stronger cross-family protocol.
- JEPA, uncertainty, export, and demo milestones remain pending; supervised TinyCNN is implemented as a local baseline.
- On this Windows path, editable installs create a `.pth` file with the non-ASCII user path encoded
  incompatibly for CPython 3.11. Use `uv sync --all-groups --no-editable` and
  `uv run --no-sync ...`.

Verification:

```text
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

Result: 11 tests passed.

Git:

```text
df4ecae chore: bootstrap project and data pipeline
```

Dataset validation:

- `data/manifests/evttc_local.yaml` generated from the local EvTTC mini subset.
- `data/splits/evttc_local.yaml` generated with full-sequence train/validation/test split.
- Real HDF5 layout validated as `prophesee/event_cam_left/{x,y,t,p,ms_map_idx}`.
- Real 100 ms window read from `CCRs-1-low-100-overlap-100`: 362,062 events, resolution 1280x720.

Baseline:

```text
uv run --no-sync e-jepa-ttc baseline trivial --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/trivial_baseline.json
```

Output is generated under ignored `artifacts/metrics/`. It is a sanity baseline only, not a model
result or project claim.

Implemented script wrappers:

```text
scripts/scan_evttc_manifest.py
scripts/validate_dataset.py
scripts/build_index.py
scripts/make_splits.py
scripts/train_baseline.py
```

## 2026-06-30 Training Pass

Additional commits implemented geometric/event-rate baselines, voxel cache generation, CUDA TinyCNN
training, raw-density metadata channels, and local result reporting.

Verification after the implemented code changes:

```text
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Result: 13 tests passed.

GPU run environment:

- PyTorch `2.11.0+cu128` in `.venv`.
- CUDA available on `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
- Full cache used 1230 indexed windows at `160x90`, `bins=5`.

Local conclusion:

- Geometric bbox expansion is strongest on labeled frames, but it uses object labels and is not a
  pure event-stream protocol.
- On indexed event windows, event-rate ridge is the strongest robust held-out test result.
- TinyCNN raw+metadata beats normalized voxels and can win validation for one seed, but five-seed
  test mean remains behind event-rate and variance is high.
- With one training sequence, local data is insufficient for a robust learned representation claim;
  the next project step should be JEPA/self-supervised pretraining or more EvTTC sequences.

See `docs/local_results.md`.

## 2026-06-30 JEPA Pass

Implemented and committed JEPA-style self-supervised pretraining:

- `TinyCNNEncoder` factored out of the supervised regressor.
- `e-jepa-ttc pretrain jepa` added with online encoder, EMA target encoder, latent predictor,
  masked context views, and variance regularization to avoid collapsed embeddings.
- `train tiny-cnn --pretrained-encoder` added for supervised fine-tuning from JEPA checkpoints.
- Smoke tests cover encoder shapes and loading a JEPA checkpoint into the supervised trainer.

Executed GPU runs on the raw+metadata `160x90`, `bins=5` cache:

- Train-only JEPA pretraining: 160 epochs, best epoch 109, best latent loss 0.0145.
- Train-only JEPA fine-tuning: validation MAE 3.297 s, test MAE 3.290 s.
- All-splits diagnostic JEPA pretraining: 160 epochs, best epoch 158, best latent loss 0.0114.
- All-splits diagnostic fine-tuning: validation MAE 3.598 s, test MAE 3.351 s.

Conclusion: JEPA runs end to end, but in this mini subset it does not improve TTC over the
non-pretrained TinyCNN seed 7 result. The all-splits diagnostic also underperforms, so the issue is
not only lack of unlabeled validation/test windows; the next useful step is larger multi-sequence
data and a stronger temporal/multi-horizon JEPA objective.

## 2026-07-01 Causal Geometry Audit

Found and documented a lookahead issue in the original centered geometric baseline: its derivative
window used future bounding boxes. That result is now treated as diagnostic only.

Implemented `baseline causal-geometry`, a detection-assisted TTC estimator that:

- uses only current and previous object boxes for apparent expansion;
- fits a two-parameter log-affine calibration using train labels only;
- reports explicit leakage audit flags in `artifacts/metrics/causal_geometry_baseline.json`;
- keeps the protocol separate from event-only models.

Run:

```text
.\.venv\Scripts\python.exe -m e_jepa_ttc baseline causal-geometry --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/causal_geometry_baseline.json --derivative-window 15
```

Result on labeled frames:

- validation MAE 0.439 s, RMSE 0.595 s, n=143;
- test MAE 0.188 s, RMSE 0.359 s, n=96.

This is the first excellent/promising local result without lookahead bias, but it is
detection-assisted, not event-only.

## 2026-07-01 Temporal Multi-Horizon JEPA

Replaced the first masked same-window JEPA objective with a temporal multi-horizon objective:

- context windows predict future target-encoder embeddings at 20, 60, 100, 240, and 500 ms;
- temporal pairs are matched only within the same sequence and selected split;
- the pretraining summary records `uses_ttc_labels=false` and no cross-sequence/split targets;
- the supervised trainer now supports frozen-encoder probes and deterministic low-label subsets.

Train-only SSL run:

```text
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7 --epochs 160 --batch-size 64 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05
```

Result:

- best temporal JEPA epoch 30, best validation latent loss 0.00544;
- full-label fine-tune seed 7: validation MAE 1.518 s, test MAE 3.183 s;
- frozen probe seed 7: validation MAE 1.916 s, test MAE 2.911 s;
- 5% labels, three seeds: scratch validation MAE 2.909 +/- 0.743 s, temporal JEPA
  validation MAE 1.548 +/- 0.176 s;
- 5% labels, three seeds: scratch test MAE 3.107 +/- 0.277 s, temporal JEPA test
  MAE 2.986 +/- 0.106 s.

Conclusion:

- This is the first positive self-supervised JEPA result in the repo, especially in low-label
  validation.
- The high-speed mini test has been inspected repeatedly, so it is no longer a sealed test. Treat
  the test numbers as exploratory and use a fresh EvTTC starter protocol for any final claim.

## 2026-07-01 EvTTC Starter Download Plan

Added `data/manifests/evttc_starter_downloads.yaml` for the six starter sequences missing from the
local handoff: CCRs-side low/medium/high and CPLA low/medium/high. The manifest records public
official Google Drive links for hdf5, gt-ttc, and bbox-segmentation assets.

Added `scripts/download_evttc_starter.py`, which prints a dry-run download plan by default and only
executes `gdown` when `--execute` is passed. Verified dry-run output contains 18 planned actions.
Installed `gdown` in the local `.venv` and verified public access by downloading all six small
`gt_ttc` assets as `ttc.csv` under ignored `datasets/evttc/...` directories. HDF5 and
bbox-segmentation assets are still pending because they are the large downloads.

The next sealed protocol should download these assets, rescan `datasets/evttc`, create a new starter
manifest/split, and avoid tuning on the new test split.

Downloaded `CCRs-side-low` HDF5 successfully and validated that the scanner detects it. Attempted
the bbox-segmentation folder, but `gdown --folder` failed after 78 JSON files because one Drive file
could not expose a public download URL. Attempts to fetch additional HDF5 files also hit gdown/Drive
public-link limits. Treat the local starter state as partial.

Created:

- `data/manifests/evttc_partial_starter.yaml`;
- `data/splits/evttc_partial_starter.yaml`;
- ignored `data/cache/evttc_partial_starter_index.json`;
- ignored `artifacts/features/evttc_partial_starter_voxel_160x90_b5_raw_meta.npz`.

Partial split:

- train: `CCRs-1-low-100-overlap-100`, `CCRs-side-low`;
- validation: `CCRs-1-medium-100-overlap-100`;
- test: `CCRs-1-high-100-overlap-100`.

Exploratory partial results:

- Event-rate ridge: validation MAE 2.816 s, test MAE 2.970 s.
- TinyCNN scratch full-label: validation MAE 3.047 s, test MAE 2.798 s.
- Temporal JEPA partial full-label, best validation LR: validation MAE 2.468 s, test MAE 2.802 s.
- TinyCNN scratch 5% labels: validation MAE 1.299 s, test MAE 3.091 s.
- Temporal JEPA partial 5% labels, best validation LR: validation MAE 1.522 s, test MAE 2.939 s.
- Temporal JEPA partial 5% labels at LR 3e-4 gives test MAE 2.489 s, but validation MAE 2.040 s;
  this is diagnostic only and not a model one would select by validation.

## 2026-07-01 Dense Motion JEPA And Available Starter

Implemented dense temporal token JEPA with causal motion conditioning:

- `TinyCNNEncoder.forward_tokens()` exposes dense spatial tokens before global pooling while keeping
  supervised checkpoint compatibility.
- `pretrain jepa` now defaults to `dense_temporal_token_motion_multihorizon` for temporal runs.
- The dense predictor is horizon-conditioned and receives context-only motion proxy features:
  event mass, temporal mass slope, centroid shift, and polarity balance.
- `--global-latent` preserves the older pooled temporal JEPA objective.
- `--no-motion-conditioning` disables the motion proxy path for ablations.

Completed additional HDF5 downloads and validation:

- `CCRs-side-medium/data.hdf5`
- `CCRs-side-high/data.hdf5`
- `CPLA-low/data.hdf5`

Google Drive quota blocked:

- `CPLA-medium/data.hdf5`
- `CPLA-high/data.hdf5`

Created and committed:

- `data/manifests/evttc_available_starter_local.yaml`
- `data/splits/evttc_available_starter_sealed.yaml`
- `docs/available_starter_results.md`

The available sealed split has 3030 windows: train 1957, validation 598, test 475.
The sealed test is the newly added `CCRs-side-high` sequence.

Key results:

- Event-rate ridge: validation MAE 3.103 s, sealed-test MAE 2.406 s.
- TinyCNN scratch full-label seed 7: validation MAE 1.171 s, sealed-test MAE 0.519 s.
- Dense motion JEPA full-label seed 7: validation MAE 1.223 s, sealed-test MAE 0.945 s.
- Dense motion JEPA frozen probe seed 7: validation MAE 1.002 s, sealed-test MAE 1.197 s.
- 5% labels, three seeds: scratch validation/test 2.413 +/- 0.480 / 2.241 +/- 0.350 s;
  dense JEPA validation/test 1.694 +/- 0.174 / 1.149 +/- 0.119 s.
- 10% labels, three seeds: scratch validation/test 1.814 +/- 0.152 / 1.400 +/- 0.035 s;
  dense JEPA validation/test 1.420 +/- 0.075 / 1.111 +/- 0.370 s.

Conclusion:

- Dense motion JEPA is now clearly useful for low-label TTC estimation on a newly sealed test:
  5% labels improves validation MAE by 29.8% and sealed-test MAE by 48.7%; 10% labels improves
  validation MAE by 21.7% and sealed-test MAE by 20.6%.
- With 100% labels, supervised scratch remains best on sealed test.
- The full-starter protocol is still pending only because Drive blocked the remaining two CPLA HDF5
  files; the committed full-starter split should be used once they are available.

## 2026-07-01 Full Starter Token JEPA

After `CPLA-medium/data.hdf5` and `CPLA-high/data.hdf5` were added locally, the full starter
protocol was scanned, validated, indexed, cached, and run end to end.

Created and committed:

- `data/manifests/evttc_full_starter_local.yaml`
- `docs/full_starter_results.md`

The full sealed split has 3972 windows: train 3019, validation 475, test 478.
Validation is `CCRs-side-high`; the sealed test is `CPLA-high`.

Implemented and used a token-transformer backbone for JEPA and supervised TTC:

- `EventTokenTransformerEncoder`
- `EventTokenTransformerRegressor`
- `--model token-transformer` for `pretrain jepa` and `train tiny-cnn`

Token JEPA pretraining:

- objective: `dense_temporal_token_motion_multihorizon`
- train split only, validation split for SSL checkpoint selection
- best epoch 56, best SSL loss 0.003049
- leakage audit: no TTC labels, target timestamps after context, no sequence/split crossing,
  motion conditioning uses context only

Key full-starter results:

- Event-rate ridge: validation MAE 2.303 s, sealed-test MAE 2.489 s.
- TinyCNN scratch full-label seed 7: validation MAE 0.549 s, sealed-test MAE 0.513 s.
- Token transformer scratch full-label, seeds 7/13/21: validation MAE 0.702 +/- 0.052 s,
  sealed-test MAE 0.844 +/- 0.008 s.
- Token JEPA full-label, seeds 7/13/21: validation MAE 0.358 +/- 0.007 s, sealed-test MAE
  0.481 +/- 0.042 s. Best single seed: validation MAE 0.350 s, sealed-test MAE 0.422 s.
- 5% labels, three seeds: token scratch validation/test 1.226 +/- 0.031 / 1.382 +/- 0.044 s;
  token JEPA validation/test 0.524 +/- 0.047 / 0.636 +/- 0.109 s.
- 10% labels, three seeds: token scratch validation/test 1.178 +/- 0.056 / 1.327 +/- 0.104 s;
  token JEPA validation/test 0.437 +/- 0.039 / 0.460 +/- 0.029 s.

Conclusion:

- Token JEPA is now the best local full-starter learned result.
- Against the matching scratch token backbone, JEPA improves 100% label sealed-test MAE by 43.0%,
  5% label sealed-test MAE by 53.9%, and 10% label sealed-test MAE by 65.4%.
- Against TinyCNN scratch with 100% labels, three-seed token JEPA improves sealed-test MAE by 6.2%;
  the best token JEPA seed improves sealed-test MAE by 17.8%.
- This is a strong sealed starter result, but not an official SOTA claim because it has not been
  compared on a published leaderboard or reproduced against the exact Event-Aided TTC baseline
  protocol.

## 2026-07-01 Deep Token JEPA Ablations

Implemented deep token supervision for the token-transformer JEPA path:

- `EventTokenTransformerEncoder.forward_intermediate_tokens()` exposes selected transformer layer
  token grids.
- `pretrain jepa --deep-supervision-layers ...` predicts dense future tokens for multiple selected
  layers.
- A second pass added layer-id conditioning to the dense predictor so intermediate and final layer
  targets are not conflated.

Verification:

```text
.\.venv\Scripts\python.exe -m pytest tests\unit\test_models.py tests\unit\test_jepa_training.py
.\.venv\Scripts\python.exe -m ruff check src\e_jepa_ttc\models\token_transformer.py src\e_jepa_ttc\training\jepa.py src\e_jepa_ttc\cli.py tests\unit\test_models.py tests\unit\test_jepa_training.py
```

Result: 4 tests passed; lint clean.

Full-starter ablations, seed 7:

- Base final-layer Token JEPA: SSL best loss 0.003049; validation MAE 0.350 s; sealed-test MAE
  0.422 s.
- Deep Token JEPA, layers 1 and 3: SSL best loss 0.003523; validation MAE 0.491 s; sealed-test MAE
  0.594 s.
- Deep layer-aware Token JEPA, layers 1 and 3: SSL best loss 0.003933; validation MAE 0.472 s;
  sealed-test MAE 0.505 s.

Conclusion:

- Deep self-supervision is now implemented and audited, but it is not the best current model.
- The layer-aware predictor recovers much of the deep-supervision penalty and nearly matches
  TinyCNN scratch on sealed test, but it remains behind final-layer Token JEPA.
- The likely next SOTA-alignment step is not simply "more intermediate layers"; it should be a
  stronger layer-specific predictor, larger token backbone, richer motion/action conditioning, or
  direct reproduction of the official Event-Aided TTC benchmark baselines.

## 2026-07-01 Large Token Transformer Ablation

Added `token-transformer-large`:

- embedding dim 256 instead of 192;
- 6 transformer layers instead of 4;
- 8 attention heads instead of 6.

Full-starter seed 7:

- Large Token JEPA pretraining: best SSL epoch 6, best SSL loss 0.003237.
- Large Token JEPA fine-tune: validation MAE 0.504 s, sealed-test MAE 0.529 s.
- Base Token JEPA remains better: validation MAE 0.350 s, sealed-test MAE 0.422 s.

Conclusion:

- Scaling the token backbone alone did not improve TTC on the full-starter split.
- The likely issue is data/objective balance: the larger encoder overfits SSL validation after early
  epochs and does not transfer better to supervised TTC.
- Keep `token-transformer-large` as an available ablation/model variant, but do not promote it as
  the default result.

## 2026-07-01 Causal Navigation Conditioning

Implemented optional causal integrated-navigation channels in the voxel cache:

- `ego_speed`
- `ego_velocity_x/y/z`
- `ego_acceleration_x/y/z`
- `ego_yaw_rate`
- `ego_navigation_valid`

The features are read only from the context window `[context_start_us, context_end_us]`, so they do
not use future target windows or TTC labels. Synthetic fixtures and files without
`integratedNavigation/data` receive zero-filled navigation channels.

Created ignored generated cache:

- `artifacts/features/evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz`

Cache summary:

- shape `[3972, 21, 90, 160]`
- train 3019, validation 475, test 478
- all full-starter windows have valid navigation channels

Full-starter 100% label results, seeds 7/13/21:

- Event-only token scratch: validation/test `0.702 +/- 0.052 / 0.844 +/- 0.008 s`.
- Event-only Token JEPA: validation/test `0.358 +/- 0.007 / 0.481 +/- 0.042 s`.
- Navigation token scratch: validation/test `0.440 +/- 0.020 / 0.465 +/- 0.021 s`.
- Navigation Token JEPA: validation/test `0.261 +/- 0.021 / 0.356 +/- 0.022 s`.

Full-starter 10% label results, seeds 7/13/21:

- Event-only token scratch: validation/test `1.178 +/- 0.056 / 1.327 +/- 0.104 s`.
- Event-only Token JEPA: validation/test `0.437 +/- 0.039 / 0.460 +/- 0.029 s`.
- Navigation token scratch: validation/test `0.578 +/- 0.026 / 0.756 +/- 0.018 s`.
- Navigation Token JEPA: validation/test `0.406 +/- 0.019 / 0.543 +/- 0.010 s`.

Conclusion:

- Causal navigation channels are the strongest improvement so far.
- Navigation alone improves token scratch sealed-test MAE by 44.9% versus event-only scratch.
- Navigation Token JEPA improves sealed-test MAE by 25.9% versus event-only Token JEPA and by
  23.3% versus navigation scratch.
- The new best local result is `0.356 +/- 0.022 s` sealed-test MAE over three seeds.
- At 10% labels, navigation helps scratch but not JEPA transfer; event-only Token JEPA remains the
  better low-label model on sealed test.

## 2026-07-01 Full Starter BBox-Assisted Reference

Recovered the missing starter `bbox_segmentation` folders for
benchmark-aligned detection-assisted baselines. `gdown --folder` listed the
folders but failed to resolve public per-file URLs. The working path was:

- save the folder listing with `gdown --folder --json`;
- download the listed JSON files through `scripts/download_gdown_listing.py`,
  which rewrites Drive `uc?id=...` links to `drive.usercontent.google.com`.

Recovered bbox JSON counts:

- `CCRs-side-low`: 149
- `CCRs-side-medium`: 141
- `CCRs-side-high`: 108
- `CPLA-low`: 152
- `CPLA-medium`: 89
- `CPLA-high`: 87

Implemented scanner support for `bbox_segmentation` as an ISAT label directory
fallback when `leftlabel` is absent, and updated target-type inference so `CP*`
families are marked as `pedestrian`.

Using a generated scan manifest with all recovered bbox labels:

```text
$env:PYTHONPATH=(Resolve-Path src).Path
.\.venv\Scripts\e-jepa-ttc.exe baseline causal-geometry --manifest artifacts\metrics\evttc_scan_full_bbox.yaml --split data\splits\evttc_full_starter_sealed.yaml --output artifacts\metrics\causal_geometry_full_starter_full_bbox.json --derivative-window 15
```

Result:

- train bbox frames: 871 labels, 856 predictions, MAE 0.512 s, RMSE 1.007 s;
- validation bbox frames: 108 labels, 106 predictions, MAE 0.279 s, RMSE 0.538 s;
- sealed `CPLA-high` bbox frames: 83 TTC-matched labels, 81 predictions, MAE 0.157 s,
  RMSE 0.331 s.

Interpretation:

- This is detection-assisted and frame-label-only, not event-only and not all-window evaluation.
- It is useful evidence that official CMax/STRTTC-style comparisons need complete bbox assets and
  benchmark-aligned frame protocols.
- It is not a replacement for the all-window Token JEPA + navigation result.

## 2026-07-01 Tubelet/V-JEPA-Like Backbone

Added `event-tubelet-transformer` and `event-tubelet-transformer-large` model
names for JEPA pretraining and supervised fine-tuning.

Design:

- first `2 * bins` channels are treated as positive/negative event bins;
- event bins are embedded with a 3D tubelet convolution over
  polarity-by-time-by-space tensors;
- remaining metadata/navigation channels are embedded as causal auxiliary
  spatial patches and added to each temporal tubelet;
- dense spatio-temporal tokens support the existing JEPA future-token objective
  and intermediate-layer supervision;
- default full-starter token count is 250 tokens at 90x160 with 5 event bins and
  `patch_size=16`, keeping it feasible on the 12 GB GPU.

SOTA alignment:

- V-JEPA 2 motivates actionless pretraining followed by action-conditioned
  predictor training.
- LeWorldModel motivates latent next-state prediction conditioned on actions and
  explicit anti-collapse regularization.
- For EvTTC, integrated navigation is treated as causal ego-action context, not
  a label. Future events remain SSL targets only, and TTC labels remain reserved
  for fine-tuning.

Next empirical step:

```text
.\.venv\Scripts\e-jepa-ttc.exe pretrain jepa --cache artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz --output-dir artifacts\runs\jepa_event_tubelet_nav_full_starter_seed7 --epochs 120 --batch-size 24 --learning-rate 0.0003 --seed 7 --device auto --model event-tubelet-transformer --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05 --deep-supervision-layers 1 5
```

Executed a 30-epoch diagnostic run instead of immediately committing to 120
epochs. Validation selected epoch 12, so longer pretraining was already starting
to overfit the SSL validation split:

- SSL best epoch: 12/30
- SSL best validation loss: 0.001577
- previous token-transformer SSL best loss: 0.003049
- relative SSL-loss reduction: 48.3%

Fine-tuned seeds 7/13/21 from the best tubelet checkpoint:

- validation MAE: `0.243 +/- 0.007 s`
- sealed-test MAE: `0.328 +/- 0.030 s`
- per-seed sealed-test MAE: seed 7 `0.323 s`, seed 13 `0.301 s`, seed 21 `0.360 s`

Percent interpretation:

- 7.9% less sealed-test error than the previous three-seed navigation Token
  JEPA mean (`0.356 s` to `0.328 s`);
- 29.5% less sealed-test error than navigation token scratch (`0.465 s` to
  `0.328 s`);
- 31.8% less sealed-test error than event-only Token JEPA (`0.481 s` to
  `0.328 s`);
- 61.1% less sealed-test error than event-only token scratch (`0.844 s` to
  `0.328 s`);
- 36.1% less sealed-test error than TinyCNN scratch seed 7 (`0.513 s` to
  `0.328 s`);
- 86.8% less sealed-test error than the event-rate baseline (`2.489 s` to
  `0.328 s`).

This is now the best robust all-window result in the local full-starter sealed
protocol. It is still not an official EvTTC SOTA claim because the official
comparisons use a bbox/ROI frame protocol.

## 2026-07-02 Action-Conditioned JEPA Predictor

Implemented the LeWorldModel/V-JEPA-2-AC-aligned predictor step for dense
temporal JEPA. When the cache contains navigation channels, the predictor now
receives a causal action vector built from:

- 6 event-motion context features: total mass, late-minus-early mass, temporal
  slope, centroid dx/dy, and polarity balance;
- 9 integrated-navigation features from the current context window: ego speed,
  velocity, acceleration, yaw-rate, and validity.

The action-conditioned objective is recorded as
`dense_temporal_token_action_multihorizon`. Checkpoints and metrics now include
`action_feature_dim`, `action_feature_names`,
`uses_navigation_action_conditioning`, and explicit leakage audit fields:

- `action_conditioning_uses_context_only=true`
- `uses_future_navigation=false`
- `uses_ttc_labels=false`

Added a unit fixture with metadata plus navigation channels to verify the 15-D
causal action vector and the anti-leakage metadata. The existing no-navigation
JEPA path remains backward compatible and still reports the motion-conditioned
objective.

Real-cache smoke, train/validation only:

```text
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa --cache artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz --output-dir artifacts\runs\jepa_event_tubelet_action_nav_full_starter_smoke_2e --epochs 2 --batch-size 24 --learning-rate 0.0003 --seed 7 --device auto --model event-tubelet-transformer --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05 --deep-supervision-layers 1 5
```

Smoke result: objective `deep_dense_temporal_token_action_multihorizon`, action
feature dimension 15, validation SSL loss `0.003594` at epoch 2, and
`uses_future_navigation=false`. This is only an implementation check, not a TTC
MAE result or a model-selection claim.

Also added `docs/evttc_official_bbox_roi_protocol.md` to pin down the official
bbox/ROI comparison requirements against STRTTC, CMax, ETTCM, FAITH,
AEB-Tracker, and Image FoE. The current conclusion remains strict: no SOTA/SOTSA
claim is valid until those baselines are reproduced on the same official
bbox/ROI protocol. All tuning continues to be validation-only, with sealed test
reserved for frozen protocols.

## 2026-07-02 Validation-Only Fine-Tuning Gate

Added `--evaluation-splits` to supervised training. The default remains
`train validation test` for final frozen reports, but tuning runs can now pass:

```text
--evaluation-splits train validation
```

In that mode, the trainer does not evaluate `test` and does not write
`test_pred`/`test_true` arrays. This lets action-conditioned JEPA fine-tuning
iterate on validation only, preserving the sealed test for a final protocol run.

## 2026-07-02 Action-Conditioned Tubelet Validation Ablation

Ran a 30-epoch action-conditioned `event-tubelet-transformer` JEPA pretrain on
the full starter `raw_meta_nav` cache:

- run: `artifacts/runs/jepa_event_tubelet_action_nav_full_starter_seed7_30e`
- objective: `deep_dense_temporal_token_action_multihorizon`
- action features: 15 total, 6 event-motion plus 9 causal navigation features
- best SSL validation epoch: 14
- best SSL validation loss: `0.0017955`
- leakage audit: no TTC labels and `uses_future_navigation=false`

Fine-tuned the same SSL checkpoint with seeds 7/13/21 using
`--evaluation-splits train validation`, so no test metrics or test predictions
were produced:

| Run | Seed | Best epoch | Validation MAE |
| --- | ---: | ---: | ---: |
| Action-conditioned tubelet JEPA, validation-only | 7 | 28 | 0.236276 s |
| Action-conditioned tubelet JEPA, validation-only | 13 | 55 | 0.247274 s |
| Action-conditioned tubelet JEPA, validation-only | 21 | 29 | 0.258325 s |

Mean validation MAE: `0.247292 +/- 0.009001 s` across the three fine-tuning
seeds.

Validation comparison against the previous best tubelet navigation JEPA:

- previous tubelet navigation JEPA validation mean: `0.242929 +/- 0.005325 s`;
- action-conditioned explicit predictor validation mean: `0.247292 +/- 0.009001 s`;
- action-conditioned explicit predictor is 1.8% worse by validation MAE.

Interpretation: the action-conditioned predictor is architecturally closer to
LeWorldModel/V-JEPA action-conditioned latent prediction, and seed 7 improved
slightly, but the three-seed validation result does not beat the previous local
best. Therefore the sealed test remains unopened for this ablation, and the
current empirical best remains the earlier event-tubelet navigation JEPA result.

## 2026-07-02 Train-Only Action Normalization And Frozen Test Check

The explicit action vector was then normalized with statistics estimated only
from SSL train context windows. This is implemented in `pretrain_jepa` and
recorded in checkpoints as:

- `action_feature_normalization=true`
- `action_feature_normalization_source=pretrain_context_indices_train_only`
- `leakage_audit.action_feature_normalization_uses_train_only=true`

Normalized action JEPA pretrain:

- run: `artifacts/runs/jepa_event_tubelet_actionnorm_nav_full_starter_seed7_30e`
- best SSL validation epoch: 29
- best SSL validation loss: `0.0018199`

Fine-tuning was first done validation-only. LR `3e-4` improved seeds 7/13 but
remained noisy; LR `1e-4` was then selected on validation and frozen before the
sealed test check.

Validation-only MAE for the frozen LR `1e-4` protocol:

| Run | Seed | Best epoch | Validation MAE |
| --- | ---: | ---: | ---: |
| Action-normalized tubelet JEPA, validation-only | 7 | 27 | 0.227108 s |
| Action-normalized tubelet JEPA, validation-only | 13 | 8 | 0.217966 s |
| Action-normalized tubelet JEPA, validation-only | 21 | 39 | 0.229651 s |

Mean validation MAE: `0.224908 +/- 0.005018 s`, a 7.42% validation improvement
over the previous tubelet navigation JEPA validation mean
`0.242929 +/- 0.005325 s`.

Because the protocol was selected by validation and then frozen, the sealed test
was opened once with `train evaluate`, without retraining:

| Run | Seed | Test MAE |
| --- | ---: | ---: |
| Action-normalized tubelet JEPA frozen test | 7 | 0.369874 s |
| Action-normalized tubelet JEPA frozen test | 13 | 0.420796 s |
| Action-normalized tubelet JEPA frozen test | 21 | 0.331366 s |

Mean sealed-test MAE: `0.374012 +/- 0.036626 s`. This is 14.06% worse than the
previous best event-tubelet navigation JEPA sealed-test mean
`0.327904 +/- 0.024231 s`.

Interpretation:

- The normalized action predictor is a validation improvement but not a sealed
  test improvement.
- The current validation split (`CCRs-side-high`) is not fully predictive of the
  sealed pedestrian test (`CPLA-high`).
- Do not tune further on `CPLA-high`; the current best local sealed result
  remains the earlier event-tubelet navigation JEPA result.
- A real next step needs a stronger validation/test design or more official
  sequences, not additional tuning against the now-opened sealed result.

## 2026-07-02 Multi-Domain Validation Split

Added a dev split that does not use `CPLA-high` for selection and exposes the
car-to-pedestrian transfer problem before opening test:

- split: `data/splits/evttc_full_starter_dev_multival.yaml`
- train: CCRs-1 low/medium/high, CCRs-side low/medium, CPLA-low
- `validation_car`: CCRs-side-high
- `validation_pedestrian`: CPLA-medium
- test: CPLA-high

Added `cache remap-splits` so the existing full-starter tensor cache can be
copied with new split labels without rereading HDF5:

```text
.\.venv\Scripts\python.exe -m e_jepa_ttc cache remap-splits --cache artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz --split data\splits\evttc_full_starter_dev_multival.yaml --output artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav_dev_multival.npz
```

Resulting split counts:

- train: 2555 windows
- validation_car: 475 windows
- validation_pedestrian: 464 windows
- test: 478 windows

Also generalized supervised training/evaluation to accept arbitrary split names
and separate `--train-splits`, `--validation-splits`, and
`--evaluation-splits`.

Dev multi-validation JEPA pretrain:

- run: `artifacts/runs/jepa_event_tubelet_actionnorm_nav_dev_multival_seed7_30e`
- pretrain split: `train`
- SSL validation splits: `validation_car validation_pedestrian`
- best SSL epoch: 3
- best SSL validation loss: `0.001549`

Fine-tune seed 7 with LR `1e-4`, selected on the combined car+pedestrian
validation set:

| Method | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| --- | ---: | ---: | ---: |
| Scratch event-tubelet | 0.619 s | 0.420 s | 0.822 s |
| Action-normalized JEPA | 0.613 s | 0.402 s | 0.828 s |

Interpretation:

- JEPA is only about 1% better than scratch under this harder dev split.
- The previous single-validation split hid a real transfer gap: when CPLA-medium
  is removed from train and used as pedestrian validation, MAE jumps above
  `0.8 s`.
- Further CPLA-high tuning is not valid. The next SOTA-oriented work should add
  official bbox/ROI comparison assets or stronger pedestrian supervision/data,
  not more architecture tweaks selected on the already-opened CPLA-high test.

## 2026-07-02 Causal ROI Event Baseline

Implemented `baseline roi-events` as the first bbox/ROI event baseline aligned
with the official CMax/STRTTC input assumption:

- RGB bbox labels now retain source image dimensions, so boxes are scaled from
  `1920x1200` into the event plane (`1280x720`) before ROI extraction.
- Event features use only the causal window
  `[timestamp - context_ms, timestamp]`; tests verify future events are ignored.
- The regressor is train-only standardized ridge on log TTC. Validation/test TTC
  labels are not used for fit or feature normalization.
- Added `--evaluation-splits` so diagnostic runs can skip sealed test entirely.

Fixed full-starter run (`context_ms=100`, `ridge_alpha=1.0`):

| Split | Labels | MAE | Mean relative error |
| --- | ---: | ---: | ---: |
| train | 871 | 0.659 s | 19.15% |
| validation | 108 | 0.293 s | 13.63% |
| sealed CPLA-high test | 83 | 0.829 s | 47.12% |

Dev multi-validation run without evaluating sealed test:

| Split | Labels | MAE | Mean relative error |
| --- | ---: | ---: | ---: |
| validation_car | 108 | 0.293 s | 14.63% |
| validation_pedestrian | 85 | 1.342 s | 65.52% |

Computed percentage-style metrics for the current best all-window JEPA from
stored predictions, without retraining:

- validation mean absolute relative error:
  `7.51 +/- 0.51%`
- sealed test mean absolute relative error:
  `6.89 +/- 0.34%`

Interpretation:

- The current best local result is still event-tubelet JEPA + causal navigation:
  `0.328 +/- 0.024 s` sealed-test MAE and `6.89 +/- 0.34%` relative error.
- ROI event ridge is protocol-useful but not competitive. Its pedestrian
  validation failure is larger than the JEPA transfer gap.
- For official SOTA comparison, next required work is still to download CCRs2 and
  CCRm HDF5/TTC/bbox assets and wrap CMax/STRTTC on the same ROI adapter.

## 2026-07-02 Dense Transformer JEPA Predictor

Implemented `--dense-predictor transformer` for JEPA pretraining. This replaces
the older per-token MLP dense predictor with a transformer predictor over all
dense tokens for each future horizon. The predictor is conditioned on the same
train-normalized causal event/navigation action vector, so it is closer to
V-JEPA 2-AC/LeWorldModel-style latent action-conditioned prediction while
keeping the no-TTC-label SSL protocol.

Unit coverage:

- `tests/unit/test_jepa_training.py::test_dense_transformer_jepa_predictor`
- verifies objective name
  `transformer_dense_temporal_token_motion_multihorizon`
- verifies checkpoint creation and `dense_predictor=transformer` metadata

Dev multi-validation pretrain:

- run:
  `artifacts/runs/jepa_event_tubelet_transformerpred_nav_dev_multival_seed7_30e`
- model: `event-tubelet-transformer`
- predictor: `transformer`
- pretrain split: `train`
- SSL validation splits: `validation_car validation_pedestrian`
- best SSL epoch: 3
- best SSL validation loss: `0.000862`
- last epoch SSL validation loss: `0.001776`

Fine-tune from the SSL-best checkpoint improved pedestrian validation but hurt
car validation, so the SSL-last checkpoint was evaluated on validation only.
Three fine-tune seeds from the SSL-last checkpoint:

| Seed | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| ---: | ---: | ---: | ---: |
| 7 | 0.471 s | 0.312 s | 0.634 s |
| 13 | 0.464 s | 0.320 s | 0.611 s |
| 21 | 0.463 s | 0.397 s | 0.531 s |

Mean dev result:

- weighted multival MAE: `0.466 +/- 0.004 s`
- validation_car MAE: `0.343 +/- 0.038 s`
- validation_pedestrian MAE: `0.592 +/- 0.044 s`
- validation_car relative error: `10.44 +/- 1.75%`
- validation_pedestrian relative error: `18.89 +/- 0.46%`

This improves the previous dev split substantially:

- versus scratch event-tubelet weighted MAE `0.619 s`: 24.7% better
- versus action-normalized JEPA weighted MAE `0.613 s`: 24.0% better
- pedestrian validation improves from `0.828 s` to `0.592 s`

The frozen dev-selected protocol was evaluated once on CPLA-high:

- test MAE: `0.430 +/- 0.027 s`
- test relative error: `8.50 +/- 1.36%`

It did not beat the best previous sealed local result
(`0.328 +/- 0.024 s`, `6.89 +/- 0.34%`).

Full-starter pretrain with CPLA-medium back in train:

- run:
  `artifacts/runs/jepa_event_tubelet_transformerpred_nav_full_starter_seed7_30e`
- best SSL validation epoch: 2
- best SSL validation loss: `0.000966`
- SSL-last epoch validation loss: `0.001535`

The full-starter SSL-last checkpoint was selected based on dev behavior and
fine-tuned on three seeds with evaluation restricted to `train validation`:

| Seed | Validation MAE | Validation relative error |
| ---: | ---: | ---: |
| 7 | 0.242 s | 8.48% |
| 13 | 0.236 s | 8.09% |
| 21 | 0.245 s | 8.26% |

Mean validation result: `0.241 +/- 0.004 s`, slightly better than the previous
best validation MAE `0.243 +/- 0.005 s`. The protocol was frozen and evaluated
once on CPLA-high:

| Seed | Test MAE | Test relative error |
| ---: | ---: | ---: |
| 7 | 0.346 s | 7.98% |
| 13 | 0.349 s | 7.75% |
| 21 | 0.356 s | 6.96% |

Mean sealed-test result: `0.351 +/- 0.004 s`, `7.56 +/- 0.44%`. This is worse
than the current best event-tubelet navigation JEPA result, so the best local
sealed model does not change.

Interpretation:

- Transformer dense prediction is the strongest dev-validation improvement so
  far and is directionally aligned with V-JEPA 2.1/V-JEPA 2-AC.
- The full-starter validation improvement still does not transfer to CPLA-high.
- Do not tune further based on these sealed-test outcomes. The next valid route
  is either official CCRs2/CCRm bbox/ROI comparison data or a validation design
  with more pedestrian diversity before any new frozen test check.

## 2026-07-02 V-JEPA 2.1 All-Token Context Loss

Implemented optional all-token context supervision for dense JEPA:

- CLI flag: `--context-token-weight`
- default: `0.0`, preserving all previous runs
- objective name prefix when active: `alltoken_`
- checkpoint metadata:
  `context_token_loss=true`,
  `context_token_weight=<value>`
- metrics:
  `future_alignment_loss`, `context_token_loss`,
  `context_token_target_count`
- leakage audit:
  `context_token_loss_uses_current_context_only=true`

This adds a same-window EMA target loss for all current context tokens in
addition to future multi-horizon dense prediction. It is inspired by V-JEPA 2.1's
all-token dense supervision, but it still uses only event context and causal
navigation/action features.

Unit coverage:

- `tests/unit/test_jepa_training.py::test_dense_alltoken_jepa_context_loss`
- verifies objective
  `alltoken_transformer_dense_temporal_token_motion_multihorizon`
- verifies context loss metrics and leakage audit

Dev multi-validation pretrain:

- run:
  `artifacts/runs/jepa_event_tubelet_alltoken_transformerpred_nav_dev_multival_w025_seed7_30e`
- model: `event-tubelet-transformer`
- predictor: `transformer`
- `context_token_weight=0.25`
- best SSL validation epoch: 2
- best SSL validation loss: `0.001287`
- last epoch SSL validation loss: `0.002258`

Fine-tune from the SSL-last checkpoint, validation only:

| Seed | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| ---: | ---: | ---: | ---: |
| 7 | 0.530 s | 0.287 s | 0.779 s |
| 13 | 0.638 s | 0.305 s | 0.979 s |
| 21 | 0.644 s | 0.261 s | 1.037 s |

Mean result:

- weighted multival MAE: `0.604 +/- 0.052 s`
- validation_car MAE: `0.284 +/- 0.018 s`
- validation_pedestrian MAE: `0.932 +/- 0.111 s`
- validation_car relative error: `9.06 +/- 1.12%`
- validation_pedestrian relative error: `28.92 +/- 2.02%`

One sanity check from the SSL-best checkpoint on seed 7 gave weighted multival
MAE `0.636 s`, validation_car MAE `0.460 s`, and validation_pedestrian MAE
`0.816 s`, so selecting SSL-best does not rescue the ablation.

Interpretation:

- The all-token context loss improves car validation relative to the transformer
  predictor without context loss (`0.284 s` vs `0.343 s`).
- It hurts pedestrian validation badly (`0.932 s` vs `0.592 s`).
- Weighted multival MAE worsens from `0.466 s` to `0.604 s`.
- The sealed CPLA-high test was not evaluated for this ablation.
- Next valid work should try smaller context-token weights or scheduling on
  multi-domain validation only, or prioritize official CCRs2/CCRm bbox/ROI data.

## 2026-07-02 Smaller All-Token Context Loss

Ran the same transformer dense predictor with `context_token_weight=0.05`,
chosen only on the dev multi-validation protocol.

Dev multi-validation pretrain:

- run:
  `artifacts/runs/jepa_event_tubelet_alltoken_transformerpred_nav_dev_multival_w005_seed7_30e`
- model: `event-tubelet-transformer`
- predictor: `transformer`
- `context_token_weight=0.05`
- best SSL validation epoch: 2
- best SSL validation loss: `0.000971`
- last epoch SSL validation loss: `0.001971`

Fine-tune from the SSL-last checkpoint, validation only:

| Seed | Weighted multival MAE | validation_car MAE | validation_pedestrian MAE |
| ---: | ---: | ---: | ---: |
| 7 | 0.453 s | 0.287 s | 0.623 s |
| 13 | 0.488 s | 0.424 s | 0.554 s |
| 21 | 0.409 s | 0.263 s | 0.558 s |

Mean dev result:

- weighted multival MAE: `0.450 +/- 0.032 s`
- validation_car MAE: `0.325 +/- 0.071 s`
- validation_pedestrian MAE: `0.578 +/- 0.032 s`
- validation_car relative error: `9.80 +/- 1.53%`
- validation_pedestrian relative error: `17.15 +/- 1.01%`

This improves weighted dev validation versus the transformer predictor without
all-token context loss (`0.466 +/- 0.004 s`) and versus the weight-0.25
ablation (`0.604 +/- 0.052 s`), so a full-starter validation-only run was
allowed.

Full-starter pretrain:

- run:
  `artifacts/runs/jepa_event_tubelet_alltoken_transformerpred_nav_full_starter_w005_seed7_30e`
- best SSL validation epoch: 2
- best SSL validation loss: `0.001283`
- last epoch SSL validation loss: `0.001838`
- last future alignment loss: `0.001775`
- last context token loss: `0.001257`

Full-starter fine-tune from SSL-last checkpoint, evaluation restricted to
`train validation`:

| Seed | Validation MAE | Validation relative error | Best epoch |
| ---: | ---: | ---: | ---: |
| 7 | 0.250 s | 8.85% | 28 |
| 13 | 0.235 s | 6.89% | 7 |
| 21 | 0.239 s | 7.02% | 3 |

Mean full-starter validation result:

- validation MAE: `0.241284 +/- 0.006477 s`
- validation relative error: `7.59 +/- 0.90%`

This is marginally worse than the selected transformer-predictor candidate
(`0.240904 +/- 0.003912 s` validation MAE), so the sealed CPLA-high test was
not evaluated. The useful conclusion is that a small all-token context loss can
help multi-domain validation, but it is not yet a full-starter model-selection
winner.

## 2026-07-02 External Dataset Triage: Markov

Checked Markov Studios / Markov AI datasets for possible world-model
pretraining. The available Markov datasets are not a priority for EvTTC:

- `markov-ai/computer-use-large`: GUI screen recordings for desktop software
  computer-use agents.
- `markov-ai/gaming-500-hours`: gameplay screen recordings with keyboard/mouse
  actions.

They are useful as conceptual examples of action-conditioned video/world-model
data, but they lack event-camera data, TTC labels, vehicle ego-motion,
EvTTC-compatible bbox/ROI labels, and driving-domain geometry. Do not download
them for the current SOTA path. Finish official EvTTC bbox/ROI assets first;
if external pretraining is needed later, prefer driving/world-model datasets
with vehicles, pedestrians, and camera/ego metadata.

## 2026-07-02 SkyJEPA Paper Triage and Latent Probers

Reviewed SkyJEPA, "Learning Long-Horizon World Models for Zero-Shot Sim-to-Real
Control of Quadrotors" (`https://arxiv.org/abs/2606.23444`). The relevant
pattern for EvTTC is not quadrotor MPPI itself, but the separation between:

- action-conditioned latent dynamics;
- anti-collapse latent regularization;
- frozen latent rollout;
- lightweight physics-inspired prober trained after the latent model is frozen.

Implemented the matching EvTTC starter pieces:

- `train latent-prober`: all-window frozen JEPA-latent residual TTC prober;
- `train roi-latent-prober`: bbox/ROI frozen-latent residual TTC prober using
  current boxes, causal event history, and a train-only ridge physics prior.

Leakage controls recorded in metrics:

- encoder frozen before prober training;
- no future events, future boxes, or future navigation;
- prober feature scaling and ridge prior fit use train split only;
- test is evaluated only when explicitly requested.

All-window frozen latent prober was negative on dev validation:

- transformer-predictor checkpoint with ridge prior: weighted multival MAE
  `0.571 s`;
- same checkpoint without prior: weighted multival MAE `0.526 s`;
- all-token checkpoint: car `0.349 s`, pedestrian `1.090 s`.

ROI latent prober is positive on bbox/ROI validation. Dev multi-validation over
seeds 7/13/21:

| Split | MAE | Mean relative error |
| --- | ---: | ---: |
| validation_car | `0.340 +/- 0.003 s` | `14.88 +/- 0.15%` |
| validation_pedestrian | `0.785 +/- 0.024 s` | `38.66 +/- 1.21%` |
| weighted | `0.528 +/- 0.011 s` | - |

Full-starter validation-only ROI prober, seeds 7/13/21:

| Seed | Validation MAE | Validation relative error | Best epoch |
| ---: | ---: | ---: | ---: |
| 7 | `0.247 s` | `11.53%` | 13 |
| 13 | `0.214 s` | `10.43%` | 22 |
| 21 | `0.218 s` | `10.39%` | 12 |

Mean full-starter validation result:

- validation MAE: `0.226 +/- 0.015 s`;
- validation relative error: `10.78 +/- 0.53%`;
- train-only ROI/ridge prior inside the same prober: `0.344 s`, `16.19%`.

This is a 34.3% validation MAE reduction versus the train-only ROI/ridge prior.
It is also stronger than the earlier local ROI event ridge validation result
(`0.293 s`), but it is still detection-assisted and matched to 98/108 validation
bbox rows. The sealed CPLA-high test was not evaluated.

Conclusion: SkyJEPA is key for the next serious architecture direction, but the
current implementation is only a partial transfer. The next valid version should
probe predicted multi-step latent rollouts, not only frozen context latents, and
should keep all tuning on multi-domain validation before any final sealed-test
check.

## 2026-07-02 Tubelet Mask JEPA

Implemented a more V-JEPA-like masking path for the event-tubelet backbone:

- new CLI flag: `pretrain jepa --mask-mode {spatial,tubelet}`;
- default remains `spatial`, preserving previous runs;
- `tubelet` masks random spatio-temporal event-channel blocks
  `[polarity, time, y, x]`;
- metadata and navigation channels are preserved during tubelet masking;
- metrics/checkpoints record `mask_mode`;
- leakage audit records:
  `tubelet_masking_uses_context_event_channels_only=true` and
  `tubelet_masking_preserves_auxiliary_channels=true`.

Unit coverage:

- direct tubelet masking test verifies event channels are masked and auxiliary
  channels are unchanged;
- pretraining smoke verifies objective
  `tubeletmask_transformer_dense_temporal_token_motion_multihorizon`.

Dev multi-validation pretraining:

- run:
  `artifacts/runs/jepa_event_tubelet_tubeletmask_transformerpred_nav_dev_multival_seed7_30e`
- model: `event-tubelet-transformer`
- predictor: `transformer`
- `mask_mode=tubelet`
- `mask_ratio=0.45`
- no all-token context loss
- best SSL validation epoch: 18
- best SSL validation loss: `0.001370`
- last SSL validation loss: `0.001652`

Negative tubelet ablations:

- tubelet plus all-token weight `0.05`, seed 7 fine-tune:
  validation_car `0.312 s`, validation_pedestrian `0.993 s`, weighted `0.649 s`;
- tubelet mask ratio `0.20`, LR `1e-4`, seed 7 fine-tune:
  validation_car `0.315 s`, validation_pedestrian `0.833 s`;
- tubelet mask ratio `0.20`, LR `3e-5`, seed 7 fine-tune:
  validation_car `0.319 s`, validation_pedestrian `0.643 s`, weighted about
  `0.479 s`.

The selected dev variant uses tubelet mask ratio `0.45`, no all-token context
loss, and supervised fine-tuning LR `3e-5`.

Dev multi-validation fine-tune from SSL-last:

| Seed | validation_car MAE | validation_pedestrian MAE | Weighted MAE |
| ---: | ---: | ---: | ---: |
| 7 | `0.240 s` | `0.580 s` | `0.408 s` |
| 13 | `0.192 s` | `0.533 s` | `0.360 s` |
| 21 | `0.293 s` | `0.626 s` | `0.457 s` |

Mean dev result:

- weighted multival MAE: `0.409 +/- 0.040 s`;
- validation_car MAE: `0.242 +/- 0.041 s`;
- validation_pedestrian MAE: `0.580 +/- 0.038 s`.

This improves over the previous best dev result, all-token transformer JEPA
weight `0.05` (`0.450 +/- 0.032 s` weighted), and over the transformer predictor
without all-token context loss (`0.466 +/- 0.004 s` weighted).

Full-starter validation-only pretraining:

- run:
  `artifacts/runs/jepa_event_tubelet_tubeletmask_transformerpred_nav_full_starter_seed7_30e`
- best SSL validation epoch: 26
- best SSL validation loss: `0.001338`
- last SSL validation loss: `0.001344`

Full-starter validation-only fine-tune, evaluation restricted to
`train validation`:

| Seed | Validation MAE | Validation relative error | Best epoch |
| ---: | ---: | ---: | ---: |
| 7 | `0.250 s` | `8.17%` | 7 |
| 13 | `0.208 s` | `7.55%` | 27 |
| 21 | `0.237 s` | `8.86%` | 29 |

Mean full-starter validation result:

- validation MAE: `0.231478 +/- 0.017632 s`;
- validation relative error: `8.192429 +/- 0.533708%`.

This is the best validation MAE so far on the full-starter validation split, but
the CPLA-high test had already been inspected in earlier branches, so this is a
local protocol result rather than a fresh sealed-test claim.

Added a reproducible metric aggregator:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_eval_metrics.py --split test --output artifacts\metrics\event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_eval_full_protocol_test_summary.json artifacts\metrics\event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_seed7_eval_full_protocol.json artifacts\metrics\event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_seed13_eval_full_protocol.json artifacts\metrics\event_tubelet_tubeletmask_transformerpred_nav_full_starter_last_lr3e5_seed21_eval_full_protocol.json
```

Full protocol evaluation, no retraining:

| Seed | CPLA-high test MAE | Test relative error | Best epoch |
| ---: | ---: | ---: | ---: |
| 7 | `0.365 s` | `7.03%` | 7 |
| 13 | `0.314 s` | `5.94%` | 27 |
| 21 | `0.257 s` | `6.27%` | 29 |

Aggregated CPLA-high test result:

- test MAE: `0.312034689 +/- 0.044063632 s`;
- test mean absolute relative error: `6.416740 +/- 0.454934%`;
- test RMSE: `0.485851757 +/- 0.090484582 s`.

This is now the best local all-window full-starter result. It improves MAE by
4.9% versus the previous event-tubelet navigation JEPA mean (`0.328 s`) and
improves relative error from `6.89%` to `6.42%`. It is still not an official SOTA
claim because the official EvTTC comparison is bbox/ROI-assisted and uses a
broader sequence protocol.

The next valid step is either a clean final protocol with additional unopened
sequences or an official bbox/ROI comparison using the complete official
sequence set.

## 2026-07-02 ROI Latent Prober Checkpoint Evaluation

Added a checkpoint-only evaluator for the SkyJEPA-style detection-assisted
prober:

```powershell
.\.venv\Scripts\python.exe -m e_jepa_ttc train roi-latent-prober-evaluate --manifest artifacts\metrics\evttc_scan_full_bbox.yaml --split data\splits\evttc_full_starter_sealed.yaml --cache artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz --checkpoint artifacts\runs\roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_seed7_160e\roi_latent_prober_best.pt --output artifacts\metrics\roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_seed7_eval_full_protocol.json --context-ms 100 --max-cache-slop-ms 12 --batch-size 64 --device auto --evaluation-splits train validation test
```

The evaluator reloads the saved ROI prober checkpoint, reloads the frozen JEPA
encoder recorded in that checkpoint, applies the train-fitted ROI and latent
feature normalization from the checkpoint, and evaluates requested splits
without retraining. It writes JSON metrics plus prediction arrays.

Full-starter validation, used for model selection before test evaluation:

| Seed | Matched validation frames | Validation MAE | Validation relative error |
| ---: | ---: | ---: | ---: |
| 7 | 98/108 | `0.247 s` | `11.53%` |
| 13 | 98/108 | `0.214 s` | `10.43%` |
| 21 | 98/108 | `0.218 s` | `10.39%` |

Aggregated validation: `0.226 +/- 0.015 s`, `10.78 +/- 0.53%`.

Frozen full-protocol evaluation, no retraining:

| Seed | Matched CPLA-high frames | Test MAE | Test relative error |
| ---: | ---: | ---: | ---: |
| 7 | 73/83 | `0.383 s` | `20.46%` |
| 13 | 73/83 | `0.431 s` | `23.91%` |
| 21 | 73/83 | `0.454 s` | `25.85%` |

Aggregated CPLA-high ROI prober result:

- test MAE: `0.422723691 +/- 0.029309195 s`;
- test mean absolute relative error: `23.407466 +/- 2.229738%`;
- test RMSE: `0.528156744 +/- 0.019810572 s`.

This result is not competitive with the current all-window JEPA test result
(`6.42 +/- 0.45%` relative error) and is also worse than the causal bbox
geometry reference (`0.157 s` MAE on 81 valid CPLA-high bbox frames). The useful
conclusion is negative and architectural: frozen context-latent probing is not
enough. The SkyJEPA-like next step should probe predicted multi-step latent
rollouts with a structured TTC/kinematic head, still selected only by
validation.

Artifacts:

- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_seed7_eval_full_protocol.json`
- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_seed13_eval_full_protocol.json`
- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_seed21_eval_full_protocol.json`
- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_eval_full_protocol_test_summary.json`
- `artifacts/metrics/roi_latent_prober_event_tubelet_transformerpred_nav_full_starter_last_eval_full_protocol_validation_summary.json`

## 2026-07-02 SkyJEPA Paper Reassessment

SkyJEPA (`https://arxiv.org/abs/2606.23444`, v2 revised 2026-06-23) is key as
an architecture guide, not as a directly transferable TTC benchmark. The paper
targets quadrotor control, but the important pattern for this project is:

- learn action-conditioned latent dynamics instead of reconstructing future
  observations;
- avoid autoregressive observation rollout drift by predicting latent dynamics;
- map frozen latent rollouts through a physics-inspired prober into
  interpretable state;
- measure long-horizon stability explicitly, not just one-step loss;
- treat data coverage as a first-class variable via a trajectory distribution
  quality concept;
- use compact anti-collapse regularization such as SIGReg/VISReg-style
  distribution regularization rather than fragile reconstruction-heavy losses.

Project implications:

1. Keep the current tubelet/dense-token JEPA direction.
2. Do not over-invest in current-frame ROI latent probing; the full-protocol
   result above is negative.
3. Implement the next prober against predicted future latents from the JEPA
   predictor, not only current frozen context latents.
4. Add validation-only diagnostics for compounding error across TTC horizons and
   split/domain coverage, analogous to SkyJEPA's long-horizon and TDQ analyses.
5. Treat VISReg/SIGReg as a serious next regularization ablation, tuned only on
   multi-domain validation.

This keeps the project aligned with SkyJEPA without claiming its quadrotor
control results transfer to event-camera TTC.

## 2026-07-02 ROI Predicted-Rollout Prober

Implemented the next SkyJEPA-aligned prober step:

- `train roi-rollout-prober` reconstructs the frozen JEPA encoder and predictor
  from a JEPA checkpoint;
- extracts causal context tokens and action/ego-motion features from the current
  event window only;
- predicts future dense latent token rollouts at the JEPA horizons;
- summarizes those predicted future tokens and trains a lightweight bbox/ROI TTC
  prober using train labels only;
- `train roi-rollout-prober-evaluate` reloads a saved prober checkpoint and
  evaluates requested splits without retraining.

Validation-only full-starter run using
`jepa_event_tubelet_tubeletmask_transformerpred_nav_full_starter_seed7_30e`:

| Seed | Validation matched frames | Validation MAE | Validation relative error |
| ---: | ---: | ---: | ---: |
| 7 | 98/108 | `0.210 s` | `10.17%` |
| 13 | 98/108 | `0.226 s` | `11.19%` |
| 21 | 98/108 | `0.242 s` | `12.09%` |

Aggregated full-starter validation:

- validation MAE: `0.226000489 +/- 0.013070400 s`;
- validation relative error: `11.148893 +/- 0.786129%`.

This is only a marginal MAE tie with the ROI latent prober
(`0.226443 s`) and is worse in relative error (`11.15%` versus `10.78%`), so
CPLA-high was not evaluated for this branch.

Checkpoint-only validation evaluation was verified with:

```powershell
.\.venv\Scripts\python.exe -m e_jepa_ttc train roi-rollout-prober-evaluate --manifest artifacts\metrics\evttc_scan_full_bbox.yaml --split data\splits\evttc_full_starter_sealed.yaml --cache artifacts\features\evttc_full_starter_voxel_160x90_b5_raw_meta_nav.npz --checkpoint artifacts\runs\roi_rollout_prober_tubeletmask_full_starter_seed7_160e\roi_rollout_prober_best.pt --output artifacts\metrics\roi_rollout_prober_tubeletmask_full_starter_seed7_eval_validation_only.json --context-ms 100 --max-cache-slop-ms 12 --batch-size 128 --device auto --evaluation-splits validation
```

It reproduced seed 7 validation MAE: `0.210115841 s`, with
`retrained_during_evaluation=false`.

Seed-7 ablations, validation only:

| Variant | Validation MAE | Relative error |
| --- | ---: | ---: |
| mean-std rollout + context | `0.210 s` | `10.17%` |
| mean rollout + context | `0.251 s` | `12.32%` |
| mean-std rollout without context | `0.219 s` | `10.73%` |
| mean rollout without context | `0.227 s` | `10.66%` |

Harder dev multi-validation, no test:

| Split | MAE | Mean relative error |
| --- | ---: | ---: |
| validation_car | `0.329 +/- 0.024 s` | `15.16 +/- 0.78%` |
| validation_pedestrian | `1.000 +/- 0.055 s` | `47.49 +/- 1.82%` |
| weighted | `0.613 +/- 0.017 s` | - |

This is worse than the previous ROI latent prober on the same dev split
(`0.528 +/- 0.011 s` weighted, `0.785 +/- 0.024 s` pedestrian), so the
flat rollout-summary prober is a negative result. The next serious version
should keep the per-horizon structure and use a kinematic/TTC head rather than
flattening all predicted horizons into one MLP feature vector.

## 2026-07-02 Rollout Dynamics Feature Ablation

Implemented `--rollout-feature-mode dynamics` for `train roi-rollout-prober`.
This mode augments flat predicted future-token summaries with:

- per-horizon latent deltas from the current context summary;
- latent velocities normalized by horizon time;
- consecutive-horizon latent velocities;
- compact scalar norms/cosine similarities.

Validation-only seed 7 result on the full-starter validation split:

| Variant | Validation MAE | Relative error | Best epoch |
| --- | ---: | ---: | ---: |
| flat rollout, mean-std + context | `0.210 s` | `10.17%` | 31 |
| dynamics rollout, mean-std + context | `0.248 s` | `11.93%` | 2 |

The dynamics feature mode overfits almost immediately and is worse than the
flat rollout prober. No CPLA-high test was run. This supports the same
conclusion as the previous rollout experiment: useful SkyJEPA-style TTC probing
needs a constrained per-horizon kinematic head, not just more flattened rollout
features.

## 2026-07-02 Final Local Package And Paper

Created the thesis-style English paper:

- `docs/e_jepa_ttc_paper.md`

Updated:

- `README.md`
- `docs/technical_report.md`
- `docs/model_card.md`
- `docs/full_starter_results.md`
- `docs/sota_jepa_world_models_2026-07-01.md`

Final verification:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest
```

Results:

- `ruff check .`: all checks passed;
- `pytest`: `50 passed in 5.39 s`.

Final local conclusion:

- best local all-window model: event-tubelet transformer JEPA with tubelet
  masking, dense transformer future-token prediction, and causal
  integrated-navigation conditioning;
- validation: `0.231477844 +/- 0.017632455 s` MAE,
  `8.192429 +/- 0.533708%` relative error;
- diagnostic CPLA-high: `0.312034689 +/- 0.044063632 s` MAE,
  `6.416740 +/- 0.454934%` relative error;
- ROI latent prober and ROI rollout prober are useful diagnostics but not
  competitive SOTA paths in their current flat-head form;
- no official EvTTC SOTA claim is justified until CCRs2/CCRm/slider assets and
  official bbox/ROI baselines are reproduced.

## 2026-07-03 Official Protocol Coverage Gate

Implemented `e_jepa_ttc.data.official_protocol` plus:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\e-jepa-ttc.exe data official-coverage --root datasets\evttc --output artifacts\metrics\evttc_official_table_v_coverage.json
```

The checker encodes the published EvTTC Table V bbox/ROI sequence set:
`CCRs-1-low/medium/high-100%`, `CCRs-2-low/medium/high-100%`,
`CCRm-low/medium-100%`, `Slider-750`, and `Slider-1000`. It requires event
HDF5, TTC labels, and nonempty bbox/ROI labels for each row.

Latest local result:

- scanned local EvTTC sequences: `9`;
- complete real-world official rows: `3/8`, `37.5%`;
- complete Table V rows including slider: `3/10`, `30.0%`;
- missing real-world rows: `CCRs-2-low-100%`, `CCRs-2-medium-100%`,
  `CCRs-2-high-100%`, `CCRm-low-100%`, `CCRm-medium-100%`;
- missing slider rows: `Slider-750`, `Slider-1000`;
- official SOTA claim allowed: `false`.

This is the final protocol conclusion for the current local data: the best
all-window JEPA result is strong locally, but official EvTTC SOTA comparison is
blocked by asset coverage and missing reproduced STRTTC/CMax/ETTCM runtime
baselines. More tuning on CPLA-high would be invalid; the next valid step is
downloading CCRs2/CCRm/slider official assets and running the same checker
before wrapping official baselines.

Final verification after adding the coverage gate:

- `ruff check .`: all checks passed;
- `pytest`: `53 passed in 4.30 s`.

## 2026-07-25 Physics-constrained FlowMimic pilot

Implemented render-then-simulate FlowMimic event generation and optional
synthetic future-alignment/inverse-TTC SSL losses. Hardened split exclusion,
cache auditing, non-finite loss handling and physical checkpoint fingerprints.
Rejected and documented one cache-exclusion failure, one AMP-navigation NaN
smoke and one downstream fingerprint collision before accepting results.

Accepted cache: format v2, 3,019 train + 475 validation, zero test, SHA-256
`22d3ef27018925aae62825f0a7f51d1420ae93cacf59aeb18b04758f5a35e88a`.

Seed-7 validation pilot:

| Variant | MAE | MARE | RMSE |
| --- | ---: | ---: | ---: |
| scratch | `0.3893 s` | `11.87%` | `0.5076 s` |
| E0 JEPA | `0.3416 s` | `9.81%` | `0.4978 s` |
| E1 physical alignment | **`0.2552 s`** | **`8.40%`** | **`0.3322 s`** |
| E2 alignment + inverse-TTC | `0.3256 s` | `9.67%` | `0.4363 s` |

E1 improves E0 by 25.29% MAE. The inverse-TTC auxiliary hurts downstream even
though it gives the best SSL loss, so it is not promoted. Signed artifact:
`artifacts/metrics/flowmimic_validation_pilot_seed7_summary.json`. Next gate:
full-schedule E0 vs E1 with independent SSL/downstream seeds 7, 13 and 21,
without opening CPLA-high.

Selected E1 batch-1 FP32 model-only latency on the RTX 5070 Ti Laptop GPU:
`2.201 ms` mean, `2.096 ms` median and `2.779 ms` p95 over 300 synchronized
iterations. This excludes voxelization and is not an official Garl-TTC runtime
comparison; it separates runtime from the `0.255 s` TTC MAE.

