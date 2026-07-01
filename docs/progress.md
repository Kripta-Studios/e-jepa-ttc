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
- Token transformer scratch full-label seed 7: validation MAE 0.709 s, sealed-test MAE 0.854 s.
- Token JEPA full-label seed 7: validation MAE 0.350 s, sealed-test MAE 0.422 s.
- 5% labels, three seeds: token scratch validation/test 1.226 +/- 0.031 / 1.382 +/- 0.044 s;
  token JEPA validation/test 0.524 +/- 0.047 / 0.636 +/- 0.109 s.
- 10% labels, three seeds: token scratch validation/test 1.178 +/- 0.056 / 1.327 +/- 0.104 s;
  token JEPA validation/test 0.437 +/- 0.039 / 0.460 +/- 0.029 s.

Conclusion:

- Token JEPA is now the best local full-starter learned result.
- Against the matching scratch token backbone, JEPA improves 100% label sealed-test MAE by 50.6%,
  5% label sealed-test MAE by 53.9%, and 10% label sealed-test MAE by 65.4%.
- Against TinyCNN scratch with 100% labels, token JEPA improves sealed-test MAE by 17.8%.
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

