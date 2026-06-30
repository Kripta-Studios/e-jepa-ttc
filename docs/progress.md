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

