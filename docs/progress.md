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

