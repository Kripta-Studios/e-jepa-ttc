# E-JEPA-TTC

Research MVP for Time-to-Contact / Time-to-Collision estimation from event camera streams.

The current repository implements the first engineering milestones from `AGENTS.md`: project
bootstrap, typed data contracts, synthetic event data with known TTC, EvTTC dataset discovery,
manifest validation, temporal indexing, sequence-level splits, dense event representations,
classical TTC baselines, voxel-cache materialization, a supervised TinyCNN TTC regressor,
dense motion-conditioned temporal JEPA pretraining, and low-label probes.

Experimental numbers must be generated from reproducible runs before being promoted to project
claims. The bundled local EvTTC data is a three-sequence mini subset, so local results are smoke and
sanity evidence rather than a broad benchmark.

## Quickstart

```bash
make setup
make smoke-data
make test
```

To scan the local EvTTC subset placed under `datasets/evttc`:

```bash
make scan-data
make validate-data
make index-data
make split-data
```

Equivalent direct CLI calls:

```bash
uv sync --all-groups --no-editable
uv run --no-sync e-jepa-ttc data scan --root datasets/evttc --output data/manifests/evttc_local.yaml
uv run --no-sync e-jepa-ttc data validate --manifest data/manifests/evttc_local.yaml
uv run --no-sync e-jepa-ttc data index --manifest data/manifests/evttc_local.yaml --output data/cache/evttc_index.json
uv run --no-sync e-jepa-ttc split create --manifest data/manifests/evttc_local.yaml --output data/splits/evttc_local.yaml
```

On Windows paths containing non-ASCII characters, `--no-editable` and `--no-sync` avoid an editable
install `.pth` encoding issue observed with CPython 3.11.

## GPU Training

The base project dependencies stay lightweight. Install PyTorch into the existing virtualenv before
using `train tiny-cnn`:

```powershell
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then build a voxel cache and train the supervised CNN:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m e_jepa_ttc cache voxel --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --width 160 --height 90 --bins 5 --no-normalize --metadata-channels
.\.venv\Scripts\python.exe -m e_jepa_ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto
```

To pretrain the encoder without TTC labels and then fine-tune the supervised head:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m e_jepa_ttc pretrain jepa --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7 --epochs 160 --batch-size 64 --learning-rate 0.0005 --seed 7 --device auto --pretrain-splits train --validation-splits validation --temporal-horizons-ms 20 60 100 240 500 --max-target-slop-ms 10 --variance-weight 1.0 --min-std 0.05
.\.venv\Scripts\python.exe -m e_jepa_ttc train tiny-cnn --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_temporal_jepa_seed7 --epochs 80 --batch-size 96 --learning-rate 0.0003 --seed 7 --device auto --pretrained-encoder artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7/jepa_encoder_best.pt
```

## Local Results

Current mini-subset results are summarized in [docs/local_results.md](docs/local_results.md).
Full-starter sealed results are summarized in
[docs/full_starter_results.md](docs/full_starter_results.md). On the full local starter protocol,
the token-transformer dense JEPA model is the strongest result so far: with 100% labels it reaches
`0.350s` validation MAE and `0.422s` sealed-test MAE, beating TinyCNN scratch (`0.549s` /
`0.513s`) and the same token transformer trained from scratch (`0.709s` / `0.854s`). With 10%
labels, token JEPA improves sealed-test MAE from `1.327 +/- 0.104s` to `0.460 +/- 0.029s`.

Earlier available-starter sealed results are kept in
[docs/available_starter_results.md](docs/available_starter_results.md).

## Script Wrappers

Implemented script wrappers mirror the CLI for current milestones:

```bash
uv run --no-sync python scripts/scan_evttc_manifest.py --root datasets/evttc --output data/manifests/evttc_local.yaml
uv run --no-sync python scripts/validate_dataset.py --manifest data/manifests/evttc_local.yaml
uv run --no-sync python scripts/build_index.py --manifest data/manifests/evttc_local.yaml --output data/cache/evttc_index.json
uv run --no-sync python scripts/make_splits.py --manifest data/manifests/evttc_local.yaml --output data/splits/evttc_local.yaml
uv run --no-sync python scripts/train_baseline.py --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --output artifacts/metrics/trivial_baseline.json
uv run --no-sync python scripts/build_voxel_cache.py --manifest data/manifests/evttc_local.yaml --split data/splits/evttc_local.yaml --index data/cache/evttc_index.json --output artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --no-normalize --metadata-channels
uv run --no-sync python scripts/pretrain_jepa.py --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/jepa_temporal_voxel_160x90_b5_raw_meta_train_seed7 --temporal-horizons-ms 20 60 100 240 500
uv run --no-sync python scripts/train_tiny_cnn.py --cache artifacts/features/evttc_voxel_160x90_b5_raw_meta.npz --output-dir artifacts/runs/tiny_cnn_voxel_160x90_b5_raw_meta_seed7 --seed 7
uv run --no-sync python scripts/download_evttc_starter.py --manifest data/manifests/evttc_starter_downloads.yaml --root .
```

## Implemented

- Synthetic expanding-object event generator with monotonic timestamps and known TTC labels.
- EvTTC local scanner for sequence folders with event HDF5, `gt.hdf5`, `ttc.csv`, and ISAT labels.
- Lazy HDF5 event-field discovery for common separate-field and compound-event layouts.
- EvTTC window reads using `ms_map_idx` when available, validated on the local HDF5 files.
- Dataset manifest validation without loading full event streams into memory.
- Temporal window index generation from TTC timestamps.
- Sequence-level split generation and validation.
- Event count, time surface, voxel grid, and sparse token representations.
- Mean/median, geometric bbox-expansion, and event-rate ridge TTC baselines.
- Causal detection-assisted geometry baseline with explicit anti-lookahead audit.
- Voxel tensor cache builder for supervised and representation-learning experiments.
- JEPA-style self-supervised pretraining with online encoder, EMA target encoder, dense temporal
  token future prediction, causal context-motion conditioning, masked context views, and leakage
  audit metadata.
- Supervised TinyCNN log-TTC regressor with CUDA AMP, checkpoints, history, metrics, and predictions.
- Frozen-encoder probes and low-label supervised runs via `--freeze-encoder` and `--train-fraction`.
- Unit and integration tests for data contracts, representations, synthetic data, manifests, splits,
  and EvTTC window reads.

## Not Implemented Yet

- Robustness suite, ONNX export, streaming demo, and project-level final report generation.

These remain in the milestone order defined in `AGENTS.md`; they should be added after the
supervised baseline is established against the local data.

## Local Dataset Notes

The handoff contains an EvTTC mini subset under `datasets/evttc/CCRs-1` with three speed buckets:
`low-100`, `medium-100`, and `high-100`. Each sequence includes `ttc.csv`, one large event HDF5,
`gt.hdf5`, ISAT JSON labels, and video/bag files. The current pipeline uses only HDF5 metadata,
`ttc.csv`, and label metadata; video and bag files are intentionally ignored for the MVP.

`data/manifests/evttc_starter_downloads.yaml` records public official links for the six starter
sequences missing from the handoff. `scripts/download_evttc_starter.py` prints a dry-run plan by
default and only calls `gdown` when `--execute` is passed; it also supports `--continue` and
`--quiet`.

The locally complete full-starter manifest is
`data/manifests/evttc_full_starter_local.yaml`, with split
`data/splits/evttc_full_starter_sealed.yaml`. It contains the three original `CCRs-1` sequences,
all three `CCRs-side` sequences, and all three `CPLA` starter sequences. The sealed test sequence is
`CPLA-high`.

The earlier available-starter manifest is
`data/manifests/evttc_available_starter_local.yaml`, with split
`data/splits/evttc_available_starter_sealed.yaml`.

See [docs/datasets_local.md](docs/datasets_local.md), [docs/progress.md](docs/progress.md),
[docs/local_results.md](docs/local_results.md), and
[docs/full_starter_results.md](docs/full_starter_results.md).



