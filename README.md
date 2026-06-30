# E-JEPA-TTC

Research MVP for Time-to-Contact / Time-to-Collision estimation from event camera streams.

The current repository implements the first engineering milestones from `AGENTS.md`: project
bootstrap, typed data contracts, synthetic event data with known TTC, EvTTC dataset discovery,
manifest validation, temporal indexing, sequence-level splits, and dense event representations.

No experimental claims or benchmark numbers are reported yet. Metrics must be generated from
reproducible runs before being added to this README.

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

## Implemented

- Synthetic expanding-object event generator with monotonic timestamps and known TTC labels.
- EvTTC local scanner for sequence folders with event HDF5, `gt.hdf5`, `ttc.csv`, and ISAT labels.
- Lazy HDF5 event-field discovery for common separate-field and compound-event layouts.
- EvTTC window reads using `ms_map_idx` when available, validated on the local HDF5 files.
- Dataset manifest validation without loading full event streams into memory.
- Temporal window index generation from TTC timestamps.
- Sequence-level split generation and validation.
- Event count, time surface, voxel grid, and sparse token representations.
- Unit and integration tests for data contracts, representations, synthetic data, manifests, and splits.

## Not Implemented Yet

- Supervised Tiny CNN training.
- E-JEPA target encoder and multi-horizon predictor.
- Fine-tuning, robustness suite, ONNX export, streaming demo, and report generation.

These remain in the milestone order defined in `AGENTS.md`; they should be added after the data
pipeline and representations remain green under tests.

## Local Dataset Notes

The handoff contains an EvTTC mini subset under `datasets/evttc/CCRs-1` with three speed buckets:
`low-100`, `medium-100`, and `high-100`. Each sequence includes `ttc.csv`, one large event HDF5,
`gt.hdf5`, ISAT JSON labels, and video/bag files. The current pipeline uses only HDF5 metadata,
`ttc.csv`, and label metadata; video and bag files are intentionally ignored for the MVP.

See [docs/datasets_local.md](docs/datasets_local.md) and [docs/progress.md](docs/progress.md).
