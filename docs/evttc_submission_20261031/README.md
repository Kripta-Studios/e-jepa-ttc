# PHASE-TTC-Adapt-20261031: submission and reproducibility package

This delivery contains one final trained adapted model and a separate reply about
the unchanged PHASE-TTC submission of 28 October. Author: **Álvaro Schwiedop Souto**,
sole author, **no institutional affiliation**. Nothing has been emailed or uploaded.

For the benchmark, attach `phase_submission/replay/results.zip` and use the draft
`repository/docs/evttc_submission_20261031/adapted_cover_note.txt`. That ZIP contains
the ten sequence directories and four-column `results.txt` files expected by EvTTC.
Use `original_reply.txt` to supply the missing details for the original listing.
The original and adapted methods have different experimental histories and names.

Read `REPORT.md` in this document's directory for measured results and remaining
limitations. The final model is `phase_submission/FINAL_MODEL`. The full package
manifest records SHA-256 and length for each payload file; the outer ZIP hash is
recorded separately in `DELIVERY.json` beside the local campaign.

## Model and data contract

The original nine TRAIN40 producer/context/fusion components remain frozen. A
three-seed ensemble adds 688 trainable parameters per router. Each router combines
the original estimate, RGB context, event context and causal bbox geometry in
inverse-TTC coordinates using physical observation-quality features. Seeds 7, 13,
21 each receive 300 fixed AdamW updates. No seed is chosen using the new replay.

The final fit contains 2,608 queries from 24 road recordings and Slider-500.
`training/CONFIG.json` lists every included recording and all eight excluded road
recordings. Slider-750 and Slider-1000 are also excluded from this fit. The
population and exclusion checks are executable, not just descriptions.

Earlier development inspected all 32 road recordings and all three Slider
recordings; the inherited recipe was selected during that exposed campaign.
**The final fit exclusions do not turn the benchmark into a virgin holdout.**
These are adapted-method research results, without a SOTA or independent
generalization claim. The training features use the native road preparation;
the competition replay uses the right camera pair. Their covariate shift remains.

The router takes `native[B,3]` = original/RGB/event estimates in seconds,
`geometry[B]` = signed geometric TTC or NaN, and `quality[B,20]` from the supplied
causal geometry implementation. History is limited to 650 ms. Supplied boxes,
RGB and events are inference inputs; TTC labels, target distance and sequence ID
are not model inputs. Missing/stale bbox history returns the original estimate.
The inverse-coordinate model bounds its internal expert magnitudes/rates as
documented in `operational/phase_startup/model.py`; submitted outputs receive no
additional sign removal, clipping or ground-truth-based correction.

The final refit does not carry a calibrated confidence interval. The rotation-only
sensor adapter omits translation because an allowed depth estimate is unavailable.
Slider inter-camera rotation is still a provisional calibration consensus. These
are material limitations, not resolved by the training procedure.

## Reproduce from the extracted package

Use Python 3.11. From the extracted `repository` directory:

```powershell
uv sync --group dev --no-editable
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
```

The non-editable install avoids Python 3.11/Windows `.pth` decoding problems in
user paths containing accented characters. `PYTHONPATH` still selects the bundled
source for the reproducible research commands. On Linux, set `PYTHONPATH=src:.`
instead. A CUDA-compatible PyTorch installation is
required for the CUDA replay. The observed delivery environment is recorded in
`phase_submission/replay/ENVIRONMENT.json`; CPU inference is also supported.

Retrain the small routers from the included frozen features, without raw sensors:

```text
uv run --no-sync python -m scripts.train_phase_submission --population ../inputs/POPULATION.npz --exclusions ../inputs/exclusions.json --previous-bundle ../phase_submission/FINAL_MODEL --output ../retrained
```

The feature population includes previously inspected benchmark Slider rows for
provenance; the training function explicitly removes both benchmark Slider
recordings before fitting. The included eligible training population contains
only the 2,608 retained rows. Original executed training source bytes are preserved
under `training/executed_source`; subsequent formatting changes do not alter the
training computation. Config paths/receipt hashes will differ after relocation.

Reproduce all 434 predictions from separately obtained, licensed raw benchmark
data. Each sequence directory must contain the original supplied bbox CSV,
HDF5 or Slider bag, and calibration expected by `SensorSequence`:

```text
uv run --no-sync python -m scripts.run_phase_submission --bundle ../phase_submission/FINAL_MODEL --data-root /path/to/EvTTC-Benchmark --output ../replay_new --device cuda
```

An interrupted run can use the same command with `--resume`; changes to sources
listed in the run freeze, weights, sensor files or query identities are rejected.
The original runner's source inventory is partial. For this delivery, a separate
post-run source supplement verifies the wider tracked Python implementation
against the pre-run commit and binds those bytes at packaging. This is an audit
of the retained checkout, not a claim of continuous file monitoring during the run.
The entire delivered implementation must be retained unchanged for reproduction.
A `--limit` run produces
no submission ZIP until all 434 queries are complete. Query costs are actual
wall-clock sensor preparation plus the original graph and router, with GPU
synchronization. One-time model loading, hashes/indexing and ledger writes are
outside those per-query costs. Cached raw chunks are lossless and the preparation
still slices each causal input window. No CUDA Graph speedup is claimed here.

Verify the frozen predictions, reload/TorchScript parity and local evaluation:

```text
uv run --no-sync python -m scripts.evaluate_phase_submission --replay ../phase_submission/replay --bundle ../phase_submission/FINAL_MODEL --historical-run ../historical_run --road-targets ../inputs/road_targets.json --slider-contract ../inputs/slider --output ../evaluation_reproduced
```

The evaluator first requires and verifies the completed prediction seal. It
retains the 11 road queries without local target support, and reports coverage
alongside scores. Negative prediction interpolation is an evaluator-only view
following the organizer's email; raw files remain unchanged. The public target
alignment does not reproduce the original Slider-750 PDF score, so these local
metrics must not be labelled official EvTTC scores.

`router/router.ts` is the exported fusion ensemble, not the entire raw-sensor
pipeline. The complete executable model also requires the bundled upstream
components and source adapter. Raw data and the large original eAP cache are not
redistributed. The source-fallback and cached corruption validation receipts
record their external input hashes so those larger checks can be repeated with
the original licensed data/cache.

## Evidence

- `audit/HISTORICAL_AUDIT.json`: original ZIP identity, hardware, per-sequence
  calibration and explicit lens evidence, plus the unresolved target alignment.
- `training/`: configuration, frozen eligible inputs, all three checkpoints,
  fitted predictions, export and resubstitution metrics.
- `replay/`: source/weight/sensor/query freeze, measured per-query ledger, exact
  submission files, environment and completion receipt.
- `evaluation/LOCAL_EVALUATION.json`: original and adapted values on identical
  supported targets, raw and organizer-processed views, explicit missing targets.
- `validation/`: exact source fallback and diagnostic cached sensor corruptions.
- `FINAL_CHECKS.json` and `verification_logs/`: final test/review evidence.

Camera descriptions use the [official sensor page](https://nail-hnu.github.io/EvTTC/about/sensor/)
and [EvTTC paper](https://arxiv.org/html/2412.05053v1). Right-camera nominal 16 mm
identification is an inference from the actual calibrated FoV and the paper's
Table II, not an undocumented mm field invented from HDF5. Slider nominal mm
remains unspecified; its actual pixel intrinsics and testbed identity are given.
