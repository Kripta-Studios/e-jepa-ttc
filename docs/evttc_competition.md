# EvTTC competition inference

The competition adapter runs the sealed nine-component TRAIN40 PHASE-TTC model
on all ten downloaded EvTTC competition sequences. It requires the existing
project environment, the preserved `FINAL_CANDIDATE` package, and the official
HDF5/ROS bag, bbox, and calibration inputs. It performs no training and does not
read TTC labels or published result curves.

The sensor policy and its limitations are documented in
[ADR-0043](decisions/ADR-0043-evttc-competition-sensor-adapter.md).

## Run

From the repository root, using its Python 3.11 environment:

```powershell
python scripts/run_evttc_competition.py `
  --data-root "../EvTTC-Benchmark" `
  --candidate "<FINAL_CANDIDATE>" `
  --output "output/evttc_competition_run" `
  --device cuda
```

`--device cpu` is also supported. Use a fresh output directory. The program
verifies model and runtime hashes, then freezes the input sources, query
identities, calibrations, and adapter policy before its first forward pass.

`--limit N` produces an explicitly partial run. Resume an interrupted or limited
inference run using the same command and `--resume`; all frozen identities must
match. An error ledger requires diagnosis and a new run directory rather than
silently skipping its failed query. A `--prepare-only` run cannot be resumed as
inference because its mode is part of the freeze.

For a revised adapter following a failed attempt, add `--parent-attempt` with
the original run directory. The new freeze links the preserved failed attempt
without reusing its predictions. Adapter v2 supersedes the first failed v1
attempt; the reason and boundary are recorded in ADR-0043.

Do not edit executed source files during a run. The program revalidates model,
runtime, preprocessing sources, and raw inputs before packaging.

## Validate

```powershell
python scripts/verify_evttc_competition_run.py `
  --run "output/evttc_competition_run" `
  --data-root "../EvTTC-Benchmark"
```

The independent verifier checks the original CSV identities, the prediction
ledger, temporal support, ZIP contents, and hashes. Its statistics describe
outputs and measured runtime; they are not accuracy metrics.

Generate diagnostic figures from a completed ledger with:

```powershell
python scripts/plot_evttc_competition_predictions.py `
  --run "output/evttc_competition_run"
```

The plots preserve all signed outputs and mark observed-prefix startup, empty
event-ROI observations, and mathematical observation rejections. A terminal
partial run displays unavailable query gaps without interpolating predictions.

Focused implementation checks:

```powershell
python -m pytest tests/unit/test_evttc_competition_readers.py `
  tests/unit/test_evttc_competition_geometry.py `
  tests/unit/test_evttc_competition_prepare.py `
  tests/unit/test_evttc_competition_runtime.py `
  tests/unit/test_evttc_competition_submission.py `
  tests/unit/test_evttc_competition_audit.py
```

The sealed CUDA parity test additionally requires the `EVTTC_CANDIDATE_ROOT`
environment variable to identify the preserved candidate. It uses controlled
synthetic tensors and compares the new adapter boundary with the unchanged
native model core. It is separate from scientific benchmark inference.

## Artifacts

- `RUN_FREEZE.json`: immutable model, data, calibration, query, policy, and source identities.
- `all_rows.jsonl`: one record per original query, including support diagnostics and timings.
- `results/<sequence>/results.txt`: index, original timestamp, signed TTC, and measured cost.
- `results.zip`: the ten result files, generated only for a complete finite population.
- `summary.json`, `manifest.json`, `environment.json`: coverage, package metadata, and environment.
- `independent_audit.json`: independent consistency checks and descriptive statistics.

Unavailable queries remain in `all_rows.jsonl`. A partial population never
produces `results.zip`; any finite-only export is explicitly marked partial.

The cost column includes sensor reading, representation preparation, all producer
and fusion stages, and completed host output. One-time source hashing, indexing,
calibration, and model loading are excluded. Internal component timers are host
spans, not isolated CUDA kernel timings.

This offline adapter reconstructs each query's selected history and recomputes
its producer observations. It does not reuse producer features across queries.
Its measured costs therefore describe this execution path, not the separate
optimized streaming runtime or a camera acquisition-to-response benchmark.

## Interpretation and submission

The supplied CSV rows are used as the prediction population: 365 road rows plus
51 Slider-750 and 18 Slider-1000 rows. Their original indices and timestamp text
are preserved. Early event windows use the actual shorter recorded prefix, and
Slider RGB uses actual frame intervals. Empty observed windows remain valid
inputs with explicit diagnostics; finite predictions do not establish reliable
sensor support. At startup, the fixed four event anchors each use the actual
prefix available at that anchor. Mathematically unrepresentable expert outputs
reject the whole observation with recorded evidence. Valid earlier observations
remain eligible; if no valid event observation remains, the query is unavailable.
This admission rule does not use confidence flags or test accuracy.

The cross-camera crop projection omits translation and is therefore approximate.
This adapter changes sensor preprocessing while keeping model weights and the
100 ms phase readout fixed. GarlTTC performance does not establish EvTTC accuracy.

The official submission format does not publicly specify whether negative TTC
values are accepted, whether every bbox row is a required query, or whether a
startup exclusion is allowed. The package preserves all finite signed model
outputs without retrospective selection or clipping. Confirm these server
details through the organizer's submission process.

Generating files does not apply for access or upload them. Official accuracy and
leaderboard placement require evaluation by the EvTTC organizers.

Official sources:

- [Competition inputs and submission entry](https://nail-hnu.github.io/EvTTC/competition/)
- [Data and result format](https://nail-hnu.github.io/EvTTC/download/data_format/)
- [Calibration](https://nail-hnu.github.io/EvTTC/calibration/)
