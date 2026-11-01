# PHASE-TTC startup-router handoff — 2026-10-31

This iteration delivers an additional trained candidate, not a replacement for
the previous robust default. Read the generated
[report and scope](evidence/phase_startup/REPORT.md),
[result arrays summarized as JSON](evidence/phase_startup/RESULTS.json), and
[independent review](evidence/phase_startup/REVIEW.md).

Local artifacts are under `output/phase_startup/`:

- `training/`: rejected seconds-coordinate trial, every checkpoint retained.
- `training_rate/`: inverse-coordinate controls, family-held-out predictions,
  selected three-seed development model, calibration and exposed confirmation.
- `all_data_refit/`: three seeds fitted on every eligible label; descriptive fit
  metrics, no independent test or validated interval remains for these weights.
- `FINAL_MODEL/`: portable upstream plus `evaluated` and `all_data_refit` router
  variants. `ROUTER_BINDING.json` verifies identity, code, calibration and exports.
- `PRE_REVIEW_MODEL/`: historical intermediate, with its exact implementation
  snapshot; use `FINAL_MODEL/` for current inference.
- `validation/` and `validation_v2/`: original failed stale-history stress test
  and corrected freshness-contract results; neither result is hidden.
- `raw_robustness/`: all 144 raw perturbation predictions, including the three
  native expert outputs, for future fusion experiments without sensor decoding.
- `verification/`: exact population reconstruction, repeated deterministic fold,
  sensor replay differences and current-runtime raw-cache checks.
- `report/`, `AUDIT.json`, `FINAL_CHECKS.json`: generated results, evaluator and
  synthetic parallax audit, complete release verification.

Use Python3.11 and the repository dependencies. The current machine's old `.venv`
is not the working environment. For local reproduction in PowerShell:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:UV_PROJECT_ENVIRONMENT=(Resolve-Path .venv-transfer).Path

.venv-transfer/Scripts/python.exe -m scripts.predict_phase_startup --help
.venv-transfer/Scripts/python.exe -m scripts.predict_phase_startup --bundle output/phase_startup/FINAL_MODEL --contract native_road --manifest SENSOR_ONLY.json --device cuda --output NEW_REPLAY.json
.venv-transfer/Scripts/python.exe -m scripts.predict_phase_startup --bundle output/phase_startup/FINAL_MODEL --contract slider --sensors SENSOR_ROOT --sequence Slider-750 --device cuda --output NEW_SLIDER.json
```

The default variant inside this candidate bundle is `evaluated`; add
`--variant all_data_refit` for the fully fitted variant. This CLI default does
not promote the candidate above the previous release. Native road and Slider
preprocessing contracts differ. Inference rejects target-bearing manifests;
provided bounding boxes must be available at their timestamps.

The final ZIP includes the small prior feature/metadata dependencies required to
rerun grouped training. Its `START_HERE.txt` contains the relative-path command.
Raw sensor replay and the 88744-row source fallback check require the original
licensed sensor data/cache; raw streams are not redistributed.

Remaining scientific requirements cannot be supplied by another fit on the same
exposed recordings: official per-query750 targets/evaluator, authoritative or
independently measured inter-camera correspondence, and fresh independent
acquisitions including Slider. The coordinate experiment does not establish that
JEPA or encoder retraining is necessary, and does not claim those experiments ran.

Keep the research branch local. No merge, publication, remote training or
benchmark submission was requested for this iteration.
