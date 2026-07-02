# EvTTC Official BBox/ROI Comparison Protocol

Checked on 2026-07-03. This is a protocol note, not a SOTA claim.

## Official Reference

Primary sources:

- EvTTC paper: https://arxiv.org/html/2412.05053v1
- Event-Aided TTC / STRTTC paper: https://arxiv.org/abs/2407.07324
- Event-Aided TTC project and code: https://nail-hnu.github.io/EventAidedTTC/
  and https://github.com/NAIL-HNU/event_aided_ttc/

EvTTC computes ground-truth TTC from LiDAR depth plus GNSS/INS vehicle motion in
the calibrated camera frame. Its benchmark section evaluates STRTTC, CMax,
ETTCM, FAITH, AEB-Tracker, and Image FoE, averaged over multiple runs.

The published EvTTC Table V reports benchmark sequences:

- `CCRs1-low`, `CCRs1-medium`, `CCRs1-high`
- `CCRs2-low`, `CCRs2-medium`, `CCRs2-high`
- `CCRm-low`, `CCRm-medium`
- `Slider-750`, `Slider-1000`

The same section states that CMax improves accuracy by using all events inside
the bounding box, at the cost of nonlinear least-squares runtime. That means
official comparison is a bbox/ROI-assisted event protocol, not the all-window
event-only protocol used by the current JEPA TTC runs.

## Current Local Status

Current local sealed full-starter split:

- Train: CCRs-1 plus CCRs-side-low/medium plus CPLA-low/medium
- Validation: `CCRs-side-high`
- Sealed test: `CPLA-high`

Local official-Table-V asset coverage is now checked by code:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\e-jepa-ttc.exe data official-coverage --root datasets\evttc --output artifacts\metrics\evttc_official_table_v_coverage.json
```

Latest local coverage result from that command:

- scanned local EvTTC sequences: `9`;
- official real-world CCRs1/CCRs2/CCRm coverage: `3/8` complete, `37.5%`;
- complete Table V coverage including slider rows: `3/10` complete, `30.0%`;
- `official_sota_claim_allowed`: `false`.

| Sequence family | Local status | Use in current sealed split |
| --- | --- | --- |
| `CCRs1-low/medium/high` | Present with HDF5, `ttc.csv`, `gt.hdf5`, and `leftlabel`; auto-check complete | Train only |
| `CCRs2-low/medium/high` | Missing locally | Not available |
| `CCRm-low/medium` | Missing locally | Not available |
| `Slider-750/1000` | Missing locally | Not available |

Therefore a partial CCRs1-only reproduction is possible as a smoke check, but
not as an official comparison: the current protocol has already used CCRs1 for
training, and the remaining official benchmark sequences are not present.

Current local bbox/ROI result:

| Method | Split | Labels | Predictions | Metric |
| --- | --- | ---: | ---: | --- |
| Causal bbox geometry | train bbox frames | 871 | 856 | MAE 0.512 s, RMSE 1.007 s |
| Causal bbox geometry | validation bbox frames | 108 | 106 | MAE 0.279 s, RMSE 0.538 s |
| Causal bbox geometry | sealed CPLA-high bbox frames | 83 | 81 | MAE 0.157 s, RMSE 0.331 s |
| ROI event ridge | train bbox frames | 871 | 871 | MAE 0.659 s, mean relative error 19.15% |
| ROI event ridge | validation bbox frames | 108 | 108 | MAE 0.293 s, mean relative error 13.63% |
| ROI event ridge | sealed CPLA-high bbox frames | 83 | 83 | MAE 0.829 s, mean relative error 47.12% |
| ROI latent prober | validation bbox frames | 108 | 98 | MAE 0.226 +/- 0.015 s, mean relative error 10.78 +/- 0.53% |
| ROI latent prober | sealed CPLA-high bbox frames | 83 | 73 | MAE 0.423 +/- 0.029 s, mean relative error 23.41 +/- 2.23% |

These are detection-assisted and frame-label-only. `causal-geometry` uses
current and past bbox scale with train-only calibration. `roi-events` uses the
current object box to crop only past/current events in a 100 ms window and fits a
train-only ridge regressor. `roi-latent-prober` loads frozen JEPA and prober
checkpoints selected before test evaluation, then evaluates matched bbox/cache
rows without retraining. They are not the official CMax/STRTTC metric table and
must not be compared as if they were.

## Compatibility Matrix

| Requirement | Official CMax/STRTTC-style benchmark | Current repo status |
| --- | --- | --- |
| Inputs | Event stream cropped/selected by bbox or ROI | All-window JEPA exists; bbox geometry and causal ROI event extraction exist |
| Baselines | STRTTC, CMax, ETTCM, FAITH, AEB-Tracker, Image FoE | Causal bbox geometry and ROI event ridge are implemented locally |
| Sequences | CCRs1/CCRs2/CCRm plus slider testbed in paper table | Full starter uses CCRs-side and CPLA validation/test |
| Metrics | Relative TTC error plus runtime, averaged over runs | MAE/RMSE seconds and relative error % for local regression outputs |
| Test discipline | Fixed benchmark sequences | Frozen final checks only; tuning only by validation |
| Claim allowed now | No | No official SOTA or SOTSA claim yet |

## Required Next Work For A Real Comparison

1. Download or recover complete bbox/segmentation assets for the official Table V
   real-world sequences: CCRs1, CCRs2, and CCRm at the required speeds.
2. Add the slider-testbed data if the goal is to reproduce the complete Table V
   including `Slider-750` and `Slider-1000`.
3. Implement or wrap the official STRTTC MATLAB code and a CMax baseline under a
   deterministic CLI, recording runtime and random seeds.
4. Promote the current `roi-events` extractor from ridge smoke baseline to a
   shared official-protocol data adapter for CMax/STRTTC wrappers.
5. Evaluate every method on the same frames/events, same sequence list, same TTC
   alignment, and same metric.
6. Keep model selection on validation. Run the sealed test once per frozen
   protocol and do not change hyperparameters after seeing it.

## Exact Download Checklist

Minimum assets for an official-style bbox/ROI reproduction:

| Sequence | Required assets | Local status |
| --- | --- | --- |
| `CCRs-1-low-100%` | already local: `hdf5`, `gt-ttc`, `bbox/leftlabel` | present |
| `CCRs-1-medium-100%` | already local: `hdf5`, `gt-ttc`, `bbox/leftlabel` | present |
| `CCRs-1-high-100%` | already local: `hdf5`, `gt-ttc`, `bbox/leftlabel` | present |
| `CCRs-2-low-100%` | `hdf5`, `gt-ttc`, `bbox-segmentation` | missing |
| `CCRs-2-medium-100%` | `hdf5`, `gt-ttc`, `bbox-segmentation` | missing |
| `CCRs-2-high-100%` | `hdf5`, `gt-ttc`, `bbox-segmentation` | missing |
| `CCRm-low-100%` | `hdf5`, `gt-ttc`, `bbox-segmentation` | missing |
| `CCRm-medium-100%` | `hdf5`, `gt-ttc`, `bbox-segmentation` | missing |

Do not download these for the first official bbox/ROI pass unless a baseline
explicitly needs them:

- `video`
- `raw bag`
- `gt-depth`
- `download-full-sequence`

They are useful for visualization, ROS replay, or depth-supervised experiments,
but STRTTC/CMax-style TTC reproduction needs event HDF5, TTC labels, and object
ROI/bbox annotations first.

Optional later assets:

- `Slider-750`
- `Slider-1000`

Those are needed only for reproducing the complete Table V including the slider
testbed. They are not part of the current local EvTTC starter folder.

Until those items are done, the correct comparison is:

> Our best local all-window JEPA result is strong on the sealed starter split,
> but it is not directly comparable with EvTTC Table V CMax/STRTTC results.
> The local bbox/ROI results currently implemented are causal geometry,
> ROI-event ridge, and frozen JEPA ROI prober diagnostics, useful for protocol
> plumbing, not for claiming official SOTA.
