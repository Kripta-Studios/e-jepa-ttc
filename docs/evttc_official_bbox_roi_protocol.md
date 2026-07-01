# EvTTC Official BBox/ROI Comparison Protocol

Checked on 2026-07-02. This is a protocol note, not a SOTA claim.

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

Local official-Table-V asset coverage:

| Sequence family | Local status | Use in current sealed split |
| --- | --- | --- |
| `CCRs1-low/medium/high` | Present with HDF5, `ttc.csv`, `gt.hdf5`, and `leftlabel` | Train only |
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

This is detection-assisted and frame-label-only. It uses current and past bbox
scale, with train-only calibration, and reports TTC error in seconds. It is not
the official CMax/STRTTC metric table and must not be compared as if it were.

## Compatibility Matrix

| Requirement | Official CMax/STRTTC-style benchmark | Current repo status |
| --- | --- | --- |
| Inputs | Event stream cropped/selected by bbox or ROI | All-window JEPA exists; bbox geometry exists |
| Baselines | STRTTC, CMax, ETTCM, FAITH, AEB-Tracker, Image FoE | Only causal bbox geometry is implemented locally |
| Sequences | CCRs1/CCRs2/CCRm plus slider testbed in paper table | Full starter uses CCRs-side and CPLA validation/test |
| Metrics | Published benchmark metric plus runtime, averaged over runs | MAE/RMSE seconds for local regression outputs |
| Test discipline | Fixed benchmark sequences | Current test remains sealed; tuning only by validation |
| Claim allowed now | No | No official SOTA or SOTSA claim yet |

## Required Next Work For A Real Comparison

1. Download or recover complete bbox/segmentation assets for the official Table V
   real-world sequences: CCRs1, CCRs2, and CCRm at the required speeds.
2. Add the slider-testbed data if the goal is to reproduce the complete Table V
   including `Slider-750` and `Slider-1000`.
3. Implement or wrap the official STRTTC MATLAB code and a CMax baseline under a
   deterministic CLI, recording runtime and random seeds.
4. Add an ROI event extractor that uses only current/past bbox information at
   inference time and never future boxes.
5. Evaluate every method on the same frames/events, same sequence list, same TTC
   alignment, and same metric.
6. Keep model selection on validation. Run the sealed test once per frozen
   protocol and do not change hyperparameters after seeing it.

Until those items are done, the correct comparison is:

> Our best local all-window JEPA result is strong on the sealed starter split,
> but it is not directly comparable with EvTTC Table V CMax/STRTTC results.
> The only bbox/ROI result currently implemented is a causal geometry reference,
> useful for protocol plumbing, not for claiming official SOTA.
