# eAP Geometry-v2 + Official GarlTTC LHR Object-JEPA TTC (v2)

This patch adds opt-in research arms without replacing the legacy `ssl`, `geo`
or `ttc` controls.

## Scientific boundary

The LHR-v2 path **requires**:

- `GarlTTC-dataset/data/train.parquet`;
- `GarlTTC-dataset/annotations/train.parquet`;
- the audited five-key join already used by `garlttc_eap.py`;
- official `ttc`, `box3d_h`, `box3d_Fcam` and `K_event` targets.

There is no fallback to TTC reconstructed from public eAP 3-D tracks.

## No privileged inputs

The estimator receives only:

- two full-frame event endpoints;
- two object event ROI endpoints;
- causal 2-D box-motion features;
- the endpoint time gap;
- optional RGB endpoints.

The following are supervision-only and are forbidden as model inputs:

- TTC;
- depth or `box3d_Fcam`;
- 3-D closing speed;
- geometry-v2 targets;
- category labels;
- foreground masks.

`geometry_v2_target` is predicted from visual features; it is never encoded
and fed back into the fusion network.

## Architecture

1. Full-frame EventTubeletTransformer encodes scene context and global change.
2. Shared ROI encoder processes object endpoints.
3. A motion encoder processes only observable 2-D box history.
4. The object-JEPA predictor estimates the second ROI target embedding from:
   first ROI + full context + global change + motion + delta-time.
5. A retained head predicts visible heights, LHR TTC and a bounded residual.
6. Geometry/category are auxiliary targets only.
7. The EMA target ROI encoder and JEPA predictor are removed at inference;
   the entire TTC estimator is retained for zero-shot EvTTC.

## Recommended ablations

- L0: height + ratio only.
- L1: L0 + official TTC log loss and bounded residual.
- L2: L1 + geometry-v2/category auxiliary supervision.
- L3: L2 + object-JEPA.
- RGB is a later isolated ablation.
- Foreground remains disabled unless real teacher/SAM masks are supplied.

Use grouped validation for selection. Do not tune on sealed test sequences.
