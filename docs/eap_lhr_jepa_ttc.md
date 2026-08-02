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

## v3 hardening protocol

The v3 hardening patch does not add another Geo2 encoder branch. It first makes
the existing experiment attributable and auditable:

* official GarlTTC endpoint pairs are selected at 100 ms with a declared tolerance;
* `track_age` is computed from the first timestamp of the same track;
* visibility is consistently clipped-area/raw-area;
* the JEPA predictor receives only ROI/full-frame information at `t1`, a causal
  `t0→t1` box-motion context, and the requested horizon; it never receives `t2`;
* bbox-motion is an explicit ablation rather than an inseparable input;
* sampling is hierarchical across stratum, track and state;
* zero-shot artifacts contain sample identities and predictions for disjoint OOF
  aggregation and paired sequence-cluster bootstrap;
* FP16 uses GradScaler and resume restores optimization/RNG/early-stopping state.

Geo2 retains its full-frame architecture in this patch. Adding an ROI tower and
category objective to Geo2 is reserved for a later, separately named ablation so
that any gain cannot be conflated with these correctness fixes.

## v4 cache-free status (2026-08-02)

The v2/v3 launchers are retired. The active event-only path is:

```text
scripts/train_e_jepa_tubelet_lhr.py
scripts/run_e_jepa_garl_final.py
configs/experiment/e_jepa_garl_event_{screen,full}_v1.yaml
```

It reads raw eAP events on demand and does not require the approximately 455 GiB
dense cache. The full profile uses all valid rows, seeds 7/13/23 and a clean-tree
freeze. RGB-E and dense JEPA pretraining remain blocked and are rejected rather
than silently mapped to an event-only or pooled legacy model.

The only completed raw high-resolution smoke used 16/16 samples and obtained
validation sequence-macro MiD `1868.3186`. It is integration evidence only and
does not validate the architecture's scientific quality.

## Semantic representation decision

A dataset-free shortcut audit found that the current variance objective and
VISReg preserve a 12-bit slow nuisance despite healthy variance/effective rank.
R²-lite failed its predeclared TTC gate and is rejected for the production path.
Temporal residual prediction passed the slow-nuisance fixture but failed the
frame-varying control, so it may only be tested as an extra dynamic channel beside
the level embedding.

The next real experiment is therefore a matched
`level` versus `level+temporal_residual` high-resolution JEPA screen, followed by
frozen probes for expansion, event rate, sequence identity and TTC. No result from
the synthetic audit is itself evidence of improved eAP/Garl performance.
