# Frozen TRAIN40 versus Garl: EvTTC transfer

Descriptive evaluation on previously observed EvTTC; not an official score or SOTA claim.

## historical46

46/46 queries have valid TTC.

| Model | Coverage | Pooled RTE % | Pooled MAE s |
|---|---:|---:|---:|
| PHASE_ONLY_TRAIN40_S7 | 46/46 | 7.7478 | 0.1409 |
| Garl_full | 46/46 | 13.3273 | 0.2501 |
| E_CTX_TRAIN40 | 46/46 | 15.8221 | 0.2864 |
| R_CTX_TRAIN40 | 46/46 | 6.3873 | 0.1127 |

Paired common finite subset: 46 queries; 0 excluded because a route has no finite output. These are conditional scores, not full-cohort scores.

| Model | Paired pooled RTE % | Paired pooled MAE s |
|---|---:|---:|
| PHASE_ONLY_TRAIN40_S7 | 7.7478 | 0.1409 |
| Garl_full | 13.3273 | 0.2501 |

Equal-sequence macro RTE difference (TRAIN40 minus Garl): -6.2301 percentage points; whole-sequence bootstrap 95% interval [-8.6816, -4.7667]. Exploratory; negative favors TRAIN40.

## extended_new

468/508 queries have valid TTC.

| Model | Coverage | Pooled RTE % | Pooled MAE s |
|---|---:|---:|---:|
| PHASE_ONLY_TRAIN40_S7 | 468/468 | 41.2267 | 1.0227 |
| Garl_full | 468/468 | 140.0689 | 3.2576 |
| E_CTX_TRAIN40 | 468/468 | 65.6955 | 1.4514 |
| R_CTX_TRAIN40 | 468/468 | 38.5555 | 1.0154 |

Paired common finite subset: 468 queries; 0 excluded because a route has no finite output. These are conditional scores, not full-cohort scores.

| Model | Paired pooled RTE % | Paired pooled MAE s |
|---|---:|---:|
| PHASE_ONLY_TRAIN40_S7 | 41.2267 | 1.0227 |
| Garl_full | 140.0689 | 3.2576 |

Equal-sequence macro RTE difference (TRAIN40 minus Garl): -94.8312 percentage points; whole-sequence bootstrap 95% interval [-210.5034, -14.4910]. Exploratory; negative favors TRAIN40.

## Interpretation limits

- **GT_gate:** Both complete seals, exact query IDs, manifest SHA and model bindings before TTC reads
- **GT_matching:** Nearest public ttc.csv row, maximum 50 ms, earlier timestamp wins ties; zero/nonfinite excluded
- **RTE_pct:** 100 * abs(prediction-target) / abs(target); signed nonzero targets, no clipping
- **macro:** Equal weight per physical sequence; pooled gives equal weight per query
- **bootstrap:** 2000 whole physical sequence draws seed 7, macro paired difference, exploratory
- **cohorts:** historical46 contains shared anchors; extended_new excludes all overlaps; no merged primary score
- **bands:** negative<0; (0,3); [3,6); [6,10]; (10,infinity). Descriptive RTE bins, NOT official MiD bins
- **missing_predictions:** Coverage retained; complete-cohort score withheld; finite and paired subsets labeled conditional
- **inputs:** Own RGB+events uses native causal sensor history up to 650 ms; Garl uses two RGB frames and two event segments dt 0.1 s
- **ROI:** Public annotated boxes and declared adapters, not detector outputs; shared query ROI differs from native per-anchor ROI
- **transfer:** EvTTC previously observed; descriptive transfer, not blind confirmation or exact paper Table VI reproduction
- **timing:** Raw instrumented wall seconds include preprocessing/caches; no isolated kernel or head-to-head latency claim
- **precision:** Both routes use float32 tensors. Own explicitly disables matmul/cuDNN TF32; Garl preserves backend defaults. Not a matched backend precision benchmark.
- **scope:** No official benchmark score, no SOTA claim, no new fitting or checkpoint selection
