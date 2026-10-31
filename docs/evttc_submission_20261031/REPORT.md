# PHASE-TTC-Adapt-20261031 — measured delivery report

Generated from frozen JSON/JSONL receipts; no table values are manually supplied.

Final model: 2,608 fit queries, 25 recordings, seeds 7/13/21, 688 parameters per router.
The eight competition road recordings and Slider-750/1000 are excluded from this fit.
Historical recipe selection and evaluation exposure remain; this is an adapted research result.
The original TRAIN40 producers are unchanged. No new model selection follows these scores.

## Submission completeness

434/434 finite outputs; 1 negative estimates.
Each original index/timestamp string and measured cost matches the inference ledger exactly.
Prediction ZIP SHA-256: `e7e41820003f18a28f2ef23e634992b05d7789ecf48364b2debe659cbb9a20e7`.
Model identity SHA-256: `fe1460c407ccd08fc670c1955435b0fe9b4719737e22e3dec7534a57e66bb005`.

## Local accuracy on supported targets

Relative error is mean |prediction-target|/target, expressed as a percentage.
The original and adapted columns below both use the organizer-email negative interpolation.
Raw and processed scores, plus geometry coverage, remain separately available in JSON.
These are NOT official EvTTC results: 11 road targets are unavailable locally and the original
Slider-750 public-target calculation does not reproduce the organizer's PDF.
CCRs1-low and CCRs1-medium also differ from the PDF; six other road scores reproduce
its rounded values. The organizer reply requests targets/clock mappings for all three.

| Population | Supported / all | Original error % | Adapted error % | Original MAE s | Adapted MAE s |
|---|---:|---:|---:|---:|---:|
| CCRm-low | 42/42 | 21.0100 | 6.9778 | 0.800198 | 0.279679 |
| CCRm-medium | 40/40 | 15.0546 | 10.1449 | 0.365496 | 0.248933 |
| CCRs1-high | 30/30 | 13.1178 | 9.4401 | 0.175780 | 0.137442 |
| CCRs1-low | 50/50 | 36.3885 | 35.9026 | 1.516138 | 1.481718 |
| CCRs1-medium | 32/43 | 70.5582 | 68.1662 | 0.993157 | 0.967736 |
| CCRs2-high | 59/59 | 7.9857 | 6.4514 | 0.148254 | 0.118841 |
| CCRs2-low | 46/46 | 15.2812 | 9.6247 | 0.439735 | 0.294427 |
| CCRs2-medium | 55/55 | 9.3429 | 7.1012 | 0.196866 | 0.143918 |
| Slider-1000 | 18/18 | 156.8813 | 48.9130 | 0.821204 | 0.356766 |
| Slider-750 | 51/51 | 283.9311 | 76.0714 | 4.544831 | 1.407064 |
| all | 423/434 | 58.9781 | 25.8452 | 1.057825 | 0.561545 |
| road | 354/365 | 21.5915 | 17.4363 | 0.567491 | 0.450145 |
| slider | 69/69 | 250.7877 | 68.9866 | 3.573450 | 1.133073 |

## Fit and validation

Resubstitution on the 2,608 fitted queries: 11.1586% relative error, 0.452961s MAE. This is training fit, not generalization.
Training reload/export exact on 2608 rows; final replay export exact on 434 rows.
Batch versus per-query FP32 maximum difference: 2.86102295e-06s.
Absent-history original-model fallback bitwise exact on 88744 source rows.
Clock shift and appended future leave outputs unchanged on 24 sampled recordings.

Cached raw-sensor perturbation diagnostics use 24 previously exposed recording midpoints;
the upstream producers are identical, so their frozen perturbed native outputs are reused.
This is a small robustness diagnostic, not an independent test or a new tuning phase.

| Perturbation | Original error % | Adapted error % |
|---|---:|---:|
| background_5 | 28.2366 | 11.3854 |
| clean | 28.2881 | 11.3983 |
| drop_70 | 29.9054 | 12.0180 |
| drop_negative | 31.1322 | 11.8184 |
| drop_positive | 30.4897 | 12.0047 |
| jitter_1000 | 28.2651 | 11.4088 |

## Measured runtime

GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU; Python 3.11.15; PyTorch 2.11.0+cu128; CUDA 12.8.
CPU: AMD Ryzen 9 9955HX, 16 cores / 32 logical processors.
Two CPU threads for PyTorch; FP32; TF32 disabled. No CUDA Graph acceleration claimed.

| Stage | Mean seconds/query |
|---|---:|
| preparation_seconds | 1.158298 |
| native_graph_seconds | 0.580139 |
| geometry_router_seconds | 0.001853 |
| cost_time_seconds | 1.740290 |

Original submission observed mean: 3.268250s/query.
This cross-run timing comparison is descriptive, not a controlled cache ablation.
Costs include sensor/cache preparation and synchronized inference; one-time loading,
hashing/indexing and ledger writes are outside the query boundary. This pipeline is not
claimed real-time. The lossless raw cache is bounded to 64 MiB events and 16 RGB frames.

## Unresolved limits and organizer action

- The adapted result is historically exposed; do not claim independent SOTA.
- Training native-camera features and competition right-camera features differ.
- The sensor projection omits translation; Slider rotation is provisional.
- No calibrated interval is carried after this final fit.
- Nominal Slider lens millimetres are not established; exact pixel intrinsics are provided.
- Ask organizers for Slider-750 aligned targets/clock to reconcile the original PDF.
- Send the two versions' descriptions separately; official evaluation/listing is pending.

No email was sent and no external upload was performed.
