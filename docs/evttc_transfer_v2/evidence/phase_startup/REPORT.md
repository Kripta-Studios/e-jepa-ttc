# PHASE-TTC startup router — 2026-10-31

Additional trained candidate; previous robust release remains the default. Inverse-rate mixing improves aggregate road OOF slightly and substantially improves Slider fitting, but does not dominate confirmation MAE or raw perturbations.

## Protocol

Frozen nine-component TRAIN40 upstream; new 688-parameter four-expert router, three seeds 7/13/21 (2064 learned parameters in ensemble). Causal boxes <=650ms; two frames >=10ms; stale endpoint >150ms or no history returns exact F0. Signed inverse rates are bounded to TTC magnitude [.1,60]s; exact rate cancellation falls back to F0. Opposite-sign experts can recover a failed component; no target or sequence identity enters the gate. Native road uses left RGB/history; Slider uses the competition adapter. These contracts are not interchangeable.

Development: 1747 road + 146 Slider = 1893 rows; five road families plus one Slider family. Each CV fold excludes the complete family. Current fit/selection excludes CCRm 580 and CPNAO 204. Every dataset is historically exposed. No clean external test. Compared with the previous gate, development now includes Slider750/1000; the old/new difference cannot be attributed solely to architecture. Both coordinates run Q/QA/NAcap with equal 688-parameter capacity and seeds 7/13/21, 300 updates, fixed AdamW .01/.01. Train-only normalization and family-balanced loss. 117 experiment fits total including both trials and all-data refit; none discarded.

The seconds-coordinate trial is rejected: road QA 15.2286%/0.541610s versus previous quality 12.8336%/0.422359s. Inverse QA is retained as a candidate only. Its raw-event midpoint stress test is worse than the previous gate; see JSON. A failed bbox-dropout test (46.4918%) exposed stale-history routing. The explicit 150ms fallback fixes that contract (28.2881%, identical F0). This stress test has now informed implementation and is not independent validation.

Table metrics pool all queries; model selection uses macro-record metrics in JSON.

| Population | Model | RAE % | MAE s | Coverage |
|---|---|---:|---:|---:|
| road_development_1747 | F0 | 27.1994 | 0.696295 | 1747/1747 |
| road_development_1747 | previous_quality_oof | 12.8336 | 0.422359 | 1747/1747 |
| road_development_1747 | router_inverse_oof | 12.6513 | 0.404882 | 1747/1747 |
| CCRm_580 | F0 | 16.7495 | 1.082968 | 580/580 |
| CCRm_580 | previous_quality | 15.6624 | 1.019652 | 580/580 |
| CCRm_580 | router_inverse | 15.1910 | 1.035545 | 580/580 |
| Slider-500 | F0 | 83.8874 | 1.178208 | 76/77 |
| Slider-500 | previous_quality | 29.3319 | 0.293020 | 77/77 |
| Slider-500 | router_all_slider_held_out | 29.9771 | 0.508801 | 77/77 |
| Slider-500 | router_fit | 13.1971 | 0.200379 | 77/77 |
| Slider-500 | router_all_data_refit | 14.4482 | 0.240190 | 77/77 |
| Slider-500 | geometry_only | 19.4946 | 0.407528 | 76/77 |
| Slider-750 | F0 | 282.4763 | 4.539549 | 51/51 |
| Slider-750 | previous_quality | 168.0380 | 3.315706 | 51/51 |
| Slider-750 | router_all_slider_held_out | 107.9245 | 1.905037 | 51/51 |
| Slider-750 | router_fit | 24.9307 | 0.344790 | 51/51 |
| Slider-750 | router_all_data_refit | 25.4621 | 0.352490 | 51/51 |
| Slider-750 | geometry_only | 18.4346 | 0.209248 | 50/51 |
| Slider-1000 | F0 | 156.8756 | 0.821170 | 18/18 |
| Slider-1000 | previous_quality | 38.4193 | 0.275653 | 18/18 |
| Slider-1000 | router_all_slider_held_out | 45.4669 | 0.289550 | 18/18 |
| Slider-1000 | router_fit | 19.4623 | 0.144392 | 18/18 |
| Slider-1000 | router_all_data_refit | 26.1413 | 0.194144 | 18/18 |
| Slider-1000 | geometry_only | 8.6708 | 0.048472 | 17/18 |

Paired family bootstrap of road OOF change (percentage points): {'families': ['CCRs-1', 'CCRs-3', 'CCRs-side', 'CPLA', 'CPNA'], 'units': 5, 'replicates': 5000, 'mean_delta_pp': -0.18235650658607483, 'ci95_delta_pp': [-1.5175654888153076, 0.8001460433006287], 'scope': 'descriptive paired family bootstrap on exposed development, only five units'}.

## All-data weights and uncertainty

All-data refit uses all 2677 eligible rows (27 recordings), retains eight-overlap exclusion. Fit metrics: 10.9488% road and 19.7372% Slider. These are resubstitution, not generalization. Encoders/producers were not retrained. The all-data model returns no calibrated interval because calibration labels were fitted.

Evaluated-candidate interval: {'claim': 'descriptive separate-fit calibration, historically exposed/correlated groups, no certified guarantee', 'confirmation_coverage80': 0.8310344827586207, 'confirmation_mean_width_seconds': 2.8369923043849825, 'constant_radius': 1.1656296253204346, 'groups': ['CPNAO-high', 'CPNAO-low', 'CPNAO-medium'], 'scaled_factor': 2.482123154885643}. Calibration uses separate CPNAO, historical exposure and correlated sequences remain; coverage is descriptive, not a safety guarantee.

## Verification and runtime

Source fallback is bitwise equal on 88744 eAP examples without bbox history; no source-domain geometry transfer claim. TorchScript eager/reload parity is exact for both 2677-row variants. Perturbations include raw event loss/polarity loss/jitter/background and bbox jitter/dropout/quantization/missing. Raw robustness caches all three native experts for future reuse; avoids repeating sensor decoding when testing another fusion. Training uses cached features and two CPU threads; native raw replay retains cached input/producer acceleration. Gate CPU batch1 timing: {'median': 0.2857500221580267, 'p95': 0.2945400250609964} (not end-to-end sensor latency).

Fresh sensor replay is distinct from tensor-export parity: native query difference 0.000000954s; eight-query Slider replay max difference 0.003544450s. The new CLI disables TF32 while the earlier Slider cache inherited TF32 settings; this is a measured precision-contract difference, not bitwise sensor parity. The complete QA inverse seed7/CPLA-held-out refit reproduced every weight exactly.

## Deployment

Install repository dependencies; set PYTHONPATH=src;. on Windows. Run from repo root:

```powershell
python -m scripts.predict_phase_startup --bundle PATH_TO_BUNDLE --contract native_road --manifest SENSOR_ONLY.json --device cuda --output NEW.json
python -m scripts.predict_phase_startup --bundle PATH_TO_BUNDLE --contract slider --sensors SENSOR_ROOT --sequence Slider-750 --device cuda --output NEW.json
# Add --variant all_data_refit for fully fitted weights (no validated uncertainty).
```

Bundle binds every upstream checkpoint/normalizer, named variant, calibration, export and implementation hashes. Raw data are not included. Model inputs are native[B,3]=(F0,RGBcontext,eventcontext) seconds, geometry[B] seconds/NaN and quality[B,20] documented in geometry.py. Example tensor inputs/outputs in validation_v2.

## Unresolved evidence

Official PDF750 is 132.0000%, while original submitted predictions on public targets yield 282.4982%; 1000 reproduces 156.8813%. PDF contains no per-query aligned targets or evaluator. No offset, clipping or denominator was fitted to reconcile it. Audit JSON retains original file hashes and synthetic parallax at seven depths. Slider extrinsic remains a candidate, not authoritative; no ground-truth depth was injected into inference. New independent acquisitions/official evaluator are still needed; this does not establish SOTA. JEPA/encoder retraining is not justified by these fusion experiments alone and has not been claimed complete.
