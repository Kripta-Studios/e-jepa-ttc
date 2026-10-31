# PHASE-TTC: robust quality fusion, 2026-10-31

Two trained variants are delivered. `evaluated` is the fixed three-seed candidate tested after selection. `all_data_refit` uses all 2,677 eligible labeled queries across 24 road and three Slider recordings; its scores are training resubstitution and it has no remaining independent confirmation.

The new trainable component is a three-seed ensemble of 401-parameter quality gates. The original nine-component PHASE graph is frozen. Geometry uses provided boxes over at most 650 ms, actual timestamps and a robust inverse-scale fit. No target depth, TTC or future observation enters inference.

## Recomputed results

| Population / role | Model | Rows | Finite | Relative absolute error (%) | MAE (s) |
|---|---|---:|---:|---:|---:|
| road / grouped development OOF | F0 | 1747 | 1747 | 27.199 | 0.6963 |
| road / grouped development OOF | quality | 1747 | 1747 | 12.834 | 0.4224 |
| road / grouped development OOF | native | 1747 | 1747 | 22.081 | 0.7544 |
| slider / grouped development OOF | F0 | 77 | 76 | 83.887 | 1.1782 |
| slider / grouped development OOF | quality | 77 | 77 | 41.344 | 0.4938 |
| slider / grouped development OOF | native | 77 | 77 | 35.009 | 0.4746 |
| CCRm / current-iteration fit/selection reserve; historical exposure | F0 | 580 | 580 | 16.749 | 1.0830 |
| CCRm / current-iteration fit/selection reserve; historical exposure | candidate | 580 | 580 | 15.662 | 1.0197 |
| Slider-750 / public GT joined after prediction freeze; historical benchmark exposure | F0 | 51 | 51 | 282.476 | 4.5395 |
| Slider-750 / public GT joined after prediction freeze; historical benchmark exposure | candidate | 51 | 51 | 168.038 | 3.3157 |
| Slider-750 / public GT joined after prediction freeze; historical benchmark exposure | geometry | 51 | 49 | 18.077 | 0.1976 |
| Slider-1000 / public GT joined after prediction freeze; historical benchmark exposure | F0 | 18 | 18 | 156.876 | 0.8212 |
| Slider-1000 / public GT joined after prediction freeze; historical benchmark exposure | candidate | 18 | 18 | 38.419 | 0.2757 |
| Slider-1000 / public GT joined after prediction freeze; historical benchmark exposure | geometry | 18 | 16 | 7.694 | 0.0374 |
| all / all-data resubstitution | F0 | 2677 | 2676 | 34.986 | 0.9289 |
| all / all-data resubstitution | candidate | 2677 | 2677 | 13.129 | 0.4779 |
| road24 / all-data resubstitution | F0 | 2531 | 2531 | 27.664 | 0.8494 |
| road24 / all-data resubstitution | candidate | 2531 | 2531 | 11.289 | 0.4654 |
| slider_all / all-data resubstitution | F0 | 146 | 145 | 162.797 | 2.3162 |
| slider_all / all-data resubstitution | candidate | 146 | 146 | 45.036 | 0.6933 |

## What the evidence supports

Quality information improved road grouped validation beyond an equal-parameter control. Both controls use the geometric estimate; this isolates gate-quality information, not geometry availability. Seeds 7, 13 and 21 are all retained and averaged. Selection required improvements in both road macro-record RAE and macro-record MAE, plus Slider500 RAE and MAE. It did not select on the reserves.

CCRm has four recording groups but is one scenario family. Its paired recording bootstrap interval for the relative-error change is [-2.061, 0.290] percentage points and includes zero. These are historically exposed data; this is not a claim of statistically established independent superiority.

Public Slider750/1000 boxes exactly match the local query CSVs. Targets join by exact frame identity and sensor clock within 2 microseconds. These are recomputed local public-GT scores, not a new official benchmark receipt. In particular, Slider750's local F0 score differs from the previously supplied official aggregate. Do not splice the old aggregate and the new candidate score into one comparison.

## Limitations that remain visible

Slider750 startup is still poor: the first two queries lack three causal boxes, and inconsistent neural signs can remain in the conservative native fallback. Full-population scores retain those failures. The standalone geometric observer is also reported and can beat the learned fusion on Slider. Provided annotation boxes are an oracle-input contract; detector-generated boxes were not evaluated. Event/RGB extrinsics remain provisional rotation-only on Slider. FCWD cannot support the full RGB-event comparison without an audited spatial mapping, so no such result is manufactured.

The 80% interval for `evaluated` uses separate CPNAO calibration records, with a fixed symmetric radius of 1.1318 s. CCRm coverage is 79.83%. This is descriptive calibration, not a safety or exchangeability guarantee. The interval must not be presented as validated for `all_data_refit`, which trained on those calibration labels.

Without geometry, 88,744 eAP native predictions are preserved bitwise. Source eAP with full box histories remains unevaluated. No end-to-end encoder training or JEPA benefit is claimed: the demonstrated intervention is quality-aware physical fusion.

## Runtime and verification

The evaluated CPU gate batch-one median was 1.273 ms; geometry plus gate median was 9.023 ms. These exclude upstream sensor decoding and neural inference. Cached external replay keeps exact prepared tensors and bounded memory; raw road replay uses the previously verified CUDA-graph producer runtime. The tiny gate trains from preloaded features on CPU; CUDA graphs would not remove sensor preprocessing cost.

Both TorchScript exports match their PyTorch models exactly. Checkpoints, configurations, source snapshots, normalizers and parent graph hashes are included. `report/VERIFICATION.json`, `validation/VALIDATION.json` and `raw_robustness/REPORT.json` contain machine-readable checks. Full test and independent review results are recorded separately after they finish.

## Reproduction

Run from the repository root, with Python 3.11 and the locked project dependencies:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
python -m scripts.predict_phase_robust --bundle output/phase_robust/FINAL_MODEL --sensors ../EvTTC-Benchmark --sequence Slider-1000 --device cuda --variant evaluated --output predictions.json
```

`--variant all_data_refit` selects the final all-data weights. The default remains the evaluated candidate. The prediction CLI never opens a target file and reports missing-history/sign-conflict flags.

The command above is restricted to Slider: it uses the audited competition adapter and event-plane provided boxes. For native road data use `scripts.predict_phase_robust_native` with `--manifest` pointing to a sensor-only native manifest. Its RGB boxes, timestamps and upstream preprocessing match the road evaluation. Target fields and target-file paths are rejected. `NATIVE_PARITY.json` records an exact raw replay against the cached candidate.

`FUSION_BINDING.json` binds named evaluated/refit variants, exports and evaluated calibration to the upstream bundle. The loader rejects a swapped variant even if its internal checkpoint hashes are otherwise consistent.

## Long-TTC tail

For the 439 grouped-CV road queries above4s, MAE changes from 0.8947 to 0.8825s; RMSE changes from 1.2275 to 1.4351s. The latter worsens, so the candidate does not dominate every tail metric. Full bucket results are in SUPPLEMENT.json.

## Completed delivery checks

Full suite: 7309 passed, 46 skipped, 17 additional subtests, no failures or errors. Final focused suite: 14 passed. Ruff and Pyright pass. The one fresh review found two deployment/identity issues; both were fixed and verified by contract/tamper tests and exact native raw replay. See FINAL_CHECKS.json for the full-suite versus final-fix verification scope.
